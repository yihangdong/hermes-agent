"""Attempt-lifecycle regression tests for #97488 / #96775.

Pins the three attempt-level lifecycle guarantees added after PR #98628
collapsed lean compaction to one auxiliary request per attempt:

1. A ceiling/idle-timeout host TEARS DOWN its worker: a cooperative worker is
   joined within the bounded grace (releasing the lease normally), while an
   uninterruptible worker is orphaned behind the poison fence — its late
   result is discarded and, on the total-ceiling path, the durable lease is
   retained until it exits so no new attempt overlaps the unchanged session.
2. A failed/stalled/cancelled attempt records a durable per-session backoff
   (strategy + failure kind stamped into the state.db cooldown row) that
   SURVIVES a gateway restart; the next automatic turn skips the same
   strategy inside the window, and a successful compression clears it.
3. Late results from a superseded attempt are discarded, never committed
   over newer state (generation counter), and a transiently-blocked no-op is
   reported as a soft defer — never compression_exhausted (false auto-reset).
"""

from __future__ import annotations

import concurrent.futures
import copy
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import conversation_compression as cc
from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.conversation_compression import (
    CompressionCommitFence,
    _claim_compressor_attempt,
    compress_context,
    compression_blocked_transiently,
    compression_skipped_due_to_lock,
    run_compress_context_with_progress_timeout,
)
from hermes_state import SessionDB


def _build_agent(tmp_path: Path, session_id: str, db: SessionDB | None = None):
    if db is None:
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id, source="cli")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    agent.context_compressor.threshold_tokens = 1_000
    return db, agent


def _messages():
    return [{"role": "user", "content": f"m{i}"} for i in range(20)]


def _slow_pool_setup(delay: float):
    """Make the wrapper's lazy pool setup outlast a tiny test ceiling.

    Models the CI cold start (first ``tools.thread_context`` import + pool
    construction) that runs BEFORE the attempt epoch is established.
    """
    real = cc._get_compress_timeout_executor

    def _slow():
        executor = real()
        time.sleep(delay)
        return executor

    return patch.object(cc, "_get_compress_timeout_executor", _slow)


def _slow_context_capture(delay: float, *, done: threading.Event, threads: list):
    """Make the ACTUAL host-side context wrapper/callback capture slow.

    ``propagate_context_to_thread`` is resolved from ``tools.thread_context``
    at call time and runs on the HOST thread: it copies the current Context
    and lazily imports the terminal approval/sudo callback API. On a cold
    process that is not a constant-time lookup — it is local submission setup
    that must finish BEFORE the attempt epoch is armed. The real helper is
    still built and used (only its cost is modelled), and
    ``tools/thread_context.py`` itself is untouched.
    """
    import tools.thread_context as thread_context

    real = thread_context.propagate_context_to_thread

    def _slow(target):
        wrapper = real(target)
        time.sleep(delay)
        threads.append(threading.current_thread())
        done.set()
        return wrapper

    return patch.object(thread_context, "propagate_context_to_thread", _slow)


class _EpochOrderingFence(CompressionCommitFence):
    """Record whether host-side setup was complete when the epoch was armed.

    No budget is altered and nothing is inferred from wall clock:
    ``begin_attempt`` samples an event the capture control sets as it
    returns, so the ordering claim is decided by happens-before.
    """

    def __init__(self, setup_done: threading.Event) -> None:
        super().__init__()
        self._setup_done = setup_done
        self.setup_done_at_epoch: bool | None = None

    def begin_attempt(self, total_ceiling_seconds: float) -> float:
        self.setup_done_at_epoch = self._setup_done.is_set()
        return super().begin_attempt(total_ceiling_seconds)


class _ScheduledFence(CompressionCommitFence):
    """Fence double that makes this file's scheduling preconditions explicit.

    The production wrapper is untouched: only the values this test hands the
    host are controlled, and every configured budget is left exactly as is.

    * the host's FIRST idle sample blocks until the supplied worker is
      provably inside its loop, so no timeout branch can be classified
      against a worker that never ran;
    * ``deadline_exceeded`` is never true before entry, so the fence-gated
      pre-start refusal (``_CompressionWorkerPreStartExpiry``) and a
      pending-future cancellation cannot masquerade as a teardown; after
      entry it is the real value;
    * ``stale_progress_seconds=None`` reports continuous progress, so the
      idle window cannot expire and the TOTAL ceiling is the only branch the
      host can reach. A value ``>= idle`` reproduces the uncontrolled stale
      sample instead (falsifier only);
    * ``hold_inside_ceiling`` shifts ONLY the elapsed-time origin handed back
      to the host, so its total term cannot win the classification race
      (falsifier only; the fence's own deadline stays armed at the configured
      ceiling, and the host's degrade log then reports a negative elapsed
      reading — that is the control, not a production miscount).
    """

    def __init__(
        self,
        entered: threading.Event,
        *,
        stale_progress_seconds: float | None = None,
        hold_inside_ceiling: float = 0.0,
    ) -> None:
        super().__init__()
        self._entered = entered
        self._stale_progress_seconds = stale_progress_seconds
        self._hold_inside_ceiling = hold_inside_ceiling
        self.entry_observed = False

    def begin_attempt(self, total_ceiling_seconds: float) -> float:
        epoch = super().begin_attempt(total_ceiling_seconds)
        return epoch + self._hold_inside_ceiling

    @property
    def deadline_exceeded(self) -> bool:
        if self._hold_inside_ceiling or not self._entered.is_set():
            return False
        return super().deadline_exceeded

    def seconds_since_progress(self) -> float:
        if not self.entry_observed:
            # Barrier, not a sleep: the host cannot evaluate a timeout branch
            # until the supplied worker has entered. The bound only turns a
            # hang into a loud failure.
            if not self._entered.wait(timeout=5.0):
                raise AssertionError(
                    "supplied worker never entered — the timeout branch "
                    "would have been classified against a worker that "
                    "never ran"
                )
            self.entry_observed = True
        if self._stale_progress_seconds is not None:
            return self._stale_progress_seconds
        return 0.0


def _cooperative_worker(entered, joined, worker_done, original):
    """Build the cooperative worker with an EXPLICIT unwind handshake.

    Same shape as before — poll the poison fence between provider phases,
    then take real time to unwind — except the unwind is ordered against the
    host's own teardown instead of a wall-clock guess: the worker returns
    only after the host has entered ``_join_cancelled_worker``. A host
    WITHOUT the bounded-grace join therefore still returns first (the #97488
    sabotage check), and a host WITH it observes an exit that its own
    bounded grace caused.
    """

    def cooperative_worker(fence: CompressionCommitFence):
        entered.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fence.is_cancelled:
                break
            fence.touch_progress()
            time.sleep(0.01)
        joined.wait(timeout=5.0)
        worker_done.set()
        return (original, "late")

    return cooperative_worker


class TestWorkerTeardownOnCeiling:
    def test_cooperative_worker_joined_within_grace(self):
        """A worker that exits promptly after cancel is joined on the
        total-ceiling path; the lease is released normally (no retention) —
        the sabotage check for this test is removing the
        `_join_cancelled_worker` call, which makes
        `worker_done.is_set()` False when the host returns.

        Both preconditions the assertion depends on are PROVED here instead
        of inferred from 0.1/0.2s wall-clock scheduling: the fence blocks the
        host's first idle sample until the worker is inside its loop, and it
        reports continuous progress so the host can only classify the TOTAL
        ceiling (the idle path deliberately skips the join). The recorded
        join proves the worker was still LIVE when the existing bounded grace
        reaped it, so a pending cancellation, a pre-start expiry or an
        already-settled future cannot pass this test. The falsifier below
        exercises the uncontrolled schedule.
        """
        original = [{"role": "user", "content": "keep"}]
        entered = threading.Event()
        joined = threading.Event()
        worker_done = threading.Event()
        causes: list[tuple] = []
        join_calls: list[tuple] = []
        real_join = cc._join_cancelled_worker

        def _record_join(future, grace_seconds):
            # Sampled BEFORE the worker is released: a not-done future here
            # is proof the join faced a live worker, not a settled one.
            join_calls.append((future.done(), grace_seconds))
            joined.set()
            outcome = real_join(future, grace_seconds)
            join_calls[-1] += (outcome,)
            return outcome

        fence = _ScheduledFence(entered)
        with patch.object(cc, "_join_cancelled_worker", _record_join):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=_cooperative_worker(
                    entered, joined, worker_done, original
                ),
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.1,
                total_ceiling_seconds=0.2,
                fence=fence,
                on_timeout_cause=lambda exhausted, progressed: causes.append(
                    (exhausted, progressed)
                ),
                stall_fallback=False,
            )
        # Precondition: the worker entered before any branch was classified.
        assert entered.is_set() and fence.entry_observed
        # Precondition: the host took the TOTAL-ceiling branch, so the join
        # branch — not the idle skip-join branch — is the one under test.
        assert len(causes) == 1 and causes[0][0], (
            "host did not reach the total-ceiling branch — the teardown "
            "assertion below would have had no precondition"
        )
        # Precondition: exactly one join, against a still-running worker,
        # inside the UNCHANGED bounded grace.
        assert len(join_calls) == 1
        assert join_calls[0][0] is False, (
            "the join observed an already-settled future — pending "
            "cancellation or pre-start expiry, not a live teardown"
        )
        assert join_calls[0][1] <= cc._CANCELLED_WORKER_TEARDOWN_GRACE_SECONDS
        assert join_calls[0][2] is True, "the live worker was not reaped"
        # The bounded-grace join must have reaped the cooperative worker
        # BEFORE the host returned.
        assert worker_done.is_set(), (
            "host returned before tearing down a cooperative cancelled "
            "worker — bounded-grace join missing (#97488)"
        )
        # Whichever return path won the race (fallback via join, or the
        # worker's own return adopted inside the final wait slice), the
        # transcript must be unchanged.
        assert msgs == [{"role": "user", "content": "keep"}]
        assert prompt in ("fallback", "late")
        # Teardown proved quiescence, so the lease must NOT stay retained.
        assert fence._retain_cancelled_lock_until_worker_done is False

    def test_falsifier_uncontrolled_idle_sample_never_reaches_the_join(self):
        """Falsifier for the control above — NOT a production defect.

        Remove the progress control (report a stale idle sample while the
        host is still inside its ceiling — the exact unchanged schedule the
        failed run could not rule out) and production deliberately takes the
        idle path, which SKIPS the bounded-grace join. The teardown assertion
        above then has nothing to stand on: that is how a wall-clock version
        of it can fail while production teardown is correct — what is missing
        is the precondition, not the join. Nothing here asserts
        ``worker_done``; that is precisely the quantity this schedule leaves
        unguaranteed.
        """
        original = [{"role": "user", "content": "keep"}]
        entered = threading.Event()
        joined = threading.Event()
        worker_done = threading.Event()
        causes: list[tuple] = []
        join_calls: list[float] = []
        real_join = cc._join_cancelled_worker

        def _record_join(future, grace_seconds):
            join_calls.append(grace_seconds)
            joined.set()
            return real_join(future, grace_seconds)

        fence = _ScheduledFence(
            entered,
            stale_progress_seconds=0.25,
            hold_inside_ceiling=30.0,
        )
        try:
            with patch.object(cc, "_join_cancelled_worker", _record_join):
                msgs, prompt = run_compress_context_with_progress_timeout(
                    worker=_cooperative_worker(
                        entered, joined, worker_done, original
                    ),
                    messages=original,
                    system_prompt_fallback="fallback",
                    idle_timeout_seconds=0.1,
                    total_ceiling_seconds=0.2,
                    fence=fence,
                    on_timeout_cause=lambda exhausted, progressed: (
                        causes.append((exhausted, progressed))
                    ),
                    stall_fallback=False,
                )
            # The worker ran: this is the idle-stall classification, not a
            # pre-start expiry and not a pending-future cancellation.
            assert entered.is_set()
            assert len(causes) == 1 and not causes[0][0], (
                "control removed: the host must classify an idle stall here"
            )
            assert join_calls == [], (
                "the idle path intentionally skips the bounded-grace join, "
                "so the teardown precondition is simply absent"
            )
            # Production is unchanged and still correct on this path.
            assert fence.is_cancelled
            assert msgs is original and prompt == "fallback"
            assert fence._retain_cancelled_lock_until_worker_done is False
        finally:
            joined.set()
            worker_done.wait(timeout=5)

    def test_uninterruptible_worker_is_orphaned_with_lease_retained(self):
        """A worker stuck in an uninterruptible provider call is orphaned:
        the host returns after the grace, the poison fence discards its late
        result, and on the total-ceiling path the durable lease release hook
        does NOT fire while the worker is alive (no overlap window)."""
        original = [{"role": "user", "content": "keep"}]
        release = threading.Event()
        worker_finished = threading.Event()
        lock_released: list[float] = []

        def stuck_worker(fence: CompressionCommitFence):
            # Continuous progress so only the TOTAL ceiling can expire
            # (the #97488 'last progress 0.0s ago' shape).
            while not release.wait(timeout=0.02):
                fence.touch_progress()
            worker_finished.set()
            if not fence.begin_commit():
                return (original, "")
            try:
                return ([{"role": "assistant", "content": "late"}], "late")
            finally:
                fence.finish_commit()

        fence = CompressionCommitFence()
        fence.register_cancelled_lock_release(
            lambda: lock_released.append(time.monotonic())
        )
        msgs, prompt = run_compress_context_with_progress_timeout(
            worker=stuck_worker,
            messages=original,
            system_prompt_fallback="fallback",
            idle_timeout_seconds=0.1,
            total_ceiling_seconds=0.3,
            fence=fence,
            stall_fallback=False,
        )
        # Precondition: the worker is genuinely still running.
        assert not worker_finished.is_set()
        assert msgs is original and prompt == "fallback"
        # Total-ceiling path: lease retained until the worker exits, so no
        # new attempt can overlap the unchanged session.
        assert not lock_released, (
            "durable lease released while the timed-out worker was still "
            "alive — overlap window reopened (#97488)"
        )
        release.set()
        assert worker_finished.wait(timeout=2)
        # Late result was fence-poisoned, never adopted.
        assert msgs == [{"role": "user", "content": "keep"}]


class TestCommonAttemptEpoch:
    """#97488 clock origin: setup latency is not provider silence."""

    def test_setup_delay_before_the_epoch_still_admits_the_worker(self):
        original = [{"role": "user", "content": "keep"}]
        entered = threading.Event()
        worker_done = threading.Event()

        def cooperative_worker(fence: CompressionCommitFence):
            entered.set()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if fence.is_cancelled:
                    break
                fence.touch_progress()
                time.sleep(0.01)
            time.sleep(0.08)
            worker_done.set()
            return (original, "late")

        fence = CompressionCommitFence()
        started = time.monotonic()
        with _slow_pool_setup(0.35):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=cooperative_worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.1,
                total_ceiling_seconds=0.2,
                fence=fence,
                stall_fallback=False,
            )
        elapsed = time.monotonic() - started
        assert entered.is_set(), (
            "worker never entered: setup latency ahead of the attempt epoch "
            "consumed the whole ceiling (#97488)"
        )
        assert elapsed >= 0.5, (
            "the ceiling must be charged from the post-setup epoch, not from "
            "fence construction"
        )
        assert worker_done.is_set(), "cooperative worker was not joined"
        assert msgs == [{"role": "user", "content": "keep"}]
        assert prompt in ("fallback", "late")
        assert fence._retain_cancelled_lock_until_worker_done is False

    def test_context_capture_delay_before_the_epoch_still_admits_the_worker(
        self,
    ):
        """The host-side context wrapper/callback capture is LOCAL submission
        setup, so the common attempt epoch must be armed after it.

        The control blocks the ACTUAL ``propagate_context_to_thread`` seam —
        not the executor getter — for longer than the whole (unchanged, tiny)
        ceiling. With the epoch armed after that capture, the supplied primary
        still enters and is never refused by the fence-gated pre-start check.
        With the reverse order the fence deadline is already exceeded when the
        worker starts: ``setup_done_at_epoch`` is False and the worker never
        runs. The ordering itself is decided by happens-before (an event
        sampled inside ``begin_attempt``), not by wall-clock scheduling.
        """
        original = [{"role": "user", "content": "keep"}]
        entered = threading.Event()
        worker_done = threading.Event()
        setup_done = threading.Event()
        released = threading.Event()
        captured_on: list = []
        releases: list = []
        pre_start_refusals: list = []

        class _CountingTicket(cc._CompressionAdmissionTicket):
            """The real ticket; records only EFFECTIVE (first) releases."""

            __slots__ = ()

            def release(self, _future=None) -> None:
                already_released = self._released
                super().release(_future)
                if not already_released and self._released:
                    releases.append(True)
                    released.set()

        def _counting_admit():
            # The unchanged bounded cap still decides admission; only the
            # ticket instance is instrumented.
            if not cc._try_admit_compression_job():
                return None
            return _CountingTicket()

        real_outcome = cc._settled_worker_outcome

        def _record_outcome(future):
            outcome = real_outcome(future)
            if outcome is not None and outcome[0] == "not_started":
                pre_start_refusals.append(outcome[1])
            return outcome

        def cooperative_worker(fence: CompressionCommitFence):
            entered.set()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if fence.is_cancelled:
                    break
                fence.touch_progress()
                time.sleep(0.01)
            worker_done.set()
            return (original, "late")

        fence = _EpochOrderingFence(setup_done)
        started = time.monotonic()
        with patch.object(cc, "_admit_compression_job", _counting_admit):
            with patch.object(cc, "_settled_worker_outcome", _record_outcome):
                with _slow_context_capture(
                    0.35, done=setup_done, threads=captured_on
                ):
                    msgs, prompt = run_compress_context_with_progress_timeout(
                        worker=cooperative_worker,
                        messages=original,
                        system_prompt_fallback="fallback",
                        idle_timeout_seconds=0.1,
                        total_ceiling_seconds=0.2,
                        fence=fence,
                        stall_fallback=False,
                    )
        elapsed = time.monotonic() - started
        # Precondition: the blocked seam really is the host-side capture.
        assert captured_on == [threading.current_thread()], (
            "the control did not block the host-side context capture"
        )
        # The ordering claim itself.
        assert fence.setup_done_at_epoch is True, (
            "the attempt epoch was armed BEFORE host-side context "
            "wrapper/callback capture finished, so cold local setup is still "
            "charged to the provider attempt"
        )
        assert not pre_start_refusals, (
            "the supplied primary was refused before start: local setup "
            "consumed the attempt ceiling"
        )
        assert entered.is_set(), (
            "worker never entered: host-side context capture ahead of the "
            "attempt epoch consumed the whole ceiling"
        )
        assert elapsed >= 0.5, (
            "the ceiling must be charged from the post-setup epoch, not from "
            "the start of host-side context capture"
        )
        assert worker_done.wait(timeout=5), "cooperative worker never exited"
        assert msgs == [{"role": "user", "content": "keep"}]
        assert prompt in ("fallback", "late")
        # Unchanged contracts: exactly-once admission-ticket release, and no
        # lease retention once the cancelled worker provably exited.
        assert released.wait(timeout=5), "admission ticket was never released"
        assert releases == [True], "admission ticket was released twice"
        assert fence._retain_cancelled_lock_until_worker_done is False

    def test_worker_thrown_timeout_error_is_a_completed_outcome(self):
        original = [{"role": "user", "content": "keep"}]
        timeouts = []
        fence = CompressionCommitFence()

        def timing_out_worker(_fence: CompressionCommitFence):
            raise concurrent.futures.TimeoutError("provider read timed out")

        with pytest.raises(
            concurrent.futures.TimeoutError, match="provider read timed out"
        ):
            run_compress_context_with_progress_timeout(
                worker=timing_out_worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.1,
                total_ceiling_seconds=0.2,
                fence=fence,
                on_timeout=lambda *args: timeouts.append(args),
                stall_fallback=False,
            )
        assert not timeouts, (
            "a settled worker TimeoutError was misread as a host wait timeout"
        )
        assert fence._retain_cancelled_lock_until_worker_done is False

    def test_live_worker_reports_live_only_when_not_done_after_grace(self):
        original = [{"role": "user", "content": "keep"}]
        release = threading.Event()
        started = threading.Event()
        joins = []
        real_join = cc._join_cancelled_worker

        def _record_join(future, grace_seconds):
            outcome = real_join(future, grace_seconds)
            joins.append((outcome, future.done()))
            return outcome

        def stuck_worker(fence: CompressionCommitFence):
            started.set()
            while not release.wait(timeout=0.02):
                fence.touch_progress()
            return ([{"role": "assistant", "content": "late"}], "late")

        fence = CompressionCommitFence()
        try:
            with patch.object(cc, "_join_cancelled_worker", _record_join):
                msgs, prompt = run_compress_context_with_progress_timeout(
                    worker=stuck_worker,
                    messages=original,
                    system_prompt_fallback="fallback",
                    idle_timeout_seconds=0.1,
                    total_ceiling_seconds=0.3,
                    fence=fence,
                    stall_fallback=False,
                )
            assert started.is_set()
            assert msgs is original and prompt == "fallback"
            assert joins == [(False, False)], (
                "a genuinely running future must be reported live"
            )
            assert fence.is_cancelled, "the timed-out fence must be poisoned"
            assert fence._retain_cancelled_lock_until_worker_done is True
            assert fence.begin_commit() is False, "late commit was admitted"
        finally:
            release.set()


class TestExistingSafetyControlsIntact:
    def test_in_flight_commit_is_never_abandoned_past_the_ceiling(self):
        original = [{"role": "user", "content": "keep"}]
        compressed = [{"role": "assistant", "content": "committed"}]
        in_commit = threading.Event()
        finish = threading.Event()
        overruns = []

        def committing_worker(fence: CompressionCommitFence):
            assert fence.begin_commit()
            try:
                in_commit.set()
                assert finish.wait(timeout=5)
                return (compressed, "committed")
            finally:
                fence.finish_commit()

        def _finish_after_ceiling():
            if in_commit.wait(timeout=5):
                time.sleep(0.35)
            finish.set()

        releaser = threading.Thread(target=_finish_after_ceiling, daemon=True)
        releaser.start()
        try:
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=committing_worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.1,
                total_ceiling_seconds=0.2,
                on_commit_overrun=lambda *args: overruns.append(args),
                stall_fallback=False,
            )
        finally:
            finish.set()
            releaser.join(timeout=5)
        assert msgs == compressed and prompt == "committed", (
            "an admitted in-flight commit must never be abandoned"
        )
        assert len(overruns) == 1, "commit overrun must surface exactly once"

    def test_pool_saturation_still_fails_closed(self):
        original = [{"role": "user", "content": "keep"}]
        entered = []

        def refused_worker(fence: CompressionCommitFence):
            entered.append(fence)
            return ([{"role": "assistant", "content": "nope"}], "nope")

        with patch.object(cc, "_try_admit_compression_job", return_value=False):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=refused_worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.1,
                total_ceiling_seconds=0.2,
                stall_fallback=False,
            )
        assert msgs is original and prompt == "fallback"
        assert not entered, "a saturation refusal must never run a worker"


class TestDurableAttemptBackoff:
    def test_backoff_row_records_strategy_and_kind(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_KIND")
        agent.context_compressor.record_timeout_failure(
            "host ceiling exhausted", failure_kind="ceiling_exhausted"
        )
        row = db.get_compression_failure_cooldown("BACKOFF_KIND")
        assert row is not None, "backoff must persist to state.db"
        assert row["remaining_seconds"] > 0
        assert "backoff:ceiling_exhausted:strategy=lean" in (row["error"] or "")

    def test_backoff_blocks_same_strategy_reentry_next_turn(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_REENTRY")
        agent.context_compressor.record_timeout_failure(
            "stall", failure_kind="stalled"
        )
        # Precondition: over threshold, so only the backoff can block.
        assert 500_000 >= agent.context_compressor.threshold_tokens
        should, reason = agent.context_compressor.should_compress_info(500_000)
        assert should is False
        assert reason and reason.startswith("cooldown"), (
            "next-turn re-entry of the same strategy must be skipped inside "
            "the backoff window (#96775)"
        )

    def test_backoff_survives_simulated_gateway_restart(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_RESTART")
        agent.context_compressor.record_timeout_failure(
            "stall before restart", failure_kind="stall_interrupted"
        )
        # Precondition: row is durable in this DB file.
        assert db.get_compression_failure_cooldown("BACKOFF_RESTART")
        # Simulated restart: brand-new SessionDB handle + brand-new agent
        # objects rebuilt from the same state.db file.
        db2 = SessionDB(db_path=tmp_path / "state.db")
        _db2, agent2 = _build_agent(tmp_path, "BACKOFF_RESTART", db=db2)
        cooldown = agent2.context_compressor.get_active_compression_failure_cooldown(
            refresh=True
        )
        assert cooldown is not None, (
            "backoff must survive a gateway restart via state.db (#96775)"
        )
        assert "stall_interrupted" in (cooldown["error"] or "")
        should, reason = agent2.context_compressor.should_compress_info(500_000)
        assert should is False and reason.startswith("cooldown")

    def test_success_clears_backoff(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_CLEAR")
        compressor = agent.context_compressor
        compressor.record_timeout_failure("stall", failure_kind="stalled")
        assert db.get_compression_failure_cooldown("BACKOFF_CLEAR")
        # What a successful compression does on commit:
        compressor._clear_compression_failure_cooldown()
        assert db.get_compression_failure_cooldown("BACKOFF_CLEAR") is None
        assert compressor.should_compress_info(500_000)[0] is True


class TestSupersessionDiscardsLateResults:
    def test_superseded_attempt_candidate_never_commits(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "SUPERSEDE")
        live = _messages()
        original = copy.deepcopy(live)

        def compress_and_get_superseded(messages, **_kwargs):
            # While this attempt's summary was in flight, a NEWER attempt
            # claimed the compressor (what a retry/fallback does).
            _claim_compressor_attempt(agent.context_compressor)
            return [{"role": "assistant", "content": "stale summary"}]

        agent.context_compressor.compress = compress_and_get_superseded
        out, _prompt = compress_context(
            agent, live, "sys", approx_tokens=500_000
        )
        assert out == original, (
            "late candidate from a superseded attempt must be discarded, "
            "never committed over newer state (#97488)"
        )
        assert live == original
        # Session stayed writable and unrotated.
        assert db.get_compression_lock_holder("SUPERSEDE") is None
        db.append_message("SUPERSEDE", "assistant", "still writable")


class TestTransientBlockIsNotExhaustion:
    def test_cooldown_blocked_noop_sets_transient_signal(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "TRANSIENT_SIGNAL")
        agent.context_compressor.record_timeout_failure(
            "host ceiling", failure_kind="ceiling_exhausted"
        )
        live = _messages()
        before = copy.deepcopy(live)
        out, _ = compress_context(agent, live, "sys", approx_tokens=500_000)
        # Preconditions: the pass no-oped and it was NOT a lock skip.
        assert out == before
        assert getattr(agent, "_compression_skipped_due_to_lock", None) is None
        assert compression_blocked_transiently(agent) is True, (
            "a cooldown-blocked no-op must be distinguishable from "
            "exhaustion or the gateway falsely auto-resets (#97488)"
        )

    def test_signal_cleared_per_attempt_and_not_set_when_unblocked(
        self, tmp_path: Path
    ):
        db, agent = _build_agent(tmp_path, "TRANSIENT_CLEAR")
        # Stale signal from a previous pass must not leak.
        agent._compression_blocked_transient = "cooldown:999"
        agent.context_compressor.compress = lambda messages, **kw: list(messages)
        live = _messages()
        compress_context(agent, live, "sys", approx_tokens=500_000)
        assert compression_blocked_transiently(agent) is False

    def test_type_pinned_against_magicmock_agents(self):
        from unittest.mock import MagicMock

        mock_agent = MagicMock()
        # MagicMock auto-attributes are truthy but not str.
        assert compression_blocked_transiently(mock_agent) is False


class _AttemptOwnershipCompressor:
    """Attribute holder — the ownership helpers touch nothing else."""


class TestGenerationOwnershipRequiresDurableOwnership:
    """#198 F1 ownership algebra — recorded claims only, no schedule.

    Each case is a pure function of the claims taken, so the ownership rule
    itself is pinned instead of a sampled interleaving: no thread, sleep,
    retry or timing tolerance is involved.
    """

    def test_retired_contender_stops_superseding_the_current_owner(self):
        compressor = _AttemptOwnershipCompressor()
        owner = _claim_compressor_attempt(compressor)
        contender = _claim_compressor_attempt(compressor)
        # F1 shape: the newer claim ALONE invalidated the durable owner.
        assert not cc._compressor_attempt_is_current(compressor, owner)
        assert cc._retire_compressor_attempt(compressor, contender) is True
        assert cc._compressor_attempt_is_current(compressor, owner), (
            "a contender that never became the durable owner must not "
            "invalidate the real lock holder (#198 F1)"
        )
        assert not cc._compressor_attempt_is_current(compressor, contender), (
            "the retired loser must sit out, not regain authority"
        )
        # Not a rollback: the monotonic counter never moves backwards.
        assert compressor._compression_attempt_generation == contender

    def test_consecutive_retirements_collapse_onto_the_owner(self):
        compressor = _AttemptOwnershipCompressor()
        owner = _claim_compressor_attempt(compressor)
        first = _claim_compressor_attempt(compressor)
        second = _claim_compressor_attempt(compressor)
        # Two overlapping contenders both lose the lock the owner holds, and
        # retire in the reverse of their claim order.
        assert cc._retire_compressor_attempt(compressor, second) is True
        assert cc._retire_compressor_attempt(compressor, first) is True
        assert cc._compressor_attempt_is_current(compressor, owner), (
            "a chain of lock losers must not invalidate the lock holder"
        )
        assert not cc._compressor_attempt_is_current(compressor, first)
        assert not cc._compressor_attempt_is_current(compressor, second)
        assert compressor._compression_attempt_generation == second

    def test_admitted_successor_keeps_authority_over_a_stale_primary(self):
        compressor = _AttemptOwnershipCompressor()
        primary = _claim_compressor_attempt(compressor)
        contender = _claim_compressor_attempt(compressor)
        # A genuinely admitted successor (stall-fallback) takes authority,
        # and only afterwards does the losing contender retire.
        successor = _claim_compressor_attempt(compressor)
        assert cc._retire_compressor_attempt(compressor, contender) is True
        assert cc._compressor_attempt_is_current(compressor, successor), (
            "retiring a lock loser must never demote the admitted successor"
        )
        assert not cc._compressor_attempt_is_current(compressor, primary), (
            "a detached stale primary must stay superseded (#96634/#97488)"
        )
        # The detached primary's late restore must still no-op.
        compressor._previous_summary = "successor state"
        cc._restore_compressor_attempt_state(
            compressor,
            {"_previous_summary": "stale primary state"},
            attempt_generation=primary,
        )
        assert compressor._previous_summary == "successor state", (
            "a stale primary rolled successor-owned state back (#96634)"
        )

    def test_disabled_guard_is_never_retired(self):
        compressor = _AttemptOwnershipCompressor()
        # Generation 0 = tracking unavailable (slotted third party).
        assert cc._retire_compressor_attempt(compressor, 0) is False
        assert cc._compressor_attempt_is_current(compressor, 0) is True


class TestFailedLockContenderCannotInvalidateTheOwner:
    """#198 F1 end-to-end: the loser of the durable lock must sit out.

    The overlap is linearized by construction — the contender runs to
    completion on the owner's own thread, at the exact point where the owner
    holds the durable lease and has a valid candidate in hand — so the
    interleaving is fixed, not sampled. No budget, sleep or retry is used.
    """

    def test_contender_that_loses_the_lock_leaves_the_owner_committable(
        self, tmp_path: Path
    ):
        db, agent = _build_agent(tmp_path, "GEN_LOCK_OWNER")
        live = _messages()
        original = copy.deepcopy(live)
        for message in original:
            db.append_message(
                "GEN_LOCK_OWNER", message["role"], message["content"]
            )
        compressor = agent.context_compressor
        generations: list[int] = []
        contender: list[tuple] = []

        def compress_with_overlapping_contender(messages, **_kwargs):
            # The owner holds the durable lease here and is about to hand
            # back a valid candidate.
            generations.append(
                int(getattr(compressor, "_compression_attempt_generation", 0))
            )
            assert db.get_compression_lock_holder("GEN_LOCK_OWNER"), (
                "precondition: the owner must hold the durable lease while "
                "its summary runs"
            )
            # A NEWER overlapping entrypoint on the SAME agent/compressor
            # (automatic + manual /compress overlap, run_agent.py) claims the
            # shared generation and then loses the durable lock.
            contender_live = copy.deepcopy(original)
            contender_out, _contender_prompt = compress_context(
                agent,
                contender_live,
                "sys",
                approx_tokens=500_000,
                force=True,
            )
            contender.append(
                (
                    contender_out is contender_live,
                    contender_live == original,
                    compression_skipped_due_to_lock(agent),
                )
            )
            generations.append(
                int(getattr(compressor, "_compression_attempt_generation", 0))
            )
            return [{"role": "assistant", "content": "owner summary"}]

        compressor.compress = compress_with_overlapping_contender
        out, _prompt = compress_context(
            agent, live, "sys", approx_tokens=500_000
        )
        # Precondition: the contender really claimed a NEWER generation than
        # the lock owner — the exact F1 ordering.
        assert len(generations) == 2 and generations[1] > generations[0] > 0, (
            "the contender did not advance the shared attempt generation"
        )
        # Precondition: the contender reached durable-lock contention (not a
        # breaker/cooldown no-op) and sat out with its transcript unchanged.
        assert contender == [(True, True, True)], (
            "the overlapping attempt did not sit out through the "
            "lock-contended path"
        )
        # The finding itself: the lock LOSER must not make the lock WINNER
        # discard a healthy candidate (both attempts no-oped before this).
        assert out != original, (
            "the durable lock holder discarded its valid candidate because a "
            "failed contender had claimed the generation first (#198 F1)"
        )
        assert any(
            isinstance(message, dict)
            and message.get("content") == "owner summary"
            for message in out
        )
        assert len(out) < len(original)
        # The owner remained eligible to commit — and did.
        assert agent._last_compaction_in_place is True
        durable = db.get_messages_as_conversation("GEN_LOCK_OWNER")
        assert 0 < len(durable) < len(original), (
            "the owner's compaction never reached durable state"
        )


class _DurableCooldownStore:
    """SessionDB stand-in for the three durable-cooldown rollback APIs.

    Only the methods the rollback path reaches are implemented, so the control
    stays hermetic at that API boundary: it records the durable write ORDER and
    the surviving row, and depends on no storage schema or workflow.
    """

    def __init__(self, row):
        self.row = dict(row)
        self.writes: list[str] = []

    def restore_compression_failure_cooldown_row(self, session_id, state):
        self.writes.append("rollback")
        self.row = dict(state)

    def record_compression_failure_cooldown(self, session_id, deadline, error):
        self.writes.append("successor")
        self.row = {"until": deadline, "error": error}

    def clear_compression_failure_cooldown(self, session_id):
        self.writes.append("clear")
        self.row = {}


class _StaleRollbackCompressor:
    """Attribute holder bound to a durable store; no provider or plugin."""

    def __init__(self, store, session_id):
        self._session_db = store
        self._session_id = session_id


class TestStaleRollbackCannotOverwriteAnAdmittedSuccessor:
    """#198 F1: ownership check and durable rollback are ONE ordered step.

    Fixed interleaving, no sleep, retry or timing tolerance: the stale primary
    parks INSIDE its authority check (after it passed, before the durable
    write), then the successor admits, claims and writes the authoritative
    cooldown row. The primary is released by the first of two causally
    guaranteed events -- the successor finishing its write (the unordered
    shape) or the successor's claim actually WAITING on this compressor's
    rollback ordering (the corrected shape) -- so both shapes run to
    completion without a timeout and the assertion, not the schedule, decides.
    """

    def test_successor_owned_durable_cooldown_survives_a_stale_rollback(
        self, monkeypatch
    ):
        original_row = {"until": 11.0, "error": "primary-original"}
        store = _DurableCooldownStore(original_row)
        compressor = _StaleRollbackCompressor(store, "STALE_ROLLBACK")
        primary = _claim_compressor_attempt(compressor)

        primary_parked = threading.Event()
        release_primary = threading.Event()
        successor_at_claim = threading.Event()
        may_release_primary = threading.Event()
        real_lock = threading.RLock()

        class _OrderingObserver:
            """Observes the correction's per-compressor rollback mutex."""

            def acquire(self, *args, **kwargs):
                if real_lock.acquire(blocking=False):
                    return True
                # A claim is really WAITING on an in-flight durable rollback.
                may_release_primary.set()
                real_lock.acquire()
                return True

            def release(self):
                real_lock.release()

        observer = _OrderingObserver()
        monkeypatch.setattr(
            cc,
            "_compressor_durable_rollback_lock",
            lambda _compressor: observer,
            raising=False,
        )

        real_is_current = cc._compressor_attempt_is_current

        def parking_is_current(target, generation):
            current = real_is_current(target, generation)
            if generation == primary and not primary_parked.is_set():
                primary_parked.set()
                release_primary.wait()
            return current

        monkeypatch.setattr(
            cc, "_compressor_attempt_is_current", parking_is_current
        )

        successor_generation: list[int] = []

        def successor_body():
            successor_at_claim.set()
            successor_generation.append(_claim_compressor_attempt(compressor))
            store.record_compression_failure_cooldown(
                "STALE_ROLLBACK", 99.0, "successor-authoritative"
            )
            may_release_primary.set()

        def primary_body():
            cc._restore_compressor_attempt_state(
                compressor,
                {
                    "_summary_failure_cooldown_until": 11.0,
                    "_last_summary_error": "primary-original",
                    "_previous_summary": "stale primary state",
                },
                durable_cooldown_authoritative=True,
                durable_cooldown_state=dict(original_row),
                attempt_generation=primary,
            )

        primary_thread = threading.Thread(target=primary_body)
        successor_thread = threading.Thread(target=successor_body)
        primary_thread.start()
        primary_parked.wait()
        # Precondition: the primary passed its ownership check and has NOT
        # written durable state yet -- exactly the F1 window.
        assert store.writes == []
        successor_thread.start()
        successor_at_claim.wait()
        may_release_primary.wait()
        release_primary.set()
        primary_thread.join()
        successor_thread.join()

        assert successor_generation and successor_generation[0] > primary, (
            "precondition: the successor must have been admitted with a newer "
            "generation than the detached primary"
        )
        assert "successor" in store.writes
        assert store.writes[-1] == "successor", (
            "a detached stale primary's durable cooldown rollback landed "
            "AFTER an admitted successor's authoritative write (#198 F1)"
        )
        assert store.row == {
            "until": 99.0,
            "error": "successor-authoritative",
        }, (
            "successor-owned durable cooldown state was rolled back to the "
            "stale primary's pre-attempt row (#198 F1)"
        )

    def test_rollback_without_a_successor_still_restores(self):
        store = _DurableCooldownStore(
            {"until": 0.0, "error": "cleared-by-summary"}
        )
        compressor = _StaleRollbackCompressor(store, "NO_SUCCESSOR")
        compressor._previous_summary = "post-summary state"
        compressor._summary_failure_cooldown_until = 0.0
        generation = _claim_compressor_attempt(compressor)
        original_row = {"until": 42.0, "error": "pre-attempt"}
        cc._restore_compressor_attempt_state(
            compressor,
            {
                "_summary_failure_cooldown_until": 42.0,
                "_last_summary_error": "pre-attempt",
                "_previous_summary": "pre-attempt state",
            },
            durable_cooldown_authoritative=True,
            durable_cooldown_state=original_row,
            attempt_generation=generation,
        )
        # No successor was admitted: the legitimate pre-commit cancellation
        # must still restore BOTH the authoritative durable row and the safe
        # in-memory snapshot, unchanged by the new ordering.
        assert store.writes == ["rollback"]
        assert store.row == original_row
        assert compressor._previous_summary == "pre-attempt state"
        assert compressor._summary_failure_cooldown_until == 42.0
