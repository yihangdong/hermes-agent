"""Compression falls back after an aborted (stalled) summary — #78981.

A summariser that keeps the connection open but never emits a real token
produces no fence progress, so the host's progress-aware timeout aborts the
worker and returns "continue without compression". Nothing raises out of the
auxiliary client on that path, so its configured ``fallback_chain`` — the
user's declared answer to "this route is unhealthy" — was never consulted for
the one failure mode that most needs it.

These tests pin the contract:

* an aborted stall re-attempts compression once with the summary route pinned
  to the configured ``auxiliary.compression.fallback_chain``;
* the pinned route reaches the summary ``call_llm`` (provider/model/base_url/
  api_key/timeout), and is single-use so the compressor's own main-model retry
  does not re-issue the same failed route;
* the historical "continue without compression" degrade survives when no chain
  is configured or the fallback attempt also stalls.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from agent import conversation_compression as cc
from agent.context_compressor import (
    ContextCompressor,
    pin_summary_route,
    take_pinned_summary_route,
)
from agent.conversation_compression import (
    CompressionCommitFence,
    resolve_compression_fallback_route,
    run_compress_context_with_progress_timeout,
)

CHAIN_ENTRY = {
    "provider": "custom",
    "model": "backup-summarizer",
    "base_url": "https://fallback.invalid/v1",
    "api_key": "sk-fallback",
    "timeout": 45,
}


def _patch_chain(chain):
    """Pin auxiliary.compression config without touching the real config.yaml."""
    return patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value={"fallback_chain": chain},
    )


class _StalledSummaryWorker:
    """A compression worker whose first attempt streams nothing at all.

    Mirrors the reported shape: the provider holds the connection open, so the
    worker never calls ``fence.touch_progress()`` and the host's idle budget
    lapses. ``stall_attempts`` controls how many attempts hang; any later
    attempt commits a real summary.
    """

    def __init__(self, compressed, *, stall_attempts=1):
        self.compressed = compressed
        self.stall_attempts = stall_attempts
        self.routes = []
        self.fences = []
        self._lock = threading.Lock()
        self.release = threading.Event()

    @property
    def attempts(self):
        return len(self.routes)

    def __call__(self, fence: CompressionCommitFence):
        with self._lock:
            self.routes.append(take_pinned_summary_route())
            self.fences.append(fence)
            attempt = len(self.routes)
        if attempt <= self.stall_attempts:
            # Connection open, zero tokens, zero fence progress.
            self.release.wait(timeout=10)
            return ([{"role": "assistant", "content": "late"}], "late-prompt")
        if not fence.begin_commit():
            return ([{"role": "assistant", "content": "cancelled"}], "cancelled")
        try:
            return (self.compressed, "summarized-prompt")
        finally:
            fence.finish_commit()


def _run(worker, *, chain, timeouts, messages, idle=0.05, ceiling=0.2):
    with _patch_chain(chain):
        return run_compress_context_with_progress_timeout(
            worker=worker,
            messages=messages,
            system_prompt_fallback="degraded-prompt",
            idle_timeout_seconds=idle,
            total_ceiling_seconds=ceiling,
            on_timeout=lambda *args: timeouts.append(args),
        )


# ---------------------------------------------------------------------------
# Fence-level contract: an aborted stall consults the configured chain
# ---------------------------------------------------------------------------


def test_stalled_summary_attempts_configured_fallback_chain():
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "summary of earlier turns"}]
    worker = _StalledSummaryWorker(compressed)
    timeouts = []

    try:
        msgs, prompt = _run(
            worker, chain=[CHAIN_ENTRY], timeouts=timeouts, messages=original
        )
    finally:
        worker.release.set()

    assert worker.attempts == 2, "the aborted stall must be retried once"
    assert worker.routes[0] is None, "the primary attempt is never pinned"
    pinned = worker.routes[1]
    assert pinned is not None, "the retry must carry the configured fallback route"
    assert pinned["provider"] == "custom"
    assert pinned["model"] == "backup-summarizer"
    assert msgs == compressed, "the fallback attempt's compression must be published"
    assert prompt == "summarized-prompt"
    assert not timeouts, "no continue-without-compression degrade after a recovery"


def test_retry_runs_on_a_host_published_fence():
    """The aborted fence vetoes every future commit, so the retry needs a new
    one — minted through the host so ``/stop`` admits against the attempt that
    is actually running."""
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "summary"}]
    worker = _StalledSummaryWorker(compressed)
    minted = []

    def _new_fence():
        fence = CompressionCommitFence()
        minted.append(fence)
        return fence

    try:
        with _patch_chain([CHAIN_ENTRY]):
            msgs, _prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=0.05,
                total_ceiling_seconds=0.2,
                new_fence=_new_fence,
            )
    finally:
        worker.release.set()

    assert msgs == compressed
    assert len(minted) == 1, "exactly one fence is minted for the one retry"
    assert worker.fences[1] is minted[0]
    assert worker.fences[1] is not worker.fences[0]
    assert worker.fences[0].is_cancelled, "the aborted attempt stays cancelled"


def test_slow_setup_does_not_consume_the_primary_attempt():
    """#97488: the ceiling is charged from the post-setup attempt epoch.

    With the ceiling armed BEFORE the wrapper's lazy setup, a cold start
    longer than the tiny test ceiling refused the PRIMARY worker pre-start, so
    the one permitted fallback became the only attempt that ever ran.
    """
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "summary of earlier turns"}]
    worker = _StalledSummaryWorker(compressed)
    timeouts = []
    real_executor = cc._get_compress_timeout_executor

    def _slow_setup():
        executor = real_executor()
        time.sleep(0.3)  # > the 0.2s ceiling of this attempt
        return executor

    try:
        with patch.object(cc, "_get_compress_timeout_executor", _slow_setup):
            msgs, prompt = _run(
                worker, chain=[CHAIN_ENTRY], timeouts=timeouts, messages=original
            )
    finally:
        worker.release.set()

    assert worker.attempts == 2, (
        "the primary attempt must still enter its worker after slow setup, "
        "and the stall must be retried exactly once"
    )
    assert worker.routes[0] is None, "the primary attempt is never pinned"
    assert worker.routes[1] is not None, "the retry carries the fallback route"
    assert worker.fences[1] is not worker.fences[0], "the retry needs a fresh fence"
    assert worker.fences[0].is_cancelled, "the aborted attempt stays cancelled"
    assert msgs == compressed and prompt == "summarized-prompt"
    assert not timeouts, "no continue-without-compression degrade after recovery"


def test_hard_interrupt_suppresses_the_fallback_attempt():
    """An explicit stop is not an unhealthy route — don't start another
    summary on the user's behalf after they asked for the turn to end."""
    original = [{"role": "user", "content": "keep-me"}]
    worker = _StalledSummaryWorker([{"role": "user", "content": "unused"}])
    stopped = threading.Event()
    stopped.set()
    agent = SimpleNamespace(_hard_interrupt_requested=stopped)
    timeouts = []

    try:
        with _patch_chain([CHAIN_ENTRY]):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=0.05,
                total_ceiling_seconds=0.2,
                on_timeout=lambda *args: timeouts.append(args),
                telemetry_agent=agent,
            )
    finally:
        worker.release.set()

    assert worker.attempts == 1
    assert msgs is original
    assert prompt == "degraded-prompt"
    assert len(timeouts) == 1


def test_no_fallback_chain_configured_degrades_without_retry():
    original = [{"role": "user", "content": "keep-me"}]
    worker = _StalledSummaryWorker([{"role": "user", "content": "unused"}])
    timeouts = []

    try:
        msgs, prompt = _run(worker, chain=[], timeouts=timeouts, messages=original)
    finally:
        worker.release.set()

    assert worker.attempts == 1, "nothing to fall back to — do not burn a retry"
    assert msgs is original
    assert prompt == "degraded-prompt"
    assert len(timeouts) == 1


def test_fallback_that_also_stalls_degrades_after_one_attempt():
    original = [{"role": "user", "content": "keep-me"}]
    worker = _StalledSummaryWorker(
        [{"role": "user", "content": "unused"}], stall_attempts=2
    )
    timeouts = []
    entry = dict(CHAIN_ENTRY, timeout=0.05)

    try:
        msgs, prompt = _run(worker, chain=[entry], timeouts=timeouts, messages=original)
    finally:
        worker.release.set()

    assert worker.attempts == 2, "the fallback is attempted once, not in a loop"
    assert msgs is original, "no messages may be dropped when both routes stall"
    assert prompt == "degraded-prompt"
    assert len(timeouts) == 1, "the degrade must be reported exactly once"


# ---------------------------------------------------------------------------
# Deep F1: a FINISHED primary must not hold its slot against the one fallback
# ---------------------------------------------------------------------------


class _CallbackWithheldFuture:
    """Future proxy that withholds the wrapper's admission done-callback.

    ``Future.set_result``/``set_exception`` mark the future FINISHED and wake
    its waiters BEFORE ``_invoke_callbacks()`` runs, so a woken host can
    legitimately observe ``done() is True`` while the admission slot is still
    held. Withholding the callback pins that schedule deterministically (a
    strictly worse lag than any real one) instead of racing for it.
    """

    def __init__(self, future, withheld):
        self._future = future
        self._withheld = withheld

    def add_done_callback(self, fn):
        self._withheld.append(fn)

    def __getattr__(self, name):
        return getattr(self._future, name)


class _CallbackWithheldExecutor:
    def __init__(self, executor, withheld):
        self._executor = executor
        self._withheld = withheld

    def submit(self, fn, *args, **kwargs):
        return _CallbackWithheldFuture(
            self._executor.submit(fn, *args, **kwargs), self._withheld
        )


class _StallsUntilHostGivesUpWorker:
    """Primary that streams progress, then FINISHES the instant the host has
    stopped waiting; the fallback attempt commits a real summary."""

    def __init__(self, compressed):
        self.compressed = compressed
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.attempts = 0
        self.admitted_at_start = []
        self.fifth_admissions = []
        self._lock = threading.Lock()

    def __call__(self, fence: CompressionCommitFence):
        with self._lock:
            self.attempts += 1
            attempt = self.attempts
        with cc._compress_admission_lock:
            self.admitted_at_start.append(cc._compress_admitted_count)
        # A fifth job must never be admissible while four are admitted.
        extra = cc._try_admit_compression_job()
        self.fifth_admissions.append(extra)
        if extra:
            cc._release_compression_admission()
        if attempt == 1:
            self.entered.set()
            # Continuous progress keeps the idle budget alive so only the
            # TOTAL ceiling expires; the host releases this worker once it has
            # stopped waiting, settling the future with its callback withheld.
            while not self.release.wait(timeout=0.02):
                fence.touch_progress()
            self.finished.set()
            return ([{"role": "assistant", "content": "late"}], "late-prompt")
        if not fence.begin_commit():
            return ([{"role": "assistant", "content": "cancelled"}], "cancelled")
        try:
            return (self.compressed, "summarized-prompt")
        finally:
            fence.finish_commit()


def test_finished_primary_slot_is_reclaimed_for_the_one_fallback():
    """The single fallback must survive FINISHED-before-callback at capacity.

    With the other three slots occupied, a host that woke on the primary's
    FINISHED future while its admission callback was still pending read a
    stale count of four and refused the one permitted fallback as
    ``pool_saturated`` — the fallback opportunity was lost with no fifth job
    ever running.
    """
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "summary of earlier turns"}]
    worker = _StallsUntilHostGivesUpWorker(compressed)
    timeouts = []
    withheld = []
    cap = cc._COMPRESS_EXECUTOR_MAX_WORKERS
    others = 0

    def _host_stopped_waiting(_total_exhausted, _progress_observed):
        # Host-side barrier: the wait loop has already ended here, so the
        # primary settles strictly before the teardown join and the fallback.
        assert worker.entered.wait(timeout=5)
        worker.release.set()
        assert worker.finished.wait(timeout=5)

    real_executor = cc._get_compress_timeout_executor
    try:
        # Precondition (existing drain idiom): start from an idle pool.
        deadline = time.time() + 5
        while time.time() < deadline:
            with cc._compress_admission_lock:
                if cc._compress_admitted_count == 0:
                    break
            time.sleep(0.02)
        with cc._compress_admission_lock:
            assert cc._compress_admitted_count == 0, "pool did not drain"
        # Occupy the other three slots: the counter is the contended
        # resource, so no real peer worker is needed to model them.
        for _ in range(cap - 1):
            assert cc._try_admit_compression_job()
            others += 1

        with patch.object(
            cc,
            "_get_compress_timeout_executor",
            lambda: _CallbackWithheldExecutor(real_executor(), withheld),
        ), _patch_chain([CHAIN_ENTRY]):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=0.1,
                total_ceiling_seconds=0.2,
                on_timeout=lambda *args: timeouts.append(args),
                on_timeout_cause=_host_stopped_waiting,
            )

        assert worker.attempts == 2, (
            "the one permitted fallback was refused while the settled "
            "primary's admission callback was still pending"
        )
        assert msgs == compressed and prompt == "summarized-prompt"
        assert not timeouts, "a recovered fallback is not a degrade"
        assert worker.admitted_at_start == [cap, cap], (
            "the cap must hold and the settled primary's slot must be "
            "reclaimed before the fallback is admitted"
        )
        assert worker.fifth_admissions == [False, False], (
            "a fifth compression job was admitted"
        )
        assert len(withheld) == 2, "one admission callback per admitted future"
        # Exactly-once: the primary's withheld callback is a no-op (the host
        # already reclaimed that slot); only the fallback's slot is freed.
        for callback in withheld:
            callback(None)
        withheld.clear()
        with cc._compress_admission_lock:
            assert cc._compress_admitted_count == others, (
                "one admitted future released more than one slot"
            )
    finally:
        worker.release.set()
        for callback in withheld:
            callback(None)
        for _ in range(others):
            cc._release_compression_admission()
    with cc._compress_admission_lock:
        assert cc._compress_admitted_count == 0


# ---------------------------------------------------------------------------
# Route resolution: a chain entry becomes an explicit summary route
# ---------------------------------------------------------------------------


def test_resolved_route_carries_entry_credentials_and_timeout():
    with _patch_chain([CHAIN_ENTRY]):
        route = resolve_compression_fallback_route()

    assert route is not None
    assert route["provider"] == "custom"
    assert route["model"] == "backup-summarizer"
    assert route["base_url"] == "https://fallback.invalid/v1"
    assert route["api_key"] == "sk-fallback"
    # Per-entry timeouts already govern aux-client fallback candidates
    # (#62452); the stall retry honours the same declaration.
    assert route["timeout"] == 45.0


def test_incomplete_chain_entries_are_skipped():
    chain = [
        "not-a-mapping",
        {"model": "orphan-model"},          # no provider
        {"provider": "custom"},             # no model
        CHAIN_ENTRY,
    ]
    with _patch_chain(chain):
        route = resolve_compression_fallback_route()

    assert route is not None
    assert route["model"] == "backup-summarizer"


def test_no_chain_resolves_to_no_route():
    with _patch_chain([]):
        assert resolve_compression_fallback_route() is None


# ---------------------------------------------------------------------------
# Injection point: the pinned route reaches the summary call
# ---------------------------------------------------------------------------


def _make_compressor(summary_model="aux-summarizer"):
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(
            model="main-model",
            quiet_mode=True,
            summary_model_override=summary_model,
        )


def _msgs():
    return [
        {"role": "user", "content": "u1 " + "x" * 200},
        {"role": "assistant", "content": "a1 " + "y" * 200},
        {"role": "user", "content": "u2 " + "z" * 200},
    ]


def _ok_response(content="SUMMARY BODY"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_pinned_route_overrides_the_summary_call_route():
    compressor = _make_compressor()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        with pin_summary_route(dict(CHAIN_ENTRY)):
            summary = compressor._generate_summary(_msgs())

    assert summary and "SUMMARY BODY" in summary
    assert len(calls) == 1
    call = calls[0]
    assert call["task"] == "compression"
    assert call["provider"] == "custom"
    assert call["model"] == "backup-summarizer"
    assert call["base_url"] == "https://fallback.invalid/v1"
    assert call["api_key"] == "sk-fallback"
    assert call["timeout"] == 45


def test_pinned_route_is_not_reissued_by_the_main_model_retry():
    """The compressor's own main-model retry must not re-run the failed route."""
    compressor = _make_compressor()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("Request timed out.")
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        with pin_summary_route(dict(CHAIN_ENTRY)):
            summary = compressor._generate_summary(_msgs())

    assert summary and "SUMMARY BODY" in summary
    assert len(calls) == 2
    assert calls[0]["provider"] == "custom"
    assert "provider" not in calls[1], (
        "the retry must route normally, not repeat the stalled fallback route"
    )


def test_unpinned_summary_call_keeps_task_routing():
    compressor = _make_compressor()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        summary = compressor._generate_summary(_msgs())

    assert summary
    assert calls and "provider" not in calls[0]
    assert calls[0]["model"] == "aux-summarizer"


# ---------------------------------------------------------------------------
# #198 F1 (unified) -- the HOST window.
#
# The idle-timeout path releases the cancelled worker's durable lease
# (``release_cancelled_compression_lock``), then runs the ONE permitted stall
# fallback, and only then calls ``on_timeout`` -- which is where
# ``AIAgent._compress_context`` records the host timeout cooldown. A successor
# can take the session in that post-release window, so that record must be
# authorized by the ownership epoch THIS attempt captured under its lease.
#
# These controls drive the real ``AIAgent._compress_context`` (its real
# ``_on_timeout`` closure), the real host wrapper, the real fallback
# opportunity, a real built-in ContextCompressor and a real SessionDB. The only
# control is the fence handed to the host (the ``_pin_attempt_fence`` idiom of
# tests/agent/test_compression_worker_isolation_76354.py); production budgets,
# ordering and callbacks are untouched. No sleep, no retry, no tolerance, no
# mocked persistence: every step is a barrier on a real signal.
# ---------------------------------------------------------------------------

_HOST_STALE_IDLE_SECONDS = 60.0
_HOST_EPOCH_SHIFT_SECONDS = 30.0
_HOST_BARRIER_TIMEOUT_SECONDS = 30.0
_D1_HOST_ROW = {
    "session_exists": True,
    "cooldown_until": 5_000_000_000.0,
    "error": "D1-successor",
}


class _HostIdleStallFence(CompressionCommitFence):
    """Classify ONE host idle timeout against a provably-entered provider.

    * the host's first idle sample blocks until the attempt has entered its
      provider call -- which is strictly after it captured the session cooldown
      ownership epoch under its lease, so no timeout is classified against an
      attempt that never owned anything;
    * the host's elapsed-time origin is shifted and ``deadline_exceeded`` is
      pinned False so the TOTAL-ceiling term cannot win: that path retains the
      lease and joins the worker, a different contract from the idle release
      under test;
    * ``release_cancelled_compression_lock`` is the exact production instant
      the successor is admitted at -- after the lease is free, before the
      fallback opportunity and before the host timeout record.
    """

    def __init__(self, provider_entered, *, on_release) -> None:
        super().__init__()
        self._provider_entered = provider_entered
        self._on_release = on_release
        self._host_ident = None
        self.worker_ident = None
        self.releases = 0

    def begin_attempt(self, total_ceiling_seconds: float) -> float:
        self._host_ident = threading.get_ident()
        epoch = super().begin_attempt(total_ceiling_seconds)
        return epoch + _HOST_EPOCH_SHIFT_SECONDS

    @property
    def deadline_exceeded(self) -> bool:
        return False

    def seconds_since_progress(self) -> float:
        ident = threading.get_ident()
        if ident == self._host_ident:
            # Barrier, not a sleep.
            assert self._provider_entered.wait(
                timeout=_HOST_BARRIER_TIMEOUT_SECONDS
            ), (
                "the compression provider never entered -- a timeout "
                "classified here would prove nothing"
            )
            return _HOST_STALE_IDLE_SECONDS
        if ident == self.worker_ident:
            return _HOST_STALE_IDLE_SECONDS
        return super().seconds_since_progress()

    def release_cancelled_compression_lock(self) -> None:
        super().release_cancelled_compression_lock()
        self.releases += 1
        if self.releases == 1:
            self._on_release()


def _pin_host_fence(real, fence):
    """Hand the production host wrapper ONE controlled fence, once.

    ``stall_fallback`` is left at its production default, so the one permitted
    fallback opportunity still runs between the lease release and ``on_timeout``.
    """
    pinned = []

    def _wrapper(**kwargs):
        if not pinned:
            pinned.append(True)
            kwargs["fence"] = fence
        return real(**kwargs)

    return _wrapper


def _build_real_host_agent(db, session_id):
    """A real AIAgent whose REAL built-in ContextCompressor stays installed."""
    import os

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
    compressor = agent.context_compressor
    assert isinstance(compressor, ContextCompressor), (
        "this control must exercise the built-in compressor, not a stub"
    )
    compressor._session_db = db
    compressor._session_id = session_id
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = "sys"
    agent.compression_in_place = True
    return agent, compressor


def _successor_takes_session(db, session_id, compressor_b, *, release):
    """A DIFFERENT compressor mints a successor epoch and writes D1."""
    holder_b = "pid:b:successor"
    assert db.try_acquire_compression_lock(
        session_id, holder_b, ttl_seconds=300
    ), "the host must have freed the cancelled attempt's lease"
    authoritative, state = cc._capture_authoritative_cooldown_under_lease(
        compressor_b, {}
    )
    assert authoritative is True and isinstance(state, dict), (
        "the successor's capture was not authoritative"
    )
    assert state.get("owner_token")
    db.record_compression_failure_cooldown(
        session_id, 5_000_000_000.0, "D1-successor"
    )
    if release:
        db.release_compression_lock(session_id, holder_b)
    return holder_b


def test_host_timeout_cooldown_cannot_write_successor_owned_row(
    tmp_path, monkeypatch
):
    """#198 F1 (host window): B takes the session after the lease release.

    Both successor lease states are run. The real host path is followed all the
    way through ``release_cancelled_compression_lock`` -> fallback opportunity
    -> the real ``_on_timeout`` record, and then through the detached worker's
    own late unwind (refused commit fence -> owner-qualified restore ->
    stall backoff). D1 must stay byte-exact throughout.
    """
    from hermes_state import SessionDB

    for successor_released in (False, True):
        case = f"successor_released={successor_released}"
        db = SessionDB(db_path=tmp_path / f"host-{int(successor_released)}.db")
        session_id = "F1_HOST_TIMEOUT_COOLDOWN"
        db.create_session(session_id, source="cli")
        db.append_message(session_id, "user", "durable original")
        db.record_compression_failure_cooldown(
            session_id, 4_000_000_000.0, "D0-primary"
        )

        agent_a, compressor_a = _build_real_host_agent(db, session_id)
        _agent_b, compressor_b = _build_real_host_agent(db, session_id)
        assert compressor_a is not compressor_b, case

        monkeypatch.setattr(
            cc,
            "resolve_context_compression_timeouts",
            lambda cfg=None: (1.0, 2.0),
        )
        provider_entered = threading.Event()
        release_provider = threading.Event()
        backoff_done = threading.Event()
        admitted: list = []
        order: list = []

        def _admit_successor():
            order.append("release")
            admitted.append(
                _successor_takes_session(
                    db, session_id, compressor_b, release=successor_released
                )
            )

        fence = _HostIdleStallFence(
            provider_entered, on_release=_admit_successor
        )
        monkeypatch.setattr(
            cc,
            "run_compress_context_with_progress_timeout",
            _pin_host_fence(
                cc.run_compress_context_with_progress_timeout, fence
            ),
        )
        real_retry = cc._retry_compression_on_fallback_chain

        def _observed_retry(**kwargs):
            order.append("fallback")
            return real_retry(**kwargs)

        monkeypatch.setattr(
            cc, "_retry_compression_on_fallback_chain", _observed_retry
        )
        real_backoff = cc._record_stall_interrupted_backoff

        def _observed_backoff(*args, **kwargs):
            try:
                return real_backoff(*args, **kwargs)
            finally:
                backoff_done.set()

        monkeypatch.setattr(
            cc, "_record_stall_interrupted_backoff", _observed_backoff
        )

        def _stalled_summary(msgs, **_kwargs):
            fence.worker_ident = threading.get_ident()
            provider_entered.set()
            assert release_provider.wait(
                timeout=_HOST_BARRIER_TIMEOUT_SECONDS
            ), case
            return msgs

        compressor_a.compress = _stalled_summary

        messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        with _patch_chain([]):
            returned, _prompt = agent_a._compress_context(
                messages, "sys", approx_tokens=120_000, force=True
            )

        assert returned is messages, case
        assert fence.cooldown_owner_token(), (
            "the attempt never published its captured session cooldown "
            f"ownership epoch, so this control is vacuous; {case}"
        )
        assert fence.releases >= 1 and admitted, case
        assert order == ["release", "fallback"], (
            "the one permitted stall fallback must still run BETWEEN the lease "
            f"release and the host timeout cooldown record; {case}"
        )
        assert (
            db.get_compression_failure_cooldown_row(session_id)
            == _D1_HOST_ROW
        ), (
            "the host timeout callback of a released, superseded attempt wrote "
            f"a successor-owned cooldown row (#198 F1); {case}"
        )

        # The detached worker now unwinds through the real refused-commit-fence
        # branch: its owner-qualified restore AND its stall backoff must be
        # no-ops too. Barrier on the production helper, not a sleep.
        release_provider.set()
        assert backoff_done.wait(timeout=_HOST_BARRIER_TIMEOUT_SECONDS), case
        assert (
            db.get_compression_failure_cooldown_row(session_id)
            == _D1_HOST_ROW
        ), (
            "the detached worker's late unwind mutated the successor-owned "
            f"cooldown row; {case}"
        )
        if successor_released:
            assert db.get_compression_lock_holder(session_id) is None, case
        else:
            assert (
                db.get_compression_lock_holder(session_id) == admitted[0]
            ), (
                "the stale attempt released the successor's durable lease "
                f"(ABA); {case}"
            )
            db.release_compression_lock(session_id, admitted[0])


def test_host_timeout_still_records_the_cooldown_for_the_current_owner(
    tmp_path, monkeypatch
):
    """#198 F1 must not be closed by suppressing the HOST timeout ladder.

    Same real host path, no successor: the attempt still owns the captured
    session epoch, so ``_on_timeout`` must still persist the first rung of the
    unchanged 60/300/900 ladder. The row is read while the worker is provably
    still blocked, so this asserts the HOST write specifically.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "host-no-successor.db")
    session_id = "F1_HOST_TIMEOUT_NO_SUCCESSOR"
    db.create_session(session_id, source="cli")
    db.append_message(session_id, "user", "durable original")
    agent, compressor = _build_real_host_agent(db, session_id)

    monkeypatch.setattr(
        cc,
        "resolve_context_compression_timeouts",
        lambda cfg=None: (1.0, 2.0),
    )
    provider_entered = threading.Event()
    release_provider = threading.Event()
    fence = _HostIdleStallFence(provider_entered, on_release=lambda: None)
    monkeypatch.setattr(
        cc,
        "run_compress_context_with_progress_timeout",
        _pin_host_fence(cc.run_compress_context_with_progress_timeout, fence),
    )

    def _stalled_summary(msgs, **_kwargs):
        fence.worker_ident = threading.get_ident()
        provider_entered.set()
        assert release_provider.wait(timeout=_HOST_BARRIER_TIMEOUT_SECONDS)
        return msgs

    compressor.compress = _stalled_summary

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    before = time.time()
    try:
        with _patch_chain([]):
            agent._compress_context(
                messages, "sys", approx_tokens=120_000, force=True
            )

        assert fence.cooldown_owner_token(), (
            "the attempt never published its captured ownership epoch, so "
            "this control proves nothing about the qualified host write"
        )
        assert fence.releases >= 1
        # Read BEFORE the detached worker is released: the worker is provably
        # still blocked, so this row is the HOST callback's own write.
        row = db.get_compression_failure_cooldown_row(session_id)
    finally:
        release_provider.set()

    assert row["session_exists"] is True
    assert str(row["error"] or "").startswith("backoff:stalled:"), (
        "an attempt that still owns the captured session cooldown epoch lost "
        "its host timeout cooldown (#198 F1 must not be closed by "
        f"suppressing the ladder); row={row!r}"
    )
    assert row["cooldown_until"] is not None
    assert float(row["cooldown_until"]) >= before + 59.0, (
        f"the first 60s rung of the 60/300/900 ladder was not persisted: {row!r}"
    )
    assert compressor._consecutive_timeout_failures >= 1
    assert db.get_compression_lock_holder(session_id) is None, (
        "the host did not free the cancelled attempt's durable lease"
    )


# ---------------------------------------------------------------------------
# #198 F1 (unified) -- the CAPTURE -> PUBLICATION window.
#
# The attempt COMMITS its durable session cooldown ownership epoch
# (SessionDB.begin_compression_cooldown_ownership) BEFORE compress_context
# publishes that epoch on the fence, and the lock-setup barrier had already
# been left by then. A host idle cancellation landing in that window released a
# durably-owning attempt while the fence still exposed NO token, admitted a
# successor, and then let the host's own timeout record take the legacy
# no-owner branch and write UNQUALIFIED over the successor's row.
#
# Same real substrate as the host controls above: the real AIAgent host path
# and its real _on_timeout closure, the real wrapper, the real fallback
# opportunity, a real built-in ContextCompressor and a real SessionDB. The
# attempt is parked INSIDE the real ownership transaction; every step is a
# barrier on a real signal -- no sleep, no retry, no tolerance.
# ---------------------------------------------------------------------------


class _CaptureWindowFence(_HostIdleStallFence):
    """Classify ONE host idle timeout INSIDE the capture -> publication window.

    The idle barrier is the DURABLE OWNERSHIP STAMP rather than the provider
    entry used by the other host controls: the window under repair opens at the
    stamp and closes at the publication, so a timeout classified after the
    provider entry would prove nothing about it. The host's FIRST cancellation
    attempt is recorded together with both what it could observe and what the
    production fence answered, so this control cannot pass by never attempting
    one, and the record is taken strictly before the parked worker is allowed
    to resume (no race decides it).
    """

    def __init__(self, stamped, *, on_release, observed) -> None:
        super().__init__(stamped, on_release=on_release)
        self._observed = observed
        self.cancel_attempts = 0
        self.cancel_attempted = threading.Event()

    def try_cancel_before_commit(self):
        self.cancel_attempts += 1
        first = self.cancel_attempts == 1
        if first:
            self._observed["token_at_first_cancel_attempt"] = (
                self.cooldown_owner_token()
            )
        outcome = super().try_cancel_before_commit()
        if first:
            self._observed["first_cancel_outcome"] = outcome
            # Only now may the parked worker leave the ownership transaction:
            # the production answer above was taken while it was provably
            # inside it.
            self.cancel_attempted.set()
        return outcome


def test_host_idle_cancel_cannot_win_between_the_cooldown_stamp_and_publication(
    tmp_path, monkeypatch
):
    """#198 F1: no host release between the durable stamp and its publication.

    The attempt is parked inside the REAL
    ``SessionDB.begin_compression_cooldown_ownership`` transaction, after it
    committed this attempt's ownership epoch and before ``compress_context``
    publishes it. The real host wrapper reaches its cancellation attempt there;
    it must be refused until the epoch is host-visible, so the release, the
    successor it admits and the host timeout record all follow the publication
    and the successor's row survives.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "host-capture-window.db")
    session_id = "F1_HOST_CAPTURE_PUBLICATION"
    db.create_session(session_id, source="cli")
    db.append_message(session_id, "user", "durable original")
    db.record_compression_failure_cooldown(
        session_id, 4_000_000_000.0, "D0-primary"
    )

    agent_a, compressor_a = _build_real_host_agent(db, session_id)
    _agent_b, compressor_b = _build_real_host_agent(db, session_id)
    assert compressor_a is not compressor_b

    monkeypatch.setattr(
        cc,
        "resolve_context_compression_timeouts",
        lambda cfg=None: (1.0, 2.0),
    )
    provider_entered = threading.Event()
    release_provider = threading.Event()
    stamped = threading.Event()
    backoff_done = threading.Event()
    observed: dict = {
        "stamp": None,
        "token_at_first_cancel_attempt": "unread",
        "first_cancel_outcome": "unread",
    }
    admitted: list = []
    order: list = []

    def _admit_successor():
        order.append("release")
        admitted.append(
            _successor_takes_session(db, session_id, compressor_b, release=True)
        )

    fence = _CaptureWindowFence(
        stamped, on_release=_admit_successor, observed=observed
    )
    monkeypatch.setattr(
        cc,
        "run_compress_context_with_progress_timeout",
        _pin_host_fence(cc.run_compress_context_with_progress_timeout, fence),
    )
    real_retry = cc._retry_compression_on_fallback_chain

    def _observed_retry(**kwargs):
        order.append("fallback")
        return real_retry(**kwargs)

    monkeypatch.setattr(
        cc, "_retry_compression_on_fallback_chain", _observed_retry
    )
    real_backoff = cc._record_stall_interrupted_backoff

    def _observed_backoff(*args, **kwargs):
        try:
            return real_backoff(*args, **kwargs)
        finally:
            backoff_done.set()

    monkeypatch.setattr(
        cc, "_record_stall_interrupted_backoff", _observed_backoff
    )

    real_begin_ownership = SessionDB.begin_compression_cooldown_ownership

    def _park_inside_the_real_stamp(self, sid):
        # The REAL ownership transaction runs first: the epoch below is
        # durably committed while the fence still exposes nothing. The
        # successor's own later capture is not parked.
        snapshot = real_begin_ownership(self, sid)
        if sid == session_id and not stamped.is_set():
            observed["stamp"] = snapshot.get("owner_token")
            stamped.set()
            assert fence.cancel_attempted.wait(
                timeout=_HOST_BARRIER_TIMEOUT_SECONDS
            ), "the host never reached its cancellation attempt"
        return snapshot

    monkeypatch.setattr(
        SessionDB,
        "begin_compression_cooldown_ownership",
        _park_inside_the_real_stamp,
    )

    def _stalled_summary(msgs, **_kwargs):
        fence.worker_ident = threading.get_ident()
        provider_entered.set()
        assert release_provider.wait(timeout=_HOST_BARRIER_TIMEOUT_SECONDS)
        return msgs

    compressor_a.compress = _stalled_summary

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    try:
        with _patch_chain([]):
            returned, _prompt = agent_a._compress_context(
                messages, "sys", approx_tokens=120_000, force=True
            )

        assert returned is messages
        # Non-vacuity: the durable stamp really happened, and the host really
        # attempted to cancel while that epoch was still unpublished.
        assert observed["stamp"], "the real durable ownership stamp never ran"
        assert observed["token_at_first_cancel_attempt"] is None, (
            "the host's cancellation attempt did not land inside the "
            "capture -> publication window; this control would be vacuous"
        )
        assert observed["first_cancel_outcome"] is None, (
            "the host's idle cancellation was ADMITTED while the attempt was "
            "still inside its durable cooldown ownership transaction, so the "
            "lease release -- and the successor it admits -- can precede the "
            "publication (#198 F1 capture -> publication window)"
        )
        assert fence.cancel_attempts >= 2, (
            "the deferred attempt was never retried, so no release was "
            "ordered behind the publication"
        )
        assert fence.cooldown_owner_token() == observed["stamp"], (
            "the host-visible state is not the epoch this attempt durably "
            "stamped (#198 F1)"
        )
        assert fence.releases >= 1 and admitted, (
            "the host must still promptly release a genuinely timed-out "
            "attempt; the fix must not retain the lease instead"
        )
        assert order == ["release", "fallback"], (
            "the one permitted stall fallback must still run BETWEEN the "
            "lease release and the host timeout cooldown record"
        )
        # Non-vacuity: the real _on_timeout reached the unchanged ladder, so
        # the stale-write path was exercised rather than bypassed.
        assert compressor_a._consecutive_timeout_failures >= 1, (
            "the host timeout record never ran, so nothing was proven about "
            "the no-token legacy path"
        )
        assert (
            db.get_compression_failure_cooldown_row(session_id) == _D1_HOST_ROW
        ), (
            "the host released a durably-owning attempt and then wrote an "
            "unqualified cooldown row through the false no-token legacy path "
            "(#198 F1 capture -> publication window)"
        )

        # The detached worker now unwinds through the real refused-commit
        # branch; its late restore and stall backoff must be no-ops too.
        release_provider.set()
        assert backoff_done.wait(timeout=_HOST_BARRIER_TIMEOUT_SECONDS)
        assert (
            db.get_compression_failure_cooldown_row(session_id) == _D1_HOST_ROW
        ), "the detached worker's late unwind mutated the successor-owned row"
    finally:
        release_provider.set()
    assert db.get_compression_lock_holder(session_id) is None


# ---------------------------------------------------------------------------
# #198 F1 residual (lease -> capture window).
#
# The host's legacy UNQUALIFIED cooldown write is reserved for an attempt that
# provably never established native durable ownership. An attempt that ACQUIRED
# a native durable lease and had that lease RELEASED before it could stamp an
# epoch is NOT that attempt: its own release is what admits the successor whose
# row the unqualified write would overwrite.
#
# These checks are bounded and synthetic. No durable store, no lease, no host
# wrapper, no compression attempt and no provider call is exercised here; only
# the publication contract, the refusal a matchless token must produce under an
# owner-qualified compare-and-write, and the preserved never-native-lease
# legacy path are asserted.
# ---------------------------------------------------------------------------


class _PublishedOwnerState:
    """Synthetic stand-in for the per-attempt fence's host-visible token."""

    def __init__(self) -> None:
        self.token = None

    def publish_cooldown_owner_token(self, token):
        self.token = token or None

    def cooldown_owner_token(self):
        return self.token


class _OwnershipMintingStore:
    """A store whose TYPE exposes the native ownership stamp API.

    Reaching the stamp itself is a durable mutation and is out of scope for
    these checks, so the method refuses to run.
    """

    def begin_compression_cooldown_ownership(self, session_id):
        raise AssertionError(
            "these checks must not reach a durable ownership transaction"
        )


class _OwnerQualifiedRow:
    """Synthetic owner-qualified cooldown row.

    Stands in for the compare-and-write SessionDB performs inside the same
    transaction as the mutation: a mismatched epoch writes nothing, while the
    legacy unqualified branch overwrites whoever owns the row. Purely
    in-memory -- nothing is persisted and no lock is taken.
    """

    def __init__(self, owner, value) -> None:
        self.owner = owner
        self.value = value

    def record(self, value, *, owner_token=None):
        if owner_token is None:
            self.value = value
            return True
        if owner_token != self.owner:
            return False
        self.value = value
        return True


def _native_lease_holder(session_db):
    """A built-in compressor bound the way a native leaseholder is bound."""
    compressor = _make_compressor()
    compressor._session_db = session_db
    compressor._session_id = "F1_SYNTHETIC_NATIVE_LEASE"
    return compressor


def test_native_lease_without_an_exact_epoch_publishes_a_matchless_token():
    """A native leaseholder that captured no epoch must not read as no-owner."""
    published = _PublishedOwnerState()
    compressor = _native_lease_holder(_OwnershipMintingStore())

    cc._publish_captured_cooldown_ownership(published, compressor, None, None)

    token = published.cooldown_owner_token()
    assert token is not None, (
        "an attempt holding a NATIVE durable lease published no ownership "
        "token, so a host cooldown mutation after its release takes the "
        "legacy UNQUALIFIED branch and can overwrite the successor that its "
        "own release admitted (#198 F1 lease -> capture window)"
    )
    assert token == cc.COOLDOWN_OWNERSHIP_UNRESOLVED

    successor_epoch = "0123456789abcdef0123456789abcdef"
    assert token != successor_epoch and len(token) != len(successor_epoch), (
        "the published stand-in must be unable to collide with a minted "
        "ownership epoch"
    )
    row = _OwnerQualifiedRow(successor_epoch, "D1-successor")
    assert row.record("D2-host-timeout", owner_token=token) is False, (
        "a no-authority token did not refuse the host cooldown mutation"
    )
    assert row.value == "D1-successor", (
        "a no-authority host cooldown write mutated a successor-owned row"
    )
    # Non-vacuity: the same row DOES accept its true owner, so the refusal
    # above is the token mismatch and not an inert stub.
    assert row.record("D2-owner", owner_token=successor_epoch) is True
    assert row.value == "D2-owner"


def test_never_native_lease_attempts_keep_the_unqualified_legacy_path():
    """Preserved legacy: an attempt that can never stamp publishes nothing."""
    unbound = _native_lease_holder(_OwnershipMintingStore())
    unbound._session_db = None
    legacy_store = _native_lease_holder(SimpleNamespace())
    third_party = SimpleNamespace(
        _session_db=_OwnershipMintingStore(),
        _session_id="F1_SYNTHETIC_THIRD_PARTY",
    )

    for compressor in (unbound, legacy_store, third_party):
        published = _PublishedOwnerState()
        cc._publish_captured_cooldown_ownership(
            published, compressor, None, None
        )
        assert published.cooldown_owner_token() is None, (
            "a never-native-lease attempt must keep publishing nothing so its "
            "host cooldown path stays byte-unchanged"
        )


def test_native_lease_publication_precedes_the_cancellation_hook():
    """The publication is bound to lease ownership, not to the capture alone.

    Structural (source-text) check only: it reads ``compress_context``'s own
    source and never runs it. The window under repair opens the instant this
    attempt owns the durable lease and can therefore have that lease released
    under it, so the host-visible ownership state must already be published
    before the holder-qualified release hook is handed to the fence.
    """
    import inspect

    source = inspect.getsource(cc.compress_context)
    guard = source.index("if _lock_holder is not None:")
    hook = source.index("register_cancelled_lock_release(", guard)
    assert source.find("_publish_captured_cooldown_ownership(", guard, hook) != -1, (
        "the native leaseholder branch does not publish its host-visible "
        "cooldown ownership state before handing the lease release hook to "
        "the fence, so a cancellation admitted in that window leaves the host "
        "reading 'no token' for an attempt that held a native lease "
        "(#198 F1 lease -> capture window)"
    )
