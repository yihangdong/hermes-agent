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
