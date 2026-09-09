"""Regressions for #76354 review F3/F4/F5 — worker isolation, durable lease
cancellation, and session ContextVar repair.

F3: a timed-out worker running an IN-PLACE-MUTATING context engine must not
be able to touch the caller's live transcript — assertions run WHILE the
worker is still blocked inside the engine (released only afterwards).

F4: the reviewer's exact 5-step regression — block summary indefinitely →
host timeout → NEW compressor acquires the durable lock while the old
summary is STILL blocked → release old worker → prove it cannot clear
cooldown / release the new holder's lease / publish state.

F5: after a successful out-of-place rotation, the CALLER's session
ContextVar resolves to the child id (get_session_env / HERMES_SESSION_ID).
"""

from __future__ import annotations

import copy
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.conversation_compression import CompressionCommitFence
from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str, **compressor_kwargs):
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

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor._last_compression_made_progress = True
    compressor._last_summary_fallback_used = False
    agent.context_compressor = compressor
    # The compressor is a stub — the one-time compression-model feasibility
    # probe would resolve a REAL auxiliary provider (credential pools, live
    # token exchange) before the engine runs. In hermetic CI there are no
    # credentials, so the probe aborts compression before the stub engine
    # ever starts and every blocked-state assertion goes vacuous. These
    # tests exercise isolation/fencing, never aux-model feasibility.
    agent._compression_feasibility_checked = True
    return agent


# Bounds for the test-only schedule controls below. They turn a hang into a
# loud failure; none of them is a production budget and none of them is the
# proof of anything.
_PROVIDER_ENTRY_WATCHDOG_SECONDS = 10.0
_STALE_IDLE_SECONDS = 60.0
_HOST_EPOCH_SHIFT_SECONDS = 30.0


class _ProviderEntryFence(CompressionCommitFence):
    """Order protected-provider ENTRY before any host timeout classification.

    Production is untouched — only the fence this file hands the host is
    controlled (the ``_ScheduledFence`` idiom of
    ``test_compression_attempt_lifecycle.py``), and every configured budget is
    left exactly as it is:

    * ``deadline_exceeded`` cannot be true while the control is engaged, so
      neither the pre-start refusal (``_CompressionWorkerPreStartExpiry``) nor
      the cancelled-fence pre-summary gate can skip summary dispatch before the
      provider callback has run. Without that edge a loaded runner makes
      ``provider_started`` a SAMPLED precondition instead of a proved one;
    * the host's FIRST idle sample blocks until ``provider_started`` is set, so
      no timeout branch is classified against an attempt whose provider never
      entered;
    * the host's elapsed-time origin is shifted (the same idiom) so the TOTAL
      ceiling term cannot win the classification race: that path deliberately
      RETAINS a live worker's durable lease, which is a different contract from
      the idle cancellation under test here;
    * after entry the host is told this attempt's real state — a blocked
      provider that has reported no progress — as a stale idle age, so the
      host's own unmodified idle classification cancels the attempt while
      ``release_provider`` is still unset.

    Only the thread that armed the epoch (the host) is answered from the
    control; worker-side readers keep the real values.
    """

    def __init__(
        self,
        provider_started: threading.Event,
        *,
        entry_timeout: float = _PROVIDER_ENTRY_WATCHDOG_SECONDS,
        stale_idle_seconds: float = _STALE_IDLE_SECONDS,
        host_epoch_shift: float = _HOST_EPOCH_SHIFT_SECONDS,
    ) -> None:
        super().__init__()
        self._provider_started = provider_started
        self._entry_timeout = entry_timeout
        self._stale_idle_seconds = stale_idle_seconds
        self._host_epoch_shift = host_epoch_shift
        self._host_thread_ident = None
        self.entry_observed = False

    def begin_attempt(self, total_ceiling_seconds: float) -> float:
        self._host_thread_ident = threading.get_ident()
        epoch = super().begin_attempt(total_ceiling_seconds)
        return epoch + self._host_epoch_shift

    @property
    def deadline_exceeded(self) -> bool:
        if self._host_epoch_shift or not self._provider_started.is_set():
            return False
        return super().deadline_exceeded

    def seconds_since_progress(self) -> float:
        if threading.get_ident() != self._host_thread_ident:
            return super().seconds_since_progress()
        if not self.entry_observed:
            # Barrier, not a sleep: the host may not evaluate a timeout branch
            # until the protected provider callback has provably entered.
            if not self._provider_started.wait(timeout=self._entry_timeout):
                raise AssertionError(
                    "the protected provider callback never entered — a "
                    "timeout classified here would prove nothing about a "
                    "blocked provider stream"
                )
            self.entry_observed = True
        return self._stale_idle_seconds


class _CancelBeforeDispatchFence(CompressionCommitFence):
    """Linearize cancellation between lock setup and the dispatch gate.

    Deterministic and clock-free: cancellation becomes visible exactly when the
    worker leaves its fenced durable-lock setup — after the holder and its
    holder-qualified release hook exist, and before the pre-summary dispatch
    gate. Production then refuses to start expensive summary work, which is the
    documented fail-closed contract, NOT a defect.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lock_setup_observed = threading.Event()
        self._cancel_armed = threading.Event()

    def begin_lock_setup(self) -> bool:
        self.lock_setup_observed.set()
        return super().begin_lock_setup()

    def finish_lock_setup(self) -> None:
        super().finish_lock_setup()
        self._cancel_armed.set()

    @property
    def deadline_exceeded(self) -> bool:
        return self._cancel_armed.is_set()


def _pin_attempt_fence(real, fence):
    """Hand the host ONE controlled fence for the FIRST attempt only.

    ``AIAgent._compress_context`` mints its own fence and takes the unpooled
    direct path when a caller supplies one, so the control is installed where
    the host RECEIVES its fence instead. The production wrapper itself runs
    unchanged. ``stall_fallback`` is pinned off for that one attempt (a
    documented parameter of the same production API, used the same way by the
    #78981/#97488 tests) so exactly ONE attempt is under observation and no
    second attempt's bookkeeping can satisfy the release assertions below; any
    later call is forwarded completely uncontrolled.
    """
    pinned = []

    def _wrapper(**kwargs):
        if not pinned:
            pinned.append(True)
            kwargs["fence"] = fence
            kwargs["stall_fallback"] = False
        return real(**kwargs)

    return _wrapper


def test_f3_mutating_engine_cannot_touch_live_transcript_after_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """In-place-mutating engine + host timeout → caller transcript untouched.

    Byte-identity is asserted WHILE the worker is still blocked inside the
    engine; the worker is released only after those assertions.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "F3_ISOLATION"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent._cached_system_prompt = "sys"

    # Fast host timeout for the owned wrapper.
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda cfg=None: (0.6, 1.2),
    )

    engine_started = threading.Event()
    release_engine = threading.Event()
    mutated_lists = []

    def _mutating_engine(msgs, **_kwargs):
        # Legacy/plugin-engine contract: mutate the input list IN PLACE.
        engine_started.set()
        msgs[:] = [{"role": "assistant", "content": "ENGINE GARBAGE"}]
        mutated_lists.append(msgs)
        assert release_engine.wait(timeout=30)
        return msgs

    agent.context_compressor.compress.side_effect = _mutating_engine

    live = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    baseline = copy.deepcopy(live)

    try:
        returned, _sp = agent._compress_context(
            live, "sys", approx_tokens=120_000
        )
        # Host timed out and returned while the engine is STILL blocked.
        assert engine_started.wait(timeout=5)
        assert not release_engine.is_set()
        assert returned is live
        # ── The core assertion, made while the worker keeps running ──────
        assert live == baseline, (
            "live transcript mutated by a detached compression worker"
        )
        # The engine did mutate a list — the SNAPSHOT, not the caller's.
        assert mutated_lists and mutated_lists[0] is not live
        # Give the blocked worker extra time to prove no delayed publication.
        time.sleep(0.2)
        assert live == baseline
    finally:
        release_engine.set()
    # After the late worker finishes, the live transcript must STILL be
    # untouched (publication only on admitted commit — which was cancelled).
    deadline = time.time() + 5
    while time.time() < deadline and db.get_compression_lock_holder(session_id):
        time.sleep(0.02)
    assert live == baseline


def test_host_timeout_releases_pool_slot_while_protected_provider_is_still_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    """Fence timeout must unwind the compression owner, not occupy the pool.

    Protected auxiliary calls isolate their provider stream on a daemon thread.
    The compression owner must observe its commit-fence cancellation and unwind
    immediately; otherwise four slow streams consume all four shared compression
    workers until the auxiliary stream's much longer absolute ceiling expires.
    """
    from agent import auxiliary_client as aux
    from agent import conversation_compression as cc

    deadline = time.time() + 5
    while time.time() < deadline:
        with cc._compress_admission_lock:
            if cc._compress_admitted_count == 0:
                break
        time.sleep(0.02)
    with cc._compress_admission_lock:
        assert cc._compress_admitted_count == 0

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "F3_PROVIDER_OWNER_RELEASE"
    db.create_session(session_id, source="cli")
    db.append_message(session_id, "user", "durable original")
    agent = _build_agent_with_db(db, session_id)
    agent._cached_system_prompt = "sys"
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda cfg=None: (0.05, 0.1),
    )

    provider_started = threading.Event()
    release_provider = threading.Event()
    provider_finished = threading.Event()

    # Happens-before control (#97488 diagnosis): the host cannot classify a
    # timeout until the protected provider callback has entered. The budgets
    # above are unchanged — only the fence handed to the host is controlled.
    fence = _ProviderEntryFence(provider_started)
    monkeypatch.setattr(
        cc,
        "run_compress_context_with_progress_timeout",
        _pin_attempt_fence(
            cc.run_compress_context_with_progress_timeout, fence
        ),
    )

    def _blocked_provider(_kwargs):
        provider_started.set()
        assert release_provider.wait(timeout=10)
        provider_finished.set()
        return "late-provider-result"

    def _compress_with_protected_provider(msgs, **_kwargs):
        aux._run_protected_sync_provider_call(_blocked_provider, {})
        return msgs

    agent.context_compressor.compress.side_effect = _compress_with_protected_provider
    live = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    baseline = copy.deepcopy(live)
    durable_before = db.get_messages_as_conversation(session_id)

    try:
        returned, _sp = agent._compress_context(
            live, "sys", approx_tokens=120_000
        )
        assert returned is live
        # Precondition, PROVED rather than sampled: the protected provider
        # callback entered before the host was allowed to classify any
        # timeout branch, and it is still blocked right now.
        assert fence.entry_observed and provider_started.is_set(), (
            "the host classified a timeout with no provider callback in "
            "flight — every assertion below would lose its precondition"
        )
        assert not release_provider.is_set()
        # The attempt was cancelled by the host's own timeout classification,
        # not by the ceiling term the control keeps out of the race.
        assert fence.is_cancelled and not fence.deadline_exceeded

        deadline = time.time() + 5
        while time.time() < deadline:
            with cc._compress_admission_lock:
                if cc._compress_admitted_count == 0:
                    break
            time.sleep(0.01)
        with cc._compress_admission_lock:
            assert cc._compress_admitted_count == 0, (
                "timed-out compression owner retained its shared pool slot "
                "while the isolated provider stream was still blocked"
            )
        # Only a SETTLED owner future frees that ticket, so the compression
        # owner provably returned while the provider stayed blocked.
        assert not release_provider.is_set()

        # A replacement attempt is admissible on the freed slot...
        replacement = cc._admit_compression_job()
        assert replacement is not None, (
            "no replacement compression could be admitted while the "
            "cancelled attempt's provider was still blocked"
        )
        replacement.release()
        # ...and the old holder's durable lease was released, so a NEW
        # compressor can take the session lock (F4 ordering) with the
        # original provider still blocked.
        replacement_holder = "pid:new:replacement"
        acquired = False
        deadline = time.time() + 5
        while time.time() < deadline:
            if db.try_acquire_compression_lock(
                session_id, replacement_holder, ttl_seconds=60
            ):
                acquired = True
                break
            time.sleep(0.02)
        assert acquired, (
            "the cancelled attempt did not release its durable holder while "
            "its protected provider stream was still blocked"
        )
        assert not release_provider.is_set()
        assert db.get_compression_lock_holder(session_id) == replacement_holder
        # Nothing was published while the provider was blocked.
        assert live == baseline
        assert agent.session_id == session_id
        assert db.get_messages_as_conversation(session_id) == durable_before
    finally:
        release_provider.set()
        deadline = time.time() + 5
        while time.time() < deadline:
            with cc._compress_admission_lock:
                if cc._compress_admitted_count == 0:
                    break
            time.sleep(0.02)
    # The late provider finished AFTER cancellation: it must not publish a
    # transcript, rotate the session, commit durable state, or delete the
    # replacement holder's lease (ABA).
    assert provider_finished.wait(timeout=5)
    time.sleep(0.2)  # settle: give the late stream every chance to misbehave
    assert live == baseline, "late provider result published over the transcript"
    assert agent.session_id == session_id
    assert db.get_messages_as_conversation(session_id) == durable_before
    assert db.get_compression_lock_holder(session_id) == replacement_holder, (
        "the late cancelled worker released the replacement holder's lease "
        "(ABA)"
    )
    db.release_compression_lock(session_id, replacement_holder)


def test_pre_dispatch_cancellation_refuses_provider_work_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    """Negative control for the schedule above — VALID production behaviour.

    Cancellation is linearized deterministically at the worker's own
    lock-setup boundary: after the durable holder and its holder-qualified
    release hook exist, and before the pre-summary dispatch gate. Production
    then refuses to start expensive summary work for an already-cancelled
    fence, so the protected provider is never dispatched and
    ``provider_started`` correctly stays false.

    That refusal is the documented fail-closed contract, NOT a production
    defect. It is exactly the schedule the uncontrolled 0.05/0.1s budgets can
    reach on a loaded runner, which is why the test above establishes provider
    entry as a happens-before precondition instead of sampling it. The
    isolation invariant still holds on this branch: the owner unwinds, its
    admission slot and durable holder are released, and nothing is published.
    """
    from agent import auxiliary_client as aux
    from agent import conversation_compression as cc

    deadline = time.time() + 5
    while time.time() < deadline:
        with cc._compress_admission_lock:
            if cc._compress_admitted_count == 0:
                break
        time.sleep(0.02)

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "F3_PRE_DISPATCH_CANCEL"
    db.create_session(session_id, source="cli")
    db.append_message(session_id, "user", "durable original")
    agent = _build_agent_with_db(db, session_id)
    agent._cached_system_prompt = "sys"
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda cfg=None: (0.05, 0.1),
    )

    provider_started = threading.Event()

    def _provider(_kwargs):
        # Must never run on this schedule; it returns at once so a breach
        # fails the assertion below instead of hanging the suite.
        provider_started.set()
        return "provider-result"

    def _compress_with_protected_provider(msgs, **_kwargs):
        aux._run_protected_sync_provider_call(_provider, {})
        return msgs

    fence = _CancelBeforeDispatchFence()
    monkeypatch.setattr(
        cc,
        "run_compress_context_with_progress_timeout",
        _pin_attempt_fence(
            cc.run_compress_context_with_progress_timeout, fence
        ),
    )
    agent.context_compressor.compress.side_effect = _compress_with_protected_provider
    live = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    baseline = copy.deepcopy(live)
    durable_before = db.get_messages_as_conversation(session_id)

    returned, _sp = agent._compress_context(live, "sys", approx_tokens=120_000)

    # The worker DID enter compress_context (it reached the fenced durable
    # lock setup): this is the cancelled-before-dispatch branch, not a
    # never-started worker and not a pre-start refusal.
    assert fence.lock_setup_observed.is_set(), (
        "the worker never entered compress_context, so this schedule is not "
        "the cancelled-before-dispatch branch it is meant to pin"
    )
    # ...and production correctly refused to dispatch expensive summary work.
    assert not provider_started.is_set(), (
        "an already-cancelled fence dispatched protected provider work"
    )
    assert fence.is_cancelled
    assert returned is live
    assert live == baseline

    deadline = time.time() + 5
    while time.time() < deadline:
        with cc._compress_admission_lock:
            if cc._compress_admitted_count == 0:
                break
        time.sleep(0.02)
    with cc._compress_admission_lock:
        assert cc._compress_admitted_count == 0, (
            "the pre-dispatch-cancelled owner retained its shared pool slot"
        )

    replacement_holder = "pid:new:replacement"
    acquired = False
    deadline = time.time() + 5
    while time.time() < deadline:
        if db.try_acquire_compression_lock(
            session_id, replacement_holder, ttl_seconds=60
        ):
            acquired = True
            break
        time.sleep(0.02)
    assert acquired, (
        "the pre-dispatch-cancelled attempt did not release its durable "
        "holder"
    )
    assert agent.session_id == session_id
    assert db.get_messages_as_conversation(session_id) == durable_before
    db.release_compression_lock(session_id, replacement_holder)


def test_slow_pool_setup_still_reaches_the_protected_provider(
    tmp_path: Path, monkeypatch
) -> None:
    """Cold-start setup longer than the ceiling must not eat the attempt.

    The wrapper's lazy setup runs BEFORE the attempt epoch; with the ceiling
    armed ahead of it, a slow CI start expired the fence before the
    compression owner could enter, so the isolated provider never started and
    the owner-slot release proved nothing (#97488). Budgets are unchanged.
    """
    from agent import auxiliary_client as aux
    from agent import conversation_compression as cc

    deadline = time.time() + 5
    while time.time() < deadline:
        with cc._compress_admission_lock:
            if cc._compress_admitted_count == 0:
                break
        time.sleep(0.02)

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "F3_SLOW_SETUP_OWNER_RELEASE"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent._cached_system_prompt = "sys"
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda cfg=None: (0.05, 0.1),
    )

    real_executor = cc._get_compress_timeout_executor

    def _slow_setup():
        executor = real_executor()
        time.sleep(0.3)  # > the 0.1s ceiling armed for this attempt
        return executor

    monkeypatch.setattr(cc, "_get_compress_timeout_executor", _slow_setup)

    provider_started = threading.Event()
    release_provider = threading.Event()

    def _blocked_provider(_kwargs):
        provider_started.set()
        assert release_provider.wait(timeout=10)
        return "late-provider-result"

    def _compress_with_protected_provider(msgs, **_kwargs):
        aux._run_protected_sync_provider_call(_blocked_provider, {})
        return msgs

    agent.context_compressor.compress.side_effect = _compress_with_protected_provider
    live = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    baseline = copy.deepcopy(live)

    try:
        returned, _sp = agent._compress_context(
            live, "sys", approx_tokens=120_000
        )
        assert returned is live
        assert provider_started.wait(timeout=2), (
            "the compression owner never entered: setup latency consumed the "
            "attempt before its epoch was established"
        )
        assert not release_provider.is_set()

        deadline = time.time() + 1
        while time.time() < deadline:
            with cc._compress_admission_lock:
                if cc._compress_admitted_count == 0:
                    break
            time.sleep(0.01)
        with cc._compress_admission_lock:
            assert cc._compress_admitted_count == 0, (
                "timed-out compression owner retained its shared pool slot "
                "while the isolated provider stream was still blocked"
            )
        assert live == baseline
    finally:
        release_provider.set()
        deadline = time.time() + 5
        while time.time() < deadline:
            with cc._compress_admission_lock:
                if cc._compress_admitted_count == 0:
                    break
            time.sleep(0.02)
    assert live == baseline, "late provider result published over the transcript"


def test_f4_five_step_stale_holder_regression(tmp_path: Path) -> None:
    """Reviewer's exact 5-step durable-lease regression (#76354 F4).

    1. Block the original summary indefinitely.
    2. Let the host time out.
    3. Prove another compressor can acquire the durable lock BEFORE the
       original summary is released.
    4. Release the old worker.
    5. Prove it cannot clear cooldown, release the new holder's lease, or
       publish stale state.
    """
    from agent.conversation_compression import (
        CompressionCommitFence,
        run_compress_context_with_progress_timeout,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "F4_FIVE_STEP"
    db.create_session(session_id, source="telegram")
    db.append_message(session_id, "user", "original durable")

    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"

    summary_started = threading.Event()
    release_summary = threading.Event()

    def _blocked_summary(*_args, **_kwargs):
        summary_started.set()
        assert release_summary.wait(timeout=30)  # step 1: blocked
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] stale summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _blocked_summary
    # Track cooldown-clear attempts on the OLD worker's compressor.
    cooldown_cleared = []
    agent.context_compressor._clear_compression_failure_cooldown = (
        lambda: cooldown_cleared.append(True)
    )

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def _worker(fence):
        return agent._compress_context(
            messages, "sys", approx_tokens=120_000, commit_fence=fence
        )

    # Step 2: host-owned progress wait times out while summary is blocked.
    result_msgs, _prompt = run_compress_context_with_progress_timeout(
        worker=_worker,
        messages=messages,
        system_prompt_fallback="fallback",
        idle_timeout_seconds=0.6,
        total_ceiling_seconds=1.2,
    )
    assert summary_started.wait(timeout=5)
    assert not release_summary.is_set()  # old worker STILL blocked
    assert result_msgs is messages

    # Step 3: a NEW compressor acquires the durable lock while the old
    # summary remains blocked. The host's holder-qualified release freed
    # the old lease (refresher stopped + row deleted, holder-scoped).
    new_holder = "pid:new:contender"
    deadline = time.time() + 5
    acquired = False
    while time.time() < deadline:
        if db.try_acquire_compression_lock(session_id, new_holder, ttl_seconds=60):
            acquired = True
            break
        time.sleep(0.02)
    assert acquired, (
        "a new compressor must be able to acquire the durable lock while "
        "the timed-out worker is still blocked in its summary"
    )
    assert not release_summary.is_set()  # provably still step-3 state
    assert db.get_compression_lock_holder(session_id) == new_holder

    pre_release_rows = db.get_messages_as_conversation(session_id)

    # Step 4: release the old worker.
    release_summary.set()
    # Wait for the late worker to fully unwind (it must NOT touch the lock).
    deadline = time.time() + 5
    while time.time() < deadline:
        if db.get_compression_lock_holder(session_id) != new_holder:
            break  # would be a failure — checked below
        if cooldown_cleared:
            break
        time.sleep(0.02)
    time.sleep(0.3)  # settle: give the stale worker every chance to misbehave

    # Step 5a: it cannot clear the cooldown.
    assert not cooldown_cleared, (
        "late cancelled worker cleared the compression failure cooldown"
    )
    # Step 5b: it cannot release the NEW holder's lease (holder-qualified).
    assert db.get_compression_lock_holder(session_id) == new_holder, (
        "late worker released the replacement holder's durable lease (ABA)"
    )
    # Step 5c: it cannot publish stale state — transcript unchanged, no
    # in-place compaction landed, session id did not rotate.
    post_release_rows = db.get_messages_as_conversation(session_id)
    assert post_release_rows == pre_release_rows
    assert agent.session_id == session_id
    db.release_compression_lock(session_id, new_holder)


def test_f5_session_contextvar_rebound_after_rotation(
    tmp_path: Path, monkeypatch
) -> None:
    """Post-compression tool reads of HERMES_SESSION_ID see the CHILD id."""
    from gateway.session_context import (
        clear_session_vars,
        get_session_env,
        set_session_vars,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "F5_CTXVAR_PARENT"
    db.create_session(parent_sid, source="telegram")
    agent = _build_agent_with_db(db, parent_sid)
    agent.compression_in_place = False  # rotation mode
    agent._cached_system_prompt = "sys"

    # Enable the owned pooled wrapper so rotation happens on a WORKER thread
    # (the caller's ContextVar can only be repaired by the caller).
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda cfg=None: (5.0, 10.0),
    )

    # Simulate the gateway's bound session context for the caller.
    tokens = set_session_vars(session_id=parent_sid, platform="telegram")
    try:
        assert get_session_env("HERMES_SESSION_ID") == parent_sid

        messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(messages, "sys", approx_tokens=120_000)

        assert agent.session_id != parent_sid  # rotation happened
        # ── The F5 contract: caller-context reads resolve to the child ──
        assert get_session_env("HERMES_SESSION_ID") == agent.session_id, (
            "caller's session ContextVar still returns the parent id after "
            "an out-of-place compression rotation"
        )
    finally:
        clear_session_vars(tokens)
