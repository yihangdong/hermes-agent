"""Tests for the gateway delivery-obligation ledger (gateway/delivery_ledger.py).

State machine, dead-owner claiming, attempts cap, stale cutoff, retention,
id stability, and the startup redelivery sweep's contract:
- pending rows redeliver plainly (send never started, no dup risk)
- attempting/failed rows carry the recovered-reply marker (honest
  at-least-once; ambiguity is labeled, never silently resent)
- rows owned by a LIVE process are never claimed
- poison rows abandon at the attempts cap / stale cutoff
"""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Isolated state.db per test (autouse HERMES_HOME isolation already
    redirects get_hermes_home; make the redirect explicit and per-test)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _record(oid="ob-1", session_key="agent:main:slack:channel:C1", **kw):
    return dl.record_obligation(
        obligation_id=oid,
        session_key=session_key,
        platform=kw.get("platform", "slack"),
        chat_id=kw.get("chat_id", "C1"),
        thread_id=kw.get("thread_id", "171.001"),
        content=kw.get("content", "the final answer"),
    )


def _row(oid):
    with dl._connect() as conn:
        r = conn.execute(
            """SELECT state, attempts, owner_pid, content, next_attempt_at,
                      generation
               FROM delivery_obligations WHERE obligation_id=?""",
            (oid,),
        ).fetchone()
    return None if r is None else {
        "state": r[0], "attempts": r[1], "owner_pid": r[2], "content": r[3],
        "next_attempt_at": r[4], "generation": r[5],
    }


def _orphan(oid):
    """Make the row look like it belongs to a dead process."""
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )


class TestStateMachine:
    def test_record_starts_pending(self):
        _record()
        assert _row("ob-1")["state"] == "pending"

    def test_full_happy_path(self):
        _record()
        dl.mark_attempting("ob-1")
        assert _row("ob-1")["state"] == "attempting"
        dl.mark_delivered("ob-1")
        assert _row("ob-1")["state"] == "delivered"

    def test_failed_records_error(self):
        _record()
        dl.mark_attempting("ob-1")
        dl.mark_failed("ob-1", "chat_not_found")
        assert _row("ob-1")["state"] == "failed"

    def test_terminal_failure_is_abandoned_without_retry(self):
        generation = _record()
        dl.mark_attempting("ob-1", generation=generation)
        dl.mark_failed(
            "ob-1", "chat_not_found", generation=generation, retryable=False
        )
        row = _row("ob-1")
        assert row["state"] == "abandoned"
        assert row["next_attempt_at"] == 0
        assert dl.sweep_recoverable() == []

    def test_failed_row_is_released_for_same_process_retry_after_backoff(self):
        _record()
        dl.mark_attempting("ob-1")
        before = time.time()
        dl.mark_failed("ob-1", "rate limited", retry_after_seconds=42)
        row = _row("ob-1")
        assert row["owner_pid"] is None
        assert row["next_attempt_at"] >= before + 42
        assert dl.sweep_recoverable(now=before + 41) == []
        claimed = dl.sweep_recoverable(now=row["next_attempt_at"] + 0.01)
        assert [item["obligation_id"] for item in claimed] == ["ob-1"]

    def test_failed_row_uses_bounded_exponential_backoff(self):
        _record()
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET attempts=? WHERE obligation_id=?",
                (2, "ob-1"),
            )
        before = time.time()
        dl.mark_failed("ob-1", "transient")
        delay = _row("ob-1")["next_attempt_at"] - before
        assert dl.RETRY_BASE_SECONDS * 4 - 1 <= delay <= dl.RETRY_BASE_SECONDS * 4 + 1

    def test_rerecord_same_id_is_idempotent(self):
        first = _record()
        dl.mark_attempting("ob-1", generation=first)
        second = _record()  # same turn re-record creates a fenced generation
        assert _row("ob-1")["state"] == "pending"
        assert second == first + 1

    def test_stale_completion_cannot_mutate_rerecorded_generation(self):
        first = _record()
        second = _record()
        dl.mark_delivered("ob-1", generation=first)
        assert _row("ob-1")["state"] == "pending"
        dl.mark_failed("ob-1", "stale", generation=first)
        assert _row("ob-1")["state"] == "pending"
        dl.mark_delivered("ob-1", generation=second)
        assert _row("ob-1")["state"] == "delivered"


class TestObligationId:
    def test_stable_and_distinct(self):
        a = dl.compute_obligation_id("sk1", "msg1", "hello")
        assert a == dl.compute_obligation_id("sk1", "msg1", "hello")
        # Different thread (baked into session_key) → different id. This is
        # the cron-topic collision class from the earlier outbox attempt.
        assert a != dl.compute_obligation_id("sk1:threadB", "msg1", "hello")
        assert a != dl.compute_obligation_id("sk1", "msg2", "hello")
        assert a != dl.compute_obligation_id("sk1", "msg1", "other")
        assert len(a) == 24


class TestSweep:
    def test_live_owner_rows_never_claimed(self):
        _record()  # owner = this (live) process
        assert dl.sweep_recoverable() == []

    def test_dead_owner_pending_claimed_without_marker(self):
        _record()
        _orphan("ob-1")
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is False
        assert claimed[0]["attempts"] == 1
        # Claim re-stamps ownership: a second sweep in the same (live)
        # process must not double-claim.
        assert dl.sweep_recoverable() == []

    def test_dead_owner_attempting_needs_marker(self):
        _record()
        dl.mark_attempting("ob-1")
        _orphan("ob-1")
        claimed = dl.sweep_recoverable()
        assert claimed[0]["needs_marker"] is True

    def test_dead_owner_failed_needs_marker(self):
        _record()
        dl.mark_failed("ob-1", "boom")
        _orphan("ob-1")
        claimed = dl.sweep_recoverable(
            now=_row("ob-1")["next_attempt_at"] + 0.01
        )
        assert claimed[0]["needs_marker"] is True

    def test_delivered_rows_ignored(self):
        _record()
        dl.mark_delivered("ob-1")
        _orphan("ob-1")
        assert dl.sweep_recoverable() == []

    def test_attempts_cap_abandons(self):
        _record()
        _orphan("ob-1")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET attempts=? WHERE obligation_id=?",
                (dl.MAX_ATTEMPTS, "ob-1"),
            )
        assert dl.sweep_recoverable() == []
        assert _row("ob-1")["state"] == "abandoned"

    def test_stale_cutoff_abandons(self):
        _record()
        _orphan("ob-1")
        future = time.time() + dl.STALE_AFTER_SECONDS + 60
        assert dl.sweep_recoverable(now=future) == []
        assert _row("ob-1")["state"] == "abandoned"

    def test_limit_claims_oldest_only(self):
        _record("ob-1", content="first")
        _record("ob-2", content="second")
        now = time.time()
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET created_at=?, owner_pid=NULL "
                "WHERE obligation_id='ob-1'", (now - 2,)
            )
            conn.execute(
                "UPDATE delivery_obligations SET created_at=?, owner_pid=NULL "
                "WHERE obligation_id='ob-2'", (now - 1,)
            )
        claimed = dl.sweep_recoverable(limit=1)
        assert [item["obligation_id"] for item in claimed] == ["ob-1"]
        assert _row("ob-2")["attempts"] == 0

    def test_global_lease_allows_only_one_in_flight_recovery(self):
        _record("ob-1", content="first")
        _record("ob-2", content="second")
        _orphan("ob-1")
        _orphan("ob-2")
        first = dl.sweep_recoverable(limit=1)
        assert [row["obligation_id"] for row in first] == ["ob-1"]
        assert dl.sweep_recoverable(limit=1) == []
        dl.mark_failed(
            "ob-1", "retry later", generation=first[0]["generation"]
        )
        second = dl.sweep_recoverable(limit=1)
        assert [row["obligation_id"] for row in second] == ["ob-2"]

    def test_durable_platform_backoff_blocks_fresh_sweep(self):
        _record(platform="weixin")
        _orphan("ob-1")
        duration = dl.record_platform_rate_limit(
            "weixin", retry_after_seconds=90
        )
        assert duration >= 90
        assert dl.sweep_recoverable(
            deliverable_platforms={"weixin"}, limit=1
        ) == []
        dl.clear_platform_backoff("weixin")
        assert len(dl.sweep_recoverable(
            deliverable_platforms={"weixin"}, limit=1
        )) == 1

    def test_hard_cap_never_deletes_live_obligations(self, monkeypatch):
        monkeypatch.setattr(dl, "_MAX_ROWS", 2)
        _record("ob-1")
        _record("ob-2")
        _record("ob-3")
        assert all(_row(oid) is not None for oid in ("ob-1", "ob-2", "ob-3"))
        dl.mark_delivered("ob-1")
        dl._prune()
        assert _row("ob-1") is None
        assert _row("ob-2") is not None
        assert _row("ob-3") is not None


class TestSchemaMigration:
    def test_legacy_table_gets_next_attempt_column_without_data_loss(self):
        path = dl._db_path()
        conn = sqlite3.connect(path)
        conn.execute("DROP TABLE IF EXISTS delivery_obligations")
        conn.execute(
            """CREATE TABLE delivery_obligations (
                obligation_id TEXT PRIMARY KEY, session_key TEXT NOT NULL,
                platform TEXT NOT NULL, chat_id TEXT NOT NULL, thread_id TEXT,
                content TEXT NOT NULL, state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, owner_pid INTEGER,
                owner_started_at INTEGER, last_error TEXT)"""
        )
        conn.execute(
            "INSERT INTO delivery_obligations VALUES "
            "('legacy','s','weixin','c',NULL,'body','failed',1,1,1,NULL,NULL,'x')"
        )
        conn.commit()
        conn.close()

        with dl._connect() as migrated:
            columns = {row[1] for row in migrated.execute("PRAGMA table_info(delivery_obligations)")}
            row = migrated.execute(
                "SELECT content, next_attempt_at, generation FROM delivery_obligations "
                "WHERE obligation_id='legacy'"
            ).fetchone()
        assert "next_attempt_at" in columns
        assert "generation" in columns
        assert row == ("body", 0.0, 1)

    def test_concurrent_legacy_migration_is_idempotent(self):
        path = dl._db_path()
        conn = sqlite3.connect(path)
        conn.execute("DROP TABLE IF EXISTS delivery_obligations")
        conn.execute(
            """CREATE TABLE delivery_obligations (
                obligation_id TEXT PRIMARY KEY, session_key TEXT NOT NULL,
                platform TEXT NOT NULL, chat_id TEXT NOT NULL, thread_id TEXT,
                content TEXT NOT NULL, state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, owner_pid INTEGER,
                owner_started_at INTEGER, last_error TEXT)"""
        )
        conn.commit()
        conn.close()

        def open_and_read():
            migrated = dl._connect()
            try:
                return {
                    row[1]
                    for row in migrated.execute(
                        "PRAGMA table_info(delivery_obligations)"
                    )
                }
            finally:
                migrated.close()

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda _: open_and_read(), range(32)))
        assert all("next_attempt_at" in columns for columns in results)


class TestPrune:
    def test_old_delivered_rows_pruned(self):
        _record()
        dl.mark_delivered("ob-1")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET updated_at=? WHERE obligation_id=?",
                (time.time() - dl._RETENTION_SECONDS - 60, "ob-1"),
            )
        dl._prune()
        assert _row("ob-1") is None

    def test_undelivered_rows_survive_retention(self):
        _record()
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET updated_at=? WHERE obligation_id=?",
                (time.time() - dl._RETENTION_SECONDS - 60, "ob-1"),
            )
        dl._prune()
        assert _row("ob-1") is not None


class TestLedgerEnabled:
    def test_default_on(self):
        assert dl.ledger_enabled({}) is True
        assert dl.ledger_enabled({"gateway": {}}) is True

    def test_explicit_off(self):
        assert dl.ledger_enabled({"gateway": {"delivery_ledger": False}}) is False
        assert dl.ledger_enabled({"gateway": {"delivery_ledger": "off"}}) is False

    def test_truthy_strings(self):
        assert dl.ledger_enabled({"gateway": {"delivery_ledger": "true"}}) is True


class TestGatewayRedeliverySweep:
    """Drive the real GatewayRunner._redeliver_pending_obligations."""

    @staticmethod
    def _runner(adapter=None):
        from gateway.config import Platform
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter} if adapter else {}
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @staticmethod
    def _adapter(success=True):
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=MagicMock(success=success, error="" if success else "nope")
        )
        return adapter

    @pytest.mark.asyncio
    async def test_pending_redelivers_plain_and_clears_resume(self):
        _record()  # pending
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        sent = adapter.send.call_args.kwargs
        assert sent["content"] == "the final answer"  # no marker
        assert sent["metadata"] == {"thread_id": "171.001"}
        assert _row("ob-1")["state"] == "delivered"
        runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )

    @pytest.mark.asyncio
    async def test_attempting_redelivers_with_marker(self):
        _record()
        dl.mark_attempting("ob-1")
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)

        await runner._redeliver_pending_obligations()

        sent = adapter.send.call_args.kwargs
        assert sent["content"].startswith(dl.RECOVERED_MARKER)
        assert sent["content"].endswith("the final answer")

    @pytest.mark.asyncio
    async def test_send_failure_marks_failed_for_next_boot(self):
        _record()
        _orphan("ob-1")
        runner = self._runner(self._adapter(success=False))

        n = await runner._redeliver_pending_obligations()

        assert n == 0
        assert _row("ob-1")["state"] == "failed"
        assert _row("ob-1")["owner_pid"] is None

    @pytest.mark.asyncio
    async def test_each_pass_sends_only_one_oldest_obligation(self):
        _record("ob-1", content="first")
        _record("ob-2", content="second")
        now = time.time()
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET created_at=?, owner_pid=NULL "
                "WHERE obligation_id='ob-1'", (now - 2,)
            )
            conn.execute(
                "UPDATE delivery_obligations SET created_at=?, owner_pid=NULL "
                "WHERE obligation_id='ob-2'", (now - 1,)
            )
        adapter = self._adapter()
        runner = self._runner(adapter)

        assert await runner._redeliver_pending_obligations() == 1

        assert adapter.send.await_count == 1
        assert adapter.send.call_args.kwargs["content"] == "first"
        assert _row("ob-2")["attempts"] == 0

    @pytest.mark.asyncio
    async def test_rate_limit_globally_pauses_backlog_without_spending_next_row(self):
        _record("ob-1", content="first")
        _record("ob-2", content="second")
        _orphan("ob-1")
        _orphan("ob-2")
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=MagicMock(
                success=False,
                error="rate limited",
                error_kind="rate_limited",
                retry_after=90,
            )
        )
        runner = self._runner(adapter)

        assert await runner._redeliver_pending_obligations() == 0
        assert await runner._redeliver_pending_obligations() == 0

        assert adapter.send.await_count == 1
        assert _row("ob-1")["state"] == "failed"
        assert _row("ob-2")["attempts"] == 0
        assert runner._delivery_redelivery_rate_limit_streak == 1

        # A brand-new runner (service restart) must observe the same durable
        # platform circuit and leave the second row untouched.
        fresh_adapter = self._adapter(success=True)
        fresh_runner = self._runner(fresh_adapter)
        assert await fresh_runner._redeliver_pending_obligations() == 0
        fresh_adapter.send.assert_not_awaited()
        assert _row("ob-2")["attempts"] == 0

    @pytest.mark.asyncio
    async def test_missing_adapter_leaves_row_recoverable(self):
        _record()
        _orphan("ob-1")
        runner = self._runner(adapter=None)  # slack not connected

        n = await runner._redeliver_pending_obligations()

        assert n == 0
        # Row still claimed by us but NOT delivered/abandoned — a later boot
        # (attempts cap permitting) can retry once the platform connects.
        assert _row("ob-1")["state"] == "pending"

    @pytest.mark.asyncio
    async def test_disabled_gate_short_circuits(self):
        _record()
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)
        with patch.object(dl, "ledger_enabled", return_value=False), patch(
            "gateway.delivery_ledger.ledger_enabled", return_value=False
        ):
            n = await runner._redeliver_pending_obligations()
        assert n == 0
        adapter.send.assert_not_awaited()


class TestAttemptsOnlySpentOnRealSends:
    """``attempts`` is the redelivery budget — it must buy a send.

    ``self.adapters`` only holds a platform after its ``connect()`` succeeded,
    and the sweep claimed every dead-owner row regardless. A platform that
    failed to connect this boot therefore burned one attempt per boot while
    the caller's ``adapter is None`` branch skipped it without sending — so
    after MAX_ATTEMPTS boots the row abandoned having never been sent once,
    losing exactly the response the ledger exists to guarantee. That failure
    correlates with the crash that created the obligation: the network
    trouble that killed the send tends to still be there on the next boot.
    """

    def test_absent_platform_does_not_burn_attempts(self):
        _record(platform="telegram")
        dl.mark_attempting("ob-1")

        for _ in range(dl.MAX_ATTEMPTS + 2):
            _orphan("ob-1")
            assert dl.sweep_recoverable(deliverable_platforms={"discord"}) == []

        row = dl.debug_rows()
        assert "abandoned" not in row
        with dl._connect() as conn:
            state, attempts = conn.execute(
                "SELECT state, attempts FROM delivery_obligations "
                "WHERE obligation_id=?", ("ob-1",),
            ).fetchone()
        assert attempts == 0, "an unsendable boot must not spend the budget"
        assert state == "attempting"

    def test_row_still_delivers_once_its_platform_returns(self):
        _record(platform="telegram")
        for _ in range(dl.MAX_ATTEMPTS + 2):
            _orphan("ob-1")
            dl.sweep_recoverable(deliverable_platforms={"discord"})

        _orphan("ob-1")
        claimed = dl.sweep_recoverable(deliverable_platforms={"telegram"})
        assert len(claimed) == 1
        assert claimed[0]["attempts"] == 1

    def test_present_platform_still_claims(self):
        _record(platform="slack")
        _orphan("ob-1")
        claimed = dl.sweep_recoverable(deliverable_platforms={"slack"})
        assert len(claimed) == 1

    def test_omitting_the_filter_claims_everything(self):
        """Back-compat: existing callers pass no platform set."""
        _record(platform="telegram")
        _orphan("ob-1")
        assert len(dl.sweep_recoverable()) == 1

    def test_stale_rows_abandon_even_when_undeliverable(self):
        """The cutoff still bounds rows whose platform never returns."""
        _record(platform="telegram")
        _orphan("ob-1")
        future = time.time() + dl.STALE_AFTER_SECONDS + 10
        assert dl.sweep_recoverable(
            now=future, deliverable_platforms={"discord"}
        ) == []
        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM delivery_obligations WHERE obligation_id=?",
                ("ob-1",),
            ).fetchone()[0]
        assert state == "abandoned"


class TestUnconnectedPlatformKeepsItsBudget:
    """End-to-end through the real runner: boots where the platform failed to
    connect must not consume the row's redelivery budget."""

    @staticmethod
    def _runner_without_slack():
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {}  # slack failed to connect this boot
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @pytest.mark.asyncio
    async def test_row_survives_boots_where_its_platform_is_down(self):
        _record(platform="slack")
        dl.mark_attempting("ob-1")

        for _ in range(dl.MAX_ATTEMPTS + 1):
            _orphan("ob-1")
            runner = self._runner_without_slack()
            assert await runner._redeliver_pending_obligations() == 0

        assert _row("ob-1")["state"] != "abandoned", (
            "the obligation was abandoned without a single send being attempted"
        )
        assert _row("ob-1")["attempts"] == 0

    @pytest.mark.asyncio
    async def test_delivers_when_the_platform_comes_back(self):
        from gateway.config import Platform

        _record(platform="slack")
        for _ in range(dl.MAX_ATTEMPTS + 1):
            _orphan("ob-1")
            await self._runner_without_slack()._redeliver_pending_obligations()

        _orphan("ob-1")
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=MagicMock(success=True, error=""))
        runner = self._runner_without_slack()
        runner.adapters = {Platform.SLACK: adapter}

        assert await runner._redeliver_pending_obligations() == 1
        assert _row("ob-1")["state"] == "delivered"
