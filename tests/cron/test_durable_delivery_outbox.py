import hashlib
import json
import multiprocessing
import os
import sqlite3
import time
from pathlib import Path

import pytest

from cron.durable_delivery import (
    ClaimLost,
    OutboxError,
    PayloadConflict,
    SpoolInvalid,
    StructuredDeliveryOutbox,
)


def _text_unit(execution_id: str = "exec-1", target: str = "telegram:-100:7") -> dict:
    content = "durable hello"
    return {
        "schema_version": 2,
        "execution_id": execution_id,
        "job_id": "job-1",
        "target_index": 0,
        "unit_index": 0,
        "canonical_target": target,
        "logical_platform": "telegram",
        "chat_id": "-100",
        "thread_id": "7",
        "transport_kind": "native",
        "transport_identity_sha256": "0" * 64,
        "relay_identity": None,
        "kind": "text",
        "content": content,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "provider_content": content,
        "provider_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "delivery_config_snapshot": {
            "wrap_response": False,
            "task_name": "job-1",
            "job_id": "job-1",
        },
        "media_ref": None,
        "continuation": {"mode": "thread", "mirror": True},
    }


def _crash_after_claim(db: str, spool: str) -> None:
    outbox = StructuredDeliveryOutbox(db, spool)
    claim = outbox.claim_next(
        owner_pid=os.getpid(), owner_started_at=1.0, now=100.0
    )
    assert claim is not None
    os._exit(0)


def _crash_after_simulated_provider_ack(db: str, spool: str, ack_marker: str) -> None:
    outbox = StructuredDeliveryOutbox(db, spool)
    claim = outbox.claim_next(
        owner_pid=os.getpid(), owner_started_at=1.0, now=100.0
    )
    assert claim is not None
    marker = Path(ack_marker)
    with marker.open("xb") as handle:
        handle.write(b"provider accepted")
        handle.flush()
        os.fsync(handle.fileno())
    os._exit(0)


def _migrate_worker(db: str, spool: str, start, results) -> None:
    start.wait(10)
    try:
        outbox = StructuredDeliveryOutbox(db, spool)
        outbox.close()
        results.put(None)
    except BaseException as exc:
        results.put(repr(exc))


def _concurrent_claim_worker(db: str, spool: str, start, release, results) -> None:
    from gateway.status import get_process_start_time

    outbox = StructuredDeliveryOutbox(db, spool)
    started_at = get_process_start_time(os.getpid())
    if started_at is None:
        started_at = time.time()
    start.wait(10)
    claim = outbox.claim_next(
        owner_pid=os.getpid(), owner_started_at=float(started_at), now=100.0
    )
    results.put(claim.claim_token if claim is not None else None)
    if claim is not None:
        release.wait(10)
    outbox.close()


def test_owner_liveness_uses_cross_platform_probe_not_os_kill_zero() -> None:
    source = (Path(__file__).parents[2] / "cron" / "durable_delivery.py").read_text()

    assert "os.kill(" not in source
    assert "_pid_exists" in source


def test_enqueue_claim_and_ack_are_durable_and_fenced(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    outbox = StructuredDeliveryOutbox(db, tmp_path / "spool")

    [record] = outbox.enqueue_batch([_text_unit()])
    assert record.state == "pending"
    assert record.generation == 1

    claim = outbox.claim_next(owner_pid=123, owner_started_at=45.0, now=100.0)
    assert claim is not None
    assert claim.state == "attempting"
    assert claim.attempts == 1
    assert claim.claim_token

    with sqlite3.connect(db) as conn:
        durable = conn.execute(
            "SELECT state, attempts, claim_token FROM cron_delivery_outbox_v2"
        ).fetchone()
    assert durable == ("attempting", 1, claim.claim_token)

    outbox.mark_delivered(
        claim,
        provider_message_id="provider-ack-1",
        now=101.0,
    )
    assert outbox.get(record.obligation_id).state == "delivered"

    with pytest.raises(ClaimLost):
        outbox.mark_failed(claim, error="late failure", retryable=True, now=102.0)
    assert outbox.get(record.obligation_id).state == "delivered"

    outbox.close()


def test_obligation_identity_binds_provider_visible_payload_hash(tmp_path: Path) -> None:
    outbox = StructuredDeliveryOutbox(tmp_path / "state.db", tmp_path / "spool")
    first_unit = _text_unit()
    second_unit = dict(first_unit)
    second_unit["content"] = "different provider-visible bytes"
    second_unit["content_sha256"] = hashlib.sha256(
        second_unit["content"].encode()
    ).hexdigest()
    second_unit["provider_content"] = second_unit["content"]
    second_unit["provider_content_sha256"] = second_unit["content_sha256"]

    [first] = outbox.enqueue_batch([first_unit], now=100.0)
    [second] = outbox.enqueue_batch([second_unit], now=101.0)

    assert first.obligation_id != second.obligation_id
    with sqlite3.connect(tmp_path / "state.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM cron_delivery_outbox_v2"
        ).fetchone()[0] == 2


def test_payload_tampering_is_terminal_before_claim_and_does_not_block_next_row(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    outbox = StructuredDeliveryOutbox(db, tmp_path / "spool")
    [corrupt] = outbox.enqueue_batch([_text_unit("exec-corrupt")], now=100.0)
    [valid] = outbox.enqueue_batch([_text_unit("exec-valid")], now=101.0)
    with sqlite3.connect(db) as conn:
        payload = json.loads(
            conn.execute(
                "SELECT payload_json FROM cron_delivery_outbox_v2 WHERE obligation_id=?",
                (corrupt.obligation_id,),
            ).fetchone()[0]
        )
        payload["content"] = "tampered bytes"
        conn.execute(
            "UPDATE cron_delivery_outbox_v2 SET payload_json=? WHERE obligation_id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), corrupt.obligation_id),
        )
        conn.commit()

    claim = outbox.claim_next(owner_pid=123, owner_started_at=45.0, now=101.0)

    assert claim is not None
    assert claim.obligation_id == valid.obligation_id
    with sqlite3.connect(db) as conn:
        corrupt_state = conn.execute(
            "SELECT state, attempts, owner_pid, claim_token FROM cron_delivery_outbox_v2 "
            "WHERE obligation_id=?",
            (corrupt.obligation_id,),
        ).fetchone()
        lease_owner = conn.execute(
            "SELECT obligation_id FROM cron_delivery_recovery_lease_v2"
        ).fetchone()
    assert corrupt_state == ("abandoned", 0, None, None)
    assert lease_owner == (valid.obligation_id,)


def test_ambiguous_dispatch_becomes_unknown_and_is_never_auto_claimed(tmp_path: Path) -> None:
    outbox = StructuredDeliveryOutbox(tmp_path / "state.db", tmp_path / "spool")
    [record] = outbox.enqueue_batch([_text_unit()])
    claim = outbox.claim_next(owner_pid=123, owner_started_at=45.0, now=100.0)
    assert claim is not None

    outbox.mark_unknown(claim, error="provider ACK timed out", now=101.0)

    assert outbox.get(record.obligation_id).state == "unknown"
    assert outbox.claim_next(owner_pid=124, owner_started_at=46.0, now=10_000.0) is None


def test_media_spool_is_content_addressed_and_corruption_fails_closed(tmp_path: Path) -> None:
    outbox = StructuredDeliveryOutbox(tmp_path / "state.db", tmp_path / "spool")
    source = tmp_path / "report.pdf"
    source.write_bytes(b"immutable report bytes")

    ref = outbox.spool_media(source, is_voice=False)
    spooled = outbox.resolve_media(ref)
    assert spooled.read_bytes() == b"immutable report bytes"
    assert spooled.parent == tmp_path / "spool"

    spooled.write_bytes(b"corrupt")
    with pytest.raises(SpoolInvalid):
        outbox.resolve_media(ref)
    spooled.unlink()
    with pytest.raises(SpoolInvalid):
        outbox.resolve_media(ref)


def test_provider_output_bytes_are_bounded_before_enqueue(tmp_path: Path) -> None:
    outbox = StructuredDeliveryOutbox(
        tmp_path / "state.db",
        tmp_path / "spool",
        max_payload_bytes=8,
    )
    unit = _text_unit("exec-output-cap")
    unit["content"] = "123456789"
    unit["content_sha256"] = hashlib.sha256(unit["content"].encode()).hexdigest()
    unit["provider_content"] = unit["content"]
    unit["provider_content_sha256"] = unit["content_sha256"]

    with pytest.raises(OutboxError, match="provider payload.*8 bytes"):
        outbox.enqueue_batch([unit])


def test_spool_gc_removes_only_expired_inactive_blobs_and_preserves_capacity(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    spool = tmp_path / "spool"
    outbox = StructuredDeliveryOutbox(
        db,
        spool,
        max_spool_bytes=20,
        spool_retention_seconds=1,
    )
    active_source = tmp_path / "active.pdf"
    orphan_source = tmp_path / "orphan.pdf"
    active_source.write_bytes(b"active")
    orphan_source.write_bytes(b"orphan")
    active_ref = outbox.spool_media(active_source, is_voice=False)
    orphan_ref = outbox.spool_media(orphan_source, is_voice=False)

    unit = _text_unit("exec-active-media")
    unit["media_ref"] = active_ref
    unit["media_refs"] = [active_ref]
    [active_record] = outbox.enqueue_batch([unit])
    active_claim = outbox.claim_next(
        owner_pid=123,
        owner_started_at=45.0,
        obligation_id=active_record.obligation_id,
    )
    assert active_claim is not None
    outbox.mark_unknown(
        active_claim,
        error="provider ACK is ambiguous",
    )
    active_path = outbox.resolve_media(active_ref)
    orphan_path = outbox.resolve_media(orphan_ref)
    old = time.time() - 10
    os.utime(active_path, (old, old))
    os.utime(orphan_path, (old, old))

    result = outbox.cleanup_spool(now=time.time())

    assert result["removed_files"] == 1
    assert active_path.read_bytes() == b"active"
    assert not orphan_path.exists()

    bounded = StructuredDeliveryOutbox(
        tmp_path / "bounded.db",
        tmp_path / "bounded-spool",
        max_spool_bytes=8,
        spool_retention_seconds=0,
    )
    bounded_ref = bounded.spool_media(active_source, is_voice=False)
    bounded_unit = _text_unit("exec-bounded-media")
    bounded_unit["media_ref"] = bounded_ref
    bounded_unit["media_refs"] = [bounded_ref]
    bounded.enqueue_batch([bounded_unit])
    extra_source = tmp_path / "extra.pdf"
    extra_source.write_bytes(b"xxx")
    with pytest.raises(SpoolInvalid, match="spool capacity"):
        bounded.spool_media(extra_source, is_voice=False)
    assert bounded.resolve_media(bounded_ref).read_bytes() == b"active"


def test_spool_gc_fails_closed_on_tampered_active_media_reference(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    outbox = StructuredDeliveryOutbox(
        db,
        tmp_path / "spool",
        spool_retention_seconds=0,
    )
    source = tmp_path / "active.pdf"
    source.write_bytes(b"active")
    ref = outbox.spool_media(source, is_voice=False)
    unit = _text_unit("exec-tampered-active-media")
    unit["media_ref"] = ref
    unit["media_refs"] = [ref]
    [record] = outbox.enqueue_batch([unit])
    durable_path = outbox.resolve_media(ref)
    old = time.time() - 10
    os.utime(durable_path, (old, old))

    with sqlite3.connect(db) as conn:
        payload = json.loads(
            conn.execute(
                "SELECT payload_json FROM cron_delivery_outbox_v2 "
                "WHERE obligation_id=?",
                (record.obligation_id,),
            ).fetchone()[0]
        )
        payload["media_ref"] = None
        payload["media_refs"] = []
        conn.execute(
            "UPDATE cron_delivery_outbox_v2 SET payload_json=? WHERE obligation_id=?",
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                record.obligation_id,
            ),
        )
        conn.commit()

    with pytest.raises(PayloadConflict, match="garbage-collect"):
        outbox.cleanup_spool(now=time.time())
    assert durable_path.read_bytes() == b"active"


def test_retry_after_and_platform_circuit_survive_restart(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    spool = tmp_path / "spool"
    outbox = StructuredDeliveryOutbox(db, spool)
    [record] = outbox.enqueue_batch([_text_unit()], now=100.0)
    claim = outbox.claim_next(owner_pid=123, owner_started_at=45.0, now=100.0)
    assert claim is not None
    outbox.mark_failed(
        claim,
        error="429 rate limited",
        retryable=True,
        retry_after=30.0,
        now=101.0,
    )
    assert outbox.get(record.obligation_id).next_attempt_at == 131.0
    outbox.record_platform_circuit(
        "telegram", blocked_until=200.0, reason="429", now=101.0
    )
    outbox.close()

    restarted = StructuredDeliveryOutbox(db, spool)
    assert restarted.claim_next(owner_pid=124, owner_started_at=46.0, now=150.0) is None
    assert restarted.claim_next(owner_pid=124, owner_started_at=46.0, now=201.0) is not None


def test_max_attempts_abandons_without_a_fourth_send(tmp_path: Path) -> None:
    outbox = StructuredDeliveryOutbox(tmp_path / "state.db", tmp_path / "spool")
    [record] = outbox.enqueue_batch([_text_unit()], now=100.0)
    now = 100.0
    for attempt in range(3):
        claim = outbox.claim_next(
            owner_pid=123, owner_started_at=45.0, now=now
        )
        assert claim is not None
        assert claim.attempts == attempt + 1
        outbox.mark_failed(
            claim,
            error="retryable",
            retryable=True,
            retry_after=0,
            now=now,
        )
        now = outbox.get(record.obligation_id).next_attempt_at

    assert outbox.claim_next(owner_pid=123, owner_started_at=45.0, now=now) is None
    assert outbox.get(record.obligation_id).state == "abandoned"


def test_hard_cap_prunes_only_terminal_rows_and_never_active_obligations(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    outbox = StructuredDeliveryOutbox(db, tmp_path / "spool", max_rows=2)
    first, second = outbox.enqueue_batch(
        [_text_unit("exec-cap-1"), _text_unit("exec-cap-2")], now=100.0
    )

    with pytest.raises(OutboxError, match="capacity"):
        outbox.enqueue_batch([_text_unit("exec-cap-3")], now=101.0)
    assert outbox.get(first.obligation_id).state == "pending"
    assert outbox.get(second.obligation_id).state == "pending"

    claim = outbox.claim_next(
        owner_pid=123,
        owner_started_at=45.0,
        now=101.0,
        obligation_id=first.obligation_id,
    )
    assert claim is not None
    outbox.mark_delivered(claim, provider_message_id="ack-cap", now=102.0)
    [third] = outbox.enqueue_batch([_text_unit("exec-cap-3")], now=103.0)

    with pytest.raises(KeyError):
        outbox.get(first.obligation_id)
    assert outbox.get(second.obligation_id).state == "pending"
    assert outbox.get(third.obligation_id).state == "pending"


def test_capacity_pruning_never_resurrects_terminal_member_of_same_batch(
    tmp_path: Path,
) -> None:
    outbox = StructuredDeliveryOutbox(
        tmp_path / "state.db",
        tmp_path / "spool",
        max_rows=1,
    )
    terminal_unit = _text_unit("exec-cap-terminal")
    [terminal] = outbox.enqueue_batch([terminal_unit], now=100.0)
    claim = outbox.claim_next(
        owner_pid=123,
        owner_started_at=45.0,
        now=100.0,
        obligation_id=terminal.obligation_id,
    )
    assert claim is not None
    outbox.mark_delivered(claim, provider_message_id="ack-terminal", now=101.0)

    with pytest.raises(OutboxError, match="capacity"):
        outbox.enqueue_batch(
            [terminal_unit, _text_unit("exec-cap-new")],
            now=102.0,
        )

    durable = outbox.get(terminal.obligation_id)
    assert durable.state == "delivered"
    with sqlite3.connect(tmp_path / "state.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM cron_delivery_outbox_v2"
        ).fetchone()[0] == 1


def test_dead_claimant_is_reclassified_unknown_and_next_row_has_one_winner(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    spool = tmp_path / "spool"
    outbox = StructuredDeliveryOutbox(db, spool)
    outbox.enqueue_batch(
        [_text_unit("exec-crash-1"), _text_unit("exec-crash-2")], now=100.0
    )
    outbox.close()

    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=_crash_after_claim, args=(str(db), str(spool)))
    child.start()
    child.join(10)
    assert child.exitcode == 0

    recovered = StructuredDeliveryOutbox(db, spool)
    winner = recovered.claim_next(
        owner_pid=os.getpid(), owner_started_at=2.0, now=101.0
    )
    assert winner is not None
    with sqlite3.connect(db) as conn:
        states = [row[0] for row in conn.execute(
            "SELECT state FROM cron_delivery_outbox_v2 ORDER BY state"
        )]
    assert states == ["attempting", "unknown"]


def test_crash_after_simulated_provider_ack_becomes_unknown_without_replay(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    spool = tmp_path / "spool"
    ack_marker = tmp_path / "provider-ack.marker"
    outbox = StructuredDeliveryOutbox(db, spool)
    [record] = outbox.enqueue_batch([_text_unit("exec-ack-crash")], now=100.0)
    outbox.close()
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(
        target=_crash_after_simulated_provider_ack,
        args=(str(db), str(spool), str(ack_marker)),
    )
    child.start()
    child.join(15)
    assert child.exitcode == 0
    assert ack_marker.read_bytes() == b"provider accepted"

    recovered = StructuredDeliveryOutbox(db, spool)
    assert recovered.claim_next(
        owner_pid=os.getpid(), owner_started_at=2.0, now=101.0
    ) is None
    durable = recovered.get(record.obligation_id)
    assert durable.state == "unknown"
    assert durable.attempts == 1


def test_schema_creation_is_idempotent_under_real_multiprocess_startup(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    spool = tmp_path / "spool"
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    workers = [
        ctx.Process(target=_migrate_worker, args=(str(db), str(spool), start, results))
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    start.set()
    observed = [results.get(timeout=15) for _ in workers]
    for worker in workers:
        worker.join(15)
        assert worker.exitcode == 0

    assert observed == [None] * 4
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute(
            "SELECT COUNT(*) FROM cron_delivery_outbox_v2"
        ).fetchone()[0] == 0


def test_real_multiprocess_claim_has_one_live_winner_for_limit_one(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    spool = tmp_path / "spool"
    outbox = StructuredDeliveryOutbox(db, spool)
    outbox.enqueue_batch([_text_unit()], now=100.0)
    outbox.close()
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    release = ctx.Event()
    results = ctx.Queue()
    workers = [
        ctx.Process(
            target=_concurrent_claim_worker,
            args=(str(db), str(spool), start, release, results),
        )
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    start.set()
    observed = [results.get(timeout=15) for _ in workers]
    release.set()
    for worker in workers:
        worker.join(15)
        assert worker.exitcode == 0

    assert len([token for token in observed if token is not None]) == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone() == ("attempting", 1)


def test_rerecord_increments_generation_and_fences_old_claim(tmp_path: Path) -> None:
    outbox = StructuredDeliveryOutbox(tmp_path / "state.db", tmp_path / "spool")
    unit = _text_unit()
    [first] = outbox.enqueue_batch([unit], now=100.0)
    old_claim = outbox.claim_next(
        owner_pid=123, owner_started_at=45.0, now=100.0
    )
    assert old_claim is not None
    outbox.mark_failed(
        old_claim,
        error="definitive retryable rejection",
        retryable=True,
        now=100.5,
    )

    rerecorded = outbox.rerecord(unit, now=101.0)
    assert rerecorded.obligation_id == first.obligation_id
    assert rerecorded.generation == 2
    assert rerecorded.state == "pending"

    with pytest.raises(ClaimLost):
        outbox.mark_delivered(old_claim, provider_message_id="stale", now=102.0)
    new_claim = outbox.claim_next(
        owner_pid=124, owner_started_at=46.0, now=102.0
    )
    assert new_claim is not None
    assert new_claim.generation == 2


def test_rerecord_rejects_an_attempting_generation_before_external_send_quiesces(
    tmp_path: Path, monkeypatch
) -> None:
    outbox = StructuredDeliveryOutbox(tmp_path / "state.db", tmp_path / "spool")
    unit = _text_unit()
    [record] = outbox.enqueue_batch([unit], now=100.0)
    claim = outbox.claim_next(owner_pid=123, owner_started_at=45.0, now=100.0)
    assert claim is not None
    monkeypatch.setattr("cron.durable_delivery._owner_alive", lambda *_args: True)

    with pytest.raises(OutboxError, match="attempting"):
        outbox.rerecord(unit, now=101.0)

    durable = outbox.get(record.obligation_id)
    assert durable.state == "attempting"
    assert durable.generation == 1
    assert durable.claim_token == claim.claim_token


def test_completion_is_fenced_by_owner_identity_as_well_as_token(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    outbox = StructuredDeliveryOutbox(db, tmp_path / "spool")
    outbox.enqueue_batch([_text_unit()])
    claim = outbox.claim_next(owner_pid=123, owner_started_at=45.0, now=100.0)
    assert claim is not None
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE cron_delivery_outbox_v2 SET owner_pid=999 WHERE obligation_id=?",
            (claim.obligation_id,),
        )
        conn.commit()

    with pytest.raises(ClaimLost):
        outbox.mark_delivered(claim, provider_message_id="stale-owner", now=101.0)


def test_stale_claim_token_cannot_complete_after_same_generation_transfers_owner(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    outbox = StructuredDeliveryOutbox(db, tmp_path / "spool")
    [record] = outbox.enqueue_batch([_text_unit()], now=100.0)
    stale = outbox.claim_next(owner_pid=123, owner_started_at=45.0, now=100.0)
    assert stale is not None
    with sqlite3.connect(db) as conn:
        conn.execute(
            "DELETE FROM cron_delivery_recovery_lease_v2 WHERE singleton=1"
        )
        conn.execute(
            """UPDATE cron_delivery_outbox_v2
               SET state='failed', owner_pid=NULL, owner_started_at=NULL,
                   claim_token=NULL, next_attempt_at=0
               WHERE obligation_id=?""",
            (record.obligation_id,),
        )
        conn.commit()

    current = outbox.claim_next(owner_pid=124, owner_started_at=46.0, now=101.0)
    assert current is not None
    assert current.generation == stale.generation
    assert current.claim_token != stale.claim_token

    with pytest.raises(ClaimLost):
        outbox.mark_delivered(stale, provider_message_id="stale-ack", now=102.0)
    outbox.mark_delivered(current, provider_message_id="current-ack", now=103.0)
    durable = outbox.get(record.obligation_id)
    assert durable.state == "delivered"
    assert durable.provider_message_id == "current-ack"
