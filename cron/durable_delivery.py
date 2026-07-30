"""Crash-safe structured outbox for cron delivery.

This is intentionally separate from gateway.delivery_ledger: cron fan-out, media,
relay and continuation semantics cannot be represented by the legacy text-only
ledger.  Schema version 2 rows are consumed only by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


_TERMINAL_STATES = frozenset({"delivered", "abandoned"})
_RECOVERABLE_STATES = frozenset({"pending", "failed"})
_SCHEMA_VERSION = 2
_SCOPE_CAPSULE_VERSION = 2
_MAX_ATTEMPTS = 3
_STALE_AFTER_SECONDS = 24 * 60 * 60
_DEFAULT_MAX_ROWS = 10_000
_DEFAULT_MAX_PAYLOAD_BYTES = 1_000_000
_DEFAULT_MAX_SPOOL_BYTES = 1_000_000_000
_DEFAULT_SPOOL_RETENTION_SECONDS = 7 * 24 * 60 * 60
_SPOOL_GC_INTERVAL_SECONDS = 60 * 60


class OutboxError(RuntimeError):
    """Base structured outbox error."""


class PayloadConflict(OutboxError):
    """An idempotency key was reused with different provider-visible bytes."""


class ClaimLost(OutboxError):
    """The generation/claim token no longer authorizes a mutation."""


class SpoolInvalid(OutboxError):
    """A durable media blob is missing, corrupt, or outside the spool."""


@dataclass(frozen=True)
class OutboxRecord:
    obligation_id: str
    generation: int
    state: str
    attempts: int
    owner_pid: Optional[int]
    owner_started_at: Optional[float]
    claim_token: Optional[str]
    payload: dict[str, Any]
    next_attempt_at: float
    last_error: Optional[str]
    provider_message_id: Optional[str]


def _owner_alive(pid: Any, started_at: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import _pid_exists, get_process_start_time

        if not _pid_exists(pid):
            return False
        actual = get_process_start_time(pid)
        if actual is not None and started_at is not None:
            return float(actual) == float(started_at)
    except Exception:
        return False
    return True


class StructuredDeliveryOutbox:
    """SQLite-backed schema-v2 cron delivery obligations.

    Each method opens its own connection so separate processes coordinate through
    SQLite rather than process-local state.  Mutating operations use IMMEDIATE
    transactions and FULL synchronous durability.
    """

    def __init__(
        self,
        db_path: os.PathLike[str] | str,
        spool_dir: os.PathLike[str] | str,
        *,
        max_rows: int = _DEFAULT_MAX_ROWS,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
        max_spool_bytes: int = _DEFAULT_MAX_SPOOL_BYTES,
        spool_retention_seconds: float = _DEFAULT_SPOOL_RETENTION_SECONDS,
    ):
        self.db_path = Path(db_path)
        self.spool_dir = Path(spool_dir)
        self.max_rows = int(max_rows)
        self.max_payload_bytes = int(max_payload_bytes)
        self.max_spool_bytes = int(max_spool_bytes)
        self.spool_retention_seconds = float(spool_retention_seconds)
        if self.max_rows < 1:
            raise ValueError("structured delivery max_rows must be positive")
        if self.max_payload_bytes < 1 or self.max_spool_bytes < 1:
            raise ValueError("structured delivery byte limits must be positive")
        if self.spool_retention_seconds < 0:
            raise ValueError("structured delivery spool retention cannot be negative")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._local_lock = threading.RLock()
        self._migrate()
        self._cleanup_spool(force=False)

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise OutboxError("structured delivery outbox is closed")
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        deadline = time.monotonic() + 30.0
        while True:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    conn.close()
                    raise
                time.sleep(0.01)
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _migrate(self) -> None:
        with self._local_lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS cron_delivery_outbox_v2 (
                        obligation_id TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL CHECK(schema_version = 2),
                        execution_id TEXT NOT NULL,
                        canonical_target TEXT NOT NULL,
                        scope_capsule_json TEXT NOT NULL,
                        scope_capsule_sha256 TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        generation INTEGER NOT NULL DEFAULT 1,
                        state TEXT NOT NULL CHECK(state IN
                            ('pending','attempting','failed','unknown','abandoned','delivered')),
                        attempts INTEGER NOT NULL DEFAULT 0,
                        next_attempt_at REAL NOT NULL DEFAULT 0,
                        owner_pid INTEGER,
                        owner_started_at REAL,
                        claim_token TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        last_error TEXT,
                        provider_message_id TEXT,
                        delivered_at REAL
                    )"""
                )
                columns = {
                    str(row[1])
                    for row in conn.execute("PRAGMA table_info(cron_delivery_outbox_v2)")
                }
                for name in ("scope_capsule_json", "scope_capsule_sha256"):
                    if name not in columns:
                        conn.execute(
                            f"ALTER TABLE cron_delivery_outbox_v2 ADD COLUMN {name} TEXT"
                        )
                for row in conn.execute("SELECT * FROM cron_delivery_outbox_v2").fetchall():
                    if (
                        row["scope_capsule_json"] is not None
                        and row["scope_capsule_sha256"] is not None
                    ):
                        try:
                            self._validated_scope_capsule(row)
                        except PayloadConflict:
                            # A legacy or damaged capsule may be replaced only
                            # from a fully content-addressed canonical row below.
                            pass
                        else:
                            continue
                    try:
                        payload = json.loads(row["payload_json"])
                        obligation_id, payload_hash, canonical_payload = (
                            self._canonical_payload(payload)
                        )
                        encoded = json.dumps(
                            canonical_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )
                    except (TypeError, ValueError, json.JSONDecodeError, OutboxError) as exc:
                        raise OutboxError(
                            "cannot migrate corrupt structured delivery row"
                        ) from exc
                    if (
                        obligation_id != row["obligation_id"]
                        or payload_hash != row["payload_hash"]
                        or encoded != row["payload_json"]
                        or str(row["execution_id"])
                        != str(canonical_payload["execution_id"])
                        or str(row["canonical_target"])
                        != str(canonical_payload["canonical_target"])
                    ):
                        raise OutboxError(
                            "cannot migrate unauthenticated structured delivery scope"
                        )
                    capsule_json, capsule_sha256 = self._scope_capsule(
                        canonical_payload, obligation_id
                    )
                    conn.execute(
                        """UPDATE cron_delivery_outbox_v2
                           SET scope_capsule_json=?, scope_capsule_sha256=?
                           WHERE obligation_id=?""",
                        (capsule_json, capsule_sha256, obligation_id),
                    )
                conn.execute(
                    """CREATE INDEX IF NOT EXISTS cron_delivery_outbox_v2_due
                       ON cron_delivery_outbox_v2(state, next_attempt_at, created_at)"""
                )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS cron_delivery_recovery_lease_v2 (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        obligation_id TEXT NOT NULL,
                        generation INTEGER NOT NULL,
                        claim_token TEXT NOT NULL,
                        owner_pid INTEGER NOT NULL,
                        owner_started_at REAL NOT NULL,
                        acquired_at REAL NOT NULL
                    )"""
                )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS cron_delivery_circuit_v2 (
                        platform TEXT PRIMARY KEY,
                        blocked_until REAL NOT NULL,
                        reason TEXT,
                        updated_at REAL NOT NULL
                    )"""
                )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS cron_delivery_spool_gc_v2 (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        last_run_at REAL NOT NULL
                    )"""
                )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS cron_delivery_job_status_sync_v2 (
                        execution_id TEXT NOT NULL,
                        job_id TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        PRIMARY KEY(execution_id, job_id)
                    )"""
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _canonical_payload(self, unit: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
        payload = dict(unit)
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise OutboxError("structured cron delivery requires schema_version=2")
        required = (
            "execution_id",
            "job_id",
            "target_index",
            "unit_index",
            "canonical_target",
            "logical_platform",
            "chat_id",
            "transport_kind",
            "transport_identity_sha256",
            "kind",
            "content_sha256",
            "provider_content",
            "provider_content_sha256",
            "delivery_config_snapshot",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise OutboxError(f"missing structured delivery fields: {', '.join(missing)}")
        transport_identity = str(payload["transport_identity_sha256"])
        if len(transport_identity) != 64 or any(
            character not in "0123456789abcdef" for character in transport_identity
        ):
            raise PayloadConflict("structured delivery transport identity integrity failure")
        content = str(payload.get("content") or "")
        expected_content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if str(payload["content_sha256"]) != expected_content_hash:
            raise PayloadConflict("structured delivery content hash integrity failure")
        provider_content = str(payload["provider_content"])
        provider_size = len(provider_content.encode("utf-8"))
        if provider_size > self.max_payload_bytes:
            raise OutboxError(
                "structured delivery provider payload exceeds "
                f"{self.max_payload_bytes} bytes"
            )
        expected_provider_hash = hashlib.sha256(
            provider_content.encode("utf-8")
        ).hexdigest()
        if str(payload["provider_content_sha256"]) != expected_provider_hash:
            raise PayloadConflict(
                "structured delivery provider content hash integrity failure"
            )
        snapshot = payload["delivery_config_snapshot"]
        if not isinstance(snapshot, dict) or not {
            "wrap_response",
            "task_name",
            "job_id",
        }.issubset(snapshot):
            raise PayloadConflict("structured delivery config snapshot integrity failure")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        payload_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        identity = "|".join(
            (
                "cron-v2",
                str(payload["execution_id"]),
                str(payload["target_index"]),
                str(payload["unit_index"]),
                str(payload["canonical_target"]),
                str(payload["transport_kind"]),
                payload_hash,
            )
        )
        obligation_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return obligation_id, payload_hash, payload

    def _scope_capsule(
        self, payload: Mapping[str, Any], obligation_id: str
    ) -> tuple[str, str]:
        try:
            target_index = int(payload["target_index"])
            unit_index = int(payload["unit_index"])
            transport_kind = str(payload["transport_kind"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PayloadConflict("invalid structured delivery scope identity") from exc
        if target_index < 0 or unit_index < 0:
            raise PayloadConflict("structured delivery scope order cannot be negative")
        payload_hash = str(payload.get("payload_hash") or "")
        if not payload_hash:
            encoded_payload = json.dumps(
                dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            payload_hash = hashlib.sha256(encoded_payload.encode("utf-8")).hexdigest()
        if len(payload_hash) != 64 or any(
            character not in "0123456789abcdef" for character in payload_hash
        ):
            raise PayloadConflict("invalid structured delivery payload identity")
        identity = "|".join(
            (
                "cron-v2",
                str(payload["execution_id"]),
                str(target_index),
                str(unit_index),
                str(payload["canonical_target"]),
                transport_kind,
                payload_hash,
            )
        )
        expected_obligation_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if expected_obligation_id != str(obligation_id):
            raise PayloadConflict("structured delivery scope identity mismatch")
        capsule = {
            "version": _SCOPE_CAPSULE_VERSION,
            "obligation_id": str(obligation_id),
            "execution_id": str(payload["execution_id"]),
            "job_id": str(payload["job_id"]),
            "canonical_target": str(payload["canonical_target"]),
            "target_index": target_index,
            "unit_index": unit_index,
            "transport_kind": transport_kind,
            "payload_hash": payload_hash,
        }
        if (
            not capsule["execution_id"]
            or not capsule["canonical_target"]
            or not capsule["transport_kind"]
        ):
            raise PayloadConflict("structured delivery scope cannot be empty")
        encoded = json.dumps(
            capsule, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        digest = hashlib.sha256(
            f"cron-scope-v{_SCOPE_CAPSULE_VERSION}\0".encode("ascii")
            + encoded.encode("utf-8")
        ).hexdigest()
        return encoded, digest

    def _validated_scope_capsule(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            capsule = json.loads(row["scope_capsule_json"])
            if not isinstance(capsule, dict):
                raise TypeError("capsule is not an object")
            expected_keys = {
                "version",
                "obligation_id",
                "execution_id",
                "job_id",
                "canonical_target",
                "target_index",
                "unit_index",
                "transport_kind",
                "payload_hash",
            }
            if set(capsule) != expected_keys:
                raise ValueError("capsule fields differ")
            encoded, digest = self._scope_capsule(
                capsule, str(capsule["obligation_id"])
            )
        except (TypeError, ValueError, json.JSONDecodeError, PayloadConflict) as exc:
            raise PayloadConflict("structured delivery scope capsule integrity failure") from exc
        if (
            capsule["version"] != _SCOPE_CAPSULE_VERSION
            or str(capsule["obligation_id"]) != str(row["obligation_id"])
            or encoded != row["scope_capsule_json"]
            or digest != row["scope_capsule_sha256"]
        ):
            raise PayloadConflict("structured delivery scope capsule integrity failure")
        return capsule

    def _record(self, row: sqlite3.Row) -> OutboxRecord:
        try:
            payload = json.loads(row["payload_json"])
            obligation_id, payload_hash, canonical_payload = self._canonical_payload(payload)
            encoded = json.dumps(
                canonical_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            capsule = self._validated_scope_capsule(row)
        except (TypeError, ValueError, json.JSONDecodeError, OutboxError) as exc:
            raise PayloadConflict("structured delivery payload integrity failure") from exc
        if (
            obligation_id != row["obligation_id"]
            or payload_hash != row["payload_hash"]
            or encoded != row["payload_json"]
            or row["schema_version"] != canonical_payload["schema_version"]
            or str(row["execution_id"]) != str(canonical_payload["execution_id"])
            or str(row["canonical_target"])
            != str(canonical_payload["canonical_target"])
            or str(capsule["execution_id"])
            != str(canonical_payload["execution_id"])
            or str(capsule["job_id"]) != str(canonical_payload["job_id"])
            or str(capsule["canonical_target"])
            != str(canonical_payload["canonical_target"])
            or int(capsule["target_index"]) != int(canonical_payload["target_index"])
            or int(capsule["unit_index"]) != int(canonical_payload["unit_index"])
            or str(capsule["transport_kind"])
            != str(canonical_payload["transport_kind"])
            or str(capsule["payload_hash"]) != payload_hash
        ):
            raise PayloadConflict("structured delivery payload integrity failure")
        return OutboxRecord(
            obligation_id=str(row["obligation_id"]),
            generation=int(row["generation"]),
            state=str(row["state"]),
            attempts=int(row["attempts"]),
            owner_pid=(int(row["owner_pid"]) if row["owner_pid"] is not None else None),
            owner_started_at=(
                float(row["owner_started_at"])
                if row["owner_started_at"] is not None
                else None
            ),
            claim_token=row["claim_token"],
            payload=canonical_payload,
            next_attempt_at=float(row["next_attempt_at"]),
            last_error=row["last_error"],
            provider_message_id=row["provider_message_id"],
        )

    def enqueue_batch(
        self, units: Iterable[Mapping[str, Any]], *, now: Optional[float] = None
    ) -> list[OutboxRecord]:
        prepared = [self._canonical_payload(unit) for unit in units]
        if not prepared:
            return []
        timestamp = time.time() if now is None else float(now)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            distinct_ids = {obligation_id for obligation_id, _hash, _payload in prepared}
            new_count = sum(
                conn.execute(
                    "SELECT NOT EXISTS(SELECT 1 FROM cron_delivery_outbox_v2 "
                    "WHERE obligation_id=?)",
                    (obligation_id,),
                ).fetchone()[0]
                for obligation_id in distinct_ids
            )
            current_count = int(
                conn.execute("SELECT COUNT(*) FROM cron_delivery_outbox_v2").fetchone()[0]
            )
            excess = current_count + int(new_count) - self.max_rows
            if excess > 0:
                # A terminal row that is also present in this enqueue batch is
                # an idempotent member of the batch, not pruning capacity. If
                # we delete it here, the INSERT OR IGNORE below becomes an
                # INSERT and silently resurrects the terminal obligation as a
                # fresh pending send in the same transaction.
                terminal_ids = []
                for row in conn.execute(
                    """SELECT obligation_id FROM cron_delivery_outbox_v2
                       WHERE state IN ('delivered','abandoned')
                       ORDER BY updated_at, obligation_id"""
                ):
                    if row[0] in distinct_ids:
                        continue
                    terminal_ids.append(row[0])
                    if len(terminal_ids) == excess:
                        break
                if len(terminal_ids) != excess:
                    raise OutboxError(
                        "structured delivery capacity reached with active obligations"
                    )
                conn.executemany(
                    "DELETE FROM cron_delivery_outbox_v2 WHERE obligation_id=? "
                    "AND state IN ('delivered','abandoned')",
                    ((obligation_id,) for obligation_id in terminal_ids),
                )
            records: list[OutboxRecord] = []
            for obligation_id, payload_hash, payload in prepared:
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                scope_capsule_json, scope_capsule_sha256 = self._scope_capsule(
                    payload, obligation_id
                )
                conn.execute(
                    """INSERT OR IGNORE INTO cron_delivery_outbox_v2
                       (obligation_id, schema_version, execution_id, canonical_target,
                        scope_capsule_json, scope_capsule_sha256,
                        payload_hash, payload_json, state, created_at, updated_at)
                       VALUES (?, 2, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                    (
                        obligation_id,
                        str(payload["execution_id"]),
                        str(payload["canonical_target"]),
                        scope_capsule_json,
                        scope_capsule_sha256,
                        payload_hash,
                        encoded,
                        timestamp,
                        timestamp,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM cron_delivery_outbox_v2 WHERE obligation_id=?",
                    (obligation_id,),
                ).fetchone()
                if row is None or row["payload_hash"] != payload_hash or row["payload_json"] != encoded:
                    raise PayloadConflict(f"conflicting payload for {obligation_id}")
                records.append(self._record(row))
            conn.commit()
            return records
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _active_spool_paths(self, conn: sqlite3.Connection) -> set[str]:
        """Return media names that may still be sent or manually adjudicated."""
        active: set[str] = set()
        rows = conn.execute(
            """SELECT * FROM cron_delivery_outbox_v2
               WHERE state NOT IN ('delivered','abandoned')"""
        )
        for row in rows:
            try:
                payload = self._record(row).payload
            except (TypeError, json.JSONDecodeError, PayloadConflict) as exc:
                raise PayloadConflict(
                    "cannot garbage-collect spool with corrupt active payload"
                ) from exc
            refs = list(payload.get("media_refs") or [])
            if payload.get("media_ref"):
                refs.append(payload["media_ref"])
            for ref in refs:
                relative = str((ref or {}).get("relative_path") or "")
                if relative and Path(relative).name == relative:
                    active.add(relative)
        return active

    def _spool_size_bytes(self) -> int:
        total = 0
        for entry in self.spool_dir.iterdir():
            try:
                if not entry.is_dir():
                    total += entry.lstat().st_size
            except FileNotFoundError:
                continue
        return total

    def _cleanup_spool_locked(
        self,
        conn: sqlite3.Connection,
        *,
        now: float,
    ) -> dict[str, int]:
        active = self._active_spool_paths(conn)
        cutoff = float(now) - self.spool_retention_seconds
        removed_files = 0
        removed_bytes = 0
        for entry in self.spool_dir.iterdir():
            if entry.name in active:
                continue
            try:
                stat = entry.lstat()
            except FileNotFoundError:
                continue
            if entry.is_dir() or stat.st_mtime > cutoff:
                continue
            try:
                entry.unlink()
            except FileNotFoundError:
                continue
            removed_files += 1
            removed_bytes += stat.st_size
        return {
            "removed_files": removed_files,
            "removed_bytes": removed_bytes,
            "active_files": len(active),
        }

    def _cleanup_spool(
        self,
        *,
        now: Optional[float] = None,
        force: bool,
    ) -> dict[str, int]:
        timestamp = time.time() if now is None else float(now)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            last = conn.execute(
                "SELECT last_run_at FROM cron_delivery_spool_gc_v2 WHERE singleton=1"
            ).fetchone()
            if (
                not force
                and last is not None
                and float(last[0]) > timestamp - _SPOOL_GC_INTERVAL_SECONDS
            ):
                conn.commit()
                return {"removed_files": 0, "removed_bytes": 0, "active_files": 0}
            result = self._cleanup_spool_locked(conn, now=timestamp)
            conn.execute(
                """INSERT INTO cron_delivery_spool_gc_v2(singleton, last_run_at)
                   VALUES (1, ?)
                   ON CONFLICT(singleton) DO UPDATE SET last_run_at=excluded.last_run_at""",
                (timestamp,),
            )
            conn.commit()
            return result
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def cleanup_spool(self, *, now: Optional[float] = None) -> dict[str, int]:
        """Remove only expired blobs not referenced by a nonterminal row."""
        return self._cleanup_spool(now=now, force=True)

    def spool_media(self, source: os.PathLike[str] | str, *, is_voice: bool) -> dict[str, Any]:
        """Copy a media artifact into the content-addressed durable spool."""
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise SpoolInvalid(f"media source is not a regular file: {source_path}")
        digest = hashlib.sha256()
        size = 0
        with source_path.open("rb") as src:
            while chunk := src.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        sha256 = digest.hexdigest()
        suffix = source_path.suffix.lower()
        relative_path = f"{sha256}{suffix}"
        destination = self.spool_dir / relative_path
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._cleanup_spool_locked(conn, now=time.time())
            if not destination.exists():
                current_size = self._spool_size_bytes()
                if current_size + size > self.max_spool_bytes:
                    raise SpoolInvalid(
                        "structured delivery spool capacity reached with active "
                        f"or retained media ({self.max_spool_bytes} bytes)"
                    )
                temporary = (
                    self.spool_dir
                    / f".{relative_path}.{secrets.token_hex(8)}.tmp"
                )
                try:
                    with source_path.open("rb") as src, temporary.open("xb") as dst:
                        while chunk := src.read(1024 * 1024):
                            dst.write(chunk)
                        dst.flush()
                        os.fsync(dst.fileno())
                    os.replace(temporary, destination)
                    directory_fd = os.open(self.spool_dir, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        ref = {
            "relative_path": relative_path,
            "sha256": sha256,
            "size": size,
            "is_voice": bool(is_voice),
            "original_name": source_path.name,
        }
        self.resolve_media(ref)
        return ref

    def rerecord(
        self, unit: Mapping[str, Any], *, now: Optional[float] = None
    ) -> OutboxRecord:
        """Replace one logical obligation as a new fenced generation."""
        obligation_id, payload_hash, payload = self._canonical_payload(unit)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        timestamp = time.time() if now is None else float(now)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute(
                "SELECT * FROM cron_delivery_outbox_v2 WHERE obligation_id=?",
                (obligation_id,),
            ).fetchone()
            if old is None:
                raise KeyError(obligation_id)
            if old["state"] == "attempting":
                raise OutboxError(
                    "cannot re-record an attempting obligation before its sender quiesces"
                )
            cursor = conn.execute(
                """UPDATE cron_delivery_outbox_v2
                   SET payload_hash=?, payload_json=?, generation=generation+1,
                       state='pending', attempts=0, next_attempt_at=0,
                       owner_pid=NULL, owner_started_at=NULL, claim_token=NULL,
                       updated_at=?, last_error=NULL, provider_message_id=NULL,
                       delivered_at=NULL
                   WHERE obligation_id=? AND generation=?""",
                (
                    payload_hash,
                    encoded,
                    timestamp,
                    obligation_id,
                    old["generation"],
                ),
            )
            if cursor.rowcount != 1:
                raise ClaimLost("generation changed while re-recording")
            if old["claim_token"]:
                conn.execute(
                    """DELETE FROM cron_delivery_recovery_lease_v2
                       WHERE singleton=1 AND obligation_id=? AND generation=?
                         AND claim_token=? AND owner_pid=? AND owner_started_at=?""",
                    (
                        obligation_id,
                        old["generation"],
                        old["claim_token"],
                        old["owner_pid"],
                        old["owner_started_at"],
                    ),
                )
            row = conn.execute(
                "SELECT * FROM cron_delivery_outbox_v2 WHERE obligation_id=?",
                (obligation_id,),
            ).fetchone()
            conn.commit()
            return self._record(row)
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def resolve_media(self, ref: Mapping[str, Any]) -> Path:
        """Resolve and verify a spool reference before any provider side effect."""
        relative = str(ref.get("relative_path") or "")
        candidate = (self.spool_dir / relative).resolve()
        spool_root = self.spool_dir.resolve()
        if candidate.parent != spool_root:
            raise SpoolInvalid("media reference escapes durable spool")
        try:
            stat = candidate.stat()
        except FileNotFoundError as exc:
            raise SpoolInvalid(f"durable media blob missing: {relative}") from exc
        expected_size = int(ref.get("size", -1))
        if not candidate.is_file() or stat.st_size != expected_size:
            raise SpoolInvalid(f"durable media blob size mismatch: {relative}")
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != str(ref.get("sha256") or ""):
            raise SpoolInvalid(f"durable media blob hash mismatch: {relative}")
        return candidate

    def get(self, obligation_id: str) -> OutboxRecord:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM cron_delivery_outbox_v2 WHERE obligation_id=?",
                (obligation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(obligation_id)
            return self._record(row)
        finally:
            conn.close()

    @staticmethod
    def _quarantine_corrupt_row(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        now: float,
    ) -> None:
        conn.execute(
            """UPDATE cron_delivery_outbox_v2
               SET state='abandoned', updated_at=?,
                   last_error='structured delivery payload integrity failure',
                   owner_pid=NULL, owner_started_at=NULL, claim_token=NULL
               WHERE obligation_id=? AND generation=?
                 AND state IN ('pending','failed')""",
            (float(now), row["obligation_id"], row["generation"]),
        )

    def _corrupt_row_scopes(self, row: sqlite3.Row) -> set[tuple[str, str]]:
        """Recover a corrupt payload's scope only from its independent capsule."""
        capsule = self._validated_scope_capsule(row)
        return {
            (
                str(capsule["execution_id"]),
                str(capsule["canonical_target"]),
            )
        }

    def claim_next(
        self,
        *,
        owner_pid: int,
        owner_started_at: float,
        now: Optional[float] = None,
        obligation_id: Optional[str] = None,
    ) -> Optional[OutboxRecord]:
        timestamp = time.time() if now is None else float(now)
        token = secrets.token_hex(32)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            lease = conn.execute(
                "SELECT * FROM cron_delivery_recovery_lease_v2 WHERE singleton=1"
            ).fetchone()
            if lease is not None:
                if _owner_alive(lease["owner_pid"], lease["owner_started_at"]):
                    conn.rollback()
                    return None
                conn.execute(
                    """UPDATE cron_delivery_outbox_v2
                       SET state='unknown', updated_at=?,
                           last_error='owner process died after dispatch claim',
                           owner_pid=NULL, owner_started_at=NULL, claim_token=NULL
                       WHERE obligation_id=? AND generation=? AND claim_token=?
                         AND owner_pid=? AND owner_started_at=?
                         AND state='attempting'""",
                    (
                        timestamp,
                        lease["obligation_id"],
                        lease["generation"],
                        lease["claim_token"],
                        lease["owner_pid"],
                        lease["owner_started_at"],
                    ),
                )
                deleted = conn.execute(
                    """DELETE FROM cron_delivery_recovery_lease_v2
                       WHERE singleton=1 AND obligation_id=? AND generation=?
                         AND claim_token=? AND owner_pid=? AND owner_started_at=?""",
                    (
                        lease["obligation_id"],
                        lease["generation"],
                        lease["claim_token"],
                        lease["owner_pid"],
                        lease["owner_started_at"],
                    ),
                )
                if deleted.rowcount != 1:
                    raise ClaimLost("stale global lease changed during reclaim")
            conn.execute(
                """UPDATE cron_delivery_outbox_v2
                   SET state='abandoned', updated_at=?,
                       last_error=COALESCE(last_error, 'maximum delivery attempts reached'),
                       owner_pid=NULL, owner_started_at=NULL, claim_token=NULL
                   WHERE state IN ('pending','failed')
                     AND (attempts >= ? OR created_at < ?)""",
                (timestamp, _MAX_ATTEMPTS, timestamp - _STALE_AFTER_SECONDS),
            )
            all_rows = conn.execute(
                """SELECT * FROM cron_delivery_outbox_v2
                   ORDER BY created_at, obligation_id"""
            ).fetchall()
            validated_rows: list[tuple[sqlite3.Row, OutboxRecord]] = []
            corrupt_scopes: set[tuple[str, str]] = set()
            unscoped_corruption = False
            for persisted in all_rows:
                try:
                    persisted_record = self._record(persisted)
                except PayloadConflict:
                    try:
                        corrupt_scopes.update(self._corrupt_row_scopes(persisted))
                    except PayloadConflict:
                        unscoped_corruption = True
                    self._quarantine_corrupt_row(conn, persisted, now=timestamp)
                    continue
                validated_rows.append((persisted, persisted_record))
            if unscoped_corruption:
                conn.commit()
                return None
            row = None
            for candidate, validated in validated_rows:
                if candidate["state"] not in ("pending", "failed"):
                    continue
                if float(candidate["next_attempt_at"]) > timestamp:
                    continue
                if obligation_id is not None and candidate["obligation_id"] != obligation_id:
                    continue
                scope = (
                    str(validated.payload["execution_id"]),
                    str(validated.payload["canonical_target"]),
                )
                if scope in corrupt_scopes:
                    continue
                siblings = [
                    (sibling, sibling_record)
                    for sibling, sibling_record in validated_rows
                    if (
                        str(sibling_record.payload["execution_id"]),
                        str(sibling_record.payload["canonical_target"]),
                    )
                    == scope
                ]
                target_index = int(validated.payload["target_index"])
                unit_index = int(validated.payload["unit_index"])
                if any(
                    int(sibling_record.payload["target_index"]) == target_index
                    and int(sibling_record.payload["unit_index"]) < unit_index
                    and sibling["state"] != "delivered"
                    for sibling, sibling_record in siblings
                ):
                    continue
                circuit = conn.execute(
                    """SELECT 1 FROM cron_delivery_circuit_v2
                       WHERE platform=? AND blocked_until > ?""",
                    (
                        str(validated.payload["logical_platform"]).lower(),
                        timestamp,
                    ),
                ).fetchone()
                if circuit is None:
                    row = candidate
                    break
            if row is None:
                conn.commit()
                return None
            cursor = conn.execute(
                """UPDATE cron_delivery_outbox_v2
                   SET state='attempting', attempts=attempts+1,
                       owner_pid=?, owner_started_at=?, claim_token=?, updated_at=?
                   WHERE obligation_id=? AND generation=?
                     AND state IN ('pending','failed')""",
                (
                    int(owner_pid),
                    float(owner_started_at),
                    token,
                    timestamp,
                    row["obligation_id"],
                    row["generation"],
                ),
            )
            if cursor.rowcount != 1:
                raise ClaimLost("row changed while claiming")
            conn.execute(
                """INSERT INTO cron_delivery_recovery_lease_v2
                   (singleton, obligation_id, generation, claim_token,
                    owner_pid, owner_started_at, acquired_at)
                   VALUES (1, ?, ?, ?, ?, ?, ?)""",
                (
                    row["obligation_id"],
                    row["generation"],
                    token,
                    int(owner_pid),
                    float(owner_started_at),
                    timestamp,
                ),
            )
            claimed = conn.execute(
                "SELECT * FROM cron_delivery_outbox_v2 WHERE obligation_id=?",
                (row["obligation_id"],),
            ).fetchone()
            claimed_record = self._record(claimed)
            conn.commit()
            return claimed_record
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _finish(
        self,
        claim: OutboxRecord,
        *,
        state: str,
        now: float,
        error: Optional[str] = None,
        provider_message_id: Optional[str] = None,
        next_attempt_at: float = 0,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """UPDATE cron_delivery_outbox_v2
                   SET state=?, updated_at=?, last_error=?, provider_message_id=?,
                       next_attempt_at=?, delivered_at=?, owner_pid=NULL,
                       owner_started_at=NULL, claim_token=NULL
                   WHERE obligation_id=? AND generation=? AND claim_token=?
                     AND owner_pid=? AND owner_started_at=?
                     AND state='attempting'""",
                (
                    state,
                    float(now),
                    error,
                    provider_message_id,
                    float(next_attempt_at),
                    float(now) if state == "delivered" else None,
                    claim.obligation_id,
                    claim.generation,
                    claim.claim_token,
                    claim.owner_pid,
                    claim.owner_started_at,
                ),
            )
            if cursor.rowcount != 1:
                raise ClaimLost(f"claim no longer owns {claim.obligation_id}")
            lease_cursor = conn.execute(
                """DELETE FROM cron_delivery_recovery_lease_v2
                   WHERE singleton=1 AND obligation_id=? AND generation=?
                     AND claim_token=? AND owner_pid=? AND owner_started_at=?""",
                (
                    claim.obligation_id,
                    claim.generation,
                    claim.claim_token,
                    claim.owner_pid,
                    claim.owner_started_at,
                ),
            )
            if lease_cursor.rowcount != 1:
                raise ClaimLost("matching global lease was lost")
            if state == "delivered":
                execution_id = str(claim.payload["execution_id"])
                job_id = str(claim.payload["job_id"])
                all_certified = True
                remaining = False
                for persisted in conn.execute(
                    "SELECT * FROM cron_delivery_outbox_v2"
                ).fetchall():
                    try:
                        capsule = self._validated_scope_capsule(persisted)
                    except PayloadConflict:
                        # Without an authenticated scope, conservatively block
                        # every completion marker in this transaction.
                        all_certified = False
                        continue
                    if str(capsule["execution_id"]) != execution_id:
                        continue
                    try:
                        sibling = self._record(persisted)
                    except PayloadConflict:
                        all_certified = False
                        continue
                    if (
                        str(capsule["job_id"]) != job_id
                        or str(sibling.payload["job_id"]) != job_id
                    ):
                        all_certified = False
                    if sibling.state != "delivered":
                        remaining = True
                if all_certified and not remaining:
                    conn.execute(
                        """INSERT OR IGNORE INTO cron_delivery_job_status_sync_v2
                           (execution_id, job_id, created_at) VALUES (?, ?, ?)""",
                        (execution_id, job_id, float(now)),
                    )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def mark_delivered(
        self,
        claim: OutboxRecord,
        *,
        provider_message_id: Optional[str],
        now: Optional[float] = None,
    ) -> None:
        self._finish(
            claim,
            state="delivered",
            now=time.time() if now is None else float(now),
            provider_message_id=provider_message_id,
        )

    def mark_unknown(
        self,
        claim: OutboxRecord,
        *,
        error: str,
        now: Optional[float] = None,
    ) -> None:
        """Record an ambiguous provider acknowledgement without making it retryable."""
        self._finish(
            claim,
            state="unknown",
            now=time.time() if now is None else float(now),
            error=error,
        )

    def mark_failed(
        self,
        claim: OutboxRecord,
        *,
        error: str,
        retryable: bool,
        now: Optional[float] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        if not retryable:
            self._finish(claim, state="abandoned", now=timestamp, error=error)
            return
        delay = min(3600.0, max(float(retry_after or 0), min(300.0, 2 ** max(0, claim.attempts - 1))))
        self._finish(
            claim,
            state="failed",
            now=timestamp,
            error=error,
            next_attempt_at=timestamp + delay,
        )

    def record_platform_circuit(
        self,
        platform: str,
        *,
        blocked_until: float,
        reason: str,
        now: Optional[float] = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO cron_delivery_circuit_v2
                   (platform, blocked_until, reason, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(platform) DO UPDATE SET
                     blocked_until=MAX(blocked_until, excluded.blocked_until),
                     reason=excluded.reason, updated_at=excluded.updated_at""",
                (str(platform).lower(), float(blocked_until), reason, timestamp),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def pending_job_status_syncs(self, *, limit: int = 100) -> list[tuple[str, str]]:
        """Return durable execution/job markers awaiting jobs.json reconciliation."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT execution_id, job_id
                   FROM cron_delivery_job_status_sync_v2
                   ORDER BY created_at, execution_id, job_id LIMIT ?""",
                (max(1, int(limit)),),
            ).fetchall()
            return [(str(row[0]), str(row[1])) for row in rows]
        finally:
            conn.close()

    def mark_job_status_synced(self, execution_id: str, job_id: str) -> None:
        """Acknowledge one idempotently reconciled mutable job-status marker."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """DELETE FROM cron_delivery_job_status_sync_v2
                   WHERE execution_id=? AND job_id=?""",
                (str(execution_id), str(job_id)),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def close(self) -> None:
        self._closed = True
