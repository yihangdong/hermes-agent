"""Durable delivery-obligation ledger for gateway final responses.

A final agent response that was generated but not yet confirmed-delivered
to the messaging platform is the one artifact the gateway can lose without
a trace: the turn already burned its tokens, the text exists only in a
Python local, and a crash / planned restart between finalize and platform
ACK drops it silently (#58818, #41696, #63695).

This module records a small durable row per outbound final response in the
shared ``state.db`` (same file and conventions as
``tools.async_delegation`` — WAL, owner pid + process-start-time liveness,
bounded retention). The gateway writes three checkpoints around the send:

    record_obligation()   state='pending'     before any send attempt
    mark_attempting()     state='attempting'  immediately before the await
    mark_delivered() /    state='delivered'   only on SendResult.success
    mark_failed()         state='failed'      on a definitive rejection

On startup and from a paced background worker, ``sweep_recoverable()`` claims
rows whose owning process is dead or whose failed send released ownership,
and hands them to the gateway for redelivery. Crash semantics are
explicit about ambiguity (the contract review of the earlier
delivery-outbox attempt, #61790, closed it for silently resending
ambiguous sends):

- ``pending``     — the send never started: redeliver plainly, no dup risk.
- ``attempting``  — crashed mid-await: the platform MAY already have the
  message. Redelivered WITH a visible recovered-reply marker so the
  contract is honest at-least-once, never a silent duplicate.
- ``failed``      — definitively rejected once; durable ``next_attempt_at``
  applies bounded exponential backoff without requiring a restart. Also
  carries the marker.
- ``delivered``   — nothing to do; retention prunes.

Poison rows cannot spin: one row is claimed per paced pass, platform-wide
rate limits pause the whole backlog, attempts are capped, stale rows expire,
and terminal rows transition to ``abandoned`` (kept briefly for inspection,
then pruned).

Everything here is best-effort by design: ledger failures must never block
or delay an actual send. Callers wrap every call in try/except.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_LOCK = threading.RLock()

# Redelivery policy knobs (module constants; deliberately not config — the
# ledger itself is gated by ``gateway.delivery_ledger`` and these bounds
# only matter in the rare recovery path).
MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
RETRY_BASE_SECONDS = 30.0
RETRY_MAX_SECONDS = 30 * 60.0
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500
_RECOVERY_LEASE_NAME = "global"

# Visible prefix for redeliveries that might duplicate an already-received
# message (crash mid-send / post-rejection retry). Honest at-least-once.
RECOVERED_MARKER = (
    "♻️ Recovered reply — the gateway restarted during delivery, "
    "so this may be a duplicate:\n\n"
)


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        # Serialize same-process WAL/schema initialization and commit it before
        # returning. RLock is required because callers commonly already hold
        # _DB_LOCK when they open their transaction.
        with _DB_LOCK:
            _initialize_schema(conn)
            conn.commit()
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (delivery_ledger)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            content TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT,
            next_attempt_at REAL NOT NULL DEFAULT 0,
            generation INTEGER NOT NULL DEFAULT 1
        )"""
    )
    # Online migration for ledgers created before durable in-process retry.
    # ALTER with a constant DEFAULT is metadata-only in SQLite and preserves
    # every existing obligation.
    def _ensure_column(name: str, ddl: str) -> None:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(delivery_obligations)")
        }
        if name in columns:
            return
        try:
            conn.execute(f"ALTER TABLE delivery_obligations ADD COLUMN {ddl}")
        except sqlite3.OperationalError:
            # Another gateway thread/process may have completed the same
            # online migration after our PRAGMA snapshot. Suppress only that
            # verified race; every other DDL failure remains fatal.
            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(delivery_obligations)"
                )
            }
            if name not in columns:
                raise

    _ensure_column("next_attempt_at", "next_attempt_at REAL NOT NULL DEFAULT 0")
    _ensure_column("generation", "generation INTEGER NOT NULL DEFAULT 1")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_recovery_lease (
            lease_name TEXT PRIMARY KEY,
            obligation_id TEXT NOT NULL,
            generation INTEGER NOT NULL,
            owner_pid INTEGER NOT NULL,
            owner_started_at INTEGER,
            acquired_at REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_platform_backoff (
            platform TEXT PRIMARY KEY,
            not_before REAL NOT NULL,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL
        )"""
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the
    transaction; they do not close the connection. Using ``with _connect()``
    alone therefore leaks a connection — and its WAL/SHM file descriptors — on
    every call, deferring the close to the garbage collector. On a long-running
    gateway that exhausts ``RLIMIT_NOFILE`` (the cron-ledger sibling of this
    bug was #69567 / PR #69594). ``record_obligation`` runs on every outbound
    final response, so this ledger is the highest-frequency leaker.
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time)."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_start = get_process_start_time(pid)
    except Exception:
        current_start = None
    if current_start is None:
        # No such process (or unreadable) — treat unreadable-but-extant
        # processes as alive only if the pid exists.
        try:
            os.kill(pid, 0)  # windows-footgun: ok — EPERM counts as alive below
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    if started_at is None:
        return True
    try:
        return int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(session_key: str, message_ref: str, content: str) -> str:
    """Stable id: same turn + same content re-records idempotently, while
    distinct threads/topics on the same chat can never collide (the
    session_key carries platform, chat and thread; ``message_ref`` is the
    triggering inbound message id, distinguishing turns in one session)."""
    payload = f"{session_key}|{message_ref}|{content}"
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


def record_obligation(
    *,
    obligation_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
) -> int:
    """Record a final response and return its monotonic row generation.

    A generation makes completion fencing explicit: a late ACK from an older
    send may not mark a same-id row that was re-recorded after restart.
    """
    now = time.time()
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at, next_attempt_at, generation)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, 0, 1)
               ON CONFLICT(obligation_id) DO UPDATE SET
                 session_key=excluded.session_key,
                 platform=excluded.platform,
                 chat_id=excluded.chat_id,
                 thread_id=excluded.thread_id,
                 content=excluded.content,
                 state='pending', attempts=0,
                 created_at=excluded.created_at,
                 updated_at=excluded.updated_at,
                 owner_pid=excluded.owner_pid,
                 owner_started_at=excluded.owner_started_at,
                 last_error=NULL, next_attempt_at=0,
                 generation=delivery_obligations.generation+1""",
            (obligation_id, session_key, platform, str(chat_id),
             str(thread_id) if thread_id else None, content, now, now,
             pid, started),
        )
        generation = int(conn.execute(
            "SELECT generation FROM delivery_obligations WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone()[0])
    _prune()
    return generation


def mark_attempting(obligation_id: str, *, generation: Optional[int] = None) -> None:
    _update_state(obligation_id, "attempting", generation=generation)


def mark_delivered(obligation_id: str, *, generation: Optional[int] = None) -> None:
    _update_state(obligation_id, "delivered", generation=generation)


def mark_failed(
    obligation_id: str,
    error: str = "",
    *,
    retry_after_seconds: Optional[float] = None,
    generation: Optional[int] = None,
    retryable: bool = True,
) -> None:
    """Record a rejected send and release it for durable in-process retry.

    ``attempts`` counts recovery sends (the original send is attempt zero).
    Each failure therefore doubles the delay, while an adapter-provided
    ``retry_after_seconds`` is treated as a lower bound.  Clearing ownership is
    intentional: the still-running gateway's paced worker may reclaim the row
    once it is due; recovery no longer depends on another process restart.
    """
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT attempts, generation FROM delivery_obligations WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone()
        if row is None or (generation is not None and int(row[1]) != int(generation)):
            return
        attempts = max(0, int(row[0] or 0))
        if not retryable:
            conn.execute(
                """UPDATE delivery_obligations
                   SET state='abandoned', updated_at=?, last_error=?,
                       owner_pid=NULL, owner_started_at=NULL, next_attempt_at=0
                   WHERE obligation_id=? AND (? IS NULL OR generation=?)""",
                (now, error[:500] if error else None, obligation_id,
                 generation, generation),
            )
            conn.execute(
                "DELETE FROM delivery_recovery_lease WHERE lease_name=? "
                "AND obligation_id=? AND (? IS NULL OR generation=?)",
                (_RECOVERY_LEASE_NAME, obligation_id, generation, generation),
            )
            return
        delay = RETRY_BASE_SECONDS * (2 ** attempts)
        if retry_after_seconds is not None:
            try:
                delay = max(delay, max(0.0, float(retry_after_seconds)))
            except (TypeError, ValueError):
                pass
        delay = min(delay, RETRY_MAX_SECONDS)
        conn.execute(
            """UPDATE delivery_obligations
               SET state='failed', updated_at=?, last_error=?,
                   owner_pid=NULL, owner_started_at=NULL, next_attempt_at=?
               WHERE obligation_id=? AND (? IS NULL OR generation=?)""",
            (now, error[:500] if error else None, now + delay, obligation_id,
             generation, generation),
        )
        conn.execute(
            "DELETE FROM delivery_recovery_lease WHERE lease_name=? "
            "AND obligation_id=? AND (? IS NULL OR generation=?)",
            (_RECOVERY_LEASE_NAME, obligation_id, generation, generation),
        )


def _update_state(
    obligation_id: str,
    state: str,
    error: str = "",
    *,
    generation: Optional[int] = None,
) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?
               WHERE obligation_id=? AND (? IS NULL OR generation=?)""",
            (state, time.time(), error[:500] if error else None, obligation_id,
             generation, generation),
        )
        if state in {"delivered", "failed", "abandoned"}:
            conn.execute(
                "DELETE FROM delivery_recovery_lease WHERE lease_name=? "
                "AND obligation_id=? AND (? IS NULL OR generation=?)",
                (_RECOVERY_LEASE_NAME, obligation_id, generation, generation),
            )


def platform_backoff_remaining(platform: str, now: Optional[float] = None) -> float:
    """Return durable platform-wide send cooldown remaining in seconds."""
    now = time.time() if now is None else now
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT not_before FROM delivery_platform_backoff WHERE platform=?",
            (str(platform),),
        ).fetchone()
    return max(0.0, float(row[0]) - now) if row else 0.0


def record_platform_rate_limit(
    platform: str,
    *,
    retry_after_seconds: float = 0.0,
    base_seconds: float = RETRY_BASE_SECONDS,
    max_seconds: float = RETRY_MAX_SECONDS,
) -> float:
    """Persist shared exponential cooldown and return the applied duration."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT consecutive_failures FROM delivery_platform_backoff "
            "WHERE platform=?",
            (str(platform),),
        ).fetchone()
        consecutive = max(0, int(row[0] or 0)) + 1 if row else 1
        duration = min(
            max(float(retry_after_seconds or 0.0),
                float(base_seconds) * (2 ** min(consecutive - 1, 20))),
            float(max_seconds),
        )
        conn.execute(
            """INSERT INTO delivery_platform_backoff
               (platform, not_before, consecutive_failures, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(platform) DO UPDATE SET
                 not_before=excluded.not_before,
                 consecutive_failures=excluded.consecutive_failures,
                 updated_at=excluded.updated_at""",
            (str(platform), now + duration, consecutive, now),
        )
    return duration


def clear_platform_backoff(platform: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "DELETE FROM delivery_platform_backoff WHERE platform=?",
            (str(platform),),
        )


def sweep_recoverable(
    now: Optional[float] = None,
    *,
    deliverable_platforms: Optional[set] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for
    redelivery.

    Claiming atomically re-stamps the owner to THIS process and increments
    ``attempts``, so a second gateway racing the same sweep cannot
    double-claim (the UPDATE is guarded on the previous owner stamp).
    Rows over the attempts cap or older than the stale cutoff transition to
    'abandoned' instead of being returned.

    ``deliverable_platforms`` (platform value strings) restricts claiming to
    platforms the caller can actually send on this boot.  ``attempts`` is the
    redelivery budget, so it must only be spent on a real send: a platform
    that failed to connect would otherwise burn one attempt per boot and hit
    the cap having never been sent once.  Rows for absent platforms are left
    untouched for a later pass; the stale cutoff still bounds them. ``limit``
    bounds claims per pass so callers can pace recovery independently from
    live traffic.
    """
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at,
                      owner_pid, owner_started_at, next_attempt_at, generation
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed')
               ORDER BY created_at ASC, obligation_id ASC"""
        ).fetchall()
        for (oid, session_key, platform, chat_id, thread_id, content, state,
             attempts, created_at, owner_pid, owner_started_at,
             next_attempt_at, generation) in rows:
            if _owner_alive(owner_pid, owner_started_at):
                continue  # a live gateway still owns this row
            if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=? WHERE obligation_id=?""",
                    (now, oid),
                )
                continue
            if float(next_attempt_at or 0) > now:
                continue
            if (
                deliverable_platforms is not None
                and platform not in deliverable_platforms
            ):
                # No adapter for this platform this boot — the caller cannot
                # send, so claiming would spend an attempt on a no-op.
                continue
            backoff = conn.execute(
                "SELECT not_before FROM delivery_platform_backoff WHERE platform=?",
                (platform,),
            ).fetchone()
            if backoff and float(backoff[0] or 0) > now:
                continue
            if limit is not None and len(claimed) >= max(0, int(limit)):
                continue
            lease = conn.execute(
                """SELECT obligation_id, generation, owner_pid, owner_started_at
                   FROM delivery_recovery_lease WHERE lease_name=?""",
                (_RECOVERY_LEASE_NAME,),
            ).fetchone()
            if lease:
                lease_row = conn.execute(
                    "SELECT state, generation FROM delivery_obligations "
                    "WHERE obligation_id=?",
                    (lease[0],),
                ).fetchone()
                lease_is_current = (
                    lease_row is not None
                    and lease_row[0] in {"pending", "attempting", "failed"}
                    and int(lease_row[1]) == int(lease[1])
                    and _owner_alive(lease[2], lease[3])
                )
                if lease_is_current:
                    return claimed
                conn.execute(
                    "DELETE FROM delivery_recovery_lease WHERE lease_name=?",
                    (_RECOVERY_LEASE_NAME,),
                )
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET owner_pid=?, owner_started_at=?, attempts=attempts+1,
                       updated_at=?
                   WHERE obligation_id=? AND generation=?
                     AND (owner_pid IS ? OR owner_pid=?)""",
                (pid, started, now, oid, generation, owner_pid, owner_pid),
            )
            if cursor.rowcount:
                conn.execute(
                    """INSERT OR REPLACE INTO delivery_recovery_lease
                       (lease_name, obligation_id, generation, owner_pid,
                        owner_started_at, acquired_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (_RECOVERY_LEASE_NAME, oid, generation, pid, started, now),
                )
                claimed.append({
                    "obligation_id": oid,
                    "session_key": session_key,
                    "platform": platform,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "content": content,
                    # pending = send never started, redeliver plainly;
                    # attempting/failed = ambiguous or rejected, carry marker.
                    "needs_marker": state != "pending",
                    "attempts": attempts + 1,
                    "generation": generation,
                })
                break
    return claimed


def _prune(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    cutoff = now - _RETENTION_SECONDS
    try:
        with _transaction() as conn:
            conn.execute(
                """DELETE FROM delivery_obligations
                   WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""",
                (cutoff,),
            )
            terminal = conn.execute(
                "SELECT COUNT(*) FROM delivery_obligations "
                "WHERE state IN ('delivered', 'abandoned')"
            ).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()[0]
            excess = min(max(0, total - _MAX_ROWS), terminal)
            if excess:
                conn.execute(
                    """DELETE FROM delivery_obligations WHERE obligation_id IN (
                         SELECT obligation_id FROM delivery_obligations
                         WHERE state IN ('delivered', 'abandoned')
                         ORDER BY updated_at ASC
                         LIMIT ?)""",
                    (excess,),
                )
    except Exception:
        logger.debug("delivery ledger prune failed", exc_info=True)


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        gw = config.get("gateway") or {}
        value = gw.get("delivery_ledger", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)
    except Exception:
        return True


def debug_rows(limit: int = 20) -> str:
    """Human-readable dump for ad-hoc inspection (sqlite3-free path)."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, state, attempts,
                      created_at, updated_at, last_error
               FROM delivery_obligations
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return json.dumps(
        [
            {
                "id": r[0], "session": r[1], "state": r[2], "attempts": r[3],
                "created_at": r[4], "updated_at": r[5], "last_error": r[6],
            }
            for r in rows
        ],
        indent=2,
    )
