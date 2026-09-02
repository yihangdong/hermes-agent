"""Stage-A Owner bridge — authenticated Owner message → local Unix socket.

A single, deliberately tiny seam that lets one already-authenticated Owner
direct message become one typed request on a **fixed local Unix-domain
socket**, and lets exactly one correlated reply come back into that same
conversation.  Nothing else.

Why it is shaped this way
-------------------------
The consumer on the other end of the socket is a deterministic controller
that owns and listens on the socket; this module is a **client only** — it
never binds, listens or accepts, so the bridge adds no network surface of
any kind.  Peer confinement is the listener's duty (``SO_PEERCRED`` or the
platform equivalent); this side supplies the typed schema and request id
that make that enforcement checkable, and additionally refuses to talk to a
socket path that is not a socket, is world-writable, or sits in a
world-writable directory.

Hard boundaries, all enforced below rather than by convention:

* **Off unless configured.**  With no Owner id configured the bridge is
  completely inert: :meth:`StageAOwnerBridge.process` returns ``None`` on
  the first check and ordinary routing is bit-identical to a tree without
  this module.
* **The Owner only.**  A request is admitted only if it *already passed*
  the adapter's own intake authorization and the sender equals the one
  configured Stage-A Owner id, in a direct message, text-only, beginning
  with the fixed :data:`REQUEST_MARKER`.  Anything else falls straight
  through to ordinary routing untouched.
* **The destination is never on the wire.**  A reply is delivered only to
  the conversation captured at ingress.  The reply frame is rejected
  outright if it carries any key outside :data:`_REPLY_KEYS`, so no
  destination, command, path or URL field can ever be smuggled in and
  honoured.
* **No credential crosses the seam.**  The frame carries the request text
  the Owner already typed plus non-secret correlation metadata.  Raw chat
  and user ids never leave this process — the peer sees only an opaque,
  non-reversible ``conversation_ref`` digest.
* **Fail closed, and say so.**  Every failure path answers the Owner in the
  same conversation with a fixed string from :data:`_REFUSAL_TEXT` and
  creates no work.  There is no second destination, no other channel, no
  retry, no queue and no scheduler.

Contract: DYHANO AI-Organization Issue #197 (Owner ruling ``5513070065``)
§§A1–A4, realized under Issue #198 Phase A as
``WEIXIN_STAGEA_DIRECT_REPLY_BRIDGE_V1``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import stat
import struct
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --- wire contract -----------------------------------------------------

SCHEMA = "dyhano.stagea.owner_bridge.v1"
PROTOCOL = 1
CLIENT_ID = "hermes.gateway.stagea_owner_bridge"

REQUEST_TYPE = "owner_request"
REPLY_TYPE = "owner_reply"

#: The only reply outcomes the Owner may be shown, per #197 §A4: an
#: accepted terminal result, or an explicit gate / failure / unresolved
#: state.  Anything else is a protocol violation and is refused.
REPLY_OUTCOMES = ("ACCEPTED_TERMINAL", "OWNER_GATE", "FAILED", "UNKNOWN")

_REQUEST_KEYS = (
    "schema",
    "protocol",
    "type",
    "client",
    "request_id",
    "conversation_ref",
    "channel",
    "chat_type",
    "text",
    "sent_at",
)

#: Exact permitted reply keys.  A frame carrying anything else is refused
#: without inspection — this is what structurally prevents the peer from
#: ever proposing its own delivery destination or side effect.
_REPLY_KEYS = frozenset(
    {"schema", "protocol", "type", "request_id", "conversation_ref", "outcome", "text"}
)

# --- fixed limits ------------------------------------------------------

#: Marker that promotes an Owner direct message to a Stage-A request.
#: Matched case-insensitively on the leading token only; everything after
#: it is the request text.
REQUEST_MARKER = "/stagea"

#: Conceptual target from #197 §A3.  The exact path stays host-compatible
#: through :data:`ENV_SOCKET_PATH`.
DEFAULT_SOCKET_PATH = "/run/dyhano-stagea/owner-bridge.sock"

ENV_OWNER_USER_ID = "HERMES_STAGEA_OWNER_WEIXIN_USER_ID"
ENV_SOCKET_PATH = "HERMES_STAGEA_BRIDGE_SOCKET"
ENV_SOCKET_UID = "HERMES_STAGEA_BRIDGE_SOCKET_UID"

MAX_REQUEST_TEXT_BYTES = 8192
MAX_FRAME_BYTES = 65536
MAX_REPLY_TEXT_CHARS = 1500

#: The peer must answer within this window.  A miss is not silence: the
#: Owner is told the outcome is unresolved, which is one of the admitted
#: reply classes and matches the human-on-exception policy.
REPLY_DEADLINE_SECONDS = 1800.0
CONNECT_TIMEOUT_SECONDS = 10.0

#: Bounds on the replay guard and on concurrent exchanges.  Both cap the
#: work a burst of Owner messages can create.
REPLAY_TTL_SECONDS = 900.0
REPLAY_CAPACITY = 256
MAX_INFLIGHT_REQUESTS = 4

REPLY_PREFIX = "[Stage-A]"

_REFUSAL_TEXT: Dict[str, str] = {
    "media_not_admitted": "attachments are not accepted; send the request as text only",
    "empty_request": "the request text was empty",
    "request_too_large": f"the request exceeded {MAX_REQUEST_TEXT_BYTES} bytes",
    "duplicate_request": "this request was already received",
    "too_many_inflight": "too many Stage-A requests are already in flight",
    "platform_unsupported": "the local bridge is unavailable on this platform",
    "socket_unavailable": "the Stage-A controller socket is unavailable",
    "socket_not_a_socket": "the configured bridge path is not a socket",
    "socket_world_writable": "the bridge socket is world-writable and was refused",
    "socket_dir_world_writable": "the bridge socket directory is world-writable and was refused",
    "socket_owner_mismatch": "the bridge socket owner did not match the expected identity",
    "connect_failed": "the Stage-A controller did not accept the connection",
    "send_failed": "the request could not be delivered to the Stage-A controller",
    "reply_timeout": "the Stage-A controller did not answer in time; the outcome is unresolved",
    "reply_truncated": "the Stage-A controller reply was incomplete",
    "reply_oversized": "the Stage-A controller reply exceeded the frame limit",
    "reply_malformed": "the Stage-A controller reply was malformed",
    "reply_unexpected_field": "the Stage-A controller reply carried an unexpected field",
    "reply_wrong_request": "the Stage-A controller reply did not match this request",
    "reply_wrong_conversation": "the Stage-A controller reply did not match this conversation",
    "reply_bad_outcome": "the Stage-A controller reply carried an unknown outcome",
    "reply_bad_text": "the Stage-A controller reply text was unusable",
    "config_invalid": "the Stage-A bridge configuration is invalid",
}


class BridgeError(Exception):
    """A fail-closed bridge outcome, identified by a fixed reason code.

    The code is the only thing that ever reaches the Owner — exception
    text is kept out of the reply so no path, address or internal detail
    can leak into the conversation.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def refusal_text(code: str) -> str:
    """Render the fixed Owner-visible sentence for a refusal ``code``."""
    detail = _REFUSAL_TEXT.get(code, "the request was refused")
    return f"{REPLY_PREFIX} request not accepted ({code}): {detail}. No Stage-A work was created."


def outcome_text(outcome: str, text: str) -> str:
    """Render an accepted peer reply for delivery."""
    return f"{REPLY_PREFIX} {outcome}\n{text}"


# --- configuration -----------------------------------------------------


@dataclass(frozen=True)
class BridgeConfig:
    """Resolved, fixed, non-secret bridge configuration."""

    owner_user_id: str
    socket_path: str
    socket_uid: Optional[int]


def load_config(getter: Callable[[str, Optional[str]], Optional[str]]) -> Optional[BridgeConfig]:
    """Resolve configuration, or ``None`` when the bridge is disabled.

    ``getter`` is the adapter's profile-scoped configuration reader, so a
    secondary multiplexed profile cannot borrow the default profile's
    Stage-A binding.

    Raises:
        BridgeError: ``config_invalid`` when the bridge is enabled but a
            value cannot be used.  Enabled-but-broken fails closed rather
            than degrading to a weaker check.
    """
    owner = (getter(ENV_OWNER_USER_ID, "") or "").strip()
    if not owner:
        return None

    socket_path = (getter(ENV_SOCKET_PATH, "") or "").strip() or DEFAULT_SOCKET_PATH

    raw_uid = (getter(ENV_SOCKET_UID, "") or "").strip()
    socket_uid: Optional[int] = None
    if raw_uid:
        try:
            socket_uid = int(raw_uid)
        except ValueError:
            raise BridgeError("config_invalid") from None
        if socket_uid < 0:
            raise BridgeError("config_invalid")

    return BridgeConfig(owner_user_id=owner, socket_path=socket_path, socket_uid=socket_uid)


# --- admission ---------------------------------------------------------


@dataclass(frozen=True)
class Admission:
    """Outcome of classifying one inbound message.

    Exactly one of three shapes:

    * ``admitted`` — a Stage-A request with ``request_text`` set;
    * ``refused`` — an Owner Stage-A attempt that cannot proceed; the
      Owner is answered and the message is **not** routed onward;
    * neither — an ordinary message; the caller routes it as before.
    """

    admitted: bool
    refused: bool
    reason: str
    request_text: str = ""

    @property
    def consumed(self) -> bool:
        return self.admitted or self.refused


_PASS_THROUGH = Admission(admitted=False, refused=False, reason="pass_through")


def _strip_marker(text: str) -> Optional[str]:
    """Return the request text when ``text`` opens with the marker.

    ``None`` when the marker is absent.  The marker must be a whole
    leading token: ``/stageant`` is not a Stage-A request.
    """
    candidate = text.lstrip()
    marker_len = len(REQUEST_MARKER)
    if candidate[:marker_len].lower() != REQUEST_MARKER:
        return None
    remainder = candidate[marker_len:]
    if remainder and not remainder[0].isspace():
        return None
    return remainder.strip()


def classify(
    config: Optional[BridgeConfig],
    *,
    chat_type: str,
    sender_id: Optional[str],
    text: str,
    has_media: bool,
) -> Admission:
    """Decide what one already-authorized inbound message is.

    Ordering matters: every check that could route an ordinary message
    differently is evaluated before any Stage-A-specific check, so a
    message that is not an Owner Stage-A request can never be consumed.
    """
    if config is None:
        return _PASS_THROUGH
    if chat_type != "dm":
        return _PASS_THROUGH
    if (sender_id or "") != config.owner_user_id:
        return _PASS_THROUGH

    request_text = _strip_marker(text or "")
    if request_text is None:
        return _PASS_THROUGH

    # From here the Owner has explicitly asked for Stage-A, so a problem
    # is answered rather than quietly handed to ordinary routing.
    if has_media:
        return Admission(admitted=False, refused=True, reason="media_not_admitted")
    if not request_text:
        return Admission(admitted=False, refused=True, reason="empty_request")
    if len(request_text.encode("utf-8")) > MAX_REQUEST_TEXT_BYTES:
        return Admission(admitted=False, refused=True, reason="request_too_large")

    return Admission(admitted=True, refused=False, reason="admitted", request_text=request_text)


# --- correlation -------------------------------------------------------


def conversation_ref(conversation_key: str) -> str:
    """Opaque, stable, non-reversible reference for one conversation.

    The peer needs to correlate replies, not to address them, so it is
    given a digest instead of the Owner's real chat identity.  Nothing
    downstream can turn this back into a chat id, and nothing downstream
    is ever used as a destination.
    """
    digest = hashlib.sha256(f"{SCHEMA}|{conversation_key}".encode("utf-8")).hexdigest()
    return digest[:32]


# --- framing -----------------------------------------------------------


def encode_frame(payload: Dict[str, Any]) -> bytes:
    """Length-prefixed compact JSON frame.

    Raises:
        BridgeError: ``send_failed`` when the payload exceeds the cap.
    """
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise BridgeError("send_failed")
    return struct.pack(">I", len(body)) + body


async def read_frame(reader: asyncio.StreamReader) -> Dict[str, Any]:
    """Read exactly one bounded frame.

    The declared length is validated *before* any body byte is read, so an
    oversized declaration costs nothing.
    """
    try:
        header = await reader.readexactly(4)
    except (asyncio.IncompleteReadError, ConnectionError) as exc:
        raise BridgeError("reply_truncated") from exc
    (length,) = struct.unpack(">I", header)
    if length == 0 or length > MAX_FRAME_BYTES:
        raise BridgeError("reply_oversized")
    try:
        body = await reader.readexactly(length)
    except (asyncio.IncompleteReadError, ConnectionError) as exc:
        raise BridgeError("reply_truncated") from exc
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BridgeError("reply_malformed") from exc
    if not isinstance(payload, dict):
        raise BridgeError("reply_malformed")
    return payload


def build_request(*, request_id: str, ref: str, channel: str, text: str) -> Dict[str, Any]:
    """Assemble the typed request payload.

    Every field is either a fixed constant, the Owner's own text, or an
    opaque correlation value.  There is deliberately no field for a
    credential, destination, path, command or URL.
    """
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "type": REQUEST_TYPE,
        "client": CLIENT_ID,
        "request_id": request_id,
        "conversation_ref": ref,
        "channel": channel,
        "chat_type": "dm",
        "text": text,
        "sent_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def validate_reply(payload: Dict[str, Any], *, request_id: str, ref: str) -> Tuple[str, str]:
    """Return ``(outcome, text)`` from a reply, or fail closed.

    Unknown keys are rejected before anything is read out of the frame:
    that is what makes it structurally impossible for the peer to propose
    a destination or a side effect.
    """
    extra = set(payload) - _REPLY_KEYS
    if extra:
        raise BridgeError("reply_unexpected_field")
    if payload.get("schema") != SCHEMA or payload.get("protocol") != PROTOCOL:
        raise BridgeError("reply_malformed")
    if payload.get("type") != REPLY_TYPE:
        raise BridgeError("reply_malformed")
    if payload.get("request_id") != request_id:
        raise BridgeError("reply_wrong_request")
    if payload.get("conversation_ref") != ref:
        raise BridgeError("reply_wrong_conversation")

    outcome = payload.get("outcome")
    if outcome not in REPLY_OUTCOMES:
        raise BridgeError("reply_bad_outcome")

    text = payload.get("text")
    if not isinstance(text, str):
        raise BridgeError("reply_bad_text")
    text = text.strip()
    if not text or len(text) > MAX_REPLY_TEXT_CHARS:
        raise BridgeError("reply_bad_text")

    return outcome, text


# --- socket preflight --------------------------------------------------


def check_socket_path(path: str, expected_uid: Optional[int]) -> None:
    """Refuse a socket that a local attacker could have substituted.

    The socket must be group-writable for a separate gateway identity to
    connect at all, so the test is world-writability — on the socket and
    on the directory that holds it — plus an optional exact owner.
    """
    try:
        st = os.stat(path)
    except OSError as exc:
        raise BridgeError("socket_unavailable") from exc
    if not stat.S_ISSOCK(st.st_mode):
        raise BridgeError("socket_not_a_socket")
    if st.st_mode & stat.S_IWOTH:
        raise BridgeError("socket_world_writable")
    if expected_uid is not None and st.st_uid != expected_uid:
        raise BridgeError("socket_owner_mismatch")

    parent = os.path.dirname(os.path.abspath(path)) or os.sep
    try:
        dir_st = os.stat(parent)
    except OSError as exc:
        raise BridgeError("socket_unavailable") from exc
    if dir_st.st_mode & stat.S_IWOTH:
        raise BridgeError("socket_dir_world_writable")


# --- replay guard ------------------------------------------------------


class ReplayGuard:
    """Bounded first-seen guard over ``(conversation, message)`` keys.

    Keys are recorded *before* the exchange starts, so a duplicate that
    arrives while the first request is still in flight cannot create a
    second one.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = REPLAY_TTL_SECONDS,
        capacity: int = REPLAY_CAPACITY,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._capacity = capacity
        self._clock = clock
        self._seen: Dict[str, float] = {}

    def check_and_record(self, key: str) -> bool:
        """``True`` when ``key`` is new (and now recorded)."""
        now = self._clock()
        expired = [k for k, seen_at in self._seen.items() if now - seen_at > self._ttl]
        for k in expired:
            self._seen.pop(k, None)
        if key in self._seen:
            return False
        while len(self._seen) >= self._capacity:
            oldest = min(self._seen, key=self._seen.__getitem__)
            self._seen.pop(oldest, None)
        self._seen[key] = now
        return True


def replay_key(ref: str, message_id: Optional[str], text: str) -> str:
    """Stable dedupe key for one inbound Owner message."""
    if message_id:
        return f"{ref}|id:{message_id}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{ref}|tx:{digest}"


# --- bridge ------------------------------------------------------------


class StageAOwnerBridge:
    """One adapter's Stage-A Owner bridge.

    Holds the resolved configuration, the replay guard and the in-flight
    counter.  Owns no socket, no task and no timer: an exchange exists
    only for the lifetime of the message that caused it.
    """

    def __init__(
        self,
        *,
        secret_getter: Callable[[str, Optional[str]], Optional[str]],
        channel: str,
    ) -> None:
        self._secret_getter = secret_getter
        self._channel = channel
        self._config: Optional[BridgeConfig] = None
        self._config_error: Optional[str] = None
        self._config_loaded = False
        self._guard = ReplayGuard()
        self._inflight = 0

    # -- configuration --

    def _resolve_config(self) -> Optional[BridgeConfig]:
        if not self._config_loaded:
            self._config_loaded = True
            try:
                self._config = load_config(self._secret_getter)
            except BridgeError as exc:
                self._config = None
                self._config_error = exc.code
                logger.error("[stagea] bridge configuration rejected: %s", exc.code)
        return self._config

    @property
    def enabled(self) -> bool:
        return self._resolve_config() is not None

    # -- main entry point --

    async def process(
        self,
        *,
        chat_type: str,
        sender_id: Optional[str],
        text: str,
        has_media: bool,
        conversation_key: str,
        message_id: Optional[str],
    ) -> Optional[str]:
        """Handle one inbound message.

        Returns:
            The exact text to deliver back into the *same* conversation,
            or ``None`` when the message is ordinary and must continue
            through the caller's normal routing.
        """
        config = self._resolve_config()
        if config is None:
            # A broken configuration must not silently look like "off"
            # for the Owner's own Stage-A attempts.
            if self._config_error and _strip_marker(text or "") is not None:
                return refusal_text(self._config_error)
            return None

        decision = classify(
            config,
            chat_type=chat_type,
            sender_id=sender_id,
            text=text,
            has_media=has_media,
        )
        if not decision.consumed:
            return None
        if decision.refused:
            logger.info("[stagea] request refused: %s", decision.reason)
            return refusal_text(decision.reason)

        ref = conversation_ref(conversation_key)
        if not self._guard.check_and_record(replay_key(ref, message_id, decision.request_text)):
            logger.info("[stagea] duplicate request suppressed")
            return refusal_text("duplicate_request")

        if self._inflight >= MAX_INFLIGHT_REQUESTS:
            logger.warning("[stagea] refusing request: %d already in flight", self._inflight)
            return refusal_text("too_many_inflight")

        request_id = uuid.uuid4().hex
        self._inflight += 1
        try:
            outcome, reply = await self._exchange(config, request_id, ref, decision.request_text)
        except BridgeError as exc:
            logger.warning("[stagea] request %s failed closed: %s", request_id, exc.code)
            return refusal_text(exc.code)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[stagea] request %s failed closed: unexpected error", request_id)
            return refusal_text("socket_unavailable")
        finally:
            self._inflight -= 1

        logger.info("[stagea] request %s answered: %s", request_id, outcome)
        return outcome_text(outcome, reply)

    # -- transport --

    async def _exchange(
        self,
        config: BridgeConfig,
        request_id: str,
        ref: str,
        request_text: str,
    ) -> Tuple[str, str]:
        """One connection: send the request, read one reply, close."""
        open_unix_connection = getattr(asyncio, "open_unix_connection", None)
        if open_unix_connection is None:
            raise BridgeError("platform_unsupported")

        await asyncio.to_thread(check_socket_path, config.socket_path, config.socket_uid)

        frame = encode_frame(
            build_request(
                request_id=request_id,
                ref=ref,
                channel=self._channel,
                text=request_text,
            )
        )

        try:
            reader, writer = await asyncio.wait_for(
                open_unix_connection(config.socket_path),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, OSError) as exc:
            raise BridgeError("connect_failed") from exc

        try:
            try:
                writer.write(frame)
                await writer.drain()
            except (OSError, ConnectionError) as exc:
                raise BridgeError("send_failed") from exc

            try:
                payload = await asyncio.wait_for(read_frame(reader), timeout=REPLY_DEADLINE_SECONDS)
            except asyncio.TimeoutError as exc:
                raise BridgeError("reply_timeout") from exc

            return validate_reply(payload, request_id=request_id, ref=ref)
        finally:
            await _close_writer(writer)


async def _close_writer(writer: Any) -> None:
    """Close a stream writer without letting teardown mask the outcome."""
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
        logger.debug("[stagea] writer close failed: %s", exc)
