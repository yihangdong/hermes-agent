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
any kind.  Peer confinement runs in *both* directions: the listener
authenticates this client (``SO_PEERCRED`` or the platform equivalent),
and this client authenticates the listener the same way before it writes
a single byte of the Owner's request — a path check describes an object,
but only a peer credential describes the process that is actually holding
the other end.

Hard boundaries, all enforced below rather than by convention:

* **Off unless configured.**  With no Owner id configured the bridge is
  completely inert: :meth:`StageAOwnerBridge.process` returns ``None`` on
  the first check and ordinary routing is bit-identical to a tree without
  this module.  All three settings are non-secret behaviour and live in
  ``config.yaml`` under the host adapter's own ``extra`` block —
  ``gateway.platforms.<platform>.extra.stagea_*`` — read through that
  adapter's profile-scoped configuration reader.  Nothing here reads the
  environment: ``.env`` is for credentials, and this bridge holds none.
* **The Owner only.**  A request is admitted only if it *already passed*
  the adapter's own intake authorization and the sender equals the one
  configured Stage-A Owner id, in a direct message, text-only, beginning
  with the fixed :data:`REQUEST_MARKER`.  Anything else falls straight
  through to ordinary routing untouched.  Admission is decided *before*
  any secondary configuration is consulted, so a misconfigured deployment
  can only ever answer the Owner — it can never consume, divert or
  respond to anybody else's message.
* **The same message is the same work.**  The Stage-A ``request_id`` is
  *derived* from the conversation and the platform's own message id, not
  generated, so a redelivery of one inbound message carries the identity
  it carried before — across a bridge restart, a gateway restart, or a
  cold host.  A message the platform cannot identify stably creates no
  work at all.
* **Text means text.**  Attachment presence is read from the inbound
  metadata the platform sent, never inferred from whether a download
  happened to succeed, so a failed media fetch cannot make an
  attachment-bearing message look text-only.
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
import socket as socket_module
import stat
import struct
import time
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
#: through :data:`CONFIG_SOCKET_PATH`.
DEFAULT_SOCKET_PATH = "/run/dyhano-stagea/owner-bridge.sock"

# Stage-A's three settings are non-secret behaviour — which Owner, which
# socket, whose uid — so they are ordinary platform configuration and
# live where every other tunable on the host adapter lives: ``config.yaml``
# under that platform's ``extra`` block, read through the adapter's own
# profile-scoped reader.  They are deliberately *not* environment
# variables: ``.env`` is for credentials, and an ambient variable is
# process-wide, so it could not be scoped to one profile even if it were
# allowed.
CONFIG_OWNER_USER_ID = "stagea_owner_user_id"
CONFIG_SOCKET_PATH = "stagea_bridge_socket_path"
CONFIG_SOCKET_UID = "stagea_bridge_socket_uid"

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

#: ``SOL_LOCAL`` / ``XUCRED_VERSION`` from the BSD socket ABI.  Python
#: does not export either, and both are fixed by that ABI.
_SOL_LOCAL = 0
_XUCRED_VERSION = 0

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
    "socket_dir_group_writable": "the bridge socket directory is group-writable and was refused",
    "socket_dir_owner_mismatch": "the bridge socket directory owner did not match the expected identity",
    "socket_replaced": "the bridge socket changed identity during connection and was refused",
    "peer_unverifiable": "the Stage-A controller identity could not be proven on this platform",
    "peer_owner_mismatch": "the connected Stage-A peer did not match the expected identity",
    "unstable_message_identity": "the request had no stable message identity and was refused",
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


# --- exchange deadlines ----------------------------------------------


def _local_step_deadline() -> float:
    """Bound for every step of an exchange except the peer's answer.

    The socket preflight stat, the connect, the post-connect identity
    re-stat, the write/drain and the close all talk to a socket on *this
    host*, so one bound is honest for all five.  Only the controller's
    thinking time deserves a long one, and that is
    :data:`REPLY_DEADLINE_SECONDS`.

    Read from :data:`CONNECT_TIMEOUT_SECONDS` at call time rather than
    copied into a constant at import, so shortening the declared connect
    deadline shortens every local step with it — and, through
    :func:`_exchange_budget`, the exchange as a whole.  A falsifier that
    tightens the declared deadlines therefore tightens what it is
    measuring, instead of leaving a second, longer bound behind.
    """
    return CONNECT_TIMEOUT_SECONDS


def _exchange_budget() -> float:
    """Total wall clock one exchange may consume.

    The sum of every declared step bound.  The exchange is bounded *as a
    whole* and not merely step by step, so a peer cannot chain several
    individually-legal delays into an unbounded hold on one of the four
    in-flight slots.
    """
    return 5 * _local_step_deadline() + REPLY_DEADLINE_SECONDS


class _Deadline:
    """One monotonic budget shared by every await in a single exchange.

    ``asyncio.wait_for`` bounds one await; this bounds their sum.  Each
    step still carries its own cap, so one slow step cannot quietly spend
    another's time, and no await in the lifecycle — teardown included —
    happens outside the budget.
    """

    __slots__ = ("_expiry",)

    def __init__(self, budget: float) -> None:
        self._expiry = time.monotonic() + budget

    def remaining(self) -> float:
        return self._expiry - time.monotonic()

    async def bounded(self, awaitable: Any, cap: float) -> Any:
        """Await under the smaller of this step's cap and what is left.

        Once the budget is spent this raises :class:`asyncio.TimeoutError`
        without awaiting at all, so an exhausted exchange cannot be
        extended by one more step.  The coroutine that will now never run
        is closed rather than abandoned, which keeps the non-blocking
        teardown path from leaving a warning behind it.
        """
        timeout = min(cap, self.remaining())
        if timeout <= 0:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise asyncio.TimeoutError
        return await asyncio.wait_for(awaitable, timeout=timeout)


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
    """Resolved, fixed, non-secret *secondary* bridge configuration.

    ``socket_uid`` is mandatory: an enabled bridge that cannot name the
    exact uid it expects to be talking to has no way to prove the peer,
    so it refuses rather than connecting to whatever answers.
    """

    owner_user_id: str
    socket_path: str
    socket_uid: int


def load_owner_user_id(getter: Callable[[str, Optional[str]], Optional[str]]) -> Optional[str]:
    """Resolve the *primary* gate: the one exact Stage-A Owner id.

    This is deliberately the only configuration consulted before a
    message has been classified.  It is a bare identifier with no
    parseable structure, so it is either present or absent — it can never
    be "invalid" and can therefore never turn a non-Owner message into a
    Stage-A outcome.  ``None`` means the bridge is off and ordinary
    routing is bit-identical to a tree without this module.

    ``getter`` is the adapter's profile-scoped configuration reader — the
    platform's own ``config.yaml`` ``extra`` block — so a secondary
    multiplexed profile cannot borrow the default profile's Stage-A
    binding.
    """
    return (getter(CONFIG_OWNER_USER_ID, "") or "").strip() or None


def load_config(
    getter: Callable[[str, Optional[str]], Optional[str]], *, owner_user_id: str
) -> BridgeConfig:
    """Resolve the *secondary* configuration for an admitted Owner request.

    Only ever called once a message is already known to be an exact Owner
    Stage-A candidate, so a broken value can only ever be reported to the
    Owner — it can never consume anyone else's traffic.

    Raises:
        BridgeError: ``config_invalid`` when a value cannot be used.
            Enabled-but-broken fails closed rather than degrading to a
            weaker check.
    """
    socket_path = (getter(CONFIG_SOCKET_PATH, "") or "").strip() or DEFAULT_SOCKET_PATH

    raw_uid = (getter(CONFIG_SOCKET_UID, "") or "").strip()
    if not raw_uid:
        raise BridgeError("config_invalid")
    try:
        socket_uid = int(raw_uid)
    except ValueError:
        raise BridgeError("config_invalid") from None
    if socket_uid < 0:
        raise BridgeError("config_invalid")

    return BridgeConfig(
        owner_user_id=owner_user_id, socket_path=socket_path, socket_uid=socket_uid
    )


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


def is_owner_candidate(
    owner_user_id: Optional[str],
    *,
    chat_type: str,
    sender_id: Optional[str],
    text: str,
) -> bool:
    """Whether this message is an exact Owner Stage-A candidate.

    These are the checks that decide *whose* message this is — bridge
    enabled, direct message, exact Owner, explicit marker — and nothing
    else.  Deliberately independent of anything that could *refuse* a
    request: media, size, message identity and configuration validity all
    describe a message that is already the Owner's, and answering those
    is :func:`classify`'s job.

    Kept separate because the intake path needs the same question one
    step earlier, before it decides whether an adapter-level cache may
    discard the message.  Sharing the predicate is what keeps that
    decision from drifting away from the one the bridge itself makes.
    """
    if owner_user_id is None:
        return False
    if chat_type != "dm":
        return False
    if (sender_id or "") != owner_user_id:
        return False
    return _strip_marker(text or "") is not None


def classify(
    owner_user_id: Optional[str],
    *,
    chat_type: str,
    sender_id: Optional[str],
    text: str,
    has_media: bool,
    message_id: Optional[str],
) -> Admission:
    """Decide what one already-authorized inbound message is.

    Ordering is the whole security property.  Every check that decides
    *whose* message this is — bridge enabled, direct message, exact Owner
    — is evaluated first and returns pass-through, so no state outside
    this function can make an ordinary message take a Stage-A path.  Only
    once the message is known to be an exact Owner Stage-A candidate may
    a problem be answered instead of routed.
    """
    if not is_owner_candidate(
        owner_user_id, chat_type=chat_type, sender_id=sender_id, text=text
    ):
        return _PASS_THROUGH

    # The gate above already proved the marker is present, so this is the
    # request text and never ``None``.
    request_text = _strip_marker(text or "") or ""

    # From here the Owner has explicitly asked for Stage-A, so a problem
    # is answered rather than quietly handed to ordinary routing.
    if has_media:
        return Admission(admitted=False, refused=True, reason="media_not_admitted")
    if not request_text:
        return Admission(admitted=False, refused=True, reason="empty_request")
    if len(request_text.encode("utf-8")) > MAX_REQUEST_TEXT_BYTES:
        return Admission(admitted=False, refused=True, reason="request_too_large")
    # Without a stable platform message id there is no work identity that
    # survives a restart, so no work may be created at all.
    if not (message_id or "").strip():
        return Admission(admitted=False, refused=True, reason="unstable_message_identity")

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


def derive_request_id(ref: str, message_id: str) -> str:
    """Stable Stage-A work identity for one authenticated inbound message.

    Derived, never random: the same platform message always yields the
    same ``request_id``, so a redelivery that arrives after this process
    (or the whole gateway) has restarted is recognisable as the *same*
    work by the controller's own request-id persistence, not merely by a
    cache that died with the process.  Distinct messages yield distinct
    ids because the platform's message id is distinct.

    The real message id is consumed by the digest and never leaves the
    process, so this adds no identifier to the wire.
    """
    digest = hashlib.sha256(
        f"{SCHEMA}|request_id|{ref}|{message_id}".encode("utf-8")
    ).hexdigest()
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


def is_exact_int(value: Any, expected: int) -> bool:
    """Whether ``value`` *is* the authoritative integer ``expected``.

    ``==`` is not a type test and neither is ``in``.  In Python
    ``True == 1`` and ``1.0 == 1``, and all three hash equal, so a JSON
    ``true`` or ``1.0`` satisfies both an equality check and membership
    in a set that names an integer.  A schema that says "integer ``1``"
    therefore has to say so in the type system, not only in the value
    comparison, or a malformed frame passes a contract it never met.

    ``type(value) is int`` is deliberate rather than ``isinstance``:
    ``bool`` is a subclass of ``int``, and so is any other subclass a
    caller might construct, and none of them is the authoritative
    representation.

    Used on both sides of the seam — the raw inbound item type on
    ingress, the protocol number on reply — so the two cannot drift.
    """
    return type(value) is int and value == expected


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
    if payload.get("schema") != SCHEMA or not is_exact_int(payload.get("protocol"), PROTOCOL):
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


def check_socket_path(path: str, expected_uid: int) -> os.stat_result:
    """Refuse a socket that a local attacker could have substituted.

    The socket itself must stay group-writable for a separate gateway
    identity to connect at all, so on the socket the test is exact
    ownership plus world-writability.  The *directory* has no such
    excuse — connecting needs only search permission on it, never write
    — so any writability beyond its owner is refused outright, and its
    owner must be the controller identity or ``root``.  That is what
    removes the ability to swap the socket out from under this client.

    Returns the socket's ``stat`` result so the caller can prove after
    connecting that it is still the same object.
    """
    try:
        st = os.stat(path)
    except OSError as exc:
        raise BridgeError("socket_unavailable") from exc
    if not stat.S_ISSOCK(st.st_mode):
        raise BridgeError("socket_not_a_socket")
    if st.st_mode & stat.S_IWOTH:
        raise BridgeError("socket_world_writable")
    if st.st_uid != expected_uid:
        raise BridgeError("socket_owner_mismatch")

    parent = os.path.dirname(os.path.abspath(path)) or os.sep
    try:
        dir_st = os.stat(parent)
    except OSError as exc:
        raise BridgeError("socket_unavailable") from exc
    if dir_st.st_mode & stat.S_IWOTH:
        raise BridgeError("socket_dir_world_writable")
    if dir_st.st_mode & stat.S_IWGRP:
        raise BridgeError("socket_dir_group_writable")
    if dir_st.st_uid not in (expected_uid, 0):
        raise BridgeError("socket_dir_owner_mismatch")
    return st


def check_socket_identity(path: str, preimage: os.stat_result) -> None:
    """Prove the socket is the very object that passed :func:`check_socket_path`.

    Closes the window between the preflight ``stat`` and the ``connect``:
    if the path now resolves to a different inode, the connection that
    was just opened cannot be trusted to be the one that was checked.
    """
    try:
        st = os.stat(path)
    except OSError as exc:
        raise BridgeError("socket_replaced") from exc
    if (st.st_dev, st.st_ino) != (preimage.st_dev, preimage.st_ino):
        raise BridgeError("socket_replaced")


# --- connected-peer confinement ----------------------------------------


def read_peer_uid(sock: Any) -> Optional[int]:
    """Effective uid of the process on the other end, or ``None``.

    ``None`` means *this platform cannot prove it* — which the caller
    treats as a refusal, never as permission.  Linux answers through
    ``SO_PEERCRED``; the BSD/macOS family answers through
    ``LOCAL_PEERCRED``.
    """
    if sock is None:
        return None

    so_peercred = getattr(socket_module, "SO_PEERCRED", None)
    if so_peercred is not None:
        try:
            raw = sock.getsockopt(socket_module.SOL_SOCKET, so_peercred, struct.calcsize("3i"))
            _pid, uid, _gid = struct.unpack("3i", raw)
        except (OSError, struct.error, ValueError):
            return None
        return int(uid)

    local_peercred = getattr(socket_module, "LOCAL_PEERCRED", None)
    if local_peercred is not None:
        # struct xucred: version, then the effective uid.
        size = struct.calcsize("2I")
        try:
            raw = sock.getsockopt(_SOL_LOCAL, local_peercred, size)
            version, uid = struct.unpack("2I", raw[:size])
        except (OSError, struct.error, ValueError):
            return None
        if version != _XUCRED_VERSION:
            return None
        return int(uid)

    return None


def verify_connected_peer(sock: Any, expected_uid: int) -> None:
    """Fail closed unless the connected peer *is* the expected identity.

    Path checks describe an object; this describes the process actually
    holding the other end of this connection, which is the only thing
    that can be trusted once the request is about to be written.  A
    platform that cannot answer is refused rather than assumed safe.
    """
    uid = read_peer_uid(sock)
    if uid is None:
        raise BridgeError("peer_unverifiable")
    if uid != expected_uid:
        raise BridgeError("peer_owner_mismatch")


# --- replay guard ------------------------------------------------------


class ReplayGuard:
    """Bounded in-process first-seen guard over derived request ids.

    Keys are recorded *before* the exchange starts, so a duplicate that
    arrives while the first request is still in flight cannot create a
    second one.

    This is an optimisation, not the idempotency boundary.  It is process
    memory and dies with the process; the guarantee that survives a
    restart is that :func:`derive_request_id` hands the controller the
    same id for the same inbound message.
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
        config_getter: Callable[[str, Optional[str]], Optional[str]],
        channel: str,
    ) -> None:
        self._config_getter = config_getter
        self._channel = channel
        self._owner_user_id: Optional[str] = None
        self._owner_loaded = False
        self._guard = ReplayGuard()
        self._inflight = 0

    # -- configuration --

    def _resolve_owner_user_id(self) -> Optional[str]:
        """The primary gate, resolved once.  Never raises."""
        if not self._owner_loaded:
            self._owner_loaded = True
            self._owner_user_id = load_owner_user_id(self._config_getter)
        return self._owner_user_id

    def _resolve_config(self, owner_user_id: str) -> BridgeConfig:
        """Secondary configuration, resolved only for an Owner candidate.

        Deliberately not cached: it is reached only after admission, so it
        runs at most once per admitted Owner request, and a corrected
        deployment takes effect without a restart.

        Raises:
            BridgeError: ``config_invalid``.
        """
        try:
            return load_config(self._config_getter, owner_user_id=owner_user_id)
        except BridgeError as exc:
            logger.error("[stagea] bridge configuration rejected: %s", exc.code)
            raise

    @property
    def enabled(self) -> bool:
        """Whether the primary gate is configured at all."""
        return self._resolve_owner_user_id() is not None

    # -- intake --

    def is_owner_candidate(
        self, *, chat_type: str, sender_id: Optional[str], text: str
    ) -> bool:
        """Whether this inbound message is an exact Owner Stage-A candidate.

        Answers the same question :meth:`process` answers first, against
        the same primary gate, but without touching the replay guard, the
        in-flight counter, the secondary configuration or the socket — so
        the intake path can ask it before it has decided what to do with
        the message, as many times as it likes, with no side effect.

        A caller uses this to keep its *own* caches from destroying a
        distinct Owner request; it grants nothing.  Whether a candidate is
        admitted, refused or passed through is still decided entirely by
        :meth:`process`.
        """
        return is_owner_candidate(
            self._resolve_owner_user_id(),
            chat_type=chat_type,
            sender_id=sender_id,
            text=text,
        )

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
        # Classification first, and against the primary gate only: until
        # a message is known to be an exact Owner Stage-A candidate, no
        # secondary configuration — valid, invalid or absent — may change
        # what happens to it.
        owner_user_id = self._resolve_owner_user_id()
        decision = classify(
            owner_user_id,
            chat_type=chat_type,
            sender_id=sender_id,
            text=text,
            has_media=has_media,
            message_id=message_id,
        )
        # ``owner_user_id is None`` already implies nothing was consumed;
        # naming it here keeps the narrowing explicit instead of relying
        # on an assertion that ``python -O`` would strip.
        if not decision.consumed or owner_user_id is None:
            return None
        if decision.refused:
            logger.info("[stagea] request refused: %s", decision.reason)
            return refusal_text(decision.reason)

        try:
            config = self._resolve_config(owner_user_id)
        except BridgeError as exc:
            return refusal_text(exc.code)

        ref = conversation_ref(conversation_key)
        request_id = derive_request_id(ref, (message_id or "").strip())
        if not self._guard.check_and_record(request_id):
            logger.info("[stagea] duplicate request suppressed")
            return refusal_text("duplicate_request")

        if self._inflight >= MAX_INFLIGHT_REQUESTS:
            logger.warning("[stagea] refusing request: %d already in flight", self._inflight)
            return refusal_text("too_many_inflight")

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
        """One connection: send the request, read one reply, close.

        Every await below draws from one :class:`_Deadline`, teardown
        included, so the exchange holds its in-flight slot for at most
        :func:`_exchange_budget` seconds however the peer or the socket
        misbehaves.  An await that is merely slow becomes a refusal the
        Owner is told about; an await that never returns would hold the
        slot and suppress that refusal, so none is left unbounded.
        """
        open_unix_connection = getattr(asyncio, "open_unix_connection", None)
        if open_unix_connection is None:
            raise BridgeError("platform_unsupported")

        deadline = _Deadline(_exchange_budget())
        step = _local_step_deadline()

        # A stat is a local call, but local is not the same as bounded: a
        # wedged filesystem holds this await for as long as it likes.  The
        # bound protects the slot and the Owner's answer — the worker
        # thread is left to finish on its own, because a thread cannot be
        # cancelled and pretending otherwise would be the wrong claim.
        try:
            preimage = await deadline.bounded(
                asyncio.to_thread(check_socket_path, config.socket_path, config.socket_uid),
                cap=step,
            )
        except asyncio.TimeoutError as exc:
            raise BridgeError("socket_unavailable") from exc

        frame = encode_frame(
            build_request(
                request_id=request_id,
                ref=ref,
                channel=self._channel,
                text=request_text,
            )
        )

        try:
            reader, writer = await deadline.bounded(
                open_unix_connection(config.socket_path), cap=step
            )
        except (asyncio.TimeoutError, OSError) as exc:
            raise BridgeError("connect_failed") from exc

        try:
            # Nothing is written until the process on the other end has
            # been proven to be the expected identity, and the path has
            # been proven not to have been swapped since the preflight.
            verify_connected_peer(writer.get_extra_info("socket"), config.socket_uid)
            try:
                await deadline.bounded(
                    asyncio.to_thread(check_socket_identity, config.socket_path, preimage),
                    cap=step,
                )
            except asyncio.TimeoutError as exc:
                raise BridgeError("socket_unavailable") from exc

            try:
                writer.write(frame)
                await deadline.bounded(writer.drain(), cap=step)
            except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
                raise BridgeError("send_failed") from exc

            try:
                payload = await deadline.bounded(
                    read_frame(reader), cap=REPLY_DEADLINE_SECONDS
                )
            except asyncio.TimeoutError as exc:
                raise BridgeError("reply_timeout") from exc

            return validate_reply(payload, request_id=request_id, ref=ref)
        finally:
            await _close_writer(writer, deadline)


async def _close_writer(writer: Any, deadline: _Deadline) -> None:
    """Close a stream writer without letting teardown mask the outcome.

    ``close()`` is synchronous and cannot block; ``wait_closed()`` is the
    only await here, so it is the one that needs a bound.  Once the
    budget is spent the wait is skipped outright rather than shortened:
    teardown is allowed to be non-blocking, never unbounded, and a peer
    that will not finish closing must not be able to withhold an answer
    this exchange has already earned.
    """
    try:
        writer.close()
        await deadline.bounded(writer.wait_closed(), cap=_local_step_deadline())
    except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
        logger.debug("[stagea] writer close failed: %s", exc)
