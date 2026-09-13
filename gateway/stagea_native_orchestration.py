"""Opt-in finite native exchange on the existing authenticated Owner bridge.

Only the Weixin adapter constructs an intake; TurnRunner supplies its ordinary
agent after normal cache selection. No canonical facts come from chat history.
The separate controller counterpart is required before runtime activation.
"""

from dataclasses import asdict
import asyncio
import concurrent.futures
import json
import threading

from agent.native_orchestration import NativeProposalRequest, produce_native_proposal
from gateway import stagea_owner_bridge as B

SCHEMA = "dyhano.stagea.native_bridge.v1"
PROTOCOL = 1
CONFIG_MODE = "stagea_native_mode"
MODE = "finite_v1"  # absent by default; no environment or installed-config write
COMMON = {"schema", "protocol", "type", "request_id", "conversation_ref"}
UNRESOLVED = "NATIVE_ATTEMPT_UNRESOLVED"


def reply_result(text):
    return {
        "final_response": text,
        "messages": [],
        "api_calls": 0,
        "tools": [],
        "completed": True,
        "_stagea_native_reply": True,
    }


def classify_native(
    owner_user_id, *, chat_type, sender_id, text, has_media, message_id
):
    if (
        owner_user_id is None
        or chat_type != "dm"
        or sender_id != owner_user_id
        or not isinstance(text, str)
        or text.lstrip().startswith("/")
    ):
        return B.Admission(False, False, "pass_through")
    if has_media:
        return B.Admission(False, True, "media_not_admitted")
    if not text.strip():
        return B.Admission(False, True, "empty_request")
    if len(text.encode("utf-8")) > 4096:
        return B.Admission(False, True, "request_too_large")
    if text != text.strip() or any(ord(c) < 32 and c not in "\n\t" for c in text):
        return B.Admission(False, True, "reply_bad_text")
    if not isinstance(message_id, str) or not message_id.strip():
        return B.Admission(False, True, "unstable_message_identity")
    return B.Admission(True, False, "admitted", text)


def native_request(legacy_shaped):
    # Both schema and type differ BEFORE the first write. The strict legacy
    # read_request rejects this before its handle_owner_request call.
    return {
        **legacy_shaped,
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "type": "native_owner_request",
    }


def encode_native_frame(payload):
    frame = B.encode_frame(payload)
    if len(frame) > B.MAX_FRAME_BYTES:  # include length header + entire envelope
        raise B.BridgeError("send_failed")
    return frame


def _envelope(payload, request, kind, fields):
    if type(payload) is not dict or set(payload) != COMMON | fields:
        raise B.BridgeError("reply_unexpected_field")
    if (
        payload["schema"] != SCHEMA
        or not B.is_exact_int(payload["protocol"], PROTOCOL)
        or payload["type"] != kind
    ):
        raise B.BridgeError("reply_malformed")
    if payload["request_id"] != request["request_id"]:
        raise B.BridgeError("reply_wrong_request")
    if payload["conversation_ref"] != request["conversation_ref"]:
        raise B.BridgeError("reply_wrong_conversation")


class NativeIngress:
    """Adapter-local admission and one-conversation unresolved-attempt fence.

    No timer/retry/reset: a native SDK timeout/interruption has no provable
    worker completion handle. That conversation stays UNKNOWN, even after
    cache eviction or config changes. Unrelated conversations keep running.
    """

    def __init__(self, bridge):
        self.bridge = bridge
        self._lock = threading.Lock()
        self._active = {}
        self._unresolved = set()

    def candidate(self, *, chat_type, sender_id, text):
        return (
            self.bridge._config_getter(CONFIG_MODE, "") not in ("", "off")
            and self.bridge._resolve_owner_user_id() is not None
            and chat_type == "dm"
            and sender_id == self.bridge._resolve_owner_user_id()
            and isinstance(text, str)
            and not text.lstrip().startswith("/")
        )

    def blocked(self, conversation_key, session_id=None):
        with self._lock:
            keys = (
                {conversation_key, "session:" + str(session_id)}
                if session_id
                else {conversation_key}
            )
            return bool(
                keys.intersection(self._active) or keys.intersection(self._unresolved)
            )

    def prepare(self, *, source, text, has_media, message_id, conversation_key):
        if not self.candidate(
            chat_type=source.chat_type, sender_id=source.user_id, text=text
        ):
            return None
        if self.bridge._config_getter(CONFIG_MODE, "") != MODE:
            return B.outcome_text("UNKNOWN", "UNSUPPORTED_NATIVE_CONFIGURATION")
        decision = classify_native(
            self.bridge._resolve_owner_user_id(),
            chat_type=source.chat_type,
            sender_id=source.user_id,
            text=text,
            has_media=has_media,
            message_id=message_id,
        )
        if decision.refused:
            return B.refusal_text(decision.reason)
        return NativeTurn(self, source, text, message_id, conversation_key)


class NativeTurn:
    def __init__(self, ingress, source, text, message_id, conversation_key):
        self.ingress, self.source = ingress, source
        self.text, self.message_id, self.key = text, message_id, conversation_key
        self._used = False
        self._cancelled = False
        self._model_started = False
        self._future = None
        self._keys = {self.key}

    def busy_reply(self):
        # Refuse this exact platform message before any ordinary busy-input
        # coalescer can discard its identity or steer the active agent.
        with self.ingress._lock:
            self._used = True
            reason = (
                UNRESOLVED if self.key in self.ingress._unresolved else "NATIVE_BUSY"
            )
        request_id = B.derive_request_id(
            B.conversation_ref(self.key), self.message_id.strip()
        )
        return B.outcome_text("UNKNOWN", f"{reason} request_id={request_id}")

    def cancel(self):
        with self.ingress._lock:
            self._cancelled = True
            if self._model_started:
                self.ingress._unresolved.update(self._keys)
            future = self._future
        if future is not None:
            future.cancel()

    def run_sync(self, agent, loop, session_id):
        with self.ingress._lock:
            self._keys.add("session:" + str(session_id))
            if (
                self._used
                or self._cancelled
                or self._keys.intersection(self.ingress._active)
                or self._keys.intersection(self.ingress._unresolved)
            ):
                return reply_result(B.outcome_text("UNKNOWN", UNRESOLVED))
            self._used = True
            self.ingress._active.update(dict.fromkeys(self._keys, self))
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.ingress.bridge.process(
                    chat_type=self.source.chat_type,
                    sender_id=self.source.user_id,
                    text=self.text,
                    has_media=False,
                    conversation_key=self.key,
                    message_id=self.message_id,
                    _native_turn=self,
                    _native_agent=agent,
                ),
                loop,
            )
            with self.ingress._lock:
                self._future = future
                if self._cancelled:
                    future.cancel()
            text = future.result(
                timeout=B._exchange_budget() + B._local_step_deadline()
            )
            return reply_result(
                text or B.outcome_text("UNKNOWN", "NATIVE_INTAKE_REFUSED")
            )
        except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
            self.cancel()
            return reply_result(B.outcome_text("UNKNOWN", UNRESOLVED))
        finally:
            with self.ingress._lock:
                for key in self._keys:
                    self.ingress._active.pop(key, None)

    def _produce(self, agent, request):
        with self.ingress._lock:
            if self._cancelled:
                from agent.native_orchestration import NativeProposalResult

                return NativeProposalResult("FAILED", reason_code="NATIVE_INTERRUPTED")
            self._model_started = True
        result = produce_native_proposal(agent, request)
        # These codes may return while an internal SDK worker is still alive.
        # No new message, wrapper return or cache replacement proves its exit.
        if result.reason_code in {
            "NATIVE_TIMEOUT",
            "NATIVE_INTERRUPTED",
            "NATIVE_FAILURE",
        }:
            with self.ingress._lock:
                self.ingress._unresolved.update(self._keys)
        return result

    async def exchange_frames(self, reader, writer, deadline, request, agent):
        try:
            payload = await deadline.bounded(
                B.read_frame(reader, strict=True), cap=B.REPLY_DEADLINE_SECONDS
            )
            # A controller may refuse before it has a valid canonical challenge.
            if payload.get("type") == "native_owner_reply":
                return self._reply(payload, request)
            _envelope(payload, request, "native_proposal_challenge", {"request"})
            try:
                canonical = NativeProposalRequest.from_json(
                    json.dumps(
                        payload["request"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
            except (ValueError, TypeError, AttributeError, RecursionError):
                raise B.BridgeError("reply_malformed") from None
            if canonical.intent != self.text:
                raise B.BridgeError("reply_wrong_request")
            result = await deadline.bounded_thread(
                self._produce, agent, canonical, cap=B.REPLY_DEADLINE_SECONDS
            )
            native = {key: request[key] for key in COMMON - {"type"}}
            native.update(type="native_proposal_result", result=asdict(result))
            writer.write(encode_native_frame(native))
            await deadline.bounded(writer.drain(), cap=B._local_step_deadline())
            payload = await deadline.bounded(
                B.read_frame(reader, strict=True), cap=B.REPLY_DEADLINE_SECONDS
            )
            return self._reply(payload, request)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            self.cancel()
            raise

    @staticmethod
    def _reply(payload, request):
        _envelope(payload, request, "native_owner_reply", {"outcome", "text"})
        # Reuse the existing bounded outcome/text validation without relaxing
        # the legacy parser or accepting legacy replies on a native exchange.
        return B.validate_reply(
            {
                **payload,
                "schema": B.SCHEMA,
                "protocol": B.PROTOCOL,
                "type": B.REPLY_TYPE,
            },
            request_id=request["request_id"],
            ref=request["conversation_ref"],
        )


def turn_for(source):
    turn = getattr(source, "_stagea_native_turn", None)
    return turn if type(turn) is NativeTurn and turn.source is source else None


def execution_blocked(runner, source, session_id):
    """Keep a fence across adapter/cache replacement and shared-session aliases.

    These are references to existing intakes, not work, watchers or a scheduler.
    Only matching conversation/session keys block; unrelated sessions proceed.
    """
    adapter = runner._adapter_for_source(source)
    ingress = getattr(adapter, "_stagea_native", None)
    fences = getattr(runner, "_stagea_native_fences", None)
    if type(fences) is not dict:
        fences = runner._stagea_native_fences = {}
    key = None
    if type(ingress) is NativeIngress:
        fences[id(ingress)] = ingress
        key = adapter._stagea_conversation_key(source)
    return any(fence.blocked(key, session_id) for fence in tuple(fences.values()))
