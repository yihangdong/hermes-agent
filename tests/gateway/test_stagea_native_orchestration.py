"""Finite native wire vectors and real ingress/TurnRunner/SDK/socket behavior.

Synthetic data only. The later controller counterpart can consume WIRE_EXAMPLES
unchanged: each value is the complete JSON envelope, preceded by a uint32 BE
UTF-8 byte count on the socket. No fixture contains operational evidence.
"""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest

from agent.native_orchestration import NativeProposalRequest
from gateway import stagea_owner_bridge as B
from gateway import stagea_native_orchestration as N
from gateway.config import Platform, PlatformConfig
from gateway.platforms import weixin
from gateway.platforms.weixin import WeixinAdapter
from gateway.session import SessionSource
from gateway.turn_context import TurnContext

INTENT = "Continue the admitted synthetic work."
CANONICAL: dict[str, Any] = {
    "schema": "hermes.native_proposal.request.v1",
    "work_item_id": "wi-synthetic",
    "intent": INTENT,
    "snapshot_id": "snapshot-synthetic",
    "facts": [
        {
            "name": name,
            "status": "KNOWN",
            "summary": "Synthetic offline fact.",
            "source_refs": ["https://example.invalid/evidence/1"],
        }
        for name in (
            "accepted_evidence",
            "capacity",
            "contract_authority",
            "dependencies",
            "pending_obligations",
            "recovery_debt",
            "source_identities",
        )
    ],
    "snapshot_sha256": "8834265f32c5660ef2150e62bfca66003f02d4afbc692a494760d4e09d53d36f",
}
# These committed envelopes are source-bound shared compatibility vectors.
WIRE_EXAMPLES: dict[str, Any] = {
    "request": {
        "schema": "dyhano.stagea.native_bridge.v1",
        "protocol": 1,
        "type": "native_owner_request",
        "client": "hermes.gateway.stagea_owner_bridge",
        "request_id": "1" * 32,
        "conversation_ref": "2" * 32,
        "channel": "weixin",
        "chat_type": "dm",
        "text": INTENT,
        "sent_at": "2026-01-01T00:00:00Z",
    },
    "challenge": {
        "schema": "dyhano.stagea.native_bridge.v1",
        "protocol": 1,
        "type": "native_proposal_challenge",
        "request_id": "1" * 32,
        "conversation_ref": "2" * 32,
        "request": CANONICAL,
    },
    "result": {
        "schema": "dyhano.stagea.native_bridge.v1",
        "protocol": 1,
        "type": "native_proposal_result",
        "request_id": "1" * 32,
        "conversation_ref": "2" * 32,
        "result": {
            "status": "PROPOSAL",
            "reason_code": None,
            "proposal": {
                "schema": "hermes.native_proposal.v1",
                "authority": "NON_AUTHORITATIVE",
                "work_item_id": "wi-synthetic",
                "snapshot_id": "snapshot-synthetic",
                "snapshot_sha256": CANONICAL["snapshot_sha256"],
                "evidence_status": "KNOWN",
                "outcome": "YIELD",
                "rationale": "No other admitted work is evidenced.",
                "proposed_steps": [],
                "capability_needs": [],
                "recovery": [],
                "next_work_item_id": None,
                "unknowns": [],
            },
        },
    },
    "reply": {
        "schema": "dyhano.stagea.native_bridge.v1",
        "protocol": 1,
        "type": "native_owner_reply",
        "request_id": "1" * 32,
        "conversation_ref": "2" * 32,
        "outcome": "UNKNOWN",
        "text": "Synthetic native handoff only; no effect.",
    },
}


def correlated(kind, request):
    value = deepcopy(WIRE_EXAMPLES[kind])
    value.update(
        request_id=request["request_id"], conversation_ref=request["conversation_ref"]
    )
    return value


def inbound(text=INTENT, sender="owner", message_id="message-1", **updates):
    return {
        "from_user_id": sender,
        "message_id": message_id,
        "context_token": "synthetic-context",
        "item_list": [{"type": weixin.ITEM_TEXT, "text_item": {"text": text}}],
        **updates,
    }


def make_adapter(path="/nonexistent", **updates):
    adapter = cast(
        Any,
        WeixinAdapter(
            PlatformConfig(
                enabled=True,
                token="synthetic-token",
                extra={
                    "account_id": "synthetic-account",
                    "dm_policy": "allowlist",
                    "allow_from": ["owner", "other"],
                    B.CONFIG_OWNER_USER_ID: "owner",
                    B.CONFIG_SOCKET_PATH: str(path),
                    B.CONFIG_SOCKET_UID: os.getuid(),
                    N.CONFIG_MODE: N.MODE,
                    **updates,
                },
            )
        ),
    )
    adapter._poll_session = object()
    adapter._send_session = object()
    adapter._send_text_chunk = AsyncMock()
    adapter._maybe_fetch_typing_ticket = AsyncMock()
    adapter._enqueue_text_event = Mock()
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.fixture
def ordinary(monkeypatch, tmp_path):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    original_connect = socket.socket.connect

    def only_local_socket(sock, address):
        if sock.family != socket.AF_UNIX:
            raise AssertionError("No network or live provider access in these tests")
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", only_local_socket)
    agent = cast(
        Any,
        AIAgent(
            api_key="synthetic-key",
            base_url="https://offline.invalid/v1",
            provider="custom",
            model="synthetic-model",
            api_mode="chat_completions",
            enabled_toolsets=[],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        ),
    )
    agent.tools = [
        {
            "type": "function",
            "function": {"name": "forbidden_tool", "parameters": {"type": "object"}},
        }
    ]
    agent._cached_system_prompt = "ordinary stable prefix"
    agent.conversation_history = [{"role": "user", "content": "old private history"}]
    wire = SimpleNamespace(
        calls=[],
        output=deepcopy(WIRE_EXAMPLES["result"]["result"]["proposal"]),
        on_request=None,
        finish="stop",
        tool_calls=None,
    )

    def respond(request):
        wire.calls.append(json.loads(request.content))
        if wire.on_request:
            wire.on_request()
        return httpx.Response(
            200,
            json={
                "id": "synthetic-completion",
                "object": "chat.completion",
                "created": 0,
                "model": agent.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": wire.finish,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(wire.output),
                            "tool_calls": wire.tool_calls,
                        },
                    }
                ],
            },
        )

    create_client = agent._create_openai_client

    def offline_client(kwargs, **options):
        return create_client(
            {
                **kwargs,
                "http_client": httpx.Client(transport=httpx.MockTransport(respond)),
            },
            **options,
        )

    monkeypatch.setattr(agent, "_create_openai_client", offline_client)

    def forbidden(*args, **kwargs):
        raise AssertionError("Secondary cognition, tool, or fallback is forbidden")

    monkeypatch.setattr(agent, "run_conversation", forbidden)
    monkeypatch.setattr(agent, "_try_activate_fallback", forbidden)
    monkeypatch.setattr("run_agent.handle_function_call", forbidden)
    yield agent, wire
    cache = getattr(agent, "_request_client_cache", None)
    if cache and cache["client"]:
        cache["client"].close()
    agent.client.close()


def runner_for(adapter, source, agent, loop, *, session_id="ordinary-session"):
    from gateway.run import TurnRunner

    runner = MagicMock()
    runner.config = SimpleNamespace(streaming=None)
    runner._provider_routing = {}
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {"ordinary-key": (agent, ("signature",), None, session_id)}
    runner._session_db = None
    runner._adapter_for_source.return_value = adapter
    runner._get_system_prompt_for_channel.return_value = None
    runner._resolve_session_agent_runtime.return_value = (
        agent.model,
        {"provider": agent.provider},
    )
    runner._resolve_session_reasoning_config.return_value = None
    runner._resolve_session_service_tier.return_value = None
    runner._resolve_turn_agent_config.return_value = {
        "model": agent.model,
        "runtime": {},
    }
    runner._agent_config_signature.return_value = ("signature",)
    runner._extract_cache_busting_config.return_value = {}
    runner._refresh_fallback_model.return_value = None
    constructor = Mock(
        side_effect=AssertionError("Must reuse the ordinary cached AIAgent")
    )
    ctx = TurnContext(
        source=source,
        message=INTENT,
        history=[],
        session_id=session_id,
        session_key="ordinary-key",
        user_config={},
        AIAgent=constructor,
        resolve_display_setting=lambda *_: False,
        _run_still_current=lambda: True,
        _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )
    ctx._loop_for_step = loop
    return TurnRunner(runner, ctx), constructor


@asynccontextmanager
async def peer(handler):
    # A short, owner-only real directory fits the native AF_UNIX path bound.
    with tempfile.TemporaryDirectory(prefix="m3nf-") as directory:
        path = Path(directory) / "s"
        finished = asyncio.Event()
        errors = []

        async def serve(reader, writer):
            try:
                await handler(reader, writer)
            except (ConnectionError, asyncio.IncompleteReadError, B.BridgeError):
                pass
            except BaseException as exc:
                errors.append(exc)
            finally:
                writer.close()
                await writer.wait_closed()
                finished.set()

        server = await asyncio.start_unix_server(serve, str(path))
        os.chmod(path, 0o600)
        try:
            yield path, finished
        finally:
            server.close()
            await server.wait_closed()
            if errors:
                raise errors[0]


async def execute(adapter, agent, *, message=None, session_id="ordinary-session"):
    results = []

    async def handle(event):
        runner, constructor = runner_for(
            adapter,
            event.source,
            agent,
            asyncio.get_running_loop(),
            session_id=session_id,
        )
        result = await asyncio.to_thread(runner.run_sync)
        constructor.assert_not_called()
        results.append((result, event, runner))
        # The production native result branch uses this same text-only sender.
        await adapter._send_stagea_reply(event.source.chat_id, result["final_response"])

    adapter.handle_message = handle
    await adapter._process_message(message or inbound())
    return results


@pytest.mark.asyncio
async def test_wire_vectors_and_real_ordinary_turn(ordinary):
    agent, wire = ordinary
    request = NativeProposalRequest.from_json(json.dumps(CANONICAL))
    assert request.intent == INTENT
    before = deepcopy({
        k: getattr(agent, k)
        for k in (
            "tools",
            "_cached_system_prompt",
            "conversation_history",
            "request_overrides",
            "provider",
            "model",
            "base_url",
            "reasoning_config",
        )
    })
    frames = []

    async def controller(reader, writer):
        incoming = await B.read_frame(reader, strict=True)
        frames.append(incoming)
        assert incoming == {
            **WIRE_EXAMPLES["request"],
            "request_id": incoming["request_id"],
            "conversation_ref": incoming["conversation_ref"],
            "sent_at": incoming["sent_at"],
        }
        writer.write(N.encode_native_frame(correlated("challenge", incoming)))
        await writer.drain()
        result = await B.read_frame(reader, strict=True)
        frames.append(result)
        assert result == correlated("result", incoming)
        writer.write(N.encode_native_frame(correlated("reply", incoming)))
        await writer.drain()

    async with peer(controller) as (path, done):
        adapter = make_adapter(path)
        results = await execute(adapter, agent)
        await asyncio.wait_for(done.wait(), 2)
    assert len(frames) == 2 and len(wire.calls) == 1
    assert wire.calls[0].get("tools") in (None, [])
    assert json.loads(wire.calls[0]["messages"][1]["content"]) == CANONICAL
    assert "old private history" not in json.dumps(wire.calls)
    assert all(getattr(agent, key) == value for key, value in before.items())
    assert results[0][2]._ctx.agent_holder[0] is agent
    results[0][2]._runner._init_cached_agent_for_turn.assert_not_called()
    results[0][2]._runner._apply_fallback_chain_to_agent.assert_not_called()
    assert adapter._send_text_chunk.await_count == 1
    assert adapter._send_text_chunk.call_args.kwargs["chat_id"] == "owner"
    assert adapter._token_store.get("synthetic-account", "owner") == "synthetic-context"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["default_off", "other", "group", "slash", "unauthorized"]
)
async def test_nonparticipants_never_enter_native(change):
    adapter = make_adapter()
    message = inbound()
    if change == "default_off":
        adapter.config.extra.pop(N.CONFIG_MODE)
    elif change == "other":
        message["from_user_id"] = "other"
    elif change == "group":
        message["room_id"] = "synthetic-room"
    elif change == "slash":
        message = inbound("/status")
    elif change == "unauthorized":
        message["from_user_id"] = "unapproved"
    await adapter._process_message(message)
    assert not any(
        N.turn_for(call.args[0].source)
        for call in adapter.handle_message.call_args_list
    )
    assert not adapter._stagea_native._active


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change,code",
    [
        ("media", "media_not_admitted"),
        ("id", "unstable_message_identity"),
        ("oversize", "request_too_large"),
        ("whitespace", "reply_bad_text"),
    ],
)
async def test_bad_intake_refuses_before_cognition_or_download(change, code):
    adapter = make_adapter()
    adapter._collect_media = AsyncMock(
        side_effect=AssertionError("No native media download")
    )
    message = inbound()
    if change == "media":
        message["item_list"].append({"type": 2, "image_item": {}})
    elif change == "id":
        message["message_id"] = ""
    elif change == "oversize":
        message = inbound("界" * 1366)
    else:
        message = inbound(" padded ")
    await adapter._process_message(message)
    adapter.handle_message.assert_not_called()
    adapter._collect_media.assert_not_called()
    assert code in adapter._send_text_chunk.call_args.kwargs["chunk"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "wrong_request",
        "wrong_conversation",
        "field",
        "protocol",
        "canonical_field",
        "canonical_intent",
        "oversized",
        "duplicate_json",
    ],
)
async def test_challenge_refusals_call_no_model(ordinary, change):
    agent, wire = ordinary

    async def controller(reader, writer):
        request = await B.read_frame(reader)
        challenge = correlated("challenge", request)
        if change == "wrong_request":
            challenge["request_id"] = "0" * 32
        elif change == "wrong_conversation":
            challenge["conversation_ref"] = "0" * 32
        elif change == "field":
            challenge["destination"] = "https://example.invalid"
        elif change == "protocol":
            challenge["protocol"] = True
        elif change == "canonical_field":
            challenge["request"]["provider"] = "forbidden"
        elif change == "canonical_intent":
            challenge["request"]["intent"] = "changed"
        if change == "oversized":
            frame = struct.pack(">I", B.MAX_FRAME_BYTES)
        elif change == "duplicate_json":
            raw = (
                json
                .dumps(challenge)
                .replace('"protocol": 1', '"protocol": 1, "protocol": 1')
                .encode()
            )
            frame = struct.pack(">I", len(raw)) + raw
        else:
            frame = N.encode_native_frame(challenge)
        writer.write(frame)
        await writer.drain()

    async with peer(controller) as (path, _):
        result = await execute(make_adapter(path), agent)
    assert not wire.calls
    assert "UNKNOWN" in result[0][0]["final_response"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change,reason,calls",
    [
        ("mode", "UNSUPPORTED_NATIVE_MODE", 0),
        ("override", "UNSUPPORTED_NATIVE_CONFIGURATION", 0),
        ("busy", "NATIVE_BUSY", 0),
        ("interrupted", "NATIVE_INTERRUPTED", 0),
        ("tools", "TOOL_OUTPUT_REJECTED", 1),
        ("output", "INVALID_JSON", 1),
        ("identity", "OUTPUT_IDENTITY_MISMATCH", 1),
        ("oversized_output", "INVALID_TEXT", 1),
    ],
)
async def test_native_failures_preserved_without_fallback(
    ordinary, change, reason, calls
):
    agent, wire = ordinary
    if change == "mode":
        agent.api_mode = "codex_responses"
    elif change == "override":
        agent.request_overrides = {"extra_body": {"anything": True}}
    elif change == "busy":
        agent._model_request_active.set()
    elif change == "interrupted":
        agent._interrupt_requested = True
    elif change == "tools":
        wire.tool_calls = [
            {
                "id": "x",
                "type": "function",
                "function": {"name": "forbidden_tool", "arguments": "{}"},
            }
        ]
    elif change == "output":
        wire.output = "not a proposal object"
    elif change == "identity":
        wire.output["snapshot_id"] = "wrong"
    elif change == "oversized_output":
        wire.output["rationale"] = "x" * 17000
    received = []

    async def controller(reader, writer):
        request = await B.read_frame(reader)
        writer.write(N.encode_native_frame(correlated("challenge", request)))
        await writer.drain()
        result = await B.read_frame(reader)
        received.append(result["result"])
        writer.write(N.encode_native_frame(correlated("reply", request)))
        await writer.drain()

    async with peer(controller) as (path, done):
        await execute(make_adapter(path), agent)
        await asyncio.wait_for(done.wait(), 2)
    assert received[0]["reason_code"] == reason
    assert received[0]["proposal"] is None
    assert len(wire.calls) == calls


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["duplicate", "legacy", "destination", "mismatch"])
async def test_invalid_final_never_repeats_cognition(ordinary, kind):
    agent, wire = ordinary

    async def controller(reader, writer):
        request = await B.read_frame(reader)
        writer.write(N.encode_native_frame(correlated("challenge", request)))
        await writer.drain()
        await B.read_frame(reader)
        final = correlated("challenge" if kind == "duplicate" else "reply", request)
        if kind == "legacy":
            final.update(schema=B.SCHEMA, type=B.REPLY_TYPE)
        elif kind == "destination":
            final["destination"] = "/tmp/forbidden"
        elif kind == "mismatch":
            final["request_id"] = "0" * 32
        writer.write(N.encode_native_frame(final))
        await writer.drain()

    async with peer(controller) as (path, _):
        result = await execute(make_adapter(path), agent)
    assert len(wire.calls) == 1
    assert "UNKNOWN" in result[0][0]["final_response"]


@pytest.mark.asyncio
async def test_wire_bound_includes_outer_envelope_and_header():
    for document in WIRE_EXAMPLES.values():
        frame = N.encode_native_frame(document)
        assert len(frame) == 4 + struct.unpack(">I", frame[:4])[0]
        reader = asyncio.StreamReader()
        reader.feed_data(frame)
        assert await B.read_frame(reader, strict=True) == document
    with pytest.raises(B.BridgeError):
        N.encode_native_frame({"text": "x" * (B.MAX_FRAME_BYTES - 14)})


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_cancellation_quarantine_blocks_replacement_only_here(
    ordinary, monkeypatch, cancel
):
    agent, wire = ordinary
    entered, release = threading.Event(), threading.Event()

    def stall():
        entered.set()
        assert release.wait(5)

    wire.on_request = stall
    monkeypatch.setattr(B, "REPLY_DEADLINE_SECONDS", 0.2 if not cancel else 2)

    async def controller(reader, writer):
        request = await B.read_frame(reader)
        writer.write(N.encode_native_frame(correlated("challenge", request)))
        await writer.drain()
        await reader.read()

    async with peer(controller) as (path, _):
        adapter = make_adapter(path)
        task = asyncio.create_task(execute(adapter, agent))
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        if cancel:
            # The production _run_agent_inner cancellation calls the same turn fence.
            active = next(iter(adapter._stagea_native._active.values()))
            active.cancel()
        await asyncio.wait_for(task, 2)
        source: Any = SessionSource(
            platform=Platform.WEIXIN, chat_id="owner", user_id="owner"
        )
        key = adapter._stagea_conversation_key(source)
        assert adapter._stagea_native.blocked(key)
        # Alias routing to the same persisted session cannot evade the fence.
        assert adapter._stagea_native.blocked("different-route", "ordinary-session")
        assert not adapter._stagea_native.blocked("different-route", "other-session")
        results = await execute(adapter, agent, message=inbound(message_id="message-2"))
        assert N.UNRESOLVED in results[0][0]["final_response"]
        assert len(wire.calls) == 1
        release.set()
        await asyncio.sleep(0.35)
        # A wrapper return never supplies missing SDK completion proof.
        assert adapter._stagea_native.blocked(key)


@pytest.mark.asyncio
async def test_gateway_inner_uses_same_turnrunner_and_no_proxy_or_second_agent(
    ordinary, monkeypatch
):
    from gateway.run import GatewayRunner

    agent, wire = ordinary

    async def controller(reader, writer):
        request = await B.read_frame(reader)
        writer.write(N.encode_native_frame(correlated("challenge", request)))
        await writer.drain()
        await B.read_frame(reader)
        reply = correlated("reply", request)
        reply["text"] = "MEDIA:/tmp/synthetic.png remains plain controller text."
        writer.write(N.encode_native_frame(reply))
        await writer.drain()

    async with peer(controller) as (path, done):
        adapter = make_adapter(path)
        source: Any = SessionSource(
            platform=Platform.WEIXIN, chat_id="owner", user_id="owner"
        )
        cast(Any, source)._stagea_native_turn = adapter._stagea_native.prepare(
            source=source,
            text=INTENT,
            has_media=False,
            message_id="message-1",
            conversation_key=adapter._stagea_conversation_key(source),
        )
        turn_runner, _ = runner_for(adapter, source, agent, asyncio.get_running_loop())
        runner = turn_runner._runner
        runner._get_proxy_url.return_value = None
        runner._resolve_enabled_toolsets_for_source.return_value = []
        runner._gateway_loop = asyncio.get_running_loop()
        runner._run_in_executor_with_context = asyncio.to_thread
        runner.hooks.loaded_hooks = False
        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {"display": {"tool_progress": "off", "streaming": False}},
        )
        result = await GatewayRunner._run_agent_inner(
            runner,
            INTENT,
            "",
            [],
            source,
            "ordinary-session",
            session_key="ordinary-key",
            inbound_message_id="message-1",
        )
        await asyncio.wait_for(done.wait(), 2)
        assert result["_stagea_native_reply"] is True
        assert "MEDIA:/tmp/synthetic.png" in result["final_response"]
        assert len(wire.calls) == 1
        runner._run_agent_via_proxy.assert_not_called()


@pytest.mark.asyncio
async def test_real_handler_holds_session_lease_and_skips_hygiene(
    ordinary, monkeypatch
):
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig
    from gateway.session import SessionEntry
    from gateway.turn_lease import SessionTurnLeaseRegistry

    agent, wire = ordinary

    async def controller(reader, writer):
        request = await B.read_frame(reader)
        writer.write(N.encode_native_frame(correlated("challenge", request)))
        await writer.drain()
        await B.read_frame(reader)
        writer.write(N.encode_native_frame(correlated("reply", request)))
        await writer.drain()

    async with peer(controller) as (path, done):
        adapter = make_adapter(path)
        source = SessionSource(
            platform=Platform.WEIXIN, chat_id="owner", user_id="owner"
        )
        turn_runner, _ = runner_for(adapter, source, agent, asyncio.get_running_loop())
        runner = turn_runner._runner
        runner.config = GatewayConfig()
        runner._recover_telegram_topic_thread_id.return_value = None
        runner._is_telegram_topic_lane.return_value = False
        runner._is_session_run_current.return_value = True
        runner._pinned_session_context_prompt.return_value = ""
        runner._get_proxy_url.return_value = None
        runner._resolve_enabled_toolsets_for_source.return_value = []
        runner._gateway_loop = asyncio.get_running_loop()
        runner._run_in_executor_with_context = asyncio.to_thread
        runner._turn_leases = SessionTurnLeaseRegistry()
        now = datetime.now(timezone.utc)
        entry = SessionEntry("ordinary-key", "ordinary-session", now, now)
        # Force the existing hygiene trigger if this native branch regresses.
        entry.last_prompt_tokens = 10_000_000
        runner.async_session_store.get_or_create_session = AsyncMock(return_value=entry)
        runner.async_session_store.load_transcript = AsyncMock(
            side_effect=AssertionError("Native does not load old history")
        )
        runner.hooks.emit = AsyncMock(
            side_effect=AssertionError("Native does not start hooks")
        )
        runner.hooks.loaded_hooks = False
        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {"display": {"tool_progress": "off", "streaming": False}},
        )

        async def run_agent(**kwargs):
            return await GatewayRunner._run_agent_inner(runner, **kwargs)

        runner._run_agent = run_agent

        async def handle(event):
            try:
                return await GatewayRunner._handle_message_with_agent(
                    runner, event, event.source, "ordinary-key", 7
                )
            finally:
                token = runner._session_state("ordinary-key").turn.lease_token
                if token is not None:
                    runner._turn_leases.release(token)

        adapter.handle_message = handle
        prior = await runner._turn_leases.acquire(
            "ordinary-session", owner_key="prior", generation=6
        )
        task = asyncio.create_task(adapter._process_message(inbound()))
        await asyncio.sleep(0.1)
        assert not wire.calls and not task.done()
        runner._turn_leases.release(prior)
        await asyncio.wait_for(task, 3)
        await asyncio.wait_for(done.wait(), 2)
        assert len(wire.calls) == 1
        assert adapter._send_text_chunk.await_count == 1
        runner.async_session_store.load_transcript.assert_not_called()
        runner.hooks.emit.assert_not_called()

        # A different route bound to the same persisted session is fenced
        # before hooks or hygiene, even without a NativeTurn on its source.
        adapter._stagea_native._unresolved.add("session:ordinary-session")
        alias_source = SessionSource(
            platform=Platform.WEIXIN, chat_id="other", user_id="other"
        )
        from gateway.platforms.base import MessageEvent, MessageType

        alias_event = MessageEvent(
            text="ordinary alias input",
            message_type=MessageType.TEXT,
            source=alias_source,
            message_id="alias-1",
        )
        await handle(alias_event)
        assert cast(Any, alias_event)._stagea_native_terminal is True
        assert len(wire.calls) == 1
        assert N.UNRESOLVED in adapter._send_text_chunk.call_args.kwargs["chunk"]
        assert not adapter._stagea_native.blocked(
            adapter._stagea_conversation_key(alias_source), "different-session"
        )
        runner.async_session_store.load_transcript.assert_not_called()
        runner.hooks.emit.assert_not_called()

        # Adapter replacement and another platform cannot evade a shared
        # persisted-session fence retained by the existing runner.
        replacement = make_adapter(path)
        runner._adapter_for_source.return_value = replacement
        assert N.execution_blocked(runner, source, "ordinary-session")
        runner._adapter_for_source.return_value = SimpleNamespace()
        assert N.execution_blocked(runner, alias_source, "ordinary-session")
        assert not N.execution_blocked(runner, alias_source, "different-session")

        # An alias can already be queued at the lease when ambiguity appears.
        runner._adapter_for_source.return_value = adapter
        adapter._stagea_native._unresolved.clear()
        entry.updated_at = datetime.now(timezone.utc)
        prior = await runner._turn_leases.acquire(
            "ordinary-session", owner_key="prior", generation=8
        )
        pending_alias = asyncio.create_task(handle(alias_event))
        await asyncio.sleep(0.1)
        assert not pending_alias.done()
        adapter._stagea_native._unresolved.add("session:ordinary-session")
        runner._turn_leases.release(prior)
        await asyncio.wait_for(pending_alias, 2)
        assert len(wire.calls) == 1
        runner.async_session_store.load_transcript.assert_not_called()
        runner.hooks.emit.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_backstop_unchanged_with_native_selected():
    async def controller(reader, writer):
        request = await B.read_frame(reader)
        assert request["schema"] == B.SCHEMA and request["type"] == B.REQUEST_TYPE
        assert request["text"] == "explicit backstop"
        writer.write(
            B.encode_frame({
                "schema": B.SCHEMA,
                "protocol": B.PROTOCOL,
                "type": B.REPLY_TYPE,
                "request_id": request["request_id"],
                "conversation_ref": request["conversation_ref"],
                "outcome": "UNKNOWN",
                "text": "Legacy source backstop retained.",
            })
        )
        await writer.drain()

    async with peer(controller) as (path, done):
        adapter = make_adapter(path)
        await adapter._process_message(inbound("/stagea explicit backstop"))
        await asyncio.wait_for(done.wait(), 2)
        adapter.handle_message.assert_not_called()
        assert adapter._send_text_chunk.await_count == 1


@pytest.mark.asyncio
async def test_unknown_opt_in_mode_refuses_without_ordinary_fallback():
    adapter = make_adapter(**{N.CONFIG_MODE: "unsupported"})
    await adapter._process_message(inbound())
    adapter.handle_message.assert_not_called()
    adapter._enqueue_text_event.assert_not_called()
    assert (
        "UNSUPPORTED_NATIVE_CONFIGURATION"
        in adapter._send_text_chunk.call_args.kwargs["chunk"]
    )


@pytest.mark.asyncio
async def test_normal_cache_miss_constructs_once_and_new_ids_stay_distinct(ordinary):
    agent, wire = ordinary
    received = []

    async def controller(reader, writer):
        request = await B.read_frame(reader)
        received.append(request)
        writer.write(N.encode_native_frame(correlated("challenge", request)))
        await writer.drain()
        await B.read_frame(reader)
        writer.write(N.encode_native_frame(correlated("reply", request)))
        await writer.drain()

    async with peer(controller) as (path, _):
        adapter = make_adapter(path)
        constructor = Mock(return_value=agent)

        async def handle(event):
            turn_runner, _ = runner_for(
                adapter, event.source, agent, asyncio.get_running_loop()
            )
            turn_runner._runner._agent_cache.clear()
            turn_runner._ctx.AIAgent = constructor
            await asyncio.to_thread(turn_runner.run_sync)

        adapter.handle_message = handle
        await adapter._process_message(inbound(message_id="new-1"))
        constructor.assert_called_once()
        # The second intentional identical text has its own platform identity.
        await execute(adapter, agent, message=inbound(message_id="new-2"))
    assert len(wire.calls) == 2
    assert received[0]["request_id"] != received[1]["request_id"]
    assert received[0]["conversation_ref"] == received[1]["conversation_ref"]


@asynccontextmanager
async def actual_gateway(adapter, ordinary, monkeypatch, tmp_path):
    """Use inherited base ingress, outer dispatch, session lease and TurnRunner.

    Only synthetic configuration, transport and consequential control/hooks
    are supplied. None of the production message/turn handlers is replaced.
    """
    from gateway.config import GatewayConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.run import GatewayRunner

    agent, _ = ordinary
    del adapter.handle_message
    del adapter._enqueue_text_event
    assert adapter.handle_message.__func__ is BasePlatformAdapter.handle_message
    adapter.config.typing_indicator = False
    adapter._text_batch_delay_seconds = 0
    adapter._text_batch_split_delay_seconds = 0
    adapter._send_with_retry = AsyncMock(return_value=SendResult(True, "inert-reply"))
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *a, **k: [])
    monkeypatch.setattr("agent.estop.paused_reply", lambda: None)
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path / "hermes")
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"display": {"tool_progress": "off", "streaming": False}},
    )
    runner = cast(
        Any,
        GatewayRunner(
            GatewayConfig(
                platforms={Platform.WEIXIN: adapter.config},
                sessions_dir=tmp_path / "gateway-sessions",
            )
        ),
    )
    runner.adapters = {Platform.WEIXIN: adapter}
    runner._gateway_loop = asyncio.get_running_loop()
    runner._is_user_authorized = lambda source: source.user_id in {"owner", "other"}
    runner._handle_restart_command = AsyncMock(return_value="INERT_CONTROL_SENTINEL")
    runner._handle_approve_command = AsyncMock(return_value="INERT_APPROVAL_SENTINEL")
    runner.hooks = SimpleNamespace(
        loaded_hooks=False, emit=AsyncMock(), emit_collect=AsyncMock(return_value=[])
    )
    runner._resolve_session_agent_runtime = lambda *a, **k: (
        agent.model,
        {"provider": agent.provider},
    )
    runner._resolve_turn_agent_config = lambda *a, **k: {
        "model": agent.model,
        "runtime": {},
    }
    runner._agent_config_signature = lambda *a, **k: ("signature",)
    runner._extract_cache_busting_config = lambda *a, **k: {}
    runner._get_system_prompt_for_channel = lambda *a, **k: None
    runner._get_proxy_url = lambda: None
    runner._refresh_fallback_model = lambda: None
    runner._resolve_session_reasoning_config = lambda *a, **k: None
    runner._resolve_session_service_tier = lambda *a, **k: None
    source = adapter.build_source(
        chat_id="owner", chat_type="dm", user_id="owner", user_name="owner"
    )
    entry = runner.session_store.get_or_create_session(source)
    key = runner._session_key_for_source(source)
    runner._agent_cache[key] = (agent, ("signature",), None, entry.session_id)
    constructor = Mock(side_effect=AssertionError("No second AIAgent"))
    monkeypatch.setattr("run_agent.AIAgent", constructor)
    adapter.set_message_handler(runner._primary_message_handler())
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    try:
        yield runner, key, constructor
    finally:
        for task in tuple(adapter._background_tasks):
            if not task.done():
                task.cancel()
        if adapter._background_tasks:
            await asyncio.gather(
                *tuple(adapter._background_tasks), return_exceptions=True
            )
        if runner._executor is not None:
            runner._executor.shutdown(wait=True)
        runner.session_store.close_all_db_handles()


async def drain_actual_ingress(adapter):
    batches = tuple(adapter._pending_text_batch_tasks.values())
    if batches:
        await asyncio.wait_for(asyncio.gather(*batches), 5)
    tasks = tuple(adapter._background_tasks)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent", ["restart gateway", "Please restart the Hermes gateway!", "approve"]
)
async def test_real_native_command_path(ordinary, monkeypatch, tmp_path, intent):
    _, wire = ordinary
    seen = []

    async def controller(reader, writer):
        request = await B.read_frame(reader, strict=True)
        seen.append(request)
        challenge = correlated("challenge", request)
        challenge["request"]["intent"] = request["text"]
        writer.write(N.encode_native_frame(challenge))
        await writer.drain()
        await B.read_frame(reader, strict=True)
        writer.write(N.encode_native_frame(correlated("reply", request)))
        await writer.drain()

    async with peer(controller) as (path, _):
        adapter = make_adapter(path)
        async with actual_gateway(adapter, ordinary, monkeypatch, tmp_path) as (
            runner,
            _,
            constructor,
        ):
            monkeypatch.setattr("tools.approval.has_blocking_approval", lambda _: True)
            await adapter._process_message(
                inbound(intent, message_id="native-command-like")
            )
            await drain_actual_ingress(adapter)
            runner._handle_restart_command.assert_not_called()
            runner._handle_approve_command.assert_not_called()
            constructor.assert_not_called()
            assert len(wire.calls) == len(seen) == 1
            assert seen[0]["text"] == intent
            assert (
                json.loads(wire.calls[0]["messages"][1]["content"])["intent"] == intent
            )
            assert adapter._send_text_chunk.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["queue", "steer", "interrupt", "drain", "approval"])
async def test_real_native_busy_is_per_message(ordinary, monkeypatch, tmp_path, mode):
    _, wire = ordinary
    entered, release = asyncio.Event(), asyncio.Event()
    received = []

    async def controller(reader, writer):
        request = await B.read_frame(reader, strict=True)
        received.append(request)
        entered.set()
        await asyncio.wait_for(release.wait(), 4)
        writer.write(N.encode_native_frame(correlated("challenge", request)))
        await writer.drain()
        await B.read_frame(reader, strict=True)
        writer.write(N.encode_native_frame(correlated("reply", request)))
        await writer.drain()

    async with peer(controller) as (path, _):
        adapter = make_adapter(path)
        adapter._busy_text_mode = "queue"
        async with actual_gateway(adapter, ordinary, monkeypatch, tmp_path) as (
            runner,
            key,
            constructor,
        ):
            runner._busy_input_mode = (
                mode if mode in {"queue", "steer", "interrupt"} else "queue"
            )
            runner._busy_text_mode = "queue"
            await adapter._process_message(inbound(message_id="active-native"))
            await asyncio.wait_for(entered.wait(), 3)
            try:
                runner._draining = mode == "drain"
                monkeypatch.setattr(
                    "tools.approval.has_blocking_approval", lambda _: mode == "approval"
                )
                for message_id, intent in [
                    ("busy-1", "restart gateway"),
                    ("busy-2", "approve"),
                ]:
                    await adapter._process_message(
                        inbound(intent, message_id=message_id)
                    )
                replies = [
                    call.kwargs["chunk"]
                    for call in adapter._send_text_chunk.call_args_list
                ]
                assert len(replies) == 2
                for reply, message_id in zip(replies, ["busy-1", "busy-2"]):
                    expected_id = B.derive_request_id(
                        received[0]["conversation_ref"], message_id
                    )
                    assert "NATIVE_BUSY" in reply and expected_id in reply
                assert not adapter._pending_messages and not adapter._text_debounce
                assert runner._queue_depth(key, adapter=adapter) == 0
                assert len(received) == 1 and not wire.calls
                runner._handle_approve_command.assert_not_called()
                if mode == "queue":
                    # Another principal retains ordinary command dispatch
                    # while the native Owner conversation remains active.
                    await adapter._process_message(
                        inbound("restart gateway", sender="other")
                    )
                    await asyncio.wait_for(
                        asyncio.gather(
                            *tuple(adapter._pending_text_batch_tasks.values())
                        ),
                        2,
                    )
                    other_key = next(k for k in adapter._session_tasks if k != key)
                    await asyncio.wait_for(adapter._session_tasks[other_key], 2)
                    runner._handle_restart_command.assert_awaited_once()
                    runner._handle_restart_command.reset_mock()
            finally:
                runner._draining = False
                release.set()
                await drain_actual_ingress(adapter)
            assert len(wire.calls) == 1
            assert len(received) == 1
            constructor.assert_not_called()
            runner._handle_restart_command.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("sender,mode", [("owner", ""), ("other", N.MODE)])
async def test_real_non_native_command_control(
    ordinary, monkeypatch, tmp_path, sender, mode
):
    _, wire = ordinary
    adapter = make_adapter(**{N.CONFIG_MODE: mode})
    if not mode:
        adapter.config.extra.pop(N.CONFIG_MODE)  # default off, not a special mode
    async with actual_gateway(adapter, ordinary, monkeypatch, tmp_path) as (
        runner,
        _,
        constructor,
    ):
        await adapter._process_message(inbound("restart gateway", sender=sender))
        await drain_actual_ingress(adapter)
        runner._handle_restart_command.assert_awaited_once()
        event = runner._handle_restart_command.call_args.args[0]
        assert event.allow_gateway_control and event.text == "/restart"
        assert N.turn_for(event.source) is None
        assert not wire.calls
        constructor.assert_not_called()
        adapter._send_text_chunk.assert_not_called()


@pytest.mark.asyncio
async def test_real_non_native_busy_retains_existing_coalescer(
    ordinary, monkeypatch, tmp_path
):
    _, wire = ordinary
    adapter = make_adapter(**{N.CONFIG_MODE: "off"})
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 30
    async with actual_gateway(adapter, ordinary, monkeypatch, tmp_path) as (
        runner,
        key,
        constructor,
    ):
        runner._busy_input_mode = "queue"
        runner._busy_text_mode = "queue"
        # Existing active-session state, with a live owner task. Native tests
        # above additionally hold a real TurnRunner/socket exchange open.
        adapter._active_sessions[key] = asyncio.Event()
        adapter._session_tasks[key] = asyncio.current_task()
        try:
            for message_id, text in [
                ("ordinary-1", "First ordinary text."),
                ("ordinary-2", "Second ordinary text."),
            ]:
                await adapter._process_message(inbound(text, message_id=message_id))
                await asyncio.wait_for(
                    asyncio.gather(*tuple(adapter._pending_text_batch_tasks.values())),
                    2,
                )
            event = adapter._text_debounce[key].event
            assert event.text == "First ordinary text.\nSecond ordinary text."
            assert event.message_id == "ordinary-2" and N.turn_for(event.source) is None
            assert event.allow_gateway_control
            await adapter._flush_text_debounce_now(key)
            assert adapter._pending_messages[key] is event
            assert not wire.calls
            constructor.assert_not_called()
        finally:
            adapter._discard_text_debounce(key)
            adapter._pending_messages.pop(key, None)
            adapter._active_sessions.pop(key, None)
            adapter._session_tasks.pop(key, None)


@pytest.mark.asyncio
async def test_real_legacy_backstop_control(ordinary, monkeypatch, tmp_path):
    _, wire = ordinary
    requests = []

    async def controller(reader, writer):
        request = await B.read_frame(reader, strict=True)
        requests.append(request)
        assert request["schema"] == B.SCHEMA and request["type"] == B.REQUEST_TYPE
        assert request["text"] == "explicit backstop"
        writer.write(
            B.encode_frame({
                "schema": B.SCHEMA,
                "protocol": B.PROTOCOL,
                "type": B.REPLY_TYPE,
                "request_id": request["request_id"],
                "conversation_ref": request["conversation_ref"],
                "outcome": "UNKNOWN",
                "text": "Inert legacy backstop.",
            })
        )
        await writer.drain()

    async with peer(controller) as (path, done):
        adapter = make_adapter(path)
        async with actual_gateway(adapter, ordinary, monkeypatch, tmp_path) as (
            runner,
            _,
            constructor,
        ):
            await adapter._process_message(inbound("/stagea explicit backstop"))
            await asyncio.wait_for(done.wait(), 2)
            assert len(requests) == 1 and not wire.calls
            adapter._send_text_chunk.assert_awaited_once()
            assert not adapter._pending_text_batches and not adapter._background_tasks
            runner._handle_restart_command.assert_not_called()
            constructor.assert_not_called()
