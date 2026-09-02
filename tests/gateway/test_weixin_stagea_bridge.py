"""Falsifiers for the Weixin adapter's Stage-A Owner bridge seam.

The bridge is one call inside ``_process_message``.  These tests pin what
that one call is allowed to change: an Owner Stage-A request never reaches
the agent, everything else routes exactly as it did before the seam
existed, and a reply lands in the initiating conversation and nowhere
else.
"""

import asyncio
import os
import stat
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.config import PlatformConfig
from gateway import stagea_owner_bridge as bridge
from gateway.platforms import weixin
from gateway.platforms.weixin import WeixinAdapter

OWNER = "wxid_owner"
OTHER = "wxid_someone_else"
ACCOUNT = "test-account"

_SOCK_STAT = os.stat_result((stat.S_IFSOCK | 0o660, 0, 0, 1, 0, 0, 0, 0, 0, 0))


def _adapter(**env) -> Any:
    """An adapter with intake open and dispatch/outbound captured.

    The doubles deliberately replace bound methods, so the adapter is held
    as ``Any``.
    """
    adapter = cast(
        Any,
        WeixinAdapter(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={
                    "account_id": ACCOUNT,
                    "dm_policy": "allowlist",
                    "allow_from": [OWNER, OTHER],
                },
            )
        ),
    )
    adapter._poll_session = object()
    adapter._send_session = object()
    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = Mock()
    adapter._send_text_chunk = AsyncMock()
    adapter._stagea_bridge = bridge.StageAOwnerBridge(
        secret_getter=lambda name, default=None: env.get(name, default),
        channel="weixin",
    )
    return adapter


def _dispatched(adapter):
    """How many times the pre-existing routing path ran.

    ``_process_message`` sends ``MessageType.TEXT`` through the debounce
    batcher and everything else — including any ``/``-prefixed
    ``MessageType.COMMAND`` — straight to ``handle_message``, so an
    untouched message shows up in exactly one of the two.
    """
    return adapter._enqueue_text_event.call_count + adapter.handle_message.await_count


def _message(text, *, sender=OWNER, message_id="msg-1"):
    return {
        "from_user_id": sender,
        "message_id": message_id,
        "item_list": [{"type": weixin.ITEM_TEXT, "text_item": {"text": text}}],
    }


def _enabled(**extra):
    return {bridge.ENV_OWNER_USER_ID: OWNER, **extra}


def _fake_peer(reply_builder):
    """Patch the socket layer for the duration of one exchange."""
    captured = []

    async def open_connection(path):
        reader = asyncio.StreamReader()
        writer = Mock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()

        def on_write(frame):
            captured.append(frame)
            import json

            request = json.loads(frame[4:].decode("utf-8"))
            reader.feed_data(bridge.encode_frame(reply_builder(request)))
            reader.feed_eof()

        writer.write = on_write
        return reader, writer

    return open_connection, captured


def _terminal(request, *, outcome="ACCEPTED_TERMINAL", text="lane advanced"):
    return {
        "schema": bridge.SCHEMA,
        "protocol": bridge.PROTOCOL,
        "type": bridge.REPLY_TYPE,
        "request_id": request["request_id"],
        "conversation_ref": request["conversation_ref"],
        "outcome": outcome,
        "text": text,
    }


class TestOrdinaryRoutingUnchanged:
    """The seam must be invisible to every message that is not the Owner's."""

    @pytest.mark.asyncio
    async def test_bridge_is_off_without_an_owner_id(self):
        adapter = _adapter()
        assert adapter._stagea_bridge.enabled is False
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            await adapter._process_message(_message("/stagea advance"))
        assert _dispatched(adapter) == 1
        adapter._send_text_chunk.assert_not_awaited()
        opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_ordinary_owner_text_still_reaches_the_agent(self):
        adapter = _adapter(**_enabled())
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            await adapter._process_message(_message("what is the current lane?"))
        adapter._enqueue_text_event.assert_called_once()
        adapter._send_text_chunk.assert_not_awaited()
        opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_owner_stagea_text_routes_normally_and_creates_no_request(self):
        adapter = _adapter(**_enabled())
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            with patch("os.stat") as stat_call:
                await adapter._process_message(_message("/stagea advance", sender=OTHER))
        assert _dispatched(adapter) == 1
        adapter._send_text_chunk.assert_not_awaited()
        opened.assert_not_called()
        stat_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_sender_is_still_dropped_before_the_bridge(self):
        adapter = _adapter(**_enabled())
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            await adapter._process_message(_message("/stagea advance", sender="wxid_stranger"))
        assert _dispatched(adapter) == 0
        adapter._send_text_chunk.assert_not_awaited()
        opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_group_message_is_never_admitted(self):
        adapter = _adapter(**_enabled())
        adapter._group_policy = "allowlist"
        adapter._group_allow_from = ["room-1"]
        message = _message("/stagea advance")
        message["room_id"] = "room-1"
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            await adapter._process_message(message)
        opened.assert_not_called()
        adapter._send_text_chunk.assert_not_awaited()
        assert _dispatched(adapter) == 1


class TestAdmittedRequest:
    @pytest.mark.asyncio
    async def test_owner_request_bypasses_the_agent_entirely(self):
        adapter = _adapter(**_enabled())
        open_connection, captured = _fake_peer(_terminal)
        with patch("os.stat", return_value=_SOCK_STAT):
            with patch.object(asyncio, "open_unix_connection", open_connection, create=True):
                await adapter._process_message(_message("/stagea advance the lane"))

        # No legacy dispatch of any kind: not the batcher, not handle_message.
        assert _dispatched(adapter) == 0
        assert len(captured) == 1

        adapter._send_text_chunk.assert_awaited_once()
        kwargs = adapter._send_text_chunk.await_args.kwargs
        assert kwargs["chat_id"] == OWNER
        assert kwargs["chunk"] == "[Stage-A] ACCEPTED_TERMINAL\nlane advanced"

    @pytest.mark.asyncio
    async def test_reply_goes_only_to_the_initiating_conversation(self):
        """A peer-named destination is refused, and cannot redirect delivery."""
        adapter = _adapter(**_enabled())
        open_connection, _ = _fake_peer(
            lambda request: {**_terminal(request), "chat_id": "wxid_attacker"}
        )
        with patch("os.stat", return_value=_SOCK_STAT):
            with patch.object(asyncio, "open_unix_connection", open_connection, create=True):
                await adapter._process_message(_message("/stagea advance"))

        adapter._send_text_chunk.assert_awaited_once()
        kwargs = adapter._send_text_chunk.await_args.kwargs
        assert kwargs["chat_id"] == OWNER
        assert "reply_unexpected_field" in kwargs["chunk"]
        assert "wxid_attacker" not in kwargs["chunk"]

    @pytest.mark.asyncio
    async def test_reply_for_another_conversation_is_refused(self):
        adapter = _adapter(**_enabled())
        open_connection, _ = _fake_peer(
            lambda request: {**_terminal(request), "conversation_ref": "f" * 32}
        )
        with patch("os.stat", return_value=_SOCK_STAT):
            with patch.object(asyncio, "open_unix_connection", open_connection, create=True):
                await adapter._process_message(_message("/stagea advance"))
        chunk = adapter._send_text_chunk.await_args.kwargs["chunk"]
        assert "reply_wrong_conversation" in chunk

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", list(bridge.REPLY_OUTCOMES))
    async def test_every_admitted_outcome_reaches_the_owner(self, outcome):
        adapter = _adapter(**_enabled())
        open_connection, _ = _fake_peer(
            lambda request: _terminal(request, outcome=outcome, text="see issue")
        )
        with patch("os.stat", return_value=_SOCK_STAT):
            with patch.object(asyncio, "open_unix_connection", open_connection, create=True):
                await adapter._process_message(_message("/stagea advance"))
        chunk = adapter._send_text_chunk.await_args.kwargs["chunk"]
        assert chunk.startswith(f"[Stage-A] {outcome}")

    @pytest.mark.asyncio
    async def test_socket_unavailable_answers_the_owner_and_dispatches_nothing(self):
        adapter = _adapter(**_enabled())
        with patch("os.stat", side_effect=FileNotFoundError()):
            await adapter._process_message(_message("/stagea advance"))
        assert _dispatched(adapter) == 0
        chunk = adapter._send_text_chunk.await_args.kwargs["chunk"]
        assert "socket_unavailable" in chunk
        assert "No Stage-A work was created." in chunk

    @pytest.mark.asyncio
    async def test_wrong_socket_peer_is_refused(self):
        adapter = _adapter(**_enabled(**{bridge.ENV_SOCKET_UID: "4242"}))
        foreign = os.stat_result((stat.S_IFSOCK | 0o660, 0, 0, 1, 1000, 0, 0, 0, 0, 0))
        with patch("os.stat", return_value=foreign):
            with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
                await adapter._process_message(_message("/stagea advance"))
        chunk = adapter._send_text_chunk.await_args.kwargs["chunk"]
        assert "socket_owner_mismatch" in chunk
        opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_replayed_message_id_creates_one_request_only(self):
        adapter = _adapter(**_enabled())
        # Defeat the adapter's own dedup so the bridge guard is what is
        # actually under test here.
        adapter._dedup.is_duplicate = Mock(return_value=False)
        open_connection, captured = _fake_peer(_terminal)
        with patch("os.stat", return_value=_SOCK_STAT):
            with patch.object(asyncio, "open_unix_connection", open_connection, create=True):
                await adapter._process_message(_message("/stagea advance"))
                await adapter._process_message(_message("/stagea advance"))

        assert len(captured) == 1
        assert adapter._send_text_chunk.await_count == 2
        second = adapter._send_text_chunk.await_args_list[1].kwargs["chunk"]
        assert "duplicate_request" in second

    @pytest.mark.asyncio
    async def test_attachment_is_refused_without_reaching_the_socket(self, monkeypatch):
        adapter = _adapter(**_enabled())

        async def _collect(item, media_paths, media_types):
            media_paths.append("/tmp/photo.jpg")
            media_types.append("image")

        monkeypatch.setattr(adapter, "_collect_media", _collect)
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            await adapter._process_message(_message("/stagea advance"))
        chunk = adapter._send_text_chunk.await_args.kwargs["chunk"]
        assert "media_not_admitted" in chunk
        opened.assert_not_called()


class TestDeliveryPath:
    def test_reply_bypasses_media_extraction(self):
        """A reply must be text, never a file delivery instruction.

        ``send()`` pulls ``MEDIA:`` tags and bare local paths out of the
        content it is handed.  The Stage-A reply path must not use it.
        """
        import inspect

        source = inspect.getsource(WeixinAdapter._send_stagea_reply)
        assert "_send_text_chunk" in source
        assert "self.send(" not in source
        assert "extract_media" not in source

    @pytest.mark.asyncio
    async def test_disconnected_adapter_drops_the_reply_without_raising(self):
        adapter = _adapter(**_enabled())
        adapter._send_session = None
        await adapter._send_stagea_reply(OWNER, "[Stage-A] UNKNOWN\nx")
        adapter._send_text_chunk.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_failure_cannot_break_the_inbound_path(self):
        adapter = _adapter(**_enabled())
        adapter._send_text_chunk = AsyncMock(side_effect=RuntimeError("iLink down"))
        with patch("os.stat", side_effect=FileNotFoundError()):
            await adapter._process_message(_message("/stagea advance"))
        adapter._send_text_chunk.assert_awaited_once()


class TestNoFallbackChannel:
    def test_seam_names_no_other_transport(self):
        import inspect

        source = "\n".join(
            inspect.getsource(fn)
            for fn in (
                WeixinAdapter._maybe_handle_stagea_owner_request,
                WeixinAdapter._send_stagea_reply,
            )
        ).lower()
        for forbidden in ("telegram", "openclaw", "webhook", "outbox", "broadcast", "home_channel"):
            assert forbidden not in source, f"stage-a seam must not reference {forbidden}"

    @pytest.mark.asyncio
    async def test_no_second_destination_is_ever_written(self):
        adapter = _adapter(**_enabled())
        adapter.send = AsyncMock()
        adapter.send_document = AsyncMock()
        adapter.send_image_file = AsyncMock()
        with patch("os.stat", side_effect=FileNotFoundError()):
            await adapter._process_message(_message("/stagea advance"))
        adapter.send.assert_not_awaited()
        adapter.send_document.assert_not_awaited()
        adapter.send_image_file.assert_not_awaited()
        assert adapter._send_text_chunk.await_count == 1
        assert adapter._send_text_chunk.await_args.kwargs["chat_id"] == OWNER


class TestCredentialCustody:
    @pytest.mark.asyncio
    async def test_no_configured_credential_reaches_the_wire(self):
        adapter = _adapter(**_enabled())
        adapter._token = "sentinel-weixin-token"
        open_connection, captured = _fake_peer(_terminal)
        with patch.dict(
            os.environ,
            {
                "WEIXIN_TOKEN": "sentinel-weixin-token",
                "TELEGRAM_BOT_TOKEN": "sentinel-telegram-token",
                "GLM_API_KEY": "sentinel-glm-key",
            },
            clear=False,
        ):
            with patch("os.stat", return_value=_SOCK_STAT):
                with patch.object(
                    asyncio, "open_unix_connection", open_connection, create=True
                ):
                    await adapter._process_message(_message("/stagea advance"))

        wire = captured[0].decode("utf-8", errors="replace")
        for sentinel in (
            "sentinel-weixin-token",
            "sentinel-telegram-token",
            "sentinel-glm-key",
        ):
            assert sentinel not in wire
        # The Owner's real channel identity stays local too.
        assert OWNER not in wire
        assert ACCOUNT not in wire
