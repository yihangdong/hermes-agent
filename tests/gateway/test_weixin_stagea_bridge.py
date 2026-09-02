"""Falsifiers for the Weixin adapter's Stage-A Owner bridge seam.

The bridge is one call inside ``_process_message``.  These tests pin what
that one call is allowed to change: an Owner Stage-A request never reaches
the agent, everything else routes exactly as it did before the seam
existed, and a reply lands in the initiating conversation and nowhere
else.
"""

import asyncio
import contextlib
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

CONTROLLER_UID = 4242

_SOCK_STAT = os.stat_result((stat.S_IFSOCK | 0o660, 101, 1, 1, CONTROLLER_UID, 0, 0, 0, 0, 0))
_DIR_STAT = os.stat_result((0o40755, 202, 1, 1, CONTROLLER_UID, 0, 0, 0, 0, 0))
_SOCKET_DIR = os.path.dirname(bridge.DEFAULT_SOCKET_PATH)


def _stat_router(path, *args, **kwargs):
    """Answer per path: the exchange stats the socket again after connecting."""
    return _DIR_STAT if str(path) == _SOCKET_DIR else _SOCK_STAT


@contextlib.contextmanager
def _socket_layer(open_connection, *, peer_uid=CONTROLLER_UID):
    """Patch the socket layer: path stats, connected-peer identity, connect."""
    with patch("os.stat", side_effect=_stat_router):
        with patch.object(bridge, "read_peer_uid", return_value=peer_uid):
            with patch.object(asyncio, "open_unix_connection", open_connection, create=True):
                yield


def _adapter(**stagea) -> Any:
    """An adapter with intake open and dispatch/outbound captured.

    Stage-A settings arrive the way a deployment actually supplies them:
    as ``extra`` keys on this profile's own ``PlatformConfig``, which is
    exactly what ``config.yaml``'s ``gateway.platforms.weixin.extra``
    block loads into.  The bridge under test is the one the adapter
    builds for itself — no test-supplied getter stands in for the real
    configuration path, so these tests fail if that wiring breaks.

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
                    **stagea,
                },
            )
        ),
    )
    adapter._poll_session = object()
    adapter._send_session = object()
    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = Mock()
    adapter._send_text_chunk = AsyncMock()
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


def _enabled(**overrides):
    """The ``config.yaml`` ``extra`` keys that enable the bridge.

    The expected controller uid is mandatory: an enabled bridge that
    cannot name its peer refuses to connect at all.
    """
    return {
        bridge.CONFIG_OWNER_USER_ID: OWNER,
        bridge.CONFIG_SOCKET_UID: str(CONTROLLER_UID),
        **overrides,
    }


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
            with patch.object(bridge, "check_socket_path") as preflight:
                await adapter._process_message(_message("/stagea advance", sender=OTHER))
        assert _dispatched(adapter) == 1
        adapter._send_text_chunk.assert_not_awaited()
        opened.assert_not_called()
        preflight.assert_not_called()

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
        with _socket_layer(open_connection):
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
        with _socket_layer(open_connection):
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
        with _socket_layer(open_connection):
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
        with _socket_layer(open_connection):
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
        adapter = _adapter(**_enabled(**{bridge.CONFIG_SOCKET_UID: "4242"}))
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
        with _socket_layer(open_connection):
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


class TestAttachmentTruth:
    """Text-only truth is read from the raw items, on an allowlist.

    Finding 3 of the first Tier-1 review: ``has_media`` was derived from
    successfully downloaded media, so an attachment-bearing message whose
    fetch failed was admitted as text-only.  Finding 2 of the re-review:
    the replacement was a *blocklist* of four known attachment types, so a
    raw item with an unknown non-text type — the one case a blocklist
    cannot get right — was still admitted as plain text.
    """

    @pytest.mark.parametrize(
        "item_type",
        [weixin.ITEM_IMAGE, weixin.ITEM_VOICE, weixin.ITEM_FILE, weixin.ITEM_VIDEO],
    )
    def test_no_known_attachment_type_is_text_only(self, item_type):
        assert weixin._is_text_only_message([{"type": item_type}]) is False

    def test_plain_text_is_text_only(self):
        assert weixin._is_text_only_message(
            [{"type": weixin.ITEM_TEXT, "text_item": {"text": "/stagea go"}}]
        ) is True

    @pytest.mark.parametrize("item_list", [[], None])
    def test_nothing_is_not_text_only(self, item_list):
        assert weixin._is_text_only_message(item_list) is False

    def test_quoted_media_is_not_text_only(self):
        assert weixin._is_text_only_message(
            [
                {
                    "type": weixin.ITEM_TEXT,
                    "text_item": {"text": "/stagea go"},
                    "ref_msg": {"message_item": {"type": weixin.ITEM_IMAGE}},
                }
            ]
        ) is False

    def test_quoted_text_stays_text_only(self):
        """The rule is an allowlist, not a ban on quoting."""
        assert weixin._is_text_only_message(
            [
                {
                    "type": weixin.ITEM_TEXT,
                    "text_item": {"text": "/stagea go"},
                    "ref_msg": {"message_item": {"type": weixin.ITEM_TEXT}},
                }
            ]
        ) is True

    @pytest.mark.parametrize(
        "item",
        [
            pytest.param({"type": 999}, id="unknown-type"),
            pytest.param({"type": None}, id="null-type"),
            pytest.param({"text_item": {"text": "x"}}, id="absent-type"),
            pytest.param({"type": "1"}, id="stringified-type"),
            pytest.param("not-a-dict", id="not-a-dict"),
        ],
    )
    def test_anything_not_provably_text_is_not_text_only(self, item):
        """The exact gap the re-review reproduced, plus its neighbours.

        A type this build has never seen carries who-knows-what; there is
        no safe way to guess, so it cannot be called text.
        """
        assert weixin._is_text_only_message([item]) is False

    @pytest.mark.parametrize(
        "ref_msg",
        [
            pytest.param({"message_item": {"type": 999}}, id="unknown-quoted-type"),
            pytest.param({"message_item": "not-a-dict"}, id="unreadable-quoted-item"),
            pytest.param("not-a-dict", id="unreadable-quote"),
        ],
    )
    def test_an_unreadable_quote_is_not_text_only(self, ref_msg):
        assert weixin._is_text_only_message(
            [{"type": weixin.ITEM_TEXT, "text_item": {"text": "/stagea go"}, "ref_msg": ref_msg}]
        ) is False

    def test_one_unknown_item_disqualifies_the_whole_message(self):
        """Mixed content is not partially text: the message is one thing."""
        assert weixin._is_text_only_message(
            [
                {"type": weixin.ITEM_TEXT, "text_item": {"text": "/stagea go"}},
                {"type": 999},
            ]
        ) is False

    @pytest.mark.asyncio
    async def test_an_unknown_item_type_never_reaches_the_socket(self):
        """End-to-end: the reviewer's exact probe, through the real intake.

        A text item carrying ``/stagea`` plus an unclassifiable item used
        to be admitted and written to the UDS.  It must now be refused
        before any connection is attempted.
        """
        adapter = _adapter(**_enabled())
        message = _message("/stagea advance")
        message["item_list"].append({"type": 999})

        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            with patch.object(bridge, "check_socket_path") as preflight:
                await adapter._process_message(message)

        chunk = adapter._send_text_chunk.await_args.kwargs["chunk"]
        assert "media_not_admitted" in chunk
        assert _dispatched(adapter) == 0
        opened.assert_not_called()
        preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_failed_image_download_is_still_an_attachment(self, monkeypatch):
        """The exact case the reviewer reproduced."""
        adapter = _adapter(**_enabled())

        async def _collect(item, media_paths, media_types):
            return  # the download produced nothing

        monkeypatch.setattr(adapter, "_collect_media", _collect)
        message = _message("/stagea advance")
        message["item_list"].append({"type": weixin.ITEM_IMAGE, "image_item": {}})

        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            with patch.object(bridge, "check_socket_path") as preflight:
                await adapter._process_message(message)

        chunk = adapter._send_text_chunk.await_args.kwargs["chunk"]
        assert "media_not_admitted" in chunk
        assert _dispatched(adapter) == 0
        opened.assert_not_called()
        preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_failed_quoted_media_download_is_still_an_attachment(self, monkeypatch):
        adapter = _adapter(**_enabled())

        async def _collect(item, media_paths, media_types):
            return

        monkeypatch.setattr(adapter, "_collect_media", _collect)
        message = _message("/stagea advance")
        message["item_list"].append(
            {
                "type": weixin.ITEM_TEXT,
                "text_item": {"text": "context"},
                "ref_msg": {"message_item": {"type": weixin.ITEM_VIDEO}},
            }
        )

        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            await adapter._process_message(message)

        assert "media_not_admitted" in adapter._send_text_chunk.await_args.kwargs["chunk"]
        opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_downloaded_media_alone_is_still_an_attachment(self, monkeypatch):
        """The two readings are unioned, so neither can weaken the other."""
        adapter = _adapter(**_enabled())

        async def _collect(item, media_paths, media_types):
            media_paths.append("/tmp/photo.jpg")
            media_types.append("image/jpeg")

        monkeypatch.setattr(adapter, "_collect_media", _collect)
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            await adapter._process_message(_message("/stagea advance"))

        assert "media_not_admitted" in adapter._send_text_chunk.await_args.kwargs["chunk"]
        opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_genuinely_text_only_request_is_still_admitted(self):
        """The stricter reading must not refuse ordinary Stage-A requests."""
        adapter = _adapter(**_enabled())
        open_connection, captured = _fake_peer(_terminal)
        with _socket_layer(open_connection):
            await adapter._process_message(_message("/stagea advance"))
        assert len(captured) == 1


class TestRestartSafeIdentity:
    """A redelivered message keeps its Stage-A work identity.

    Finding 1 of the Tier-1 review: the id was random per admission, so a
    redelivery after a restart looked like new work to the controller.
    """

    @staticmethod
    def _request_id(frame):
        import json

        return json.loads(frame[4:].decode("utf-8"))["request_id"]

    @pytest.mark.asyncio
    async def test_the_same_message_survives_a_restart_with_one_identity(self):
        open_connection, captured = _fake_peer(_terminal)
        for _ in range(2):
            # A brand-new adapter each time: fresh bridge, empty guard,
            # empty dedupe cache — exactly what a restarted gateway looks
            # like.  Nothing is stubbed out.
            adapter = _adapter(**_enabled())
            with _socket_layer(open_connection):
                await adapter._process_message(_message("/stagea advance", message_id="msg-7"))

        assert len(captured) == 2
        assert self._request_id(captured[0]) == self._request_id(captured[1])

    @pytest.mark.asyncio
    async def test_distinct_messages_keep_distinct_identities(self):
        """Two Owner requests are two requests — through the real deduper.

        Finding 1 of the re-review: the adapter's content fingerprint
        collapsed two distinct ``message_id`` values with identical text
        *before* the bridge saw the second one, and the original version
        of this test replaced ``_dedup.is_duplicate`` with a constant
        false, so it could not see that.  Nothing is stubbed here: the
        adapter's own ``MessageDeduplicator`` is live for the whole run.
        """
        open_connection, captured = _fake_peer(_terminal)
        adapter = _adapter(**_enabled())
        with _socket_layer(open_connection):
            await adapter._process_message(_message("/stagea advance", message_id="msg-a"))
            await adapter._process_message(_message("/stagea advance", message_id="msg-b"))

        assert len(captured) == 2
        assert self._request_id(captured[0]) != self._request_id(captured[1])

    @pytest.mark.asyncio
    async def test_a_redelivered_message_still_converges_to_one_request(self):
        """The other half of invariant 1, and the reason the id check stays.

        Skipping the *content* fingerprint must not turn one message into
        two units of work when the platform redelivers it under its own
        id.
        """
        open_connection, captured = _fake_peer(_terminal)
        adapter = _adapter(**_enabled())
        with _socket_layer(open_connection):
            for _ in range(3):
                await adapter._process_message(
                    _message("/stagea advance", message_id="msg-same")
                )

        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_the_owner_is_answered_once_per_distinct_request(self):
        """Two requests, two replies, in the initiating conversation only."""
        open_connection, _ = _fake_peer(_terminal)
        adapter = _adapter(**_enabled())
        with _socket_layer(open_connection):
            await adapter._process_message(_message("/stagea advance", message_id="msg-a"))
            await adapter._process_message(_message("/stagea advance", message_id="msg-b"))

        assert adapter._send_text_chunk.await_count == 2
        assert {c.kwargs["chat_id"] for c in adapter._send_text_chunk.await_args_list} == {OWNER}
        assert _dispatched(adapter) == 0


class TestSharedDedupeSemanticsPreserved:
    """The Stage-A skip is local: everything else still dedupes as before.

    Invariant 1 is explicit that shared dedupe semantics must not be
    weakened globally, so each of these pins a neighbour of the exact
    Owner Stage-A DM case and proves it still collapses.
    """

    @staticmethod
    async def _send_two(adapter, text, *, sender=OWNER, mutate=None):
        """Two distinct message ids carrying identical text."""
        for message_id in ("dup-1", "dup-2"):
            message = _message(text, sender=sender, message_id=message_id)
            if mutate:
                mutate(message)
            with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True):
                await adapter._process_message(message)

    @pytest.mark.asyncio
    async def test_ordinary_owner_text_is_still_collapsed(self):
        adapter = _adapter(**_enabled())
        await self._send_two(adapter, "what is the current lane?")
        assert _dispatched(adapter) == 1

    @pytest.mark.asyncio
    async def test_a_non_owner_stagea_lookalike_is_still_collapsed(self):
        """The skip is bound to the exact Owner, not to the marker text."""
        adapter = _adapter(**_enabled())
        await self._send_two(adapter, "/stagea advance", sender=OTHER)
        assert _dispatched(adapter) == 1
        adapter._send_text_chunk.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_owner_stagea_text_in_a_group_is_still_collapsed(self):
        """The skip is bound to the direct message, not to the sender alone."""
        adapter = _adapter(**_enabled())
        adapter._group_policy = "allowlist"
        adapter._group_allow_from = ["room-1"]

        def into_a_room(message):
            message["room_id"] = "room-1"

        await self._send_two(adapter, "/stagea advance", mutate=into_a_room)
        assert _dispatched(adapter) == 1
        adapter._send_text_chunk.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_skip_is_off_when_the_bridge_is_off(self):
        """A tree with no Owner configured behaves exactly as it did before."""
        adapter = _adapter()
        await self._send_two(adapter, "/stagea advance")
        assert _dispatched(adapter) == 1

    @pytest.mark.asyncio
    async def test_a_stagea_request_does_not_poison_the_ordinary_fingerprint(self):
        """Skipping the record costs nothing, because a candidate never routes."""
        open_connection, captured = _fake_peer(_terminal)
        adapter = _adapter(**_enabled())
        with _socket_layer(open_connection):
            await adapter._process_message(_message("/stagea advance", message_id="msg-a"))

        # Same sender, same text, ordinary routing — unaffected either way.
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True):
            await adapter._process_message(_message("plain follow-up", message_id="msg-b"))

        assert len(captured) == 1
        assert _dispatched(adapter) == 1

    @pytest.mark.asyncio
    async def test_a_message_without_an_id_creates_no_work(self):
        adapter = _adapter(**_enabled())
        message = _message("/stagea advance")
        message["message_id"] = ""
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            with patch.object(bridge, "check_socket_path") as preflight:
                await adapter._process_message(message)

        chunk = adapter._send_text_chunk.await_args.kwargs["chunk"]
        assert "unstable_message_identity" in chunk
        assert _dispatched(adapter) == 0
        opened.assert_not_called()
        preflight.assert_not_called()


class TestBrokenConfigurationKeepsOrdinaryRouting:
    """A broken Stage-A deployment must be invisible to everyone else.

    Finding 2 of the Tier-1 review: an invalid configuration answered any
    marker-bearing text before the sender and chat type were classified,
    so it consumed non-Owner and group traffic.
    """

    @staticmethod
    def _adapter_with_broken_config():
        return _adapter(
            **{bridge.CONFIG_OWNER_USER_ID: OWNER, bridge.CONFIG_SOCKET_UID: "not-a-uid"}
        )

    @pytest.mark.asyncio
    async def test_a_non_owner_is_still_routed_normally(self):
        adapter = self._adapter_with_broken_config()
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            await adapter._process_message(_message("/stagea advance", sender=OTHER))
        assert _dispatched(adapter) == 1
        adapter._send_text_chunk.assert_not_awaited()
        opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_group_message_is_still_routed_normally(self):
        adapter = self._adapter_with_broken_config()
        adapter._group_policy = "allowlist"
        adapter._group_allow_from = ["room-1"]
        message = _message("/stagea advance")
        message["room_id"] = "room-1"
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            await adapter._process_message(message)
        assert _dispatched(adapter) == 1
        adapter._send_text_chunk.assert_not_awaited()
        opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_ordinary_owner_text_is_still_routed_normally(self):
        adapter = self._adapter_with_broken_config()
        await adapter._process_message(_message("what is the current lane?"))
        assert _dispatched(adapter) == 1
        adapter._send_text_chunk.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_only_the_owner_stagea_candidate_is_told(self):
        adapter = self._adapter_with_broken_config()
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            await adapter._process_message(_message("/stagea advance"))
        assert _dispatched(adapter) == 0
        assert "config_invalid" in adapter._send_text_chunk.await_args.kwargs["chunk"]
        opened.assert_not_called()


class TestDeliveryPath:
    @pytest.mark.asyncio
    async def test_a_reply_is_delivered_as_text_not_as_a_file(self, tmp_path):
        """A reply must be text, never a file delivery instruction.

        ``send()`` pulls ``MEDIA:`` tags and bare local paths out of the
        content it is handed, so routing a peer-supplied string through it
        would turn a reply into a file upload.  The peer answers with
        exactly that bait — a real, existing local path, both tagged and
        bare — and it has to come back out as characters.
        """
        bait = tmp_path / "not-for-delivery.txt"
        bait.write_text("private")
        payload = f"MEDIA:{bait}\n{bait}"

        adapter = _adapter(**_enabled())
        adapter.send = AsyncMock()
        adapter.send_document = AsyncMock()
        adapter.send_image_file = AsyncMock()
        adapter.send_image = AsyncMock()
        adapter._send_file = AsyncMock()

        open_connection, _ = _fake_peer(lambda request: _terminal(request, text=payload))
        with _socket_layer(open_connection):
            await adapter._process_message(_message("/stagea advance"))

        chunk = adapter._send_text_chunk.await_args.kwargs["chunk"]
        assert str(bait) in chunk
        adapter.send.assert_not_awaited()
        adapter.send_document.assert_not_awaited()
        adapter.send_image_file.assert_not_awaited()
        adapter.send_image.assert_not_awaited()
        adapter._send_file.assert_not_awaited()

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
    @pytest.mark.asyncio
    async def test_the_seam_opens_no_transport_but_the_one_unix_socket(self):
        """A whole request performs one UDS connection and one text send.

        Every other way a second destination could exist — a TCP socket, a
        listener, an HTTP call, a shelled-out command — is armed to fail
        the test if the seam reaches for it.
        """
        adapter = _adapter(**_enabled())
        open_connection, captured = _fake_peer(_terminal)
        opened = []

        async def counting_open(path):
            opened.append(path)
            return await open_connection(path)

        detonate = Mock(side_effect=AssertionError("second transport opened"))
        detonate_async = AsyncMock(side_effect=AssertionError("second transport opened"))
        with patch.object(asyncio, "open_connection", detonate_async, create=True):
            with patch.object(asyncio, "start_unix_server", detonate_async, create=True):
                with patch("subprocess.run", detonate):
                    with patch("subprocess.Popen", detonate):
                        with patch("urllib.request.urlopen", detonate):
                            with _socket_layer(counting_open):
                                await adapter._process_message(_message("/stagea advance"))

        assert opened == [bridge.DEFAULT_SOCKET_PATH]
        assert len(captured) == 1
        assert adapter._send_text_chunk.await_count == 1
        assert adapter._send_text_chunk.await_args.kwargs["chat_id"] == OWNER

    @pytest.mark.asyncio
    async def test_a_dead_controller_produces_a_refusal_not_a_fallback(self):
        """There is no second channel to fall back to, by construction."""
        adapter = _adapter(**_enabled())
        adapter.send = AsyncMock()
        detonate_async = AsyncMock(side_effect=AssertionError("second transport opened"))
        with patch.object(asyncio, "open_connection", detonate_async, create=True):
            with patch("os.stat", side_effect=FileNotFoundError()):
                await adapter._process_message(_message("/stagea advance"))

        assert adapter._send_text_chunk.await_count == 1
        assert "socket_unavailable" in adapter._send_text_chunk.await_args.kwargs["chunk"]
        assert adapter._send_text_chunk.await_args.kwargs["chat_id"] == OWNER
        adapter.send.assert_not_awaited()
        assert _dispatched(adapter) == 0

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
            with _socket_layer(open_connection):
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


class TestNativeConfiguration:
    """The three Stage-A settings come from ``config.yaml``, not the environment.

    Finding 3 of the re-review: they existed only as newly invented
    ``HERMES_*`` environment variables.  The repository contract reserves
    ``.env`` for credentials and puts behavioural settings in
    ``config.yaml``; which Owner, which socket and whose uid are
    behaviour, so they belong in this platform's own ``extra`` block.

    Every adapter here is built through ``PlatformConfig.from_dict`` —
    the function the gateway's YAML loader itself calls — so these prove
    the real loading path, not a hand-assembled dataclass.
    """

    #: The ambient names the correction removed.
    _REMOVED_ENV = {
        "HERMES_STAGEA_OWNER_WEIXIN_USER_ID": OWNER,
        "HERMES_STAGEA_BRIDGE_SOCKET": "/run/dyhano-stagea/from-the-environment.sock",
        "HERMES_STAGEA_BRIDGE_SOCKET_UID": str(CONTROLLER_UID),
    }

    @staticmethod
    def _from_yaml(extra) -> Any:
        """An adapter built the way the loader builds one from ``config.yaml``."""
        adapter = cast(
            Any,
            WeixinAdapter(
                PlatformConfig.from_dict(
                    {
                        "enabled": True,
                        "token": "test-token",
                        "extra": {
                            "account_id": ACCOUNT,
                            "dm_policy": "allowlist",
                            "allow_from": [OWNER, OTHER],
                            **extra,
                        },
                    }
                )
            ),
        )
        adapter._poll_session = object()
        adapter._send_session = object()
        adapter.handle_message = AsyncMock()
        adapter._enqueue_text_event = Mock()
        adapter._send_text_chunk = AsyncMock()
        return adapter

    def test_the_owner_binding_comes_from_the_platform_config(self):
        assert self._from_yaml({})._stagea_bridge.enabled is False
        assert self._from_yaml(_enabled())._stagea_bridge.enabled is True

    def test_the_environment_cannot_enable_the_bridge(self):
        """The ambient names are gone: setting them changes nothing at all."""
        with patch.dict(os.environ, self._REMOVED_ENV, clear=False):
            assert self._from_yaml({})._stagea_bridge.enabled is False

    @pytest.mark.asyncio
    async def test_the_environment_cannot_redirect_a_configured_bridge(self):
        """A configured socket cannot be moved by an ambient variable."""
        adapter = self._from_yaml(_enabled())
        open_connection, _ = _fake_peer(_terminal)
        opened = []

        async def counting_open(path):
            opened.append(path)
            return await open_connection(path)

        with patch.dict(os.environ, self._REMOVED_ENV, clear=False):
            with _socket_layer(counting_open):
                await adapter._process_message(_message("/stagea advance"))

        assert opened == [bridge.DEFAULT_SOCKET_PATH]

    @pytest.mark.asyncio
    async def test_the_configured_socket_path_is_the_one_connected_to(self):
        host_path = "/run/dyhano-stagea/host-compatible.sock"
        adapter = self._from_yaml(_enabled(**{bridge.CONFIG_SOCKET_PATH: host_path}))
        open_connection, _ = _fake_peer(_terminal)
        opened = []

        async def counting_open(path):
            opened.append(path)
            return await open_connection(path)

        with _socket_layer(counting_open):
            await adapter._process_message(_message("/stagea advance"))

        assert opened == [host_path]

    @pytest.mark.parametrize("raw_uid", [CONTROLLER_UID, str(CONTROLLER_UID)])
    def test_yaml_native_types_are_accepted(self, raw_uid):
        """YAML yields an int for a bare number; a quoted one yields a str."""
        adapter = self._from_yaml(_enabled(**{bridge.CONFIG_SOCKET_UID: raw_uid}))
        resolved = bridge.load_config(adapter._stagea_config, owner_user_id=OWNER)
        assert resolved.socket_uid == CONTROLLER_UID

    def test_a_root_owned_socket_uid_is_not_dropped(self):
        """``0`` is a legitimate uid and must not be read as "unset"."""
        adapter = self._from_yaml(_enabled(**{bridge.CONFIG_SOCKET_UID: 0}))
        resolved = bridge.load_config(adapter._stagea_config, owner_user_id=OWNER)
        assert resolved.socket_uid == 0

    def test_an_enabled_bridge_without_a_uid_still_fails_closed(self):
        adapter = self._from_yaml({bridge.CONFIG_OWNER_USER_ID: OWNER})
        with pytest.raises(bridge.BridgeError) as caught:
            bridge.load_config(adapter._stagea_config, owner_user_id=OWNER)
        assert caught.value.code == "config_invalid"

    def test_one_profiles_binding_does_not_reach_another(self):
        """``extra`` belongs to one profile, so the binding is scoped for free.

        This is what an ambient variable could not do: a process-wide name
        is visible to every profile in the process.
        """
        configured = self._from_yaml(_enabled())
        secondary = self._from_yaml({})
        other_owner = self._from_yaml(_enabled(**{bridge.CONFIG_OWNER_USER_ID: OTHER}))

        assert configured._stagea_bridge.enabled is True
        assert secondary._stagea_bridge.enabled is False
        assert configured._stagea_bridge.is_owner_candidate(
            chat_type="dm", sender_id=OWNER, text="/stagea go"
        ) is True
        assert other_owner._stagea_bridge.is_owner_candidate(
            chat_type="dm", sender_id=OWNER, text="/stagea go"
        ) is False
