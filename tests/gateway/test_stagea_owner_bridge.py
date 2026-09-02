"""Falsifiers for the Stage-A Owner bridge.

Each test states a boundary the bridge must hold even when the peer on the
other end of the socket is hostile: only the Owner gets in, only the
initiating conversation gets an answer, and nothing on the wire can widen
what the bridge will do.
"""

import asyncio
import json
import os
import stat
import struct
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway import stagea_owner_bridge as bridge
from gateway.stagea_owner_bridge import (
    BridgeConfig,
    BridgeError,
    ReplayGuard,
    StageAOwnerBridge,
    build_request,
    check_socket_path,
    classify,
    conversation_ref,
    encode_frame,
    load_config,
    read_frame,
    replay_key,
    validate_reply,
)

OWNER = "owner-user-id"
SOCKET = "/run/dyhano-stagea/owner-bridge.sock"
CONFIG = BridgeConfig(owner_user_id=OWNER, socket_path=SOCKET, socket_uid=None)


def _getter(values):
    def get(name, default=None):
        return values.get(name, default)

    return get


def _reader_for(payload):
    """A StreamReader preloaded with one encoded frame."""
    reader = asyncio.StreamReader()
    reader.feed_data(encode_frame(payload))
    reader.feed_eof()
    return reader


def _reply(request_id, ref, *, outcome="ACCEPTED_TERMINAL", text="done", **overrides):
    payload = {
        "schema": bridge.SCHEMA,
        "protocol": bridge.PROTOCOL,
        "type": bridge.REPLY_TYPE,
        "request_id": request_id,
        "conversation_ref": ref,
        "outcome": outcome,
        "text": text,
    }
    payload.update(overrides)
    return payload


class TestConfiguration:
    def test_absent_owner_id_disables_the_bridge(self):
        assert load_config(_getter({})) is None
        assert load_config(_getter({bridge.ENV_OWNER_USER_ID: "   "})) is None

    def test_owner_id_enables_with_the_conceptual_socket_default(self):
        config = load_config(_getter({bridge.ENV_OWNER_USER_ID: OWNER}))
        assert config == BridgeConfig(
            owner_user_id=OWNER,
            socket_path=bridge.DEFAULT_SOCKET_PATH,
            socket_uid=None,
        )

    def test_socket_path_is_host_compatible(self):
        config = load_config(
            _getter(
                {
                    bridge.ENV_OWNER_USER_ID: OWNER,
                    bridge.ENV_SOCKET_PATH: "/tmp/hostpath/owner-bridge.sock",
                }
            )
        )
        assert config is not None
        assert config.socket_path == "/tmp/hostpath/owner-bridge.sock"

    @pytest.mark.parametrize("raw", ["not-a-number", "-1", "1.5"])
    def test_unusable_expected_uid_fails_closed(self, raw):
        with pytest.raises(BridgeError) as excinfo:
            load_config(
                _getter({bridge.ENV_OWNER_USER_ID: OWNER, bridge.ENV_SOCKET_UID: raw})
            )
        assert excinfo.value.code == "config_invalid"

    @pytest.mark.asyncio
    async def test_broken_configuration_never_looks_like_disabled_to_the_owner(self):
        instance = StageAOwnerBridge(
            secret_getter=_getter(
                {bridge.ENV_OWNER_USER_ID: OWNER, bridge.ENV_SOCKET_UID: "nope"}
            ),
            channel="weixin",
        )
        assert instance.enabled is False
        reply = await instance.process(
            chat_type="dm",
            sender_id=OWNER,
            text="/stagea advance the lane",
            has_media=False,
            conversation_key="k",
            message_id="m1",
        )
        assert reply is not None
        assert "config_invalid" in reply


class TestAdmission:
    """Only an Owner direct message carrying the marker is ever consumed."""

    def test_disabled_bridge_consumes_nothing(self):
        assert not classify(
            None, chat_type="dm", sender_id=OWNER, text="/stagea go", has_media=False
        ).consumed

    def test_non_owner_sender_is_not_admitted(self):
        decision = classify(
            CONFIG, chat_type="dm", sender_id="someone-else", text="/stagea go", has_media=False
        )
        assert not decision.consumed
        assert not decision.admitted

    def test_missing_sender_is_not_admitted(self):
        assert not classify(
            CONFIG, chat_type="dm", sender_id=None, text="/stagea go", has_media=False
        ).consumed

    @pytest.mark.parametrize("chat_type", ["group", "channel", "thread"])
    def test_only_direct_messages_are_admitted(self, chat_type):
        assert not classify(
            CONFIG, chat_type=chat_type, sender_id=OWNER, text="/stagea go", has_media=False
        ).consumed

    @pytest.mark.parametrize(
        "text",
        [
            "please advance the lane",
            "later /stagea go",
            "/stageant go",
            "/stage a go",
            "[Voice transcription provided by Weixin]\n/stagea go",
        ],
    )
    def test_ordinary_text_is_untouched(self, text):
        assert not classify(
            CONFIG, chat_type="dm", sender_id=OWNER, text=text, has_media=False
        ).consumed

    @pytest.mark.parametrize("text", ["/stagea go", "  /stagea go", "/STAGEA go", "/StageA\ngo"])
    def test_marker_forms_are_admitted(self, text):
        decision = classify(CONFIG, chat_type="dm", sender_id=OWNER, text=text, has_media=False)
        assert decision.admitted
        assert decision.request_text == "go"

    def test_attachments_are_refused_not_routed(self):
        decision = classify(
            CONFIG, chat_type="dm", sender_id=OWNER, text="/stagea go", has_media=True
        )
        assert decision.refused and not decision.admitted
        assert decision.reason == "media_not_admitted"

    @pytest.mark.parametrize("text", ["/stagea", "/stagea   ", "/stagea\n\n"])
    def test_empty_request_is_refused(self, text):
        decision = classify(CONFIG, chat_type="dm", sender_id=OWNER, text=text, has_media=False)
        assert decision.refused
        assert decision.reason == "empty_request"

    def test_oversized_request_is_refused_whole(self):
        payload = "x" * (bridge.MAX_REQUEST_TEXT_BYTES + 1)
        decision = classify(
            CONFIG, chat_type="dm", sender_id=OWNER, text=f"/stagea {payload}", has_media=False
        )
        assert decision.refused
        assert decision.reason == "request_too_large"
        assert decision.request_text == ""

    def test_multibyte_request_is_measured_in_bytes(self):
        payload = "字" * bridge.MAX_REQUEST_TEXT_BYTES  # 3 bytes each
        decision = classify(
            CONFIG, chat_type="dm", sender_id=OWNER, text=f"/stagea {payload}", has_media=False
        )
        assert decision.reason == "request_too_large"


class TestCorrelation:
    def test_reference_is_stable_and_hides_the_conversation(self):
        key = "weixin|acct|chat-1234|user-5678"
        ref = conversation_ref(key)
        assert ref == conversation_ref(key)
        assert len(ref) == 32
        for secret in ("chat-1234", "user-5678", "acct"):
            assert secret not in ref

    def test_distinct_conversations_do_not_collide(self):
        assert conversation_ref("weixin|a|chat-1|u") != conversation_ref("weixin|a|chat-2|u")


class TestRequestFrame:
    def test_request_carries_only_the_declared_fields(self):
        payload = build_request(
            request_id="r" * 32, ref="c" * 32, channel="weixin", text="advance"
        )
        assert set(payload) == set(bridge._REQUEST_KEYS)
        assert payload["schema"] == bridge.SCHEMA
        assert payload["type"] == bridge.REQUEST_TYPE
        assert payload["chat_type"] == "dm"
        assert payload["text"] == "advance"

    def test_request_leaks_no_credential_or_identifier(self):
        """The frame is the whole export surface — nothing else crosses."""
        sentinels = {
            "WEIXIN_TOKEN": "sentinel-weixin-token",
            "TELEGRAM_BOT_TOKEN": "sentinel-telegram-token",
            "GLM_API_KEY": "sentinel-glm-key",
            "WEIXIN_ACCOUNT_ID": "sentinel-account",
        }
        with patch.dict(os.environ, sentinels, clear=False):
            wire = json.dumps(
                build_request(
                    request_id="r" * 32,
                    ref=conversation_ref("weixin|sentinel-account|chat-1|user-1"),
                    channel="weixin",
                    text="advance the lane",
                )
            )
        for value in sentinels.values():
            assert value not in wire
        for identifier in ("chat-1", "user-1", OWNER):
            assert identifier not in wire

    def test_request_never_carries_a_destination_or_command_field(self):
        payload = build_request(request_id="r" * 32, ref="c" * 32, channel="weixin", text="go")
        for forbidden in ("chat_id", "user_id", "to", "destination", "command", "url", "path"):
            assert forbidden not in payload


class TestFraming:
    def test_round_trip(self):
        payload = {"schema": bridge.SCHEMA, "text": "hello"}
        reader = _reader_for(payload)
        assert asyncio.run(read_frame(reader)) == payload

    def test_encoder_refuses_an_oversized_payload(self):
        with pytest.raises(BridgeError) as excinfo:
            encode_frame({"text": "x" * (bridge.MAX_FRAME_BYTES + 10)})
        assert excinfo.value.code == "send_failed"

    def test_oversized_declared_length_is_rejected_before_the_body(self):
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", bridge.MAX_FRAME_BYTES + 1))
        reader.feed_data(b"{}")
        reader.feed_eof()
        with pytest.raises(BridgeError) as excinfo:
            asyncio.run(read_frame(reader))
        assert excinfo.value.code == "reply_oversized"

    def test_zero_length_frame_is_rejected(self):
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", 0))
        reader.feed_eof()
        with pytest.raises(BridgeError) as excinfo:
            asyncio.run(read_frame(reader))
        assert excinfo.value.code == "reply_oversized"

    @pytest.mark.parametrize(
        "raw, code",
        [
            (b"\x00\x00", "reply_truncated"),
            (struct.pack(">I", 16) + b"short", "reply_truncated"),
            (struct.pack(">I", 5) + b"notjs", "reply_malformed"),
            (struct.pack(">I", 2) + b"[]", "reply_malformed"),
            (struct.pack(">I", 2) + b"\xff\xfe", "reply_malformed"),
        ],
    )
    def test_malformed_frames_fail_closed(self, raw, code):
        reader = asyncio.StreamReader()
        reader.feed_data(raw)
        reader.feed_eof()
        with pytest.raises(BridgeError) as excinfo:
            asyncio.run(read_frame(reader))
        assert excinfo.value.code == code


class TestReplyValidation:
    RID = "r" * 32
    REF = "c" * 32

    def test_valid_reply(self):
        outcome, text = validate_reply(
            _reply(self.RID, self.REF), request_id=self.RID, ref=self.REF
        )
        assert (outcome, text) == ("ACCEPTED_TERMINAL", "done")

    @pytest.mark.parametrize("outcome", list(bridge.REPLY_OUTCOMES))
    def test_every_admitted_outcome_is_accepted(self, outcome):
        got, _ = validate_reply(
            _reply(self.RID, self.REF, outcome=outcome), request_id=self.RID, ref=self.REF
        )
        assert got == outcome

    def test_unknown_outcome_is_rejected(self):
        with pytest.raises(BridgeError) as excinfo:
            validate_reply(
                _reply(self.RID, self.REF, outcome="MERGED"), request_id=self.RID, ref=self.REF
            )
        assert excinfo.value.code == "reply_bad_outcome"

    def test_reply_for_another_request_is_rejected(self):
        with pytest.raises(BridgeError) as excinfo:
            validate_reply(_reply("z" * 32, self.REF), request_id=self.RID, ref=self.REF)
        assert excinfo.value.code == "reply_wrong_request"

    def test_reply_for_another_conversation_is_rejected(self):
        with pytest.raises(BridgeError) as excinfo:
            validate_reply(_reply(self.RID, "z" * 32), request_id=self.RID, ref=self.REF)
        assert excinfo.value.code == "reply_wrong_conversation"

    @pytest.mark.parametrize(
        "field",
        ["chat_id", "to", "destination", "platform", "media", "command", "url", "notify"],
    )
    def test_reply_cannot_smuggle_an_extra_field(self, field):
        """An unknown key is refused before any value in the frame is read."""
        with pytest.raises(BridgeError) as excinfo:
            validate_reply(
                _reply(self.RID, self.REF, **{field: "anything"}),
                request_id=self.RID,
                ref=self.REF,
            )
        assert excinfo.value.code == "reply_unexpected_field"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"schema": "other.schema"},
            {"protocol": 2},
            {"type": "owner_request"},
        ],
    )
    def test_wrong_envelope_is_rejected(self, overrides):
        with pytest.raises(BridgeError) as excinfo:
            validate_reply(
                _reply(self.RID, self.REF, **overrides), request_id=self.RID, ref=self.REF
            )
        assert excinfo.value.code == "reply_malformed"

    @pytest.mark.parametrize("text", ["", "   ", 42, None, "x" * (bridge.MAX_REPLY_TEXT_CHARS + 1)])
    def test_unusable_reply_text_is_rejected(self, text):
        with pytest.raises(BridgeError) as excinfo:
            validate_reply(
                _reply(self.RID, self.REF, text=text), request_id=self.RID, ref=self.REF
            )
        assert excinfo.value.code == "reply_bad_text"

    def test_accepted_reply_stays_within_one_message(self):
        """Prefix plus the largest admitted reply must not need splitting."""
        from gateway.platforms.weixin import WeixinAdapter

        largest = bridge.outcome_text("ACCEPTED_TERMINAL", "x" * bridge.MAX_REPLY_TEXT_CHARS)
        assert len(largest) < WeixinAdapter._SPLIT_THRESHOLD


class TestSocketPreflight:
    """A local attacker must not be able to substitute the socket."""

    @staticmethod
    def _stat(mode, uid=0):
        return os.stat_result((mode, 0, 0, 1, uid, 0, 0, 0, 0, 0))

    def _run(self, sock_mode, dir_mode=0o40755, uid=0, expected_uid=None):
        results = [self._stat(sock_mode, uid), self._stat(dir_mode)]
        with patch("os.stat", side_effect=results):
            check_socket_path(SOCKET, expected_uid)

    def test_group_writable_socket_is_accepted(self):
        """Group-writable is required: the gateway is a separate identity."""
        self._run(stat.S_IFSOCK | 0o660)

    def test_missing_socket_fails_closed(self):
        with patch("os.stat", side_effect=FileNotFoundError()):
            with pytest.raises(BridgeError) as excinfo:
                check_socket_path(SOCKET, None)
        assert excinfo.value.code == "socket_unavailable"

    def test_regular_file_is_not_a_socket(self):
        with pytest.raises(BridgeError) as excinfo:
            self._run(stat.S_IFREG | 0o660)
        assert excinfo.value.code == "socket_not_a_socket"

    def test_world_writable_socket_is_refused(self):
        with pytest.raises(BridgeError) as excinfo:
            self._run(stat.S_IFSOCK | 0o666)
        assert excinfo.value.code == "socket_world_writable"

    def test_world_writable_directory_is_refused(self):
        with pytest.raises(BridgeError) as excinfo:
            self._run(stat.S_IFSOCK | 0o660, dir_mode=0o40777)
        assert excinfo.value.code == "socket_dir_world_writable"

    def test_unexpected_owner_is_refused(self):
        with pytest.raises(BridgeError) as excinfo:
            self._run(stat.S_IFSOCK | 0o660, uid=1234, expected_uid=999)
        assert excinfo.value.code == "socket_owner_mismatch"

    def test_expected_owner_is_accepted(self):
        self._run(stat.S_IFSOCK | 0o660, uid=999, expected_uid=999)


class TestReplayGuard:
    def test_first_use_wins_and_repeats_lose(self):
        guard = ReplayGuard(clock=lambda: 100.0)
        assert guard.check_and_record("k") is True
        assert guard.check_and_record("k") is False

    def test_entries_expire(self):
        now = [100.0]
        guard = ReplayGuard(ttl_seconds=10.0, clock=lambda: now[0])
        assert guard.check_and_record("k") is True
        now[0] = 200.0
        assert guard.check_and_record("k") is True

    def test_capacity_is_bounded(self):
        guard = ReplayGuard(capacity=4, clock=lambda: 100.0)
        for i in range(50):
            guard.check_and_record(f"k{i}")
        assert len(guard._seen) <= 4

    def test_key_prefers_message_id_and_is_conversation_scoped(self):
        assert replay_key("ref-a", "m1", "text") == "ref-a|id:m1"
        assert replay_key("ref-a", "m1", "text") != replay_key("ref-b", "m1", "text")

    def test_key_falls_back_to_content(self):
        assert replay_key("ref", None, "text").startswith("ref|tx:")
        assert replay_key("ref", None, "a") != replay_key("ref", None, "b")


class TestExchange:
    """End-to-end behaviour of ``process`` with the socket mocked out."""

    @staticmethod
    def _bridge():
        return StageAOwnerBridge(
            secret_getter=_getter({bridge.ENV_OWNER_USER_ID: OWNER}), channel="weixin"
        )

    @staticmethod
    def _connection(reply_builder=None):
        """Patch the socket layer: capture the request, answer it in kind."""
        builder = reply_builder or (lambda p: _reply(p["request_id"], p["conversation_ref"]))
        captured = []

        async def open_connection(path):
            reader = asyncio.StreamReader()
            writer = Mock()
            writer.drain = AsyncMock()
            writer.wait_closed = AsyncMock()

            def on_write(frame):
                captured.append(frame)
                request = json.loads(frame[4:].decode("utf-8"))
                reader.feed_data(encode_frame(builder(request)))
                reader.feed_eof()

            writer.write = on_write
            return reader, writer

        return open_connection, captured

    async def _run(self, instance, *, text="/stagea advance", message_id="m1", **kwargs):
        return await instance.process(
            chat_type="dm",
            sender_id=OWNER,
            text=text,
            has_media=False,
            conversation_key="weixin|acct|chat-1|owner-user-id",
            message_id=message_id,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_accepted_terminal_reaches_the_owner(self):
        open_connection, captured = self._connection()
        with patch("os.stat", return_value=os.stat_result(
            (stat.S_IFSOCK | 0o660, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        )):
            with patch.object(asyncio, "open_unix_connection", open_connection, create=True):
                reply = await self._run(self._bridge())
        assert reply == "[Stage-A] ACCEPTED_TERMINAL\ndone"
        request = json.loads(captured[0][4:].decode("utf-8"))
        assert request["text"] == "advance"
        assert request["type"] == bridge.REQUEST_TYPE

    @pytest.mark.asyncio
    async def test_missing_socket_answers_the_owner_and_creates_nothing(self):
        instance = self._bridge()
        with patch("os.stat", side_effect=FileNotFoundError()):
            reply = await self._run(instance)
        assert "socket_unavailable" in reply
        assert "No Stage-A work was created." in reply

    @pytest.mark.asyncio
    async def test_refused_connection_fails_closed(self):
        async def refuse(path):
            raise ConnectionRefusedError()

        with patch("os.stat", return_value=os.stat_result(
            (stat.S_IFSOCK | 0o660, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        )):
            with patch.object(asyncio, "open_unix_connection", refuse, create=True):
                reply = await self._run(self._bridge())
        assert "connect_failed" in reply

    @pytest.mark.asyncio
    async def test_duplicate_message_does_not_reach_the_socket_twice(self):
        instance = self._bridge()
        calls = []

        open_connection, _ = self._connection()

        async def counting(path):
            calls.append(path)
            return await open_connection(path)

        with patch("os.stat", return_value=os.stat_result(
            (stat.S_IFSOCK | 0o660, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        )):
            with patch.object(asyncio, "open_unix_connection", counting, create=True):
                first = await self._run(instance)
                second = await self._run(instance)
        assert first.startswith("[Stage-A] ACCEPTED_TERMINAL")
        assert "duplicate_request" in second
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_reply_bound_to_another_conversation_is_refused(self):
        open_connection, _ = self._connection(
            reply_builder=lambda p: _reply(p["request_id"], "f" * 32)
        )
        with patch("os.stat", return_value=os.stat_result(
            (stat.S_IFSOCK | 0o660, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        )):
            with patch.object(asyncio, "open_unix_connection", open_connection, create=True):
                reply = await self._run(self._bridge())
        assert "reply_wrong_conversation" in reply

    @pytest.mark.asyncio
    async def test_reply_carrying_a_destination_is_refused(self):
        open_connection, _ = self._connection(
            reply_builder=lambda p: _reply(
                p["request_id"], p["conversation_ref"], chat_id="attacker-chat"
            )
        )
        with patch("os.stat", return_value=os.stat_result(
            (stat.S_IFSOCK | 0o660, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        )):
            with patch.object(asyncio, "open_unix_connection", open_connection, create=True):
                reply = await self._run(self._bridge())
        assert "reply_unexpected_field" in reply
        assert "attacker-chat" not in reply

    @pytest.mark.asyncio
    async def test_slow_peer_becomes_an_unresolved_answer_not_silence(self):
        async def never_answers(path):
            reader = asyncio.StreamReader()  # never fed
            writer = Mock()
            writer.drain = AsyncMock()
            writer.wait_closed = AsyncMock()
            return reader, writer

        with patch("os.stat", return_value=os.stat_result(
            (stat.S_IFSOCK | 0o660, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        )):
            with patch.object(asyncio, "open_unix_connection", never_answers, create=True):
                with patch.object(bridge, "REPLY_DEADLINE_SECONDS", 0.05):
                    reply = await self._run(self._bridge())
        assert "reply_timeout" in reply

    @pytest.mark.asyncio
    async def test_inflight_requests_are_capped(self):
        instance = self._bridge()
        instance._inflight = bridge.MAX_INFLIGHT_REQUESTS
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            reply = await self._run(instance)
        assert "too_many_inflight" in reply
        opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_owner_never_opens_the_socket(self):
        instance = self._bridge()
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            with patch("os.stat") as stat_call:
                reply = await instance.process(
                    chat_type="dm",
                    sender_id="not-the-owner",
                    text="/stagea advance",
                    has_media=False,
                    conversation_key="weixin|acct|chat-9|not-the-owner",
                    message_id="m9",
                )
        assert reply is None
        opened.assert_not_called()
        stat_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_bridge_never_touches_the_socket(self):
        instance = StageAOwnerBridge(secret_getter=_getter({}), channel="weixin")
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            reply = await self._run(instance)
        assert reply is None
        opened.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsupported_platform_fails_closed(self):
        instance = self._bridge()
        with patch.object(asyncio, "open_unix_connection", None, create=True):
            reply = await self._run(instance)
        assert "platform_unsupported" in reply


class TestNoAlternateDestination:
    """The bridge owns exactly one delivery idea: reply where it was asked."""

    def test_module_names_no_other_channel_or_transport(self):
        source = open(bridge.__file__, encoding="utf-8").read().lower()
        for forbidden in (
            "telegram",
            "discord",
            "slack",
            "webhook",
            "outbox",
            "broadcast",
            "http://",
            "https://",
            "subprocess",
            "create_unix_server",
            "start_server",
            "socket.bind",
        ):
            assert forbidden not in source, f"bridge must not reference {forbidden}"

    def test_module_declares_no_listener_api(self):
        for forbidden in ("serve", "listen", "bind", "accept"):
            assert not hasattr(bridge, forbidden)
