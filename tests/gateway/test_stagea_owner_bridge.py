"""Falsifiers for the Stage-A Owner bridge.

Each test states a boundary the bridge must hold even when the peer on the
other end of the socket is hostile: only the Owner gets in, only the
initiating conversation gets an answer, and nothing on the wire can widen
what the bridge will do.
"""

import asyncio
import contextlib
import json
import os
import shutil
import stat
import struct
import tempfile
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway import stagea_owner_bridge as bridge
from gateway.stagea_owner_bridge import (
    BridgeConfig,
    BridgeError,
    ReplayGuard,
    StageAOwnerBridge,
    build_request,
    check_socket_identity,
    check_socket_path,
    classify,
    conversation_ref,
    derive_request_id,
    encode_frame,
    load_config,
    load_owner_user_id,
    read_frame,
    validate_reply,
    verify_connected_peer,
)

OWNER = "owner-user-id"
SOCKET = "/run/dyhano-stagea/owner-bridge.sock"
SOCKET_DIR = os.path.dirname(SOCKET)
CONTROLLER_UID = 4242
CONFIG = BridgeConfig(owner_user_id=OWNER, socket_path=SOCKET, socket_uid=CONTROLLER_UID)

#: The configuration an enabled bridge needs now that an exact expected
#: controller uid is mandatory rather than optional.
ENABLED_ENV = {
    bridge.ENV_OWNER_USER_ID: OWNER,
    bridge.ENV_SOCKET_UID: str(CONTROLLER_UID),
}


def _getter(values):
    def get(name, default=None):
        return values.get(name, default)

    return get


def _stat(mode, *, uid=CONTROLLER_UID, ino=1, dev=1):
    return os.stat_result((mode, ino, dev, 1, uid, 0, 0, 0, 0, 0))


def _stat_router(*, sock=None, directory=None):
    """``os.stat`` replacement that answers per path, not per call order.

    The exchange stats the socket again after connecting, so an ordered
    ``side_effect`` list would silently depend on how many times each
    check runs.
    """
    sock_st = sock if sock is not None else _stat(stat.S_IFSOCK | 0o660, ino=101)
    dir_st = directory if directory is not None else _stat(0o40755, ino=202)

    def stat_fn(path, *args, **kwargs):
        return dir_st if str(path) == SOCKET_DIR else sock_st

    return stat_fn


@contextlib.contextmanager
def _socket_layer(open_connection, *, peer_uid=CONTROLLER_UID, stat_fn=None):
    """Patch the whole socket layer at once: stats, peer identity, connect."""
    with patch("os.stat", side_effect=stat_fn or _stat_router()):
        with patch.object(bridge, "read_peer_uid", return_value=peer_uid):
            with patch.object(asyncio, "open_unix_connection", open_connection, create=True):
                yield


def _classify(owner=OWNER, **kwargs):
    """``classify`` with the ordinary Owner Stage-A shape as the default."""
    params: Dict[str, Any] = {
        "chat_type": "dm",
        "sender_id": OWNER,
        "text": "/stagea go",
        "has_media": False,
        "message_id": "m1",
    }
    params.update(kwargs)
    return classify(owner, **params)


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
    """The primary gate and the secondary configuration are separate."""

    def test_absent_owner_id_disables_the_bridge(self):
        assert load_owner_user_id(_getter({})) is None
        assert load_owner_user_id(_getter({bridge.ENV_OWNER_USER_ID: "   "})) is None

    def test_owner_id_is_the_only_primary_gate(self):
        assert load_owner_user_id(_getter({bridge.ENV_OWNER_USER_ID: OWNER})) == OWNER

    def test_primary_gate_cannot_fail(self):
        """A bare identifier has no parse step, so it can never be 'invalid'.

        This is what keeps a broken deployment from ever reaching a
        non-Owner message: the only configuration consulted before
        classification has no failure mode at all.
        """
        for value in ("not-a-number", "-1", "1.5", "\x00", "  spaced  "):
            assert load_owner_user_id(_getter({bridge.ENV_OWNER_USER_ID: value})) is not None

    def test_secondary_config_defaults_to_the_conceptual_socket(self):
        config = load_config(_getter(ENABLED_ENV), owner_user_id=OWNER)
        assert config == BridgeConfig(
            owner_user_id=OWNER,
            socket_path=bridge.DEFAULT_SOCKET_PATH,
            socket_uid=CONTROLLER_UID,
        )

    def test_socket_path_is_host_compatible(self):
        config = load_config(
            _getter({**ENABLED_ENV, bridge.ENV_SOCKET_PATH: "/tmp/hostpath/owner-bridge.sock"}),
            owner_user_id=OWNER,
        )
        assert config.socket_path == "/tmp/hostpath/owner-bridge.sock"

    @pytest.mark.parametrize("raw", ["not-a-number", "-1", "1.5"])
    def test_unusable_expected_uid_fails_closed(self, raw):
        with pytest.raises(BridgeError) as excinfo:
            load_config(
                _getter({**ENABLED_ENV, bridge.ENV_SOCKET_UID: raw}), owner_user_id=OWNER
            )
        assert excinfo.value.code == "config_invalid"

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_expected_uid_is_mandatory(self, raw):
        """An enabled bridge that cannot name its peer must not connect."""
        with pytest.raises(BridgeError) as excinfo:
            load_config(
                _getter({bridge.ENV_OWNER_USER_ID: OWNER, bridge.ENV_SOCKET_UID: raw}),
                owner_user_id=OWNER,
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
        assert instance.enabled is True  # the primary gate is set
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


class TestAdmissionOrdering:
    """Broken secondary configuration must never widen who is consumed.

    Finding 2 of the Tier-1 review: a config failure was answered before
    the sender and chat type were classified, so an invalid deployment
    consumed non-Owner and group ``/stagea`` traffic that ordinary
    routing owns.
    """

    @staticmethod
    def _broken():
        return StageAOwnerBridge(
            secret_getter=_getter(
                {bridge.ENV_OWNER_USER_ID: OWNER, bridge.ENV_SOCKET_UID: "nope"}
            ),
            channel="weixin",
        )

    @pytest.mark.asyncio
    async def test_non_owner_dm_is_routed_ordinarily(self):
        reply = await self._broken().process(
            chat_type="dm",
            sender_id="someone-else",
            text="/stagea advance",
            has_media=False,
            conversation_key="k",
            message_id="m1",
        )
        assert reply is None

    @pytest.mark.asyncio
    async def test_owner_in_a_group_is_routed_ordinarily(self):
        reply = await self._broken().process(
            chat_type="group",
            sender_id=OWNER,
            text="/stagea advance",
            has_media=False,
            conversation_key="k",
            message_id="m1",
        )
        assert reply is None

    @pytest.mark.asyncio
    async def test_owner_without_the_marker_is_routed_ordinarily(self):
        reply = await self._broken().process(
            chat_type="dm",
            sender_id=OWNER,
            text="please advance the lane",
            has_media=False,
            conversation_key="k",
            message_id="m1",
        )
        assert reply is None

    @pytest.mark.asyncio
    async def test_only_the_exact_owner_candidate_sees_the_failure(self):
        reply = await self._broken().process(
            chat_type="dm",
            sender_id=OWNER,
            text="/stagea advance",
            has_media=False,
            conversation_key="k",
            message_id="m1",
        )
        assert "config_invalid" in reply

    @pytest.mark.asyncio
    async def test_broken_configuration_opens_no_socket_for_anyone(self):
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            with patch.object(bridge, "check_socket_path") as preflight:
                for chat_type, sender in (("dm", "someone-else"), ("group", OWNER), ("dm", OWNER)):
                    await self._broken().process(
                        chat_type=chat_type,
                        sender_id=sender,
                        text="/stagea advance",
                        has_media=False,
                        conversation_key="k",
                        message_id="m1",
                    )
        opened.assert_not_called()
        preflight.assert_not_called()


class TestAdmission:
    """Only an Owner direct message carrying the marker is ever consumed."""

    def test_disabled_bridge_consumes_nothing(self):
        assert not _classify(None).consumed

    def test_non_owner_sender_is_not_admitted(self):
        decision = _classify(sender_id="someone-else")
        assert not decision.consumed
        assert not decision.admitted

    def test_missing_sender_is_not_admitted(self):
        assert not _classify(sender_id=None).consumed

    @pytest.mark.parametrize("chat_type", ["group", "channel", "thread"])
    def test_only_direct_messages_are_admitted(self, chat_type):
        assert not _classify(chat_type=chat_type).consumed

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
        assert not _classify(text=text).consumed

    @pytest.mark.parametrize("text", ["/stagea go", "  /stagea go", "/STAGEA go", "/StageA\ngo"])
    def test_marker_forms_are_admitted(self, text):
        decision = _classify(text=text)
        assert decision.admitted
        assert decision.request_text == "go"

    def test_attachments_are_refused_not_routed(self):
        decision = _classify(has_media=True)
        assert decision.refused and not decision.admitted
        assert decision.reason == "media_not_admitted"

    @pytest.mark.parametrize("text", ["/stagea", "/stagea   ", "/stagea\n\n"])
    def test_empty_request_is_refused(self, text):
        decision = _classify(text=text)
        assert decision.refused
        assert decision.reason == "empty_request"

    def test_oversized_request_is_refused_whole(self):
        payload = "x" * (bridge.MAX_REQUEST_TEXT_BYTES + 1)
        decision = _classify(text=f"/stagea {payload}")
        assert decision.refused
        assert decision.reason == "request_too_large"
        assert decision.request_text == ""

    def test_multibyte_request_is_measured_in_bytes(self):
        payload = "\u5b57" * bridge.MAX_REQUEST_TEXT_BYTES  # 3 bytes each
        decision = _classify(text=f"/stagea {payload}")
        assert decision.reason == "request_too_large"

    @pytest.mark.parametrize("message_id", [None, "", "   "])
    def test_message_without_a_stable_identity_creates_no_work(self, message_id):
        """No stable id means no restart-safe work identity, so no work."""
        decision = _classify(message_id=message_id)
        assert decision.refused and not decision.admitted
        assert decision.reason == "unstable_message_identity"

    @pytest.mark.parametrize("message_id", [None, ""])
    def test_missing_identity_still_does_not_consume_a_non_owner(self, message_id):
        assert not _classify(sender_id="someone-else", message_id=message_id).consumed


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

    def _run(self, sock_mode, dir_mode=0o40755, uid=CONTROLLER_UID, dir_uid=CONTROLLER_UID,
             expected_uid=CONTROLLER_UID):
        with patch(
            "os.stat",
            side_effect=_stat_router(
                sock=_stat(sock_mode, uid=uid, ino=101),
                directory=_stat(dir_mode, uid=dir_uid, ino=202),
            ),
        ):
            return check_socket_path(SOCKET, expected_uid)

    def test_group_writable_socket_is_accepted(self):
        """Group-writable is required: the gateway is a separate identity."""
        assert self._run(stat.S_IFSOCK | 0o660).st_ino == 101

    def test_preflight_returns_the_identity_it_checked(self):
        """The caller needs the preimage to prove the peer did not change."""
        result = self._run(stat.S_IFSOCK | 0o660)
        assert (result.st_dev, result.st_ino) == (1, 101)

    def test_missing_socket_fails_closed(self):
        with patch("os.stat", side_effect=FileNotFoundError()):
            with pytest.raises(BridgeError) as excinfo:
                check_socket_path(SOCKET, CONTROLLER_UID)
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

    def test_group_writable_directory_is_refused(self):
        """Connecting needs search on the directory, never write.

        Finding 4: a group-writable directory lets any member of that
        group unlink the socket and put their own in its place.
        """
        with pytest.raises(BridgeError) as excinfo:
            self._run(stat.S_IFSOCK | 0o660, dir_mode=0o40775)
        assert excinfo.value.code == "socket_dir_group_writable"

    def test_directory_owned_by_a_stranger_is_refused(self):
        with pytest.raises(BridgeError) as excinfo:
            self._run(stat.S_IFSOCK | 0o660, dir_uid=CONTROLLER_UID + 1)
        assert excinfo.value.code == "socket_dir_owner_mismatch"

    def test_directory_owned_by_root_is_accepted(self):
        assert self._run(stat.S_IFSOCK | 0o660, dir_uid=0) is not None

    def test_unexpected_socket_owner_is_refused(self):
        with pytest.raises(BridgeError) as excinfo:
            self._run(stat.S_IFSOCK | 0o660, uid=1234, expected_uid=999, dir_uid=999)
        assert excinfo.value.code == "socket_owner_mismatch"

    def test_expected_owner_is_accepted(self):
        assert self._run(stat.S_IFSOCK | 0o660, uid=999, expected_uid=999, dir_uid=999) is not None


class TestSocketIdentity:
    """The object that was checked must be the object that was connected."""

    def test_same_inode_passes(self):
        preimage = _stat(stat.S_IFSOCK | 0o660, ino=101, dev=1)
        with patch("os.stat", side_effect=_stat_router(sock=preimage)):
            check_socket_identity(SOCKET, preimage)

    @pytest.mark.parametrize(
        "swapped",
        [
            _stat(stat.S_IFSOCK | 0o660, ino=999, dev=1),
            _stat(stat.S_IFSOCK | 0o660, ino=101, dev=9),
        ],
    )
    def test_a_swapped_socket_is_refused(self, swapped):
        preimage = _stat(stat.S_IFSOCK | 0o660, ino=101, dev=1)
        with patch("os.stat", side_effect=_stat_router(sock=swapped)):
            with pytest.raises(BridgeError) as excinfo:
                check_socket_identity(SOCKET, preimage)
        assert excinfo.value.code == "socket_replaced"

    def test_a_vanished_socket_is_refused(self):
        preimage = _stat(stat.S_IFSOCK | 0o660, ino=101)
        with patch("os.stat", side_effect=FileNotFoundError()):
            with pytest.raises(BridgeError) as excinfo:
                check_socket_identity(SOCKET, preimage)
        assert excinfo.value.code == "socket_replaced"


class TestConnectedPeerConfinement:
    """Only the expected controller process may receive the Owner's text."""

    def test_expected_peer_is_accepted(self):
        with patch.object(bridge, "read_peer_uid", return_value=CONTROLLER_UID):
            verify_connected_peer(object(), CONTROLLER_UID)

    def test_a_stranger_on_the_other_end_is_refused(self):
        with patch.object(bridge, "read_peer_uid", return_value=CONTROLLER_UID + 1):
            with pytest.raises(BridgeError) as excinfo:
                verify_connected_peer(object(), CONTROLLER_UID)
        assert excinfo.value.code == "peer_owner_mismatch"

    def test_an_unprovable_peer_is_refused_not_assumed(self):
        """A platform that cannot answer must fail closed, not fall through."""
        with patch.object(bridge, "read_peer_uid", return_value=None):
            with pytest.raises(BridgeError) as excinfo:
                verify_connected_peer(object(), CONTROLLER_UID)
        assert excinfo.value.code == "peer_unverifiable"

    def test_a_missing_socket_object_is_unprovable(self):
        assert bridge.read_peer_uid(None) is None

    def test_a_socket_that_cannot_answer_is_unprovable(self):
        sock = Mock()
        sock.getsockopt.side_effect = OSError("unsupported")
        assert bridge.read_peer_uid(sock) is None

    def test_no_credential_option_at_all_is_unprovable(self):
        sock = Mock()
        with patch.object(bridge.socket_module, "SO_PEERCRED", None, create=True):
            with patch.object(bridge.socket_module, "LOCAL_PEERCRED", None, create=True):
                assert bridge.read_peer_uid(sock) is None

    def test_a_wrong_xucred_version_is_unprovable(self):
        """A struct this code does not understand must not be believed."""
        sock = Mock()
        sock.getsockopt.return_value = struct.pack("2I", 99, CONTROLLER_UID)
        with patch.object(bridge.socket_module, "SO_PEERCRED", None, create=True):
            with patch.object(bridge.socket_module, "LOCAL_PEERCRED", 1, create=True):
                assert bridge.read_peer_uid(sock) is None


@pytest.mark.skipif(
    not hasattr(asyncio, "open_unix_connection"), reason="Unix-domain sockets are POSIX-only"
)
class TestPeerConfinementOnARealSocket:
    """The peer checks are exercised against a real kernel, not a mock.

    Mocking ``getsockopt`` would prove only that this file agrees with
    itself.  These tests open an actual Unix socket pair so the platform
    credential readout — ``SO_PEERCRED`` on Linux, ``LOCAL_PEERCRED`` on
    the BSD/macOS family — is the thing under test.
    """

    @staticmethod
    def _tmpdir():
        # Kept short: a Unix socket path has a hard ~104-byte ceiling.
        return tempfile.mkdtemp(prefix="sa")

    @pytest.mark.asyncio
    async def test_read_peer_uid_reports_the_real_connected_process(self):
        directory = self._tmpdir()
        path = os.path.join(directory, "s")
        server = await asyncio.start_unix_server(lambda r, w: None, path=path)
        try:
            reader, writer = await asyncio.open_unix_connection(path)
            try:
                assert bridge.read_peer_uid(writer.get_extra_info("socket")) == os.getuid()
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()
            shutil.rmtree(directory, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_the_owner_reaches_a_genuine_controller_end_to_end(self):
        """Real socket, real stat, real peer credentials, real framing."""
        directory = self._tmpdir()
        path = os.path.join(directory, "s")
        received = []

        async def handle(reader, writer):
            request = await read_frame(reader)
            received.append(request)
            writer.write(
                encode_frame(_reply(request["request_id"], request["conversation_ref"]))
            )
            await writer.drain()
            writer.close()

        server = await asyncio.start_unix_server(handle, path=path)
        try:
            instance = StageAOwnerBridge(
                secret_getter=_getter(
                    {
                        bridge.ENV_OWNER_USER_ID: OWNER,
                        bridge.ENV_SOCKET_PATH: path,
                        bridge.ENV_SOCKET_UID: str(os.getuid()),
                    }
                ),
                channel="weixin",
            )
            reply = await instance.process(
                chat_type="dm",
                sender_id=OWNER,
                text="/stagea advance",
                has_media=False,
                conversation_key="weixin|acct|chat-1|owner",
                message_id="m-real",
            )
            assert reply == "[Stage-A] ACCEPTED_TERMINAL\ndone"
            assert received[0]["text"] == "advance"
        finally:
            server.close()
            await server.wait_closed()
            shutil.rmtree(directory, ignore_errors=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "peer_uid, code",
        [(None, "peer_unverifiable"), ("stranger", "peer_owner_mismatch")],
    )
    async def test_an_unproven_peer_gets_no_bytes_at_all(self, peer_uid, code):
        """The refusal happens *before* the request is written, not after.

        This is the property finding 4 turned on: a substituted listener
        must never receive the Owner's text, let alone get to answer it.
        """
        directory = self._tmpdir()
        path = os.path.join(directory, "s")
        received = []

        async def handle(reader, writer):
            data = await reader.read(1)
            received.append(data)
            writer.close()

        server = await asyncio.start_unix_server(handle, path=path)
        try:
            instance = StageAOwnerBridge(
                secret_getter=_getter(
                    {
                        bridge.ENV_OWNER_USER_ID: OWNER,
                        bridge.ENV_SOCKET_PATH: path,
                        bridge.ENV_SOCKET_UID: str(os.getuid()),
                    }
                ),
                channel="weixin",
            )
            resolved = None if peer_uid is None else os.getuid() + 1
            with patch.object(bridge, "read_peer_uid", return_value=resolved):
                reply = await instance.process(
                    chat_type="dm",
                    sender_id=OWNER,
                    text="/stagea advance",
                    has_media=False,
                    conversation_key="weixin|acct|chat-1|owner",
                    message_id="m-real",
                )
            assert reply is not None
            assert code in reply
            assert "No Stage-A work was created." in reply
            await asyncio.sleep(0.05)
            assert received in ([], [b""]), "the peer must not have received request bytes"
        finally:
            server.close()
            await server.wait_closed()
            shutil.rmtree(directory, ignore_errors=True)


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


class TestDerivedWorkIdentity:
    """One inbound message is one Stage-A work identity, forever.

    Finding 1 of the Tier-1 review: the id was random per admission and
    the only dedupe was process memory, so a redelivery after a restart
    became fresh work that no controller-side persistence could match.
    """

    def test_identity_is_derived_not_random(self):
        first = derive_request_id("ref-a", "m1")
        second = derive_request_id("ref-a", "m1")
        assert first == second

    def test_distinct_messages_stay_distinct(self):
        assert derive_request_id("ref-a", "m1") != derive_request_id("ref-a", "m2")

    def test_distinct_conversations_stay_distinct(self):
        assert derive_request_id("ref-a", "m1") != derive_request_id("ref-b", "m1")

    def test_identity_hides_the_message_id(self):
        derived = derive_request_id("ref-a", "wxid-secret-message-42")
        assert "wxid" not in derived
        assert "secret" not in derived
        assert len(derived) == 32
        assert all(c in "0123456789abcdef" for c in derived)


class TestExchange:
    """End-to-end behaviour of ``process`` with the socket mocked out."""

    @staticmethod
    def _bridge():
        return StageAOwnerBridge(secret_getter=_getter(ENABLED_ENV), channel="weixin")

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
        with _socket_layer(open_connection):
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

        with _socket_layer(refuse):
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

        with _socket_layer(counting):
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
        with _socket_layer(open_connection):
            reply = await self._run(self._bridge())
        assert "reply_wrong_conversation" in reply

    @pytest.mark.asyncio
    async def test_reply_carrying_a_destination_is_refused(self):
        open_connection, _ = self._connection(
            reply_builder=lambda p: _reply(
                p["request_id"], p["conversation_ref"], chat_id="attacker-chat"
            )
        )
        with _socket_layer(open_connection):
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

        with _socket_layer(never_answers):
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
            with patch.object(bridge, "check_socket_path") as preflight:
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
        preflight.assert_not_called()

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

    @pytest.mark.asyncio
    async def test_a_restart_does_not_turn_one_message_into_new_work(self):
        """The controller sees the same id after the bridge is rebuilt.

        A fresh instance has an empty guard — that is the restart being
        simulated — so the only thing that can still identify the
        redelivery is the derived id on the wire.
        """
        open_connection, captured = self._connection()
        with _socket_layer(open_connection):
            first = await self._run(self._bridge(), message_id="m-restart")
            second = await self._run(self._bridge(), message_id="m-restart")

        assert first.startswith("[Stage-A] ACCEPTED_TERMINAL")
        assert second.startswith("[Stage-A] ACCEPTED_TERMINAL")
        ids = [json.loads(frame[4:].decode("utf-8"))["request_id"] for frame in captured]
        assert len(ids) == 2
        assert ids[0] == ids[1], "a redelivered message must keep its work identity"

    @pytest.mark.asyncio
    async def test_distinct_messages_still_get_distinct_identities(self):
        open_connection, captured = self._connection()
        with _socket_layer(open_connection):
            await self._run(self._bridge(), message_id="m-a")
            await self._run(self._bridge(), message_id="m-b")
        ids = [json.loads(frame[4:].decode("utf-8"))["request_id"] for frame in captured]
        assert ids[0] != ids[1]

    @pytest.mark.asyncio
    async def test_distinct_conversations_still_get_distinct_identities(self):
        open_connection, captured = self._connection()
        with _socket_layer(open_connection):
            await self._run(self._bridge(), message_id="m-same")
            await self._bridge().process(
                chat_type="dm",
                sender_id=OWNER,
                text="/stagea advance",
                has_media=False,
                conversation_key="weixin|acct|chat-2|owner-user-id",
                message_id="m-same",
            )
        ids = [json.loads(frame[4:].decode("utf-8"))["request_id"] for frame in captured]
        assert ids[0] != ids[1]

    @pytest.mark.asyncio
    async def test_a_stranger_on_the_socket_never_receives_the_request(self):
        open_connection, captured = self._connection()
        with _socket_layer(open_connection, peer_uid=CONTROLLER_UID + 1):
            reply = await self._run(self._bridge())
        assert "peer_owner_mismatch" in reply
        assert captured == [], "no byte of the Owner's text may be written"

    @pytest.mark.asyncio
    async def test_an_unprovable_platform_never_sends_the_request(self):
        open_connection, captured = self._connection()
        with _socket_layer(open_connection, peer_uid=None):
            reply = await self._run(self._bridge())
        assert "peer_unverifiable" in reply
        assert captured == []

    @pytest.mark.asyncio
    async def test_a_socket_swapped_after_the_preflight_is_refused(self):
        """Bind the checked inode to the connection that was opened."""
        open_connection, captured = self._connection()
        stats = [
            _stat(stat.S_IFSOCK | 0o660, ino=101),
            _stat(0o40755, ino=202),
            _stat(stat.S_IFSOCK | 0o660, ino=777),  # re-stat after connect
        ]

        def drifting(path, *args, **kwargs):
            return stats.pop(0) if stats else _stat(stat.S_IFSOCK | 0o660, ino=777)

        with _socket_layer(open_connection, stat_fn=drifting):
            reply = await self._run(self._bridge())
        assert "socket_replaced" in reply
        assert captured == []

    @pytest.mark.asyncio
    async def test_an_enabled_bridge_without_an_expected_uid_never_connects(self):
        instance = StageAOwnerBridge(
            secret_getter=_getter({bridge.ENV_OWNER_USER_ID: OWNER}), channel="weixin"
        )
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            with patch.object(bridge, "check_socket_path") as preflight:
                reply = await self._run(instance)
        assert "config_invalid" in reply
        opened.assert_not_called()
        preflight.assert_not_called()


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
