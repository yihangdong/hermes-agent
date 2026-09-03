"""Falsifiers for the Stage-A Owner bridge.

Each test states a boundary the bridge must hold even when the peer on the
other end of the socket is hostile: only the Owner gets in, only the
initiating conversation gets an answer, and nothing on the wire can widen
what the bridge will do.
"""

import asyncio
import concurrent.futures
import contextlib
import gc
import json
import os
import shutil
import stat
import struct
import tempfile
import threading
import time
import warnings
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
    is_owner_candidate,
    load_config,
    load_owner_user_id,
    read_frame,
    validate_reply,
    verify_connected_peer,
)

OWNER = "owner-user-id"
OTHER = "someone-else-user-id"
SOCKET = "/run/dyhano-stagea/owner-bridge.sock"
SOCKET_DIR = os.path.dirname(SOCKET)
CONTROLLER_UID = 4242
CONFIG = BridgeConfig(owner_user_id=OWNER, socket_path=SOCKET, socket_uid=CONTROLLER_UID)

#: The configuration an enabled bridge needs now that an exact expected
#: controller uid is mandatory rather than optional.
ENABLED_CONFIG = {
    bridge.CONFIG_OWNER_USER_ID: OWNER,
    bridge.CONFIG_SOCKET_UID: str(CONTROLLER_UID),
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


@contextlib.contextmanager
def _reserved_cleanup():
    """One exchange's cleanup reservation, taken and released as ``process`` does.

    :class:`bridge._Deadline` is handed the reservation its exchange was
    admitted with, so a test that drives one directly has to hold a real
    one.  Borrowing the module's own ledger rather than a stand-in keeps
    these tests measuring the ceiling that ships.
    """
    budget = bridge._CleanupBudget.reserve()
    assert budget is not None, "the cleanup ceiling was already exhausted"
    try:
        yield budget
    finally:
        budget.close()


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
        assert load_owner_user_id(_getter({bridge.CONFIG_OWNER_USER_ID: "   "})) is None

    def test_owner_id_is_the_only_primary_gate(self):
        assert load_owner_user_id(_getter({bridge.CONFIG_OWNER_USER_ID: OWNER})) == OWNER

    def test_primary_gate_cannot_fail(self):
        """A bare identifier has no parse step, so it can never be 'invalid'.

        This is what keeps a broken deployment from ever reaching a
        non-Owner message: the only configuration consulted before
        classification has no failure mode at all.
        """
        for value in ("not-a-number", "-1", "1.5", "\x00", "  spaced  "):
            assert load_owner_user_id(_getter({bridge.CONFIG_OWNER_USER_ID: value})) is not None

    def test_secondary_config_defaults_to_the_conceptual_socket(self):
        config = load_config(_getter(ENABLED_CONFIG), owner_user_id=OWNER)
        assert config == BridgeConfig(
            owner_user_id=OWNER,
            socket_path=bridge.DEFAULT_SOCKET_PATH,
            socket_uid=CONTROLLER_UID,
        )

    def test_socket_path_is_host_compatible(self):
        config = load_config(
            _getter({**ENABLED_CONFIG, bridge.CONFIG_SOCKET_PATH: "/tmp/hostpath/owner-bridge.sock"}),
            owner_user_id=OWNER,
        )
        assert config.socket_path == "/tmp/hostpath/owner-bridge.sock"

    @pytest.mark.parametrize("raw", ["not-a-number", "-1", "1.5"])
    def test_unusable_expected_uid_fails_closed(self, raw):
        with pytest.raises(BridgeError) as excinfo:
            load_config(
                _getter({**ENABLED_CONFIG, bridge.CONFIG_SOCKET_UID: raw}), owner_user_id=OWNER
            )
        assert excinfo.value.code == "config_invalid"

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_expected_uid_is_mandatory(self, raw):
        """An enabled bridge that cannot name its peer must not connect."""
        with pytest.raises(BridgeError) as excinfo:
            load_config(
                _getter({bridge.CONFIG_OWNER_USER_ID: OWNER, bridge.CONFIG_SOCKET_UID: raw}),
                owner_user_id=OWNER,
            )
        assert excinfo.value.code == "config_invalid"

    @pytest.mark.asyncio
    async def test_broken_configuration_never_looks_like_disabled_to_the_owner(self):
        instance = StageAOwnerBridge(
            config_getter=_getter(
                {bridge.CONFIG_OWNER_USER_ID: OWNER, bridge.CONFIG_SOCKET_UID: "nope"}
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
            config_getter=_getter(
                {bridge.CONFIG_OWNER_USER_ID: OWNER, bridge.CONFIG_SOCKET_UID: "nope"}
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


class TestOwnerCandidateGate:
    """The "whose message is this" predicate the intake path shares.

    The Weixin adapter asks this one step before ``classify``, to keep its
    own content-fingerprint cache from discarding a distinct Owner
    request.  Sharing the predicate is what stops that decision from
    drifting away from the one the bridge makes.
    """

    @pytest.mark.parametrize(
        "owner,kwargs",
        [
            pytest.param(None, {}, id="bridge-off"),
            pytest.param(OWNER, {"chat_type": "group"}, id="not-a-dm"),
            pytest.param(OWNER, {"sender_id": "someone-else"}, id="not-the-owner"),
            pytest.param(OWNER, {"sender_id": None}, id="no-sender"),
            pytest.param(OWNER, {"text": "what is the lane?"}, id="no-marker"),
            pytest.param(OWNER, {"text": "/stageant go"}, id="marker-is-a-prefix"),
            pytest.param(OWNER, {"text": "please /stagea go"}, id="marker-not-leading"),
        ],
    )
    def test_a_pass_through_message_is_not_a_candidate(self, owner, kwargs):
        params = {"chat_type": "dm", "sender_id": OWNER, "text": "/stagea go"}
        params.update(kwargs)
        assert is_owner_candidate(owner, **params) is False
        # …and ``classify`` agrees: it routes the message onward untouched.
        assert _classify(owner, **params, has_media=False, message_id="m1").consumed is False

    @pytest.mark.parametrize(
        "kwargs,reason",
        [
            pytest.param({"has_media": True}, "media_not_admitted", id="media"),
            pytest.param({"text": "/stagea"}, "empty_request", id="empty"),
            pytest.param({"message_id": ""}, "unstable_message_identity", id="no-id"),
        ],
    )
    def test_a_refusable_request_is_still_the_owners_message(self, kwargs, reason):
        """Refusable and pass-through are different questions.

        Everything that can *refuse* a request describes a message that is
        already the Owner's, so the gate must still call it a candidate —
        otherwise the intake path would treat a refused Stage-A request as
        ordinary traffic.
        """
        params = {"chat_type": "dm", "sender_id": OWNER, "text": "/stagea go"}
        params.update({k: v for k, v in kwargs.items() if k == "text"})
        assert is_owner_candidate(OWNER, **params) is True

        decision = _classify(OWNER, **{**params, **kwargs})
        assert decision.refused is True
        assert decision.reason == reason

    def test_the_ordinary_case_is_a_candidate(self):
        assert (
            is_owner_candidate(OWNER, chat_type="dm", sender_id=OWNER, text="  /stagea  go  ")
            is True
        )

    def test_the_gate_never_reads_the_secondary_configuration(self):
        """Asking "is this the Owner's?" must not touch socket settings.

        The admission ordering property depends on it: a broken deployment
        must not be able to change what happens to a message before that
        message is known to be the Owner's.
        """

        def explode_on_secondary(name, default=None):
            if name != bridge.CONFIG_OWNER_USER_ID:
                raise AssertionError(f"secondary configuration read: {name}")
            return OWNER

        instance = StageAOwnerBridge(config_getter=explode_on_secondary, channel="weixin")
        assert (
            instance.is_owner_candidate(chat_type="dm", sender_id=OWNER, text="/stagea go")
            is True
        )
        assert (
            instance.is_owner_candidate(chat_type="dm", sender_id=OTHER, text="/stagea go")
            is False
        )

    @pytest.mark.asyncio
    async def test_the_gate_has_no_side_effect_on_the_request_that_follows(self):
        """It is asked about every inbound message, so it must cost nothing.

        Two ways a side effect would show up, both fatal: consuming the
        replay guard would make the real request one step later look like
        a duplicate of the question the intake path just asked, and
        counting against the in-flight ceiling would refuse it outright.
        The gate is asked more times than that ceiling allows before the
        real request is made.
        """
        instance = StageAOwnerBridge(config_getter=_getter(ENABLED_CONFIG), channel="weixin")
        for _ in range(bridge.MAX_INFLIGHT_REQUESTS + 5):
            assert (
                instance.is_owner_candidate(chat_type="dm", sender_id=OWNER, text="/stagea go")
                is True
            )

        async def request(message_id):
            open_connection, captured = TestExchange._connection()
            with _socket_layer(open_connection):
                reply = await instance.process(
                    chat_type="dm",
                    sender_id=OWNER,
                    text="/stagea go",
                    has_media=False,
                    conversation_key="weixin|acct|chat-1|owner-user-id",
                    message_id=message_id,
                )
            return reply, captured

        reply, captured = await request("m1")
        assert "ACCEPTED_TERMINAL" in reply
        assert len(captured) == 1

        # A busy channel now asks the gate about more messages than the
        # guard can hold.  If the gate recorded any of them, the accepted
        # request's own entry would be evicted and its replay would be
        # admitted as new work.
        for index in range(bridge.REPLAY_CAPACITY + 10):
            assert (
                instance.is_owner_candidate(
                    chat_type="dm", sender_id=OWNER, text=f"/stagea go {index}"
                )
                is True
            )

        replayed, captured = await request("m1")
        assert "duplicate_request" in replayed
        assert captured == []


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

    @pytest.mark.parametrize(
        "protocol",
        [
            pytest.param(True, id="json-true"),
            pytest.param(False, id="json-false"),
            pytest.param(1.0, id="json-float-one"),
            pytest.param("1", id="stringified"),
            pytest.param(2, id="wrong-number"),
            pytest.param(None, id="null"),
        ],
    )
    def test_only_the_exact_integer_protocol_is_accepted(self, protocol):
        """The deep review's second finding, and its neighbours.

        ``!= 1`` is a value test, and ``True`` and ``1.0`` both pass it.
        A frame whose protocol number is a boolean or a float never met
        the schema this bridge claims to enforce, so no text in it may be
        trusted, let alone delivered to the Owner.
        """
        with pytest.raises(BridgeError) as excinfo:
            validate_reply(
                _reply(self.RID, self.REF, protocol=protocol),
                request_id=self.RID,
                ref=self.REF,
            )
        assert excinfo.value.code == "reply_malformed"

    def test_the_exact_integer_protocol_is_still_accepted(self):
        """The tightening must reject aliases, not the contract itself."""
        outcome, text = validate_reply(
            _reply(self.RID, self.REF, protocol=1), request_id=self.RID, ref=self.REF
        )
        assert (outcome, text) == ("ACCEPTED_TERMINAL", "done")

    def test_an_aliased_protocol_is_refused_before_its_text_is_read(self):
        """Rejection has to happen before the frame's text is trusted.

        A reply carrying ``protocol=True`` *and* unusable text must fail
        on the protocol, not on the text: that ordering is what stops a
        malformed frame from being read at all.
        """
        with pytest.raises(BridgeError) as excinfo:
            validate_reply(
                _reply(self.RID, self.REF, protocol=True, text=""),
                request_id=self.RID,
                ref=self.REF,
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


class TestExactIntegerTyping:
    """One rule for "an integer enum is an integer", used on both sides.

    The deep review found the same type confusion at ingress and at
    reply, which is what a duplicated rule buys.  :func:`is_exact_int` is
    the single definition both now use, so the falsifiers below pin the
    rule itself rather than one of its two call sites.
    """

    def test_the_authoritative_integer_passes(self):
        assert bridge.is_exact_int(1, 1) is True

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(True, id="bool-true"),
            pytest.param(False, id="bool-false"),
            pytest.param(1.0, id="float"),
            pytest.param("1", id="str"),
            pytest.param(2, id="other-int"),
            pytest.param(None, id="none"),
            pytest.param([1], id="list"),
        ],
    )
    def test_no_alias_passes(self, value):
        assert bridge.is_exact_int(value, 1) is False

    def test_an_int_subclass_is_not_the_authoritative_representation(self):
        """``isinstance`` would admit this; ``type(...) is int`` does not.

        Excluding ``bool`` by name would leave every other subclass in,
        which is the same defect with a different constructor.
        """

        class Sneaky(int):
            pass

        assert Sneaky(1) == 1
        assert bridge.is_exact_int(Sneaky(1), 1) is False

    def test_equality_and_membership_would_both_have_been_fooled(self):
        """Why the helper exists at all — the Python facts it corrects.

        Stated as a test rather than a comment so that a future Python
        in which these stop being true would say so here, at the rule,
        instead of somewhere further downstream.
        """
        for alias in (True, 1.0):
            assert alias == 1
            assert alias in {1}
            assert hash(alias) == hash(1)


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
                config_getter=_getter(
                    {
                        bridge.CONFIG_OWNER_USER_ID: OWNER,
                        bridge.CONFIG_SOCKET_PATH: path,
                        bridge.CONFIG_SOCKET_UID: str(os.getuid()),
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
                config_getter=_getter(
                    {
                        bridge.CONFIG_OWNER_USER_ID: OWNER,
                        bridge.CONFIG_SOCKET_PATH: path,
                        bridge.CONFIG_SOCKET_UID: str(os.getuid()),
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
        return StageAOwnerBridge(config_getter=_getter(ENABLED_CONFIG), channel="weixin")

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
        instance = StageAOwnerBridge(config_getter=_getter({}), channel="weixin")
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
            config_getter=_getter({bridge.CONFIG_OWNER_USER_ID: OWNER}), channel="weixin"
        )
        with patch.object(asyncio, "open_unix_connection", AsyncMock(), create=True) as opened:
            with patch.object(bridge, "check_socket_path") as preflight:
                reply = await self._run(instance)
        assert "config_invalid" in reply
        opened.assert_not_called()
        preflight.assert_not_called()


class TestNoAlternateDestination:
    """The bridge owns exactly one delivery idea: reply where it was asked."""

    @staticmethod
    async def _request(instance):
        return await instance.process(
            chat_type="dm",
            sender_id=OWNER,
            text="/stagea advance",
            has_media=False,
            conversation_key="weixin|acct|chat-1|owner-user-id",
            message_id="m1",
        )

    @staticmethod
    @contextlib.contextmanager
    def _other_transports_armed():
        """Make every transport but the one Unix socket fail loudly if used.

        Each of these is a way a second destination could exist: another
        socket, a listener, an HTTP call, a shelled-out command.  Arming
        them turns "the bridge must not do that" into something the run
        itself proves, instead of a substring the source happens not to
        contain.
        """
        detonate = Mock(side_effect=AssertionError("second transport opened"))
        detonate_async = AsyncMock(side_effect=AssertionError("second transport opened"))
        with patch.object(asyncio, "open_connection", detonate_async, create=True):
            with patch.object(asyncio, "start_server", detonate_async, create=True):
                with patch.object(asyncio, "start_unix_server", detonate_async, create=True):
                    with patch("subprocess.run", detonate):
                        with patch("subprocess.Popen", detonate):
                            with patch("urllib.request.urlopen", detonate):
                                yield

    @pytest.mark.asyncio
    async def test_a_whole_exchange_uses_the_one_unix_socket_and_nothing_else(self):
        """One request, one connection, one frame out — and no other transport."""
        open_connection, captured = TestExchange._connection()
        opened = []

        async def counting_open(path):
            opened.append(path)
            return await open_connection(path)

        instance = StageAOwnerBridge(config_getter=_getter(ENABLED_CONFIG), channel="weixin")
        with self._other_transports_armed():
            with _socket_layer(counting_open):
                reply = await self._request(instance)

        assert "ACCEPTED_TERMINAL" in reply
        assert opened == [SOCKET]
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_the_reply_is_returned_to_the_caller_never_delivered(self):
        """The bridge has no delivery power: it hands the text back and stops.

        Whoever called it owns the conversation and does the sending, which
        is what makes "same conversation only" enforceable at all.
        """
        open_connection, _ = TestExchange._connection()
        instance = StageAOwnerBridge(config_getter=_getter(ENABLED_CONFIG), channel="weixin")
        with self._other_transports_armed():
            with _socket_layer(open_connection):
                reply = await self._request(instance)

        assert isinstance(reply, str) and reply.startswith(bridge.REPLY_PREFIX)

    @pytest.mark.asyncio
    async def test_a_failing_exchange_still_opens_no_other_transport(self):
        """No fallback: a dead controller produces a refusal, not a retry elsewhere."""
        instance = StageAOwnerBridge(config_getter=_getter(ENABLED_CONFIG), channel="weixin")
        with self._other_transports_armed():
            with patch("os.stat", side_effect=FileNotFoundError()):
                reply = await self._request(instance)

        assert "socket_unavailable" in reply

    def test_module_declares_no_listener_api(self):
        for forbidden in ("serve", "listen", "bind", "accept"):
            assert not hasattr(bridge, forbidden)


class TestExchangeDeadlines:
    """No await in one exchange may outlive that exchange's own budget.

    The deep/adversarial review's third finding: ``drain()`` and
    ``wait_closed()`` were awaited with no deadline at all.  A peer that
    accepted the connection and then simply stopped could hold one of the
    four in-flight slots indefinitely and suppress the fail-closed answer
    the Owner is owed — and ``except TimeoutError`` around an unbounded
    await cannot create a deadline.

    Every test here stalls one await *forever* and asserts the exchange
    still finishes.  The outer :func:`asyncio.wait_for` is what makes
    that deterministic: an unbounded implementation fails on the outer
    bound instead of hanging the suite.
    """

    #: The declared deadlines, tightened the way the reviewer tightened
    #: them.  Every local step derives from the connect deadline, so this
    #: shortens the whole exchange rather than two steps of it.
    STEP = 0.02

    #: Outer observation bound.  Generous on purpose — it exists to turn
    #: "hangs forever" into a failed assertion, not to measure anything.
    OUTER = 5.0

    #: What "the bridge's own deadline fired" means here: two orders of
    #: magnitude under the shortest *unpatched* declared bound, and two
    #: orders of magnitude over the ~40 ms this actually takes, so it is
    #: neither timing-fragile nor vacuous.
    BOUNDED = 1.0

    @staticmethod
    @contextlib.contextmanager
    def _declared_deadlines(step=STEP, reply=STEP):
        with patch.object(bridge, "CONNECT_TIMEOUT_SECONDS", step):
            with patch.object(bridge, "REPLY_DEADLINE_SECONDS", reply):
                yield

    @staticmethod
    def _connection(*, stall=None, reply_builder=None):
        """A peer that answers, but stalls forever on one named await."""
        builder = reply_builder or (lambda p: _reply(p["request_id"], p["conversation_ref"]))

        async def never(*args, **kwargs):
            await asyncio.Event().wait()

        async def open_connection(path):
            reader = asyncio.StreamReader()
            writer = Mock()
            writer.drain = AsyncMock()
            writer.wait_closed = AsyncMock()
            if stall is not None:
                setattr(writer, stall, AsyncMock(side_effect=never))

            def on_write(frame):
                request = json.loads(frame[4:].decode("utf-8"))
                reader.feed_data(encode_frame(builder(request)))
                reader.feed_eof()

            writer.write = on_write
            return reader, writer

        return open_connection

    @staticmethod
    def _bridge():
        return StageAOwnerBridge(config_getter=_getter(ENABLED_CONFIG), channel="weixin")

    @classmethod
    async def _timed(cls, instance, *, message_id="m1", outer=None):
        """Run one request under an outer bound, returning ``(reply, elapsed)``."""
        started = time.monotonic()
        reply = await asyncio.wait_for(
            instance.process(
                chat_type="dm",
                sender_id=OWNER,
                text="/stagea advance",
                has_media=False,
                conversation_key="weixin|acct|chat-1|owner-user-id",
                message_id=message_id,
            ),
            timeout=outer or cls.OUTER,
        )
        return reply, time.monotonic() - started

    @pytest.mark.asyncio
    async def test_a_stalled_drain_becomes_a_bounded_refusal(self):
        """The reviewer's first F-3 probe, at this head."""
        instance = self._bridge()
        with _socket_layer(self._connection(stall="drain")):
            with self._declared_deadlines():
                reply, elapsed = await self._timed(instance)

        assert "send_failed" in reply
        assert "No Stage-A work was created." in reply
        assert elapsed < self.BOUNDED
        assert instance._inflight == 0

    @pytest.mark.asyncio
    async def test_a_stalled_close_cannot_withhold_an_answer_already_earned(self):
        """The reviewer's second F-3 probe, at this head.

        The reply was read and validated before teardown began, so
        teardown has nothing left to decide.  A peer that refuses to
        finish closing must not be able to take that answer back.
        """
        instance = self._bridge()
        with _socket_layer(self._connection(stall="wait_closed")):
            with self._declared_deadlines():
                reply, elapsed = await self._timed(instance)

        assert reply == "[Stage-A] ACCEPTED_TERMINAL\ndone"
        assert elapsed < self.BOUNDED
        assert instance._inflight == 0

    @pytest.mark.asyncio
    async def test_a_stalled_close_cannot_withhold_a_refusal_either(self):
        """Teardown must not swallow the failure path any more than the
        success path: both are answers the Owner is owed."""
        instance = self._bridge()
        connection = self._connection(
            stall="wait_closed",
            reply_builder=lambda p: _reply(p["request_id"], "f" * 32),
        )
        with _socket_layer(connection):
            with self._declared_deadlines():
                reply, elapsed = await self._timed(instance)

        assert "reply_wrong_conversation" in reply
        assert elapsed < self.BOUNDED
        assert instance._inflight == 0

    @pytest.mark.asyncio
    async def test_a_stalled_socket_stat_is_bounded_too(self):
        """"Local" is not "bounded": a wedged filesystem is a stall.

        The preflight and the post-connect identity re-check both run in
        a worker thread, and a thread cannot be cancelled — so the await
        is what carries the bound, and the slot is freed on time whether
        or not the thread ever finishes.
        """

        async def never(*args, **kwargs):
            await asyncio.Event().wait()

        instance = self._bridge()
        with patch.object(asyncio, "to_thread", never):
            with self._declared_deadlines():
                reply, elapsed = await self._timed(instance)

        assert "socket_unavailable" in reply
        assert elapsed < self.BOUNDED
        assert instance._inflight == 0

    @pytest.mark.asyncio
    async def test_a_stalled_identity_recheck_is_bounded_too(self):
        """The second stat gets its own bound, not the first one's.

        Bounding the preflight and leaving the post-connect re-check open
        would move the hole rather than close it — and move it somewhere
        strictly worse, because by then a connection is open and an
        in-flight slot is already held.
        """

        async def never(*args, **kwargs):
            await asyncio.Event().wait()

        real_to_thread = asyncio.to_thread
        calls = []

        async def stall_after_the_first(func, *args, **kwargs):
            calls.append(func)
            if len(calls) == 1:
                return await real_to_thread(func, *args, **kwargs)
            return await never()

        instance = self._bridge()
        with _socket_layer(self._connection()):
            with patch.object(asyncio, "to_thread", stall_after_the_first):
                with self._declared_deadlines():
                    reply, elapsed = await self._timed(instance)

        assert len(calls) == 2, "the identity re-check never ran, so nothing was proven"
        assert "socket_unavailable" in reply
        assert elapsed < self.BOUNDED
        assert instance._inflight == 0

    @pytest.mark.asyncio
    async def test_a_stalled_peer_cannot_exhaust_the_in_flight_slots(self):
        """The availability claim, stated as the reviewer stated it.

        Filling every slot with a peer that never finishes used to be
        permanent: the slots were held by awaits that could not time out.
        Bounded, the same burst drains and the next request is admitted.
        """
        instance = self._bridge()
        with _socket_layer(self._connection(stall="drain")):
            with self._declared_deadlines():
                burst = await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            instance.process(
                                chat_type="dm",
                                sender_id=OWNER,
                                text="/stagea advance",
                                has_media=False,
                                conversation_key="weixin|acct|chat-1|owner-user-id",
                                message_id=f"burst-{n}",
                            )
                            for n in range(bridge.MAX_INFLIGHT_REQUESTS)
                        )
                    ),
                    timeout=self.OUTER,
                )
                assert instance._inflight == 0
                after, _ = await self._timed(instance, message_id="after-the-burst")

        assert all("send_failed" in reply for reply in burst)
        assert "too_many_inflight" not in after

    @pytest.mark.asyncio
    async def test_the_declared_deadline_is_what_governs(self):
        """Bounded by the declared deadline, not by luck or by the outer bound.

        A lower bound is the honest test here: an exchange whose send
        deadline is 300 ms cannot answer a stalled peer in less, so this
        fails if the bound came from anywhere else.
        """
        instance = self._bridge()
        with _socket_layer(self._connection(stall="drain")):
            with self._declared_deadlines(step=0.3, reply=0.3):
                reply, elapsed = await self._timed(instance)

        assert "send_failed" in reply
        assert elapsed >= 0.3
        assert elapsed < 3.0

    def test_the_whole_exchange_budget_is_the_sum_of_the_declared_bounds(self):
        """Bounded as a whole, not merely step by step.

        Five local steps — preflight stat, connect, identity re-stat,
        write/drain, close — plus the one long wait for the controller's
        answer.  Chaining individually-legal delays cannot exceed it.
        """
        assert bridge._exchange_budget() == (
            5 * bridge._local_step_deadline() + bridge.REPLY_DEADLINE_SECONDS
        )

    def test_the_declared_bounds_are_read_at_call_time(self):
        """A copy taken at import would leave a second, longer bound behind.

        This is what lets a falsifier tighten the declared deadlines and
        actually shorten what it is measuring — including the local steps
        that have no constant of their own.
        """
        with patch.object(bridge, "CONNECT_TIMEOUT_SECONDS", 0.5):
            assert bridge._local_step_deadline() == 0.5
            with patch.object(bridge, "REPLY_DEADLINE_SECONDS", 1.0):
                assert bridge._exchange_budget() == pytest.approx(3.5)

    @pytest.mark.asyncio
    async def test_an_exhausted_budget_skips_the_wait_rather_than_shortening_it(self):
        """Teardown is allowed to be non-blocking; it is never unbounded.

        With nothing left in the budget the close must not await at all —
        and must not leave an un-awaited coroutine behind when it
        declines to.
        """
        writer = Mock()
        writer.close = Mock()
        writer.wait_closed = AsyncMock()
        with _reserved_cleanup() as cleanup:
            spent = bridge._Deadline(-1.0, cleanup)

            await bridge._close_writer(writer, spent)

            # Declining to await costs no cleanup permit either: a step
            # that never started cannot straggle.
            assert cleanup._permits == bridge._CLEANUP_PERMITS_PER_EXCHANGE

        writer.close.assert_called_once_with()
        writer.wait_closed.assert_not_awaited()


class _Stall:
    """Bookkeeping for one adversarial awaitable.

    A small object rather than a dict so each field keeps its own type:
    a heterogeneous ``dict`` collapses to the union of its values, and the
    counters below stop type-checking as counters.
    """

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.cancellations = 0
        self.finished = 0


class TestCancellationUncooperativeAwaitables:
    """A step that will not acknowledge cancellation must not extend the exchange.

    The deep/adversarial rereview's finding.  ``_Deadline.bounded``
    delegated its timeout to :func:`asyncio.wait_for`, which reaches that
    timeout by cancelling the inner await and then *waiting for the
    acknowledgement*.  An awaitable that catches ``CancelledError`` and
    stays pending therefore keeps the wrapper pending with it — and one
    of the four in-flight slots with that — for as long as it likes, with
    no external bound at all.

    The committed deadline falsifiers stall on ``asyncio.Event().wait()``,
    which *cooperates*: it ends the moment it is cancelled, ``wait_for``
    returns promptly, and those tests pass without ever reaching this
    boundary.  The doubles below are the ones that reach it — they
    swallow every ``CancelledError`` until the test itself releases them,
    which is the only thing that separates a deadline the bridge enforces
    from one it merely asks for.
    """

    #: The declared deadlines, tightened so one exchange is short.
    STEP = 0.02

    #: Scheduling allowance added to the bridge's own declared budget, so
    #: a loaded runner cannot turn a correct bound into a flake.  It hides
    #: nothing: unbounded, none of these exchanges finishes *at all* until
    #: the test releases the adversarial awaitable, so any finite bound
    #: separates the two implementations.
    SLACK = 0.5

    #: Outer stop, so a regression fails an assertion instead of hanging
    #: the suite.  Deliberately far above the bound being asserted.
    OUTER = 5.0

    @staticmethod
    @contextlib.contextmanager
    def _declared_deadlines(step=STEP, reply=STEP):
        with patch.object(bridge, "CONNECT_TIMEOUT_SECONDS", step):
            with patch.object(bridge, "REPLY_DEADLINE_SECONDS", reply):
                yield

    @staticmethod
    def _uncooperative():
        """An await that absorbs cancellation until the test releases it.

        A plain ``async def`` on purpose: :class:`AsyncMock` only awaits a
        side effect that :func:`asyncio.iscoroutinefunction` recognises,
        and an object with an async ``__call__`` is not one — it would be
        returned un-awaited and the double would quietly become
        cooperative, proving nothing.
        """
        state = _Stall()

        async def stall(*args, **kwargs):
            while True:
                try:
                    await state.release.wait()
                except asyncio.CancelledError:
                    state.cancellations += 1
                    continue
                state.finished += 1
                return None

        return stall, state

    @classmethod
    def _connection(cls, stall_attr):
        """A peer that answers, then refuses to acknowledge cancellation."""
        stall, state = cls._uncooperative()
        calls = {"connects": 0, "writes": 0}

        async def open_connection(path):
            calls["connects"] += 1
            reader = asyncio.StreamReader()
            writer = Mock()
            writer.close = Mock()
            writer.drain = AsyncMock()
            writer.wait_closed = AsyncMock()
            setattr(writer, stall_attr, AsyncMock(side_effect=stall))

            def on_write(frame):
                calls["writes"] += 1
                request = json.loads(frame[4:].decode("utf-8"))
                reader.feed_data(
                    encode_frame(_reply(request["request_id"], request["conversation_ref"]))
                )
                reader.feed_eof()

            writer.write = on_write
            return reader, writer

        return open_connection, state, calls

    @staticmethod
    def _bridge():
        return StageAOwnerBridge(config_getter=_getter(ENABLED_CONFIG), channel="weixin")

    @staticmethod
    def _request(instance, message_id="m1"):
        return instance.process(
            chat_type="dm",
            sender_id=OWNER,
            text="/stagea advance",
            has_media=False,
            conversation_key="weixin|acct|chat-1|owner-user-id",
            message_id=message_id,
        )

    @classmethod
    async def _finished_within(cls, coro, *, bound):
        """Run one request under a stop that does not rely on cancellation.

        An outer :func:`asyncio.wait_for` is useless against this double:
        reaching its timeout means cancelling the request, and a request
        wedged behind an uncooperative await will not acknowledge that
        either — the suite would hang rather than fail.
        :func:`asyncio.wait` returns on time whatever the task does, which
        is what makes the assertion below possible to state at all.
        """
        task = asyncio.ensure_future(coro)
        started = time.monotonic()
        done, _pending = await asyncio.wait({task}, timeout=cls.OUTER)
        elapsed = time.monotonic() - started
        assert done, f"the exchange was still unfinished after {cls.OUTER}s"
        assert elapsed < bound, f"finished in {elapsed:.3f}s, past the {bound:.3f}s bound"
        return task.result(), elapsed

    @staticmethod
    async def _settle(state, *, expected, deadline=2.0):
        """Release the stall and let every abandoned copy of it end.

        Cleanup, and deliberately assertion-free: it runs in a ``finally``,
        so an assertion here would mask whichever failure sent the test
        into it.  What the harvest actually has to guarantee is asserted
        by :meth:`_drain_stragglers` in the tests that own that claim.
        """
        state.release.set()
        stop = time.monotonic() + deadline
        while state.finished < expected and time.monotonic() < stop:
            await asyncio.sleep(0.01)
        # One more turn so the loop can run each abandoned task's callback.
        await asyncio.sleep(0)

    @staticmethod
    async def _drain_stragglers(before, *, deadline=2.0):
        """Wait for every straggler this test created to be harvested."""
        stop = time.monotonic() + deadline
        while (set(bridge._stragglers) - before) and time.monotonic() < stop:
            await asyncio.sleep(0.01)
        return set(bridge._stragglers) - before

    @staticmethod
    @contextlib.contextmanager
    def _no_unobserved_tasks():
        """Fail if the loop or the interpreter reports an unobserved task.

        Abandoning a task is only legitimate while its end is still read.
        Both ways that stops being true are watched here: the loop's
        exception handler, where an unretrieved task exception surfaces,
        and the ``RuntimeWarning`` the interpreter emits for a coroutine
        that was never awaited or a task destroyed while still pending.
        """
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        reported = []
        loop.set_exception_handler(lambda _loop, context: reported.append(context))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                yield
            finally:
                loop.set_exception_handler(previous)
        leaked = [
            str(w.message)
            for w in caught
            if issubclass(w.category, RuntimeWarning)
            and ("never awaited" in str(w.message) or "destroyed" in str(w.message))
        ]
        assert reported == [], f"the loop reported an unobserved task: {reported}"
        assert leaked == [], f"unobserved-task warnings: {leaked}"

    # -- the bound itself, proved without releasing the stall --

    @pytest.mark.asyncio
    async def test_an_uncooperative_drain_cannot_outlive_the_exchange_budget(self):
        """The rereviewer's first probe: a ``drain()`` that ignores cancellation.

        Nothing releases the stall before the assertions — that is the
        whole point.  The exchange has to end on its own declared budget,
        hand the Owner the fail-closed answer, and give the slot back.
        """
        instance = self._bridge()
        connection, state, calls = self._connection("drain")
        try:
            with _socket_layer(connection):
                with self._declared_deadlines():
                    bound = bridge._exchange_budget() + self.SLACK
                    reply, _elapsed = await self._finished_within(
                        self._request(instance), bound=bound
                    )

                    assert not state.release.is_set(), "the stall was released; nothing was proven"
                    assert state.cancellations >= 1, "the double was never asked to cancel"
                    assert "send_failed" in reply
                    assert "No Stage-A work was created." in reply
                    assert instance._inflight == 0
                    assert calls == {"connects": 1, "writes": 1}
        finally:
            await self._settle(state, expected=1)

    @pytest.mark.asyncio
    async def test_an_uncooperative_close_cannot_withhold_an_answer_already_earned(self):
        """The rereviewer's second probe: a ``wait_closed()`` that ignores cancellation.

        The reply was read and validated before teardown began, so a peer
        that will not finish closing has nothing left to decide — and must
        not be able to take that answer back or keep the slot while it
        declines to acknowledge.
        """
        instance = self._bridge()
        connection, state, calls = self._connection("wait_closed")
        try:
            with _socket_layer(connection):
                with self._declared_deadlines():
                    bound = bridge._exchange_budget() + self.SLACK
                    reply, _elapsed = await self._finished_within(
                        self._request(instance), bound=bound
                    )

                    assert not state.release.is_set(), "the stall was released; nothing was proven"
                    assert state.cancellations >= 1, "the double was never asked to cancel"
                    assert reply == "[Stage-A] ACCEPTED_TERMINAL\ndone"
                    assert instance._inflight == 0
                    assert calls == {"connects": 1, "writes": 1}
        finally:
            await self._settle(state, expected=1)

    @pytest.mark.asyncio
    async def test_four_uncooperative_peers_cannot_retain_the_four_slots(self):
        """The availability claim, against the adversarial double this time.

        Four peers, one per admitted slot, none of them acknowledging
        cancellation.  Every one has to be gone inside the budget, the
        counter has to be back at zero, and the next Owner message has to
        be admitted rather than refused as too many in flight.
        """
        instance = self._bridge()
        connection, state, calls = self._connection("drain")
        peers = bridge.MAX_INFLIGHT_REQUESTS
        try:
            with _socket_layer(connection):
                with self._declared_deadlines():
                    bound = bridge._exchange_budget() + self.SLACK
                    burst, _elapsed = await self._finished_within(
                        asyncio.gather(
                            *(
                                self._request(instance, message_id=f"burst-{n}")
                                for n in range(peers)
                            )
                        ),
                        bound=bound,
                    )

                    assert not state.release.is_set(), "the stall was released; nothing was proven"
                    assert all("send_failed" in reply for reply in burst)
                    assert instance._inflight == 0
                    assert calls == {"connects": peers, "writes": peers}

                    after, _ = await self._finished_within(
                        self._request(instance, message_id="after-the-burst"), bound=bound
                    )
                    assert "too_many_inflight" not in after
                    assert calls == {"connects": peers + 1, "writes": peers + 1}
        finally:
            await self._settle(state, expected=peers + 1)

    @pytest.mark.asyncio
    async def test_the_bound_is_the_declared_budget_and_not_the_outer_stop(self):
        """A lower bound, so the upper one cannot pass by accident.

        An exchange whose declared send deadline is 300 ms cannot answer
        an uncooperative peer in less than that, so this fails if the
        bound that actually fired came from anywhere but the bridge's own
        declared deadlines.
        """
        instance = self._bridge()
        connection, state, _calls = self._connection("drain")
        try:
            with _socket_layer(connection):
                with self._declared_deadlines(step=0.3, reply=0.3):
                    bound = bridge._exchange_budget() + self.SLACK
                    reply, elapsed = await self._finished_within(
                        self._request(instance), bound=bound
                    )

                    assert not state.release.is_set()
                    assert "send_failed" in reply
                    assert elapsed >= 0.3
        finally:
            await self._settle(state, expected=1)

    @pytest.mark.asyncio
    async def test_an_uncooperative_step_does_not_outlive_a_cancelled_exchange(self):
        """Cancelling the Owner's request must not wait on the stall either.

        The acknowledgement the deadline refuses to wait for is the same
        one an outer cancellation would otherwise block on, so the
        abandonment has to cover that path too — otherwise a shutdown
        inherits the hang the deadline just stopped inheriting.
        """
        instance = self._bridge()
        connection, state, _calls = self._connection("drain")
        try:
            with _socket_layer(connection):
                # Long enough that no deadline can be what ends this.
                with self._declared_deadlines(step=30.0, reply=30.0):
                    task = asyncio.ensure_future(self._request(instance))
                    while instance._inflight == 0:
                        await asyncio.sleep(0.005)
                    await asyncio.sleep(0.02)

                    started = time.monotonic()
                    task.cancel()
                    done, _pending = await asyncio.wait({task}, timeout=self.OUTER)
                    elapsed = time.monotonic() - started

                    assert done, "the cancelled exchange never finished"
                    assert task.cancelled()
                    assert not state.release.is_set()
                    assert elapsed < self.SLACK
                    assert instance._inflight == 0
        finally:
            await self._settle(state, expected=1)

    # -- the abandoned step, and how it is accounted for --

    @pytest.mark.asyncio
    async def test_an_abandoned_step_is_harvested_once_it_is_released(self):
        """Abandonment is bookkeeping, not forgetting.

        The step that overran is held — so the collector cannot reclaim it
        while it is still pending — and dropped the moment it ends, so
        holding it is not itself the leak.
        """
        instance = self._bridge()
        before = set(bridge._stragglers)
        connection, state, _calls = self._connection("drain")

        with self._no_unobserved_tasks():
            with _socket_layer(connection):
                with self._declared_deadlines():
                    await self._finished_within(
                        self._request(instance), bound=bridge._exchange_budget() + self.SLACK
                    )
                    assert set(bridge._stragglers) - before, "nothing was detached"

            await self._settle(state, expected=1)
            assert await self._drain_stragglers(before) == set()

    @pytest.mark.asyncio
    async def test_an_abandoned_step_that_fails_is_still_observed(self):
        """Harvesting reads the outcome; it does not merely drop the reference.

        A straggler that ends in an exception nobody retrieves is reported
        by the loop, which would dress a deliberate abandonment up as a
        leak.  This releases the stall into a failure rather than a
        return, which is the case that would surface it.
        """
        state = _Stall()

        async def stall_then_fail():
            while True:
                try:
                    await state.release.wait()
                except asyncio.CancelledError:
                    continue
                raise OSError("the abandoned step failed after it was let go")

        before = set(bridge._stragglers)

        with self._no_unobserved_tasks():
            with _reserved_cleanup() as cleanup:
                deadline = bridge._Deadline(0.02, cleanup)

                with pytest.raises(asyncio.TimeoutError):
                    await deadline.bounded(stall_then_fail(), cap=0.02)

                assert len(set(bridge._stragglers) - before) == 1
                state.release.set()
                assert await self._drain_stragglers(before) == set()

    @pytest.mark.asyncio
    async def test_the_number_of_abandoned_steps_is_bounded(self):
        """Bounded by construction, not by policy.

        One exchange can leave at most two steps behind — the one that
        spent the budget and the teardown after it — and only an in-flight
        exchange can leave any, so four admitted slots cap the whole set.
        """
        instance = self._bridge()
        before = set(bridge._stragglers)
        connection, state, _calls = self._connection("drain")
        peers = bridge.MAX_INFLIGHT_REQUESTS

        with _socket_layer(connection):
            with self._declared_deadlines():
                await self._finished_within(
                    asyncio.gather(
                        *(self._request(instance, message_id=f"burst-{n}") for n in range(peers))
                    ),
                    bound=bridge._exchange_budget() + self.SLACK,
                )
                detached = set(bridge._stragglers) - before
                assert 0 < len(detached) <= 2 * peers

        await self._settle(state, expected=peers)
        assert await self._drain_stragglers(before) == set()

    @pytest.mark.asyncio
    async def test_a_step_that_finished_is_never_parked(self):
        """The ordinary path leaves nothing behind.

        A step that completes inside its bound is returned rather than
        detached, and a step that finished just as the bound arrived is
        read rather than parked — either way the set is where it started,
        and so is the reservation that step drew on.
        """
        before = set(bridge._stragglers)

        async def prompt():
            return "answered"

        with _reserved_cleanup() as cleanup:
            deadline = bridge._Deadline(1.0, cleanup)

            assert await deadline.bounded(prompt(), cap=1.0) == "answered"
            assert set(bridge._stragglers) == before

            finished = asyncio.ensure_future(prompt())
            await finished
            cleanup.abandon(finished)
            assert set(bridge._stragglers) == before

            # Nothing was spent, so an exchange of ordinary steps leaves
            # the ceiling exactly where it found it.
            assert cleanup._permits == bridge._CLEANUP_PERMITS_PER_EXCHANGE


#: The adversarial doubles above are exactly what the section below needs.
#: An alias borrows them; inheriting would re-run that whole class under a
#: second name.
_Uncooperative = TestCancellationUncooperativeAwaitables


class TestGlobalDetachedCleanupBudget:
    """The *population* of abandoned steps is bounded, not merely each exchange.

    Tier-1 rereview-4A's finding.  Correction 4 stopped an uncooperative
    step from holding one of the four in-flight slots — but ``process``
    releases that slot the moment it answers the Owner, while the step it
    let go is still alive.  So the slots stopped bounding anything the
    instant the exchange ended: the next request took the freed slot and
    stranded another straggler, sequentially, without limit.  Twelve
    sequential requests left twelve live tasks against a source-claimed cap
    of eight.

    What these falsify is the population itself.  None of them reads the
    source to decide: each one drives real requests through real
    ``process`` calls and counts what the bridge actually created.
    """

    #: How many sequential requests the ceiling has to survive.  The
    #: rereviewer used twelve; three times the ceiling is a stronger stress
    #: and still finishes in well under a second at these deadlines.
    SEQUENTIAL = 12

    #: A hard stop on the fill loops, so a ceiling that never engages fails
    #: an assertion instead of running forever.
    FILL_LIMIT = 64

    @staticmethod
    async def _quiesced(*, deadline=2.0):
        """Wait for the process-global ledger to be empty, and prove it is.

        Saturation is an absolute condition, not a delta, so these tests
        need an absolute starting point.  A predecessor that left work
        behind surfaces here rather than silently redefining what
        saturation means.
        """
        stop = time.monotonic() + deadline
        while (bridge._stragglers or bridge._cleanup_charged) and time.monotonic() < stop:
            await asyncio.sleep(0.01)
        assert not bridge._stragglers, f"stragglers left over: {len(bridge._stragglers)}"
        assert bridge._cleanup_charged == 0, f"permits left over: {bridge._cleanup_charged}"

    @staticmethod
    @contextlib.contextmanager
    def _counted_threads(counter):
        """Count worker-thread hand-offs, delegating to the real one.

        "Creates no new work" has to include the background kind: the
        preflight stat and the post-connect re-stat are the only two, and
        a refused request must make neither.
        """
        real = asyncio.to_thread

        def counting(func, /, *args, **kwargs):
            counter["threads"] += 1
            return real(func, *args, **kwargs)

        with patch.object(asyncio, "to_thread", counting):
            yield

    @staticmethod
    @contextlib.contextmanager
    def _ledger_samples(samples):
        """Record the ledger at every reservation, granted or refused.

        An end-state assertion cannot see a cap that was briefly exceeded
        and then recovered.  Sampling at the only place capacity is ever
        committed can.
        """
        real = bridge._charge_cleanup

        def sampling(permits):
            granted = real(permits)
            samples.append((granted, bridge._cleanup_charged, len(bridge._stragglers)))
            return granted

        with patch.object(bridge, "_charge_cleanup", sampling):
            yield

    @staticmethod
    def _spent(calls, threads):
        return (calls["connects"], calls["writes"], threads["threads"])

    async def _sequential(self, stall_attr, accepted_fragment):
        """Drive ``SEQUENTIAL`` distinct requests past an unreleasable stall.

        Returns the tally so each caller can state its own claim about it.
        Every per-request invariant that must hold *during* the run is
        asserted here, while the run is still going — an end-state check
        could not see a ceiling that was passed and then recovered.
        """
        instance = _Uncooperative._bridge()
        connection, state, calls = _Uncooperative._connection(stall_attr)
        threads = {"threads": 0}
        admitted, refused, peak = 0, 0, 0

        try:
            with self._counted_threads(threads):
                with _socket_layer(connection):
                    with _Uncooperative._declared_deadlines():
                        bound = bridge._exchange_budget() + _Uncooperative.SLACK
                        for n in range(self.SEQUENTIAL):
                            before = self._spent(calls, threads)
                            reply, _elapsed = await _Uncooperative._finished_within(
                                _Uncooperative._request(instance, message_id=f"seq-{n}"),
                                bound=bound,
                            )
                            spent = tuple(
                                now - was
                                for now, was in zip(self._spent(calls, threads), before)
                            )

                            if "cleanup_capacity_exhausted" in reply:
                                refused += 1
                                assert reply == bridge.refusal_text("cleanup_capacity_exhausted")
                                assert spent == (0, 0, 0), f"a refused request created {spent}"
                            else:
                                admitted += 1
                                assert accepted_fragment in reply
                                # One connect, one write, and the two
                                # preflight stats: no retry, no second
                                # socket request, no extra background work.
                                assert spent == (1, 1, 2), f"an admitted request created {spent}"

                            assert instance._inflight == 0, "the visible slot was not returned"
                            peak = max(peak, len(bridge._stragglers))
                            assert len(bridge._stragglers) <= bridge.MAX_DETACHED_CLEANUP_TASKS
                            assert bridge._cleanup_charged <= bridge.MAX_DETACHED_CLEANUP_TASKS

                        assert not state.release.is_set(), "the stall was released"
        finally:
            await _Uncooperative._settle(state, expected=admitted)
            assert await _Uncooperative._drain_stragglers(set()) == set()
            await self._quiesced()

        return admitted, refused, peak

    # -- the population bound, proved without releasing the stall --

    @pytest.mark.asyncio
    async def test_sequential_requests_cannot_accumulate_past_twice_the_slots(self):
        """The claim, stated in the terms the defect was found in.

        Deliberately written with nothing the correction introduced: the
        straggler set, the admitted in-flight cap and what the bridge put
        on the socket all existed before this ceiling did.  So this fails
        on *observed behaviour* rather than on a missing attribute — run
        against the source it corrects, it reports twelve live detached
        steps after twelve sequential requests, against a bound of eight.

        Nothing is released before the assertion, and no request repeats a
        message id, so nothing here is absorbed by dedupe or by the
        in-flight cap.
        """
        instance = _Uncooperative._bridge()
        connection, state, calls = _Uncooperative._connection("drain")
        claimed = 2 * bridge.MAX_INFLIGHT_REQUESTS
        before = set(bridge._stragglers)
        peak = 0

        try:
            with _socket_layer(connection):
                with _Uncooperative._declared_deadlines():
                    bound = bridge._exchange_budget() + _Uncooperative.SLACK
                    for n in range(self.SEQUENTIAL):
                        reply, _elapsed = await _Uncooperative._finished_within(
                            _Uncooperative._request(instance, message_id=f"acc-{n}"),
                            bound=bound,
                        )

                        assert reply.startswith(bridge.REPLY_PREFIX), "the Owner was not answered"
                        assert instance._inflight == 0, "the visible slot was not returned"

                        peak = max(peak, len(set(bridge._stragglers) - before))
                        assert peak <= claimed, (
                            f"{peak} live detached steps after {n + 1} sequential requests, "
                            f"against a bound of {claimed}"
                        )

                    assert not state.release.is_set(), "the stall was released"
                    # One socket and one frame for each request the bridge
                    # admitted, and nothing at all for one it refused: no
                    # retry, no second request, no replacement work.
                    assert calls["connects"] == calls["writes"]
                    assert 0 < calls["connects"] <= self.SEQUENTIAL
        finally:
            # Assertion-free: this runs on the failure path too, and the
            # harvest claim belongs to the tests below that own it.
            await _Uncooperative._settle(state, expected=calls["connects"])
            await _Uncooperative._drain_stragglers(before)

    @pytest.mark.asyncio
    async def test_twelve_sequential_uncooperative_drains_stay_under_the_ceiling(self):
        """The rereviewer's probe, as an assertion.

        Twelve distinct requests, each with a ``drain()`` that absorbs
        cancellation until this test releases it — and nothing is released
        before the assertions.  Every request still answers the Owner and
        gives its visible slot back; what changes is that the tasks left
        behind stop accumulating.
        """
        admitted, refused, peak = await self._sequential("drain", "send_failed")

        assert admitted + refused == self.SEQUENTIAL
        assert peak <= bridge.MAX_DETACHED_CLEANUP_TASKS
        assert refused, "twelve sequential stalls never met the ceiling"
        # The ceiling engages one exchange's reservation early, which is
        # the fail-closed direction: capacity is refused before it is
        # overspent, never after.
        assert admitted == (
            bridge.MAX_DETACHED_CLEANUP_TASKS - bridge._CLEANUP_PERMITS_PER_EXCHANGE + 1
        )
        assert peak == admitted

    @pytest.mark.asyncio
    async def test_twelve_sequential_uncooperative_closes_stay_under_the_ceiling(self):
        """The same population bound where the stall is teardown instead.

        An uncooperative ``wait_closed()`` strands its step *after* the
        reply was read and validated, so these exchanges each hand the
        Owner an accepted terminal and still leave a task behind.  The
        ceiling has to hold on the success path too.
        """
        admitted, refused, peak = await self._sequential("wait_closed", "ACCEPTED_TERMINAL")

        assert admitted + refused == self.SEQUENTIAL
        assert peak <= bridge.MAX_DETACHED_CLEANUP_TASKS
        assert refused, "twelve sequential stalls never met the ceiling"
        assert peak == admitted

    # -- what a saturated bridge does, and does not do --

    async def _saturate(self, instance, connection, threads, *, state):
        """Fill the ceiling with unreleasable stragglers.  Returns the tally."""
        admitted = 0
        bound = bridge._exchange_budget() + _Uncooperative.SLACK
        for n in range(self.FILL_LIMIT):
            reply, _elapsed = await _Uncooperative._finished_within(
                _Uncooperative._request(instance, message_id=f"fill-{n}"), bound=bound
            )
            if "cleanup_capacity_exhausted" in reply:
                assert not state.release.is_set(), "the stall was released"
                return admitted
            assert "send_failed" in reply
            admitted += 1
        raise AssertionError(f"the ceiling never engaged in {self.FILL_LIMIT} requests")

    @pytest.mark.asyncio
    async def test_a_saturated_bridge_starts_no_new_work_whatsoever(self):
        """Refusing has to happen before anything is created, not after.

        Once the ceiling is reached, further admitted Owner requests must
        produce the fixed refusal without opening a socket, writing a
        frame, handing anything to a worker thread, or adding one more
        entry to the set that is being bounded.
        """
        await self._quiesced()
        instance = _Uncooperative._bridge()
        connection, state, calls = _Uncooperative._connection("drain")
        threads = {"threads": 0}
        admitted = 0

        try:
            with self._counted_threads(threads):
                with _socket_layer(connection):
                    with _Uncooperative._declared_deadlines():
                        admitted = await self._saturate(
                            instance, connection, threads, state=state
                        )
                        saturated = self._spent(calls, threads)
                        parked = set(bridge._stragglers)
                        charged = bridge._cleanup_charged
                        bound = bridge._exchange_budget() + _Uncooperative.SLACK

                        for extra in range(5):
                            reply, _elapsed = await _Uncooperative._finished_within(
                                _Uncooperative._request(instance, message_id=f"after-{extra}"),
                                bound=bound,
                            )

                            assert reply == bridge.refusal_text("cleanup_capacity_exhausted")
                            assert self._spent(calls, threads) == saturated
                            assert set(bridge._stragglers) == parked
                            assert bridge._cleanup_charged == charged
                            assert instance._inflight == 0

                        assert not state.release.is_set(), "the stall was released"
                        assert len(parked) <= bridge.MAX_DETACHED_CLEANUP_TASKS
                        assert saturated == (admitted, admitted, 2 * admitted)
        finally:
            await _Uncooperative._settle(state, expected=admitted)
            assert await _Uncooperative._drain_stragglers(set()) == set()
            await self._quiesced()

    @pytest.mark.asyncio
    async def test_capacity_comes_back_only_when_the_abandoned_steps_end(self):
        """Saturation is not permanent, and it does not lift early.

        Nothing but the stalls actually finishing gives capacity back: the
        permits return as the tasks end and are harvested, the ledger goes
        back to empty, and the request that was refused a moment ago now
        completes normally through a healthy peer.  Neither the loop nor
        the interpreter may report an unobserved task along the way.
        """
        await self._quiesced()
        instance = _Uncooperative._bridge()
        connection, state, _calls = _Uncooperative._connection("drain")
        threads = {"threads": 0}

        with _Uncooperative._no_unobserved_tasks():
            with self._counted_threads(threads):
                with _socket_layer(connection):
                    with _Uncooperative._declared_deadlines():
                        admitted = await self._saturate(
                            instance, connection, threads, state=state
                        )

            assert admitted >= 1
            assert len(bridge._stragglers) == admitted
            assert bridge._cleanup_charged == admitted

            # The only thing that lifts it: the abandoned steps ending.
            await _Uncooperative._settle(state, expected=admitted)
            assert await _Uncooperative._drain_stragglers(set()) == set()
            await self._quiesced()

            # And the bridge is ordinary again — a real exchange, start to
            # finish, through a peer that behaves.
            healthy, captured = TestExchange._connection()
            with _socket_layer(healthy):
                reply = await _Uncooperative._request(instance, message_id="after-the-harvest")

            assert reply == "[Stage-A] ACCEPTED_TERMINAL\ndone"
            assert len(captured) == 1
            assert instance._inflight == 0
            await self._quiesced()

    # -- the race --

    @pytest.mark.asyncio
    async def test_a_concurrent_burst_cannot_oversubscribe_the_ceiling(self):
        """Interleaving cannot beat a reservation taken before the work.

        Part of the ceiling is already held by earlier stragglers, then a
        full burst of concurrent requests races for what is left.  However
        the loop orders them, the ledger may never show more committed than
        the ceiling allows, and whichever requests lose the race must lose
        it closed — with nothing created.
        """
        await self._quiesced()
        instance = _Uncooperative._bridge()
        connection, state, calls = _Uncooperative._connection("drain")
        threads = {"threads": 0}
        samples = []
        stranded = 3
        admitted = 0

        try:
            with self._ledger_samples(samples):
                with self._counted_threads(threads):
                    with _socket_layer(connection):
                        with _Uncooperative._declared_deadlines():
                            bound = bridge._exchange_budget() + _Uncooperative.SLACK
                            for n in range(stranded):
                                reply, _elapsed = await _Uncooperative._finished_within(
                                    _Uncooperative._request(instance, message_id=f"pre-{n}"),
                                    bound=bound,
                                )
                                assert "send_failed" in reply
                            assert len(bridge._stragglers) == stranded

                            before = self._spent(calls, threads)
                            burst, _elapsed = await _Uncooperative._finished_within(
                                asyncio.gather(
                                    *(
                                        _Uncooperative._request(instance, message_id=f"race-{n}")
                                        for n in range(bridge.MAX_INFLIGHT_REQUESTS)
                                    )
                                ),
                                bound=bound,
                            )

            assert not state.release.is_set(), "the stall was released"

            refused = [r for r in burst if "cleanup_capacity_exhausted" in r]
            accepted = [r for r in burst if "send_failed" in r]
            admitted = stranded + len(accepted)

            assert len(accepted) + len(refused) == bridge.MAX_INFLIGHT_REQUESTS
            assert refused, "the burst never met the ceiling"
            assert all(r == bridge.refusal_text("cleanup_capacity_exhausted") for r in refused)

            # The racers that lost created nothing at all.
            spent = tuple(now - was for now, was in zip(self._spent(calls, threads), before))
            assert spent == (len(accepted), len(accepted), 2 * len(accepted))

            # No reservation ever committed past the ceiling, at the only
            # point where capacity is committed.
            over = [s for s in samples if s[1] > bridge.MAX_DETACHED_CLEANUP_TASKS]
            assert over == [], f"the ceiling was oversubscribed: {over}"
            assert [s for s in samples if s[0]] , "no reservation was granted at all"
            assert len(bridge._stragglers) <= bridge.MAX_DETACHED_CLEANUP_TASKS
            assert instance._inflight == 0
        finally:
            await _Uncooperative._settle(state, expected=admitted)
            assert await _Uncooperative._drain_stragglers(set()) == set()
            await self._quiesced()

    # -- the ordinary path is untouched --

    @pytest.mark.asyncio
    async def test_a_saturated_bridge_still_routes_everybody_else_normally(self):
        """Backpressure is Stage-A's alone; it cannot reach ordinary traffic.

        The ceiling is consulted only after a message has already been
        classified as an exact Owner Stage-A request, so a saturated bridge
        must still be invisible to everything else: a non-Owner, a group
        message and the Owner's ordinary text all keep falling through to
        the caller's own routing, exactly as with no bridge at all.
        """
        await self._quiesced()
        instance = _Uncooperative._bridge()
        connection, state, calls = _Uncooperative._connection("drain")
        threads = {"threads": 0}
        admitted = 0

        try:
            with self._counted_threads(threads):
                with _socket_layer(connection):
                    with _Uncooperative._declared_deadlines():
                        admitted = await self._saturate(
                            instance, connection, threads, state=state
                        )
                        saturated = self._spent(calls, threads)

                        passthrough = [
                            {"chat_type": "dm", "sender_id": OTHER, "text": "/stagea go"},
                            {"chat_type": "group", "sender_id": OWNER, "text": "/stagea go"},
                            {"chat_type": "dm", "sender_id": OWNER, "text": "ordinary text"},
                        ]
                        for n, shape in enumerate(passthrough):
                            assert (
                                await instance.process(
                                    has_media=False,
                                    conversation_key="weixin|acct|chat-1|owner-user-id",
                                    message_id=f"ordinary-{n}",
                                    **shape,
                                )
                                is None
                            ), f"a saturated bridge consumed {shape}"

                        assert self._spent(calls, threads) == saturated
                        assert not state.release.is_set(), "the stall was released"
        finally:
            await _Uncooperative._settle(state, expected=admitted)
            assert await _Uncooperative._drain_stragglers(set()) == set()
            await self._quiesced()

    @pytest.mark.asyncio
    async def test_the_ceiling_never_engages_when_nothing_straggles(self):
        """The reservation is inert on the path that behaves.

        Four concurrent exchanges is the whole admitted concurrency, and
        four reservations is the whole ceiling, so a bridge with no
        stragglers can still run every slot it has.  Repeated bursts must
        not drift the ledger upward either — a reservation that leaked
        would show as saturation that nothing caused.
        """
        await self._quiesced()
        instance = _Uncooperative._bridge()
        healthy, captured = TestExchange._connection()

        with _socket_layer(healthy):
            for round_number in range(3):
                burst = await asyncio.gather(
                    *(
                        _Uncooperative._request(
                            instance, message_id=f"clean-{round_number}-{n}"
                        )
                        for n in range(bridge.MAX_INFLIGHT_REQUESTS)
                    )
                )

                assert all(reply == "[Stage-A] ACCEPTED_TERMINAL\ndone" for reply in burst)
                assert instance._inflight == 0
                assert bridge._cleanup_charged == 0, "a completed exchange kept its reservation"
                assert not bridge._stragglers

        assert len(captured) == 3 * bridge.MAX_INFLIGHT_REQUESTS
        await self._quiesced()


class _RealWorkers:
    """Real executor work, counted from inside the worker threads.

    A cancellation-cooperative coroutine cannot falsify anything about a
    worker thread, because cancelling it ends it — which is precisely the
    behaviour under test.  So every stall below is a genuine synchronous
    call blocking a genuine thread on an event only the test can set,
    reached through the genuine :func:`asyncio.to_thread`.  ``live`` is
    therefore work that is really running, not a task that is merely
    unfinished.
    """

    #: A blocked worker gives up eventually so a regression fails the
    #: suite instead of wedging it.  Far above every bound asserted here.
    ABANDON_AFTER = 10.0

    def __init__(self):
        self._lock = threading.Lock()
        self._entered = 0
        self._exited = 0
        self.peak = 0
        self.release = threading.Event()

    @property
    def live(self):
        with self._lock:
            return self._entered - self._exited

    @property
    def started(self):
        with self._lock:
            return self._entered

    async def reaches(self, count, *, deadline=2.0):
        """Wait for exactly ``count`` live workers, then report what there is.

        Handing work to the executor and that work reaching the first line
        of a thread are two different instants.  Waiting for the second
        removes a start-up race from the assertion without weakening it:
        the caller still states an exact number, and a wrong number still
        fails — it just is not allowed to fail because a thread was slow
        to be born.
        """
        stop = time.monotonic() + deadline
        while self.live != count and time.monotonic() < stop:
            await asyncio.sleep(0.005)
        return self.live

    def block(self):
        """Occupy one real worker thread until the test lets it go."""
        with self._lock:
            self._entered += 1
            self.peak = max(self.peak, self._entered - self._exited)
        try:
            self.release.wait(self.ABANDON_AFTER)
        finally:
            with self._lock:
                self._exited += 1

    async def settled(self, *, deadline=5.0):
        """Release every blocked thread and wait for each to really exit.

        Cleanup, and deliberately assertion-free: it runs in a ``finally``,
        where an assertion would mask whichever failure sent the test into
        it.  The claims about what harvest must guarantee belong to the
        tests that own them.
        """
        self.release.set()
        stop = time.monotonic() + deadline
        while self.live and time.monotonic() < stop:
            await asyncio.sleep(0.01)
        # Turns for each wrapper to resolve from the worker's thread and
        # for its done callback to run on this loop.
        for _ in range(3):
            await asyncio.sleep(0)


def _blocking_socket_stat(workers, *, block):
    """An ``os.stat`` that blocks a real worker thread on one chosen step.

    An exchange stats the socket path exactly twice: once in the preflight
    that runs before anything is connected, and once in the post-connect
    identity re-check that runs with a connection already open.  ``block``
    picks which of the two blocks, so each :func:`asyncio.to_thread`
    hand-off can be falsified on its own rather than as a pair.

    A blocked worker never reaches a second stat, so counting socket-path
    stats stays in step with the requests even as blocked work piles up:
    under ``"preflight"`` every socket stat is a preflight, and under
    ``"recheck"`` the odd ones are preflights that must be let through.
    """
    real_stat = os.stat
    sock_st = _stat(stat.S_IFSOCK | 0o660, ino=101)
    dir_st = _stat(0o40755, ino=202)
    seen = {"n": 0}
    counter_lock = threading.Lock()

    def stat_fn(path, *args, **kwargs):
        target = str(path)
        if target == SOCKET_DIR:
            return dir_st
        if target != SOCKET:
            return real_stat(path, *args, **kwargs)
        with counter_lock:
            seen["n"] += 1
            nth = seen["n"]
        if block == "preflight" or nth % 2 == 0:
            workers.block()
        return sock_st

    return stat_fn


class TestRealExecutorWorkIsCharged:
    """The ceiling has to bound work, not the loop's handle on it.

    ``asyncio.to_thread`` is the one step of an exchange whose real work
    runs outside the loop.  Cancelling the future the loop is awaiting
    ends *that future* and nothing else: the call inside the thread runs
    on, uninterruptible, for as long as it likes.  A budget that watched
    the future would hand capacity back while the work it stands for was
    still running, and a bridge already holding a wedged thread apiece
    could then start unboundedly many more — which is exactly what
    twelve sequential requests demonstrated before this correction.

    Every falsifier here blocks real threads and releases them only from
    the test, so none of them can be satisfied by a double that ends the
    moment it is asked to.
    """

    #: Comfortably more requests than the ceiling admits, so the bound is
    #: reached rather than merely approached.
    SEQUENTIAL = 12

    @staticmethod
    def _connection(calls):
        """A peer that connects and answers; every stall is on this side."""

        async def open_connection(path):
            calls["connects"] += 1
            reader = asyncio.StreamReader()
            writer = Mock()
            writer.close = Mock()
            writer.drain = AsyncMock()
            writer.wait_closed = AsyncMock()

            def on_write(frame):
                calls["writes"] += 1
                request = json.loads(frame[4:].decode("utf-8"))
                reader.feed_data(
                    encode_frame(_reply(request["request_id"], request["conversation_ref"]))
                )
                reader.feed_eof()

            writer.write = on_write
            return reader, writer

        return open_connection

    @staticmethod
    @contextlib.contextmanager
    def _real_threads(workers, *, block, calls):
        """The whole socket layer, with one step blocking a real thread.

        The loop's executor is sized past everything these tests ask of it
        first.  The default pool is ``min(32, cpu_count + 4)`` wide, so on
        a small runner it — and not the bridge — would be what stopped the
        thirteenth worker from running, and a ceiling that only holds
        because the host ran out of threads is not the ceiling under test.
        Sized this way the falsifier measures
        :data:`bridge.MAX_DETACHED_CLEANUP_TASKS` on every host, and the
        uncorrected module really does reach twelve live workers.
        """
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=TestRealExecutorWorkIsCharged.SEQUENTIAL + bridge.MAX_INFLIGHT_REQUESTS,
            thread_name_prefix="stagea-falsifier",
        )
        asyncio.get_running_loop().set_default_executor(executor)
        # Left installed for the rest of this test's loop, which closes it:
        # shutting it down here would break the ordinary exchange some of
        # these tests run *after* the stall, to show the bridge still works.
        # Every test releases its threads in its own ``finally``, so that
        # close has nothing to wait for.
        with patch("os.stat", side_effect=_blocking_socket_stat(workers, block=block)):
            with patch.object(bridge, "read_peer_uid", return_value=CONTROLLER_UID):
                with patch.object(
                    asyncio,
                    "open_unix_connection",
                    TestRealExecutorWorkIsCharged._connection(calls),
                    create=True,
                ):
                    yield

    @classmethod
    async def _sequential(cls, block, *, expected_connects):
        """Drive ``SEQUENTIAL`` distinct requests past unreleasable threads.

        Every per-request invariant that has to hold *during* the run is
        asserted here while it is still running: an end-state check could
        not see a ceiling that was passed and then recovered, which is the
        shape the defect actually had.
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}
        threads = {"threads": 0}
        cap = bridge.MAX_DETACHED_CLEANUP_TASKS
        admitted, refused = 0, 0

        try:
            with TestGlobalDetachedCleanupBudget._counted_threads(threads):
                with cls._real_threads(workers, block=block, calls=calls):
                    with _Uncooperative._declared_deadlines():
                        bound = bridge._exchange_budget() + _Uncooperative.SLACK
                        for n in range(cls.SEQUENTIAL):
                            was = (calls["connects"], calls["writes"], threads["threads"])
                            was_live = workers.live
                            reply, _elapsed = await _Uncooperative._finished_within(
                                _Uncooperative._request(instance, message_id=f"thread-{n}"),
                                bound=bound,
                            )
                            spent = (
                                calls["connects"] - was[0],
                                calls["writes"] - was[1],
                                threads["threads"] - was[2],
                            )

                            if reply == bridge.refusal_text("cleanup_capacity_exhausted"):
                                refused += 1
                                # Nothing at all: no worker, no connect, no
                                # write, and no growth in live real work.
                                assert spent == (0, 0, 0), f"a refused request created {spent}"
                                assert workers.live == was_live
                            else:
                                admitted += 1
                                assert "socket_unavailable" in reply
                                assert spent == (expected_connects, 0, expected_connects + 1)
                                assert await workers.reaches(was_live + 1) == was_live + 1

                            assert instance._inflight == 0, "the visible slot was not returned"
                            # The claim, stated about work rather than about
                            # tasks: this is what 12 sequential requests
                            # broke, with 12 live workers against a cap of 8.
                            assert workers.live <= cap, (
                                f"{workers.live} live worker threads against a cap of {cap}"
                            )
                            assert bridge._cleanup_charged <= cap

                        assert not workers.release.is_set(), "the stall was released"
                        assert workers.live == admitted, (
                            "every admitted request should still own one live worker"
                        )
                        await workers.settled()
        finally:
            await workers.settled()
            assert await _Uncooperative._drain_stragglers(set()) == set()
            await TestGlobalDetachedCleanupBudget._quiesced()

        return admitted, refused, workers

    @pytest.mark.asyncio
    async def test_twelve_sequential_blocked_preflight_threads_stay_under_the_ceiling(self):
        """The reviewer's probe, on the preflight hand-off.

        Twelve distinct requests whose socket preflight blocks a real
        thread that will not come back.  Before the correction all twelve
        were admitted and twelve worker calls ran at once against a
        declared cap of eight, with not one capacity refusal; the wrapper
        each one had been charged to was cancelled and harvested while its
        thread was still running.
        """
        admitted, refused, workers = await self._sequential("preflight", expected_connects=0)

        assert admitted + refused == self.SEQUENTIAL
        assert refused, "the ceiling never engaged, so nothing was bounded"
        assert workers.peak <= bridge.MAX_DETACHED_CLEANUP_TASKS
        assert workers.peak == admitted
        assert workers.live == 0, "a released worker never exited"

    @pytest.mark.asyncio
    async def test_twelve_sequential_blocked_identity_rechecks_stay_under_the_ceiling(self):
        """The same, on the hand-off that runs with a socket already open.

        Bounding the preflight and leaving the post-connect re-check
        uncharged would move the defect rather than close it, and move it
        somewhere strictly worse: by then a connection exists.  Each
        admitted request here connects exactly once, writes nothing, and
        hands off twice — the preflight that returns and the re-check that
        does not.
        """
        admitted, refused, workers = await self._sequential("recheck", expected_connects=1)

        assert admitted + refused == self.SEQUENTIAL
        assert refused, "the ceiling never engaged, so nothing was bounded"
        assert workers.peak <= bridge.MAX_DETACHED_CLEANUP_TASKS
        assert workers.peak == admitted
        assert workers.live == 0, "a released worker never exited"

    @pytest.mark.asyncio
    async def test_a_let_go_wrapper_keeps_its_permit_while_its_thread_runs(self):
        """Losing the wrapper must not be mistaken for the work ending.

        The exchange is over, the Owner has been answered and the in-flight
        slot is back — and the thread the exchange handed off is still
        running.  Before the correction the ledger read empty at exactly
        this point, which is what let the next request start another one.
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}

        try:
            with self._real_threads(workers, block="preflight", calls=calls):
                with _Uncooperative._declared_deadlines():
                    reply, _elapsed = await _Uncooperative._finished_within(
                        _Uncooperative._request(instance, message_id="one"),
                        bound=bridge._exchange_budget() + _Uncooperative.SLACK,
                    )

                    assert "socket_unavailable" in reply
                    assert instance._inflight == 0
                    assert await workers.reaches(1) == 1

                    # Not a single sample: the wrapper is cancelled or
                    # resolved on some later turn of the loop, and the
                    # refund this test forbids would happen then.  So the
                    # claim is checked over a window that outlives every
                    # turn the wrapper could possibly end on.
                    for _ in range(50):
                        await asyncio.sleep(0.002)
                        assert workers.live == 1, "the real worker ended on its own"
                        assert bridge._cleanup_charged >= 1, (
                            "capacity came back while the worker thread was still running"
                        )
                        assert len(bridge._stragglers) >= 1

                    await workers.settled()
                    assert await _Uncooperative._drain_stragglers(set()) == set()
                    assert bridge._cleanup_charged == 0, "the permit was never given back"
        finally:
            await workers.settled()
            await TestGlobalDetachedCleanupBudget._quiesced()

    @pytest.mark.asyncio
    async def test_a_saturated_bridge_hands_nothing_to_a_worker_thread(self):
        """At the ceiling the door closes before the executor is touched.

        The refusal has to come before a worker is scheduled, a socket is
        opened, anything is written or any other background work is made —
        otherwise "bounded" would only mean "bounded eventually".
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}
        threads = {"threads": 0}

        try:
            with TestGlobalDetachedCleanupBudget._counted_threads(threads):
                with self._real_threads(workers, block="preflight", calls=calls):
                    with _Uncooperative._declared_deadlines():
                        bound = bridge._exchange_budget() + _Uncooperative.SLACK
                        n = 0
                        while (
                            bridge._cleanup_charged + bridge._CLEANUP_PERMITS_PER_EXCHANGE
                            <= bridge.MAX_DETACHED_CLEANUP_TASKS
                        ):
                            await _Uncooperative._finished_within(
                                _Uncooperative._request(instance, message_id=f"fill-{n}"),
                                bound=bound,
                            )
                            n += 1
                            assert n <= self.SEQUENTIAL, "the bridge never saturated"

                        saturated = (
                            calls["connects"],
                            calls["writes"],
                            threads["threads"],
                            workers.started,
                            bridge._cleanup_charged,
                            len(bridge._stragglers),
                        )
                        assert workers.live, "nothing was actually holding the ceiling"

                        for extra in range(5):
                            reply, _elapsed = await _Uncooperative._finished_within(
                                _Uncooperative._request(instance, message_id=f"over-{extra}"),
                                bound=bound,
                            )
                            assert reply == bridge.refusal_text("cleanup_capacity_exhausted")
                            assert instance._inflight == 0
                            assert (
                                calls["connects"],
                                calls["writes"],
                                threads["threads"],
                                workers.started,
                                bridge._cleanup_charged,
                                len(bridge._stragglers),
                            ) == saturated, "a refused request changed something"
        finally:
            await workers.settled()
            assert await _Uncooperative._drain_stragglers(set()) == set()
            await TestGlobalDetachedCleanupBudget._quiesced()

    @pytest.mark.asyncio
    async def test_capacity_comes_back_only_when_the_real_threads_exit(self):
        """Saturation lifts on the workers' exit, and on nothing else.

        A ceiling that never recovers is as wrong as one that never binds,
        so this proves both directions: refused while the threads run,
        harvested to empty once they exit, and an ordinary exchange
        succeeding afterwards on the same bridge.
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}

        try:
            with self._real_threads(workers, block="preflight", calls=calls):
                with _Uncooperative._declared_deadlines():
                    bound = bridge._exchange_budget() + _Uncooperative.SLACK
                    replies = []
                    for n in range(self.SEQUENTIAL):
                        reply, _elapsed = await _Uncooperative._finished_within(
                            _Uncooperative._request(instance, message_id=f"recover-{n}"),
                            bound=bound,
                        )
                        replies.append(reply)

                    refusal = bridge.refusal_text("cleanup_capacity_exhausted")
                    assert replies[-1] == refusal, "the bridge never saturated"
                    held = workers.live
                    assert held and bridge._cleanup_charged >= held

                    # Releasing the threads is the only thing that happens.
                    await workers.settled()
                    assert await _Uncooperative._drain_stragglers(set()) == set()
                    assert workers.live == 0
                    assert bridge._cleanup_charged == 0, "capacity did not harvest down"
                    assert not bridge._stragglers

            healthy, captured = TestExchange._connection()
            with _socket_layer(healthy):
                after = await _Uncooperative._request(instance, message_id="after-release")

            assert after == "[Stage-A] ACCEPTED_TERMINAL\ndone"
            assert len(captured) == 1
            assert bridge._cleanup_charged == 0
        finally:
            await workers.settled()
            await TestGlobalDetachedCleanupBudget._quiesced()

    @pytest.mark.asyncio
    async def test_a_concurrent_burst_cannot_oversubscribe_the_real_threads(self):
        """Racing for the last permits must not hand out more than exist.

        Sequential saturation cannot catch a check-then-act: only bursts
        that reserve at the same moment can.  The ledger is sampled at the
        one place capacity is ever committed, and the live worker count is
        read on every turn, so a ceiling that was passed and then
        recovered still fails here.
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}
        cap = bridge.MAX_DETACHED_CLEANUP_TASKS
        samples = []
        refusal = bridge.refusal_text("cleanup_capacity_exhausted")
        refused = 0

        try:
            with TestGlobalDetachedCleanupBudget._ledger_samples(samples):
                with self._real_threads(workers, block="preflight", calls=calls):
                    with _Uncooperative._declared_deadlines():
                        for burst in range(4):
                            tasks = [
                                asyncio.ensure_future(
                                    _Uncooperative._request(
                                        instance, message_id=f"burst-{burst}-{n}"
                                    )
                                )
                                for n in range(bridge.MAX_INFLIGHT_REQUESTS)
                            ]
                            done, pending = await asyncio.wait(tasks, timeout=_Uncooperative.OUTER)
                            assert not pending, "an exchange never finished"
                            for task in done:
                                if task.result() == refusal:
                                    refused += 1
                            assert workers.live <= cap, (
                                f"{workers.live} live worker threads against a cap of {cap}"
                            )
                            assert bridge._cleanup_charged <= cap
                            assert instance._inflight == 0

                        assert refused, "the ceiling never engaged, so no race was run"
                        assert workers.peak <= cap
                        assert calls["writes"] == 0

            assert samples, "no reservation was ever attempted"
            assert all(charged <= cap for _granted, charged, _live in samples), (
                "the ceiling was oversubscribed at a reservation"
            )
        finally:
            await workers.settled()
            assert await _Uncooperative._drain_stragglers(set()) == set()
            await TestGlobalDetachedCleanupBudget._quiesced()


class TestExternalCancellationCannotReleaseLiveWork:
    """Cancelling the wrapper is not the same as finishing the work.

    Tier-1 rereview-4C's finding, ``T1R4C-F1``.  Correction 4C stopped
    *this module* from cancelling a thread step's wrapper and parked that
    wrapper in the ledger uncancelled, which held for exactly as long as
    nobody else cancelled it.  But the parked object was still an ordinary
    cancellable task, reachable by anything able to enumerate the loop's
    tasks — the loop's own shutdown included.  One cancellation from
    outside resolved it, the harvest watching it returned the permit, and
    the uninterruptible call it stood for went on running while the room
    charged for it was handed to somebody else.

    So nothing below cancels through the bridge, which has no such path at
    all.  Each falsifier reaches past it, finds the task the loop is
    holding for a hand-off, and cancels *that* — while a real thread is
    really blocked inside the call it wraps, released only by the test.
    """

    #: Comfortably more requests than the ceiling admits, so the bound is
    #: reached rather than merely approached.  Matches the count the
    #: rereviewers used.
    SEQUENTIAL = 12

    @staticmethod
    @contextlib.contextmanager
    def _handoffs(record):
        """Count thread hand-offs and keep the coroutine each one returned.

        The wrapper the loop actually awaits is the task
        ``_Deadline.bounded`` builds around that coroutine, and the bridge
        never exposes it.  Recording the coroutine is what lets a
        falsifier find that task among the loop's own and cancel it from
        outside — reaching the object under test without reaching into the
        implementation, and without weakening the hand-off, which remains
        the genuine :func:`asyncio.to_thread`.
        """
        real = asyncio.to_thread

        def capture(func, /, *args, **kwargs):
            coro = real(func, *args, **kwargs)
            record["coros"].append(coro)
            record["threads"] += 1
            return coro

        with patch.object(asyncio, "to_thread", capture):
            yield

    @staticmethod
    async def _cancel_wrappers(record):
        """Cancel every wrapper still awaiting a hand-off, from outside.

        This is the vector, stated exactly: a third party that can see the
        loop's tasks cancels one the bridge deliberately never cancels.
        The turns afterwards let the cancellation be delivered and let
        every callback the bridge attached to that wrapper run, so an
        early refund has happened by the time the caller asserts.
        """
        live = [
            task
            for task in asyncio.all_tasks()
            if not task.done() and task.get_coro() in record["coros"]
        ]
        for task in live:
            task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        return len(live)

    @staticmethod
    @contextlib.contextmanager
    def _one_worker(workers, *, calls):
        """The socket layer over an executor with exactly one thread.

        A second hand-off then has nowhere to run: its work item waits in
        the executor's queue, entered by no thread at all.  That is the
        one state in which cancelling a wrapper really does end everything
        the step ever was, and there the permit must come back — a ledger
        holding capacity against work nobody will ever do would saturate
        the bridge for good, which is the opposite failure and just as
        forbidden.
        """
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stagea-queued"
        )
        asyncio.get_running_loop().set_default_executor(executor)
        with patch("os.stat", side_effect=_blocking_socket_stat(workers, block="preflight")):
            with patch.object(bridge, "read_peer_uid", return_value=CONTROLLER_UID):
                with patch.object(
                    asyncio,
                    "open_unix_connection",
                    TestRealExecutorWorkIsCharged._connection(calls),
                    create=True,
                ):
                    yield

    @pytest.mark.asyncio
    async def test_cancelling_a_parked_wrapper_keeps_its_permit_until_the_thread_exits(self):
        """The finding itself, stated as the one thing that must not happen.

        A real worker is blocked in the call the step handed off, the
        exchange is over, and the wrapper is then cancelled by somebody
        who is not the bridge.  Before this correction the ledger emptied
        on that cancellation — permit returned, straggler forgotten —
        while the thread went on running, which is what let a later
        request reserve the same room.  Here the permit stays charged for
        as long as the call is really running, and comes back on the
        worker's own exit and on nothing else.
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}
        record = {"coros": [], "threads": 0}

        try:
            with _Uncooperative._no_unobserved_tasks():
                with self._handoffs(record):
                    with TestRealExecutorWorkIsCharged._real_threads(
                        workers, block="preflight", calls=calls
                    ):
                        with _Uncooperative._declared_deadlines():
                            reply, _elapsed = await _Uncooperative._finished_within(
                                _Uncooperative._request(instance, message_id="cancel-me"),
                                bound=bridge._exchange_budget() + _Uncooperative.SLACK,
                            )

                            assert "socket_unavailable" in reply
                            assert instance._inflight == 0
                            assert await workers.reaches(1) == 1
                            assert bridge._cleanup_charged == 1
                            assert len(bridge._stragglers) == 1

                            cancelled = await self._cancel_wrappers(record)
                            assert cancelled == 1, (
                                "no live wrapper was found to cancel, so the vector "
                                "under test was never applied"
                            )

                            # Not one sample: a refund arrives on whichever
                            # later turn the cancellation is delivered on,
                            # so the claim is checked across a window that
                            # outlives every turn it could arrive on.
                            for _ in range(50):
                                await asyncio.sleep(0.002)
                                assert workers.live == 1, "the real worker ended on its own"
                                assert bridge._cleanup_charged == 1, (
                                    "capacity came back on the wrapper's cancellation "
                                    "while the worker thread was still running"
                                )
                                assert len(bridge._stragglers) == 1
                                # And nothing was started in its place: no
                                # retry, no second request, no replacement
                                # cleanup worker.
                                assert (calls["connects"], calls["writes"]) == (0, 0)
                                assert record["threads"] == 1

                            await workers.settled()
                            assert await _Uncooperative._drain_stragglers(set()) == set()
                            assert bridge._cleanup_charged == 0, "the permit was never given back"
                            assert workers.live == 0
        finally:
            await workers.settled()
            await TestGlobalDetachedCleanupBudget._quiesced()

    @classmethod
    async def _sequential(cls, block, *, expected_connects):
        """Drive ``SEQUENTIAL`` requests, cancelling every wrapper as it parks.

        The cancellation is applied after *each* request rather than once
        at the end, because that is the shape that compounds: one early
        refund frees room the next request takes, and the run that
        followed reached twelve live workers against a declared cap of
        eight.  Every invariant that has to hold during the run is
        asserted while it is still running.
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}
        record = {"coros": [], "threads": 0}
        cap = bridge.MAX_DETACHED_CLEANUP_TASKS
        admitted, refused, cancelled = 0, 0, 0

        try:
            with cls._handoffs(record):
                with TestRealExecutorWorkIsCharged._real_threads(
                    workers, block=block, calls=calls
                ):
                    with _Uncooperative._declared_deadlines():
                        bound = bridge._exchange_budget() + _Uncooperative.SLACK
                        for n in range(cls.SEQUENTIAL):
                            was = (calls["connects"], calls["writes"], record["threads"])
                            was_live = workers.live
                            reply, _elapsed = await _Uncooperative._finished_within(
                                _Uncooperative._request(instance, message_id=f"extcancel-{n}"),
                                bound=bound,
                            )
                            spent = (
                                calls["connects"] - was[0],
                                calls["writes"] - was[1],
                                record["threads"] - was[2],
                            )

                            if reply == bridge.refusal_text("cleanup_capacity_exhausted"):
                                refused += 1
                                # Nothing at all: no executor hand-off, no
                                # connect, no write, no new live work.
                                assert spent == (0, 0, 0), f"a refused request created {spent}"
                                assert workers.live == was_live
                            else:
                                admitted += 1
                                assert "socket_unavailable" in reply
                                assert spent == (expected_connects, 0, expected_connects + 1)
                                assert await workers.reaches(was_live + 1) == was_live + 1

                            cancelled += await cls._cancel_wrappers(record)

                            assert instance._inflight == 0, "the visible slot was not returned"
                            assert workers.live <= cap, (
                                f"{workers.live} live worker threads against a cap of {cap}"
                            )
                            assert bridge._cleanup_charged <= cap

                        assert not workers.release.is_set(), "the stall was released"
                        assert workers.live == admitted, (
                            "every admitted request should still own one live worker"
                        )
                        await workers.settled()
        finally:
            await workers.settled()
            assert await _Uncooperative._drain_stragglers(set()) == set()
            await TestGlobalDetachedCleanupBudget._quiesced()

        return admitted, refused, cancelled, workers

    @pytest.mark.asyncio
    async def test_twelve_sequential_cancelled_preflight_wrappers_stay_under_the_ceiling(self):
        """Twelve requests on the preflight hand-off, each wrapper cancelled.

        The rereviewer's count and the rereviewer's vector together.  Each
        admitted request blocks one real thread that will not come back
        and then has its wrapper cancelled from outside; the ceiling has
        to keep binding anyway, and the requests past it have to create
        nothing whatsoever.
        """
        admitted, refused, cancelled, workers = await self._sequential(
            "preflight", expected_connects=0
        )

        assert admitted + refused == self.SEQUENTIAL
        assert refused, "the ceiling never engaged, so nothing was bounded"
        assert cancelled == admitted, "a parked wrapper escaped the external cancellation"
        assert workers.peak <= bridge.MAX_DETACHED_CLEANUP_TASKS
        assert workers.peak == admitted
        assert workers.live == 0, "a released worker never exited"

    @pytest.mark.asyncio
    async def test_twelve_sequential_cancelled_recheck_wrappers_stay_under_the_ceiling(self):
        """The same on the hand-off that runs with a connection already open.

        Closing the preflight and leaving the post-connect identity
        re-check open would move the finding rather than answer it, and
        move it somewhere strictly worse, because by then a socket exists.
        Each admitted request connects once, writes nothing, and hands off
        twice — the preflight that returns, and the re-check that does not
        and is then cancelled.
        """
        admitted, refused, cancelled, workers = await self._sequential(
            "recheck", expected_connects=1
        )

        assert admitted + refused == self.SEQUENTIAL
        assert refused, "the ceiling never engaged, so nothing was bounded"
        assert cancelled == admitted, "a parked wrapper escaped the external cancellation"
        assert workers.peak <= bridge.MAX_DETACHED_CLEANUP_TASKS
        assert workers.peak == admitted
        assert workers.live == 0, "a released worker never exited"

    @pytest.mark.asyncio
    async def test_a_concurrent_burst_of_cancelled_wrappers_cannot_oversubscribe(self):
        """Racing for the last permits, with every wrapper cancelled as it parks.

        Sequential saturation cannot catch a check-then-act; only requests
        that reserve at the same moment can.  The ledger is sampled at the
        one place capacity is ever committed, so a ceiling that was passed
        and then recovered still fails here, and the live worker count is
        read on every turn of the burst.
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}
        record = {"coros": [], "threads": 0}
        cap = bridge.MAX_DETACHED_CLEANUP_TASKS
        samples = []
        refusal = bridge.refusal_text("cleanup_capacity_exhausted")
        refused, cancelled = 0, 0

        try:
            with TestGlobalDetachedCleanupBudget._ledger_samples(samples):
                with self._handoffs(record):
                    with TestRealExecutorWorkIsCharged._real_threads(
                        workers, block="preflight", calls=calls
                    ):
                        with _Uncooperative._declared_deadlines():
                            for burst in range(4):
                                tasks = [
                                    asyncio.ensure_future(
                                        _Uncooperative._request(
                                            instance, message_id=f"race-{burst}-{n}"
                                        )
                                    )
                                    for n in range(bridge.MAX_INFLIGHT_REQUESTS)
                                ]
                                done, pending = await asyncio.wait(
                                    tasks, timeout=_Uncooperative.OUTER
                                )
                                assert not pending, "an exchange never finished"
                                for task in done:
                                    if task.result() == refusal:
                                        refused += 1

                                cancelled += await self._cancel_wrappers(record)

                                assert workers.live <= cap, (
                                    f"{workers.live} live worker threads against a cap of {cap}"
                                )
                                assert bridge._cleanup_charged <= cap
                                assert instance._inflight == 0

                            assert refused, "the ceiling never engaged, so no race was run"
                            assert cancelled, "no wrapper was ever cancelled"
                            assert workers.peak <= cap
                            assert calls["writes"] == 0

            assert samples, "no reservation was ever attempted"
            assert all(charged <= cap for _granted, charged, _live in samples), (
                "the ceiling was oversubscribed at a reservation"
            )
        finally:
            await workers.settled()
            assert await _Uncooperative._drain_stragglers(set()) == set()
            await TestGlobalDetachedCleanupBudget._quiesced()

    @pytest.mark.asyncio
    async def test_capacity_returns_after_the_threads_behind_cancelled_wrappers_exit(self):
        """Recovery, on the workers' exit and on nothing before it.

        A ceiling that never lifts is as wrong as one that never binds, so
        this proves both directions with every wrapper already cancelled:
        refused while the threads run, harvested to empty once they exit,
        and an ordinary exchange succeeding afterwards on the same bridge.
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}
        record = {"coros": [], "threads": 0}
        refusal = bridge.refusal_text("cleanup_capacity_exhausted")

        try:
            with self._handoffs(record):
                with TestRealExecutorWorkIsCharged._real_threads(
                    workers, block="preflight", calls=calls
                ):
                    with _Uncooperative._declared_deadlines():
                        bound = bridge._exchange_budget() + _Uncooperative.SLACK
                        replies = []
                        for n in range(self.SEQUENTIAL):
                            reply, _elapsed = await _Uncooperative._finished_within(
                                _Uncooperative._request(instance, message_id=f"back-{n}"),
                                bound=bound,
                            )
                            replies.append(reply)
                            await self._cancel_wrappers(record)

                        assert replies[-1] == refusal, "the bridge never saturated"
                        held = workers.live
                        assert held, "nothing was actually holding the ceiling"
                        assert bridge._cleanup_charged >= held

                        # A saturated bridge whose wrappers are all gone
                        # still refuses before it touches the executor.
                        was = (calls["connects"], calls["writes"], record["threads"])
                        again, _elapsed = await _Uncooperative._finished_within(
                            _Uncooperative._request(instance, message_id="still-shut"),
                            bound=bound,
                        )
                        assert again == refusal
                        assert (calls["connects"], calls["writes"], record["threads"]) == was

                        # Releasing the threads is the only thing that happens.
                        await workers.settled()
                        assert await _Uncooperative._drain_stragglers(set()) == set()
                        assert workers.live == 0
                        assert bridge._cleanup_charged == 0, "capacity did not harvest down"
                        assert not bridge._stragglers

            healthy, captured = TestExchange._connection()
            with _socket_layer(healthy):
                after = await _Uncooperative._request(instance, message_id="after-cancel")

            assert after == "[Stage-A] ACCEPTED_TERMINAL\ndone"
            assert len(captured) == 1
            assert bridge._cleanup_charged == 0
        finally:
            await workers.settled()
            await TestGlobalDetachedCleanupBudget._quiesced()

    @pytest.mark.asyncio
    async def test_a_wrapper_cancelled_before_its_call_began_returns_its_permit(self):
        """The other direction: room held for work nobody will ever do.

        Tying the permit to the call is only correct if a call that never
        begins releases it.  With a single-thread executor the second
        hand-off is still queued when its wrapper is cancelled, so that
        step ends without any worker having entered it — and its permit
        has to come back at once, or eight such requests would saturate
        the bridge permanently.  The first step's worker is really running
        throughout, so the same instant that must return one permit must
        also keep the other.
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}
        record = {"coros": [], "threads": 0}

        try:
            with self._handoffs(record):
                with self._one_worker(workers, calls=calls):
                    with _Uncooperative._declared_deadlines():
                        bound = bridge._exchange_budget() + _Uncooperative.SLACK
                        for name in ("runs", "queued"):
                            reply, _elapsed = await _Uncooperative._finished_within(
                                _Uncooperative._request(instance, message_id=name),
                                bound=bound,
                            )
                            assert "socket_unavailable" in reply

                        assert record["threads"] == 2, "both steps must have handed off"
                        assert await workers.reaches(1) == 1
                        assert workers.started == 1, "the queued call ran after all"
                        assert bridge._cleanup_charged == 2
                        assert len(bridge._stragglers) == 2

                        assert await self._cancel_wrappers(record) == 2

                        # One call is running and keeps its permit; one
                        # never began and gives its permit straight back.
                        assert bridge._cleanup_charged == 1, (
                            "one running call and one that never began should leave "
                            f"exactly one permit charged, not {bridge._cleanup_charged}"
                        )
                        assert len(bridge._stragglers) == 1
                        assert workers.live == 1, "the running worker was disturbed"

                        # Freeing the only thread must not let the skipped
                        # call start behind a ledger no longer holding room
                        # for it.
                        await workers.settled()
                        assert await _Uncooperative._drain_stragglers(set()) == set()
                        assert workers.started == 1, "the disowned call ran anyway"
                        assert bridge._cleanup_charged == 0
                        assert not bridge._stragglers
        finally:
            await workers.settled()
            await TestGlobalDetachedCleanupBudget._quiesced()


class TestAnAbandonedThreadStepThatRaisesIsStillObserved:
    """Deep rereview-4D's finding, ``D4D-F1``.

    A thread step's permit is held against the call, and the wrapper the
    loop awaits is watched only so that call's outcome is read.  The
    watch was conditional: :meth:`_CleanupBudget.abandon` attached it
    when the call had been parked, or when the wrapper had still to
    resolve, and in neither case otherwise.  The state that is neither is
    reachable — a call that has already ended, whose wrapper is therefore
    already resolved — but only through the *cancellation* path, never
    the timeout: ``asyncio.wait`` reports a wrapper that completed on the
    turn after it completed, and an outer cancellation can win that turn.
    The coroutine branch a few lines below harvests exactly this case;
    the thread branch dropped it.

    ``asyncio.wait`` reads no child's outcome, so what fell through was
    the exception the call really raised, read by nobody at all — and the
    loop reports a deliberate, correctly-accounted abandonment as ``Task
    exception was never retrieved``.  The ledger was never wrong here;
    the observation was missing, which is the whole of the finding and
    the whole of what the falsifier below states.

    Nothing in it is sampled or approximated: a genuine synchronous call
    blocks a genuine worker thread and exits by *raising*, and the
    exchange is cancelled from outside the loop's view of that wrapper
    the instant it resolves — necessarily before ``bounded``, which
    cannot resume until a later turn, has consumed it.
    """

    @staticmethod
    @contextlib.contextmanager
    def _preflight_that_raises(workers, *, calls, failures):
        """The socket layer, with the preflight stat failing in its thread.

        The stat blocks a real worker until the test releases it and then
        raises, so the wrapper's exception was really produced inside the
        executor rather than handed to it by a double.
        :func:`check_socket_path` turns it into the bridge's own
        fail-closed error, which is what a filesystem that wedges and
        then fails really produces at this step.

        The connect and peer layers are doubled as everywhere else, so
        ``calls`` staying at zero is evidence that the cancelled exchange
        opened no socket and wrote nothing — not merely that it had no
        chance to.
        """
        real_stat = os.stat

        def stat_fn(path, *args, **kwargs):
            if str(path) != SOCKET:
                return real_stat(path, *args, **kwargs)
            workers.block()
            failures["n"] += 1
            raise OSError("the preflight stat failed inside its worker thread")

        with patch("os.stat", side_effect=stat_fn):
            with patch.object(bridge, "read_peer_uid", return_value=CONTROLLER_UID):
                with patch.object(
                    asyncio,
                    "open_unix_connection",
                    TestRealExecutorWorkIsCharged._connection(calls),
                    create=True,
                ):
                    yield

    @staticmethod
    def _wrapper_of(record):
        """The task the loop is holding for the single hand-off made here.

        Found the way the external-cancellation falsifiers find it —
        among the loop's own tasks, by the coroutine
        :func:`asyncio.to_thread` returned — so the vector reaches the
        object under test without reaching inside :class:`_Deadline`.
        """
        live = [
            task
            for task in asyncio.all_tasks()
            if not task.done() and task.get_coro() in record["coros"]
        ]
        assert len(live) == 1, f"expected exactly one live wrapper, found {len(live)}"
        return live[0]

    @classmethod
    async def _cancelled_the_turn_its_wrapper_failed(cls, workers, record, instance):
        """Run the vector, and let go of both tasks before returning.

        The interleaving is not sampled.  The cancellation is issued from
        the wrapper's own completion callback, so it is necessarily
        issued after that wrapper has resolved and before ``bounded`` —
        which cannot resume before a later turn — has consumed it.  Both
        orderings that produces are the state under test: either the
        waiter ``asyncio.wait`` is holding is still pending and is
        cancelled outright, or it has already been resolved and the
        cancellation is delivered into the resume itself.  Either way
        ``asyncio.wait`` raises instead of returning the completed
        wrapper, and the release path is entered with a wrapper that is
        already done and a call that is already over.

        Neither task escapes this frame, which is the point of it: a
        wrapper the caller still references is never finalized, and it is
        the finalizer that reports an exception nobody read.
        """
        exchange = asyncio.ensure_future(
            _Uncooperative._request(instance, message_id="raises-then-cancelled")
        )
        assert await workers.reaches(1) == 1, "the preflight never reached a real thread"
        assert bridge._cleanup_charged == bridge._CLEANUP_PERMITS_PER_EXCHANGE
        assert instance._inflight == 1

        wrapper = cls._wrapper_of(record)
        wrapper.add_done_callback(lambda _task: exchange.cancel())

        workers.release.set()
        done, _pending = await asyncio.wait({exchange}, timeout=_Uncooperative.OUTER)
        assert done, f"the exchange was still unfinished after {_Uncooperative.OUTER}s"
        assert exchange.cancelled(), "the outer cancellation never landed on the exchange"
        # Deliberately not read here: reading it *is* the harvest, so a
        # falsifier that asked what the wrapper carried would perform the
        # very observation whose absence it exists to detect.
        assert wrapper.done(), "the wrapper had not resolved when the exchange ended"
        assert not wrapper.cancelled(), "the wrapper was cancelled rather than failed"

    @pytest.mark.asyncio
    async def test_a_thread_step_that_raises_is_read_when_cancellation_wins_the_resume(self):
        """The finding itself, stated as the one thing that must not happen.

        Everything the accepted ledger promises is asserted alongside it,
        because the correction is only allowed to add an observation: the
        call really ran and really ended, its permits came back, nothing
        was left in the straggler set, and no socket, write or second
        hand-off was made in place of the step that was let go.
        """
        await TestGlobalDetachedCleanupBudget._quiesced()
        workers = _RealWorkers()
        instance = _Uncooperative._bridge()
        calls = {"connects": 0, "writes": 0}
        record = {"coros": [], "threads": 0}
        failures = {"n": 0}

        try:
            # Anything an earlier test left uncollected is collected here,
            # so the only unobserved task the oracle can report below is
            # one this test created.
            gc.collect()
            with _Uncooperative._no_unobserved_tasks():
                with TestExternalCancellationCannotReleaseLiveWork._handoffs(record):
                    with self._preflight_that_raises(workers, calls=calls, failures=failures):
                        with _Uncooperative._declared_deadlines(step=5.0, reply=5.0):
                            await self._cancelled_the_turn_its_wrapper_failed(
                                workers, record, instance
                            )

                        assert failures["n"] == 1, "the blocking call did not exit by raising"
                        assert workers.started == 1, "no real worker ever entered the call"
                        assert workers.live == 0, "the call did not really end"
                        assert record["threads"] == 1, (
                            "a replacement hand-off was made; observing an outcome may "
                            "start no new work"
                        )
                        assert (calls["connects"], calls["writes"]) == (0, 0), (
                            "the cancelled exchange retried or opened a second socket"
                        )
                        assert instance._inflight == 0
                        assert bridge._cleanup_charged == 0, (
                            "the exchange's permits were not all returned"
                        )
                        assert not bridge._stragglers

                # Both tasks died with the frame that made them, so this
                # is where an exception nobody read is reported — by the
                # loop's own handler, from the wrapper's finalizer, while
                # the oracle above is still watching.
                gc.collect()
        finally:
            await workers.settled()
            await TestGlobalDetachedCleanupBudget._quiesced()
