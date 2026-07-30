"""Slack media delivery for send_message.

Covers ``plugins/platforms/slack/adapter.py::_standalone_send`` media path:
text+file, media-only, caption-on-upload, missing-file warnings.

``slack_sdk`` is optional in CI, so tests inject a fake module into
``sys.modules`` (same pattern as ``tests/gateway/test_slack.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter, _standalone_send
from tools.send_message_tool import _send_to_platform


def _pconfig(token: str = "xoxb-test"):
    return SimpleNamespace(token=token, extra={})


def _tmpfile(suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"%PDF-1.4 test")
    f.close()
    return f.name


def _mock_client(*, post_ok=True, upload_ok=True):
    client = MagicMock()
    client.chat_postMessage = AsyncMock(
        return_value={
            "ok": post_ok,
            "ts": "111.222",
            "error": None if post_ok else "channel_not_found",
        }
    )
    if upload_ok:
        client.files_upload_v2 = AsyncMock(
            return_value={
                "ok": True,
                "file": {
                    "id": "F123",
                    "timestamp": 1234567890,
                    "shares": {"public": {"C012AB3CD": [{"ts": "333.444"}]}},
                },
            }
        )
    else:
        client.files_upload_v2 = AsyncMock(
            return_value={"ok": False, "error": "not_in_channel"}
        )
    return client


def _live_adapter(client):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._app = SimpleNamespace(client=client)
    adapter._get_client = lambda _chat_id, team_id=None: client
    return adapter


@contextlib.contextmanager
def _fake_slack_sdk(client):
    """Make ``from slack_sdk.web.async_client import AsyncWebClient`` resolve to a factory."""
    sdk = ModuleType("slack_sdk")
    web = ModuleType("slack_sdk.web")
    async_client = ModuleType("slack_sdk.web.async_client")
    async_client.AsyncWebClient = MagicMock(return_value=client)
    sdk.web = web
    web.async_client = async_client

    modules = {
        "slack_sdk": sdk,
        "slack_sdk.web": web,
        "slack_sdk.web.async_client": async_client,
    }
    old = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, prev in old.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def test_text_plus_pdf_uploads_via_files_upload_v2():
    pdf = _tmpfile(".pdf")
    client = _mock_client()
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "C012AB3CD",
                    "Here is the report",
                    media_files=[(pdf, False)],
                )
            )
        assert result["success"] is True
        assert result["platform"] == "slack"
        client.chat_postMessage.assert_awaited_once()
        client.files_upload_v2.assert_awaited_once()
        upload_kwargs = client.files_upload_v2.await_args.kwargs
        assert upload_kwargs["channel"] == "C012AB3CD"
        assert upload_kwargs["file"] == pdf
        assert upload_kwargs["filename"] == os.path.basename(pdf)
        assert upload_kwargs["initial_comment"] == ""
    finally:
        os.unlink(pdf)


def test_media_only_skips_text_post():
    pdf = _tmpfile(".pdf")
    client = _mock_client()
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "C012AB3CD",
                    "",
                    media_files=[(pdf, False)],
                )
            )
        assert result["success"] is True
        client.chat_postMessage.assert_not_awaited()
        client.files_upload_v2.assert_awaited_once()
    finally:
        os.unlink(pdf)


def test_caption_rides_initial_comment_no_separate_text():
    pdf = _tmpfile(".pdf")
    client = _mock_client()
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "C012AB3CD",
                    "",
                    media_files=[(pdf, False)],
                    caption="Q3 summary PDF",
                )
            )
        assert result["success"] is True
        client.chat_postMessage.assert_not_awaited()
        upload_kwargs = client.files_upload_v2.await_args.kwargs
        assert upload_kwargs["initial_comment"] == "Q3 summary PDF"
    finally:
        os.unlink(pdf)


def test_missing_media_file_warns_and_falls_back_caption():
    client = _mock_client()
    with _fake_slack_sdk(client):
        result = asyncio.run(
            _standalone_send(
                _pconfig(),
                "C012AB3CD",
                "",
                media_files=[("/no/such/file.pdf", False)],
                caption="still deliver this",
            )
        )
    assert result["success"] is True
    assert result.get("warnings")
    assert any("not found" in w.lower() for w in result["warnings"])
    client.chat_postMessage.assert_awaited_once()
    assert client.chat_postMessage.await_args.kwargs["text"] == "still deliver this"
    client.files_upload_v2.assert_not_awaited()


def test_durable_atomic_standalone_text_is_exactly_one_preformatted_call(monkeypatch):
    import aiohttp

    durable_config = _pconfig()
    durable_config._durable_delivery = True
    durable_config._durable_preformatted_atomic_text = True
    literal = "**already-final Slack mrkdwn** & literal"
    provider_payloads = []

    class _Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            return {"ok": True, "ts": "111.222"}

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, _url, *, json, **_kwargs):
            provider_payloads.append(json)
            return _Response()

    monkeypatch.setattr(aiohttp, "ClientSession", lambda **_kwargs: _Session())

    result = asyncio.run(
        _standalone_send(
            durable_config,
            "C012AB3CD",
            literal,
            media_files=[],
        )
    )

    assert result.get("success") is True, result
    assert provider_payloads == [
        {"channel": "C012AB3CD", "text": literal, "mrkdwn": True}
    ]


def test_durable_atomic_standalone_never_falls_back_to_second_token(monkeypatch):
    import aiohttp

    durable_config = _pconfig("token-a,token-b")
    durable_config._durable_delivery = True
    durable_config._durable_preformatted_atomic_text = True
    provider_tokens = []

    class _Response:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            if len(provider_tokens) == 1:
                return {"ok": False, "error": "channel_not_found"}
            return {"ok": True, "ts": "must-not-happen"}

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, _url, *, headers, **_kwargs):
            provider_tokens.append(headers["Authorization"])
            return _Response()

    monkeypatch.setattr(aiohttp, "ClientSession", lambda **_kwargs: _Session())

    result = asyncio.run(
        _standalone_send(
            durable_config,
            "C012AB3CD",
            "frozen provider bytes",
            media_files=[],
        )
    )

    assert provider_tokens == ["Bearer token-a"]
    assert result == {"error": "Slack API error: channel_not_found"}


def test_durable_atomic_standalone_propagates_provider_retry_after(monkeypatch):
    import aiohttp

    durable_config = _pconfig("token-a")
    durable_config._durable_delivery = True
    durable_config._durable_preformatted_atomic_text = True
    provider_calls = []

    class _Response:
        headers = {"Retry-After": "1534.4"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            return {"ok": False, "error": "ratelimited"}

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            provider_calls.append(1)
            return _Response()

    monkeypatch.setattr(aiohttp, "ClientSession", lambda **_kwargs: _Session())

    result = asyncio.run(
        _standalone_send(
            durable_config,
            "C012AB3CD",
            "frozen provider bytes",
            media_files=[],
        )
    )

    assert provider_calls == [1]
    assert result == {
        "error": "Slack API error: ratelimited",
        "error_kind": "rate_limited",
        "retry_after": 1534.4,
    }


def test_durable_atomic_send_tool_does_not_resplit_provider_chunk(monkeypatch):
    from gateway.platform_registry import platform_registry

    durable_config = _pconfig()
    durable_config._durable_delivery = True
    durable_config._durable_preformatted_atomic_text = True
    literal = "*final* Slack mrkdwn"
    entry = SimpleNamespace(max_message_length=SlackAdapter.MAX_MESSAGE_LENGTH)
    monkeypatch.setattr(
        platform_registry,
        "get",
        lambda name: entry if name == "slack" else None,
    )
    send_once = AsyncMock(return_value={"success": True, "message_id": "m-1"})
    monkeypatch.setattr("tools.send_message_tool._send_via_adapter", send_once)

    result = asyncio.run(
        _send_to_platform(
            Platform.SLACK,
            durable_config,
            "C012AB3CD",
            literal,
            media_files=[],
        )
    )

    assert result.get("success") is True, result
    send_once.assert_awaited_once_with(
        Platform.SLACK,
        durable_config,
        "C012AB3CD",
        literal,
        thread_id=None,
        media_files=[],
        force_document=False,
    )


def test_upload_exception_after_provider_call_is_ack_unknown():
    pdf = _tmpfile(".pdf")
    client = _mock_client()
    client.files_upload_v2 = AsyncMock(
        side_effect=RuntimeError("provider response lost after upload")
    )
    durable_config = _pconfig()
    durable_config._durable_delivery = True
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(
                _standalone_send(
                    durable_config,
                    "C012AB3CD",
                    "",
                    media_files=[(pdf, False)],
                    caption="durable caption",
                )
            )
        assert result["error"].startswith("ACK_UNKNOWN:")
        assert "provider response lost after upload" in result["error"]
        client.files_upload_v2.assert_awaited_once()
    finally:
        os.unlink(pdf)


def test_receiptless_upload_is_ack_unknown():
    pdf = _tmpfile(".pdf")
    client = _mock_client()
    client.files_upload_v2 = AsyncMock(return_value={"ok": True, "file": {}})
    durable_config = _pconfig()
    durable_config._durable_delivery = True
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(
                _standalone_send(
                    durable_config,
                    "C012AB3CD",
                    "",
                    media_files=[(pdf, False)],
                    caption="durable caption",
                )
            )
        assert result["error"].startswith("ACK_UNKNOWN:")
        assert "receipt" in result["error"].lower()
        client.files_upload_v2.assert_awaited_once()
    finally:
        os.unlink(pdf)


def test_files_array_id_preserves_nondurable_upload_success_contract():
    pdf = _tmpfile(".pdf")
    client = _mock_client()
    client.files_upload_v2 = AsyncMock(
        return_value={"ok": True, "files": [{"id": "F-files-array-123"}]}
    )
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "C012AB3CD",
                    "",
                    media_files=[(pdf, False)],
                    caption="exact caption",
                )
            )
        assert result["success"] is True
        assert result["message_id"] == "F-files-array-123"
        client.chat_postMessage.assert_not_awaited()
        assert client.files_upload_v2.await_args.kwargs["initial_comment"] == "exact caption"
    finally:
        os.unlink(pdf)


def test_receiptless_upload_preserves_nondurable_success_contract():
    pdf = _tmpfile(".pdf")
    client = _mock_client()
    client.files_upload_v2 = AsyncMock(return_value={"ok": True, "file": {}})
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "C012AB3CD",
                    "",
                    media_files=[(pdf, False)],
                    caption="exact caption",
                )
            )
        assert result["success"] is True
        assert result["message_id"] is None
        client.chat_postMessage.assert_not_awaited()
        assert client.files_upload_v2.await_args.kwargs["initial_comment"] == "exact caption"
    finally:
        os.unlink(pdf)


def test_live_text_post_dispatch_exception_is_ack_unknown_for_durable_send():
    client = _mock_client()
    client.chat_postMessage = AsyncMock(
        side_effect=RuntimeError("provider response lost after send")
    )
    adapter = _live_adapter(client)

    result = asyncio.run(
        adapter.send(
            "C012AB3CD",
            "durable text",
            metadata={"_durable_delivery": True},
        )
    )

    assert result.success is False
    assert result.error.startswith("ACK_UNKNOWN:")
    client.chat_postMessage.assert_awaited_once()


def test_live_text_receiptless_success_is_ack_unknown_for_durable_send():
    client = _mock_client()
    client.chat_postMessage = AsyncMock(return_value={"ok": True})
    adapter = _live_adapter(client)

    result = asyncio.run(
        adapter.send(
            "C012AB3CD",
            "durable text",
            metadata={"_durable_delivery": True},
        )
    )

    assert result.success is False
    assert result.error.startswith("ACK_UNKNOWN:")
    assert "receipt" in result.error.lower()
    client.chat_postMessage.assert_awaited_once()


def test_live_text_receiptless_success_preserves_nondurable_adapter_contract():
    client = _mock_client()
    client.chat_postMessage = AsyncMock(return_value={"ok": True})
    adapter = _live_adapter(client)

    result = asyncio.run(adapter.send("C012AB3CD", "interactive text"))

    assert result.success is True
    assert result.message_id is None
    client.chat_postMessage.assert_awaited_once()


def test_missing_token_errors(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    result = asyncio.run(
        _standalone_send(
            _pconfig(token=""),
            "C012AB3CD",
            "hi",
            media_files=[("/tmp/x.pdf", False)],
        )
    )
    assert "error" in result
    assert "SLACK_BOT_TOKEN" in result["error"]


def test_thread_id_passed_to_upload():
    pdf = _tmpfile(".pdf")
    client = _mock_client()
    try:
        with _fake_slack_sdk(client):
            asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "C012AB3CD",
                    "",
                    thread_id="999.000",
                    media_files=[(pdf, False)],
                )
            )
        assert client.files_upload_v2.await_args.kwargs["thread_ts"] == "999.000"
    finally:
        os.unlink(pdf)


def test_send_to_platform_routes_slack_media():
    """_send_to_platform must call Slack standalone_sender with media_files."""
    import httpx

    if not hasattr(httpx, "Proxy") or not hasattr(httpx, "URL"):
        pytest.skip("httpx type annotations incompatible with telegram library")

    from gateway.config import Platform
    from hermes_cli.plugins import discover_plugins
    from gateway.platform_registry import platform_registry
    from tools.send_message_tool import _send_to_platform

    pdf = _tmpfile(".pdf")
    discover_plugins()
    entry = platform_registry.get("slack")
    assert entry is not None and entry.standalone_sender_fn is not None
    original = entry.standalone_sender_fn
    mock_sender = AsyncMock(
        return_value={"success": True, "platform": "slack", "message_id": "1.2"}
    )
    entry.standalone_sender_fn = mock_sender
    try:
        result = asyncio.run(
            _send_to_platform(
                Platform.SLACK,
                _pconfig(),
                "C012AB3CD",
                "Here is the report",
                media_files=[(pdf, False)],
            )
        )
        assert result["success"] is True
        mock_sender.assert_awaited()
        call_kwargs = mock_sender.await_args.kwargs
        assert call_kwargs.get("media_files") == [(pdf, False)]
        # Single captionable file + short text → caption rides the upload.
        assert call_kwargs.get("caption") == "Here is the report"
        assert not result.get("warnings")
    finally:
        entry.standalone_sender_fn = original
        os.unlink(pdf)
