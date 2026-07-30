import asyncio
import hashlib
import inspect
import json
import os
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cron import scheduler


def _path_identity(path: Path) -> dict:
    if not path.exists() and not path.is_symlink():
        return {"exists": False}
    metadata = os.lstat(path)
    is_regular = stat.S_ISREG(metadata.st_mode)
    return {
        "exists": True,
        "dev": metadata.st_dev,
        "inode": metadata.st_ino,
        "type": stat.S_IFMT(metadata.st_mode),
        "target": os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else None,
        "size": metadata.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if is_regular else None,
        "mode": stat.S_IMODE(metadata.st_mode),
        "mtime_ns": metadata.st_mtime_ns,
    }


def test_physical_home_isolates_launchd_resolution_children_and_outside_sentinel(
    tmp_path: Path,
) -> None:
    import pwd

    physical_home = Path(os.environ["HERMES_TEST_PHYSICAL_HOME"])
    sentinel = Path(os.environ["HERMES_TEST_OUTSIDE_SENTINEL"])
    baseline = json.loads(os.environ["HERMES_TEST_OUTSIDE_SENTINEL_BASELINE"])

    assert physical_home.is_dir()
    assert not physical_home.is_symlink()
    assert physical_home.absolute() == physical_home.resolve()
    assert physical_home == Path.home() == Path(os.path.expanduser("~"))
    assert physical_home == Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert physical_home == Path(os.environ["HOME"])
    assert physical_home == Path(os.environ["USERPROFILE"])
    assert physical_home == Path(os.environ["HERMES_HOME"]).parent
    assert physical_home == tmp_path / "physical-home"

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,os,pwd; from pathlib import Path; "
                "print(json.dumps([str(Path.home()), os.path.expanduser('~'), "
                "pwd.getpwuid(os.getuid()).pw_dir, os.environ['HOME'], "
                "os.environ['HERMES_HOME']]))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    child_home, child_expanded, child_pwd, child_env, child_hermes = json.loads(
        child.stdout
    )
    assert [child_home, child_expanded, child_pwd, child_env] == [
        str(physical_home)
    ] * 4
    assert child_hermes == str(physical_home / ".hermes")

    from hermes_cli import gateway as gateway_cli

    assert gateway_cli._launchd_user_home() == physical_home
    assert gateway_cli.get_launchd_plist_path().is_relative_to(physical_home)
    with pytest.raises(RuntimeError, match="physical-home guard"):
        subprocess.run(
            ["launchctl", "unload", str(sentinel)],
            check=False,
            capture_output=True,
        )
    assert _path_identity(sentinel) == baseline


def test_scheduler_commits_all_fanout_rows_before_first_send_and_checkpoints_each_target(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    targets = [
        {"platform": "telegram", "chat_id": "chat-a", "thread_id": "topic-1"},
        {"platform": "discord", "chat_id": "chat-b", "thread_id": None},
    ]
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: targets)
    observed = []

    def fake_legacy(job, content, adapters=None, loop=None):
        db = tmp_path / "home" / "state.db"
        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT canonical_target, state FROM cron_delivery_outbox_v2 "
                "ORDER BY canonical_target"
            ).fetchall()
        observed.append((job["deliver"], content, rows))
        if "telegram" in job["deliver"]:
            job["_durable_provider_message_id"] = "provider-ack-telegram"
            return None
        return "definitive rejection"

    monkeypatch.setattr(scheduler, "_deliver_result_legacy", fake_legacy, raising=False)
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})

    error = scheduler._deliver_result(
        {"id": "job-1", "name": "fanout", "deliver": "ignored"},
        "durable payload",
        execution_id="exec-1",
    )

    assert error == "definitive rejection"
    assert len(observed) == 2
    # The whole fanout is durably recorded before the first provider side effect.
    assert len(observed[0][2]) == 2
    assert sorted(state for _target, state in observed[0][2]) == ["attempting", "pending"]

    with sqlite3.connect(tmp_path / "home" / "state.db") as conn:
        states = dict(
            conn.execute(
                "SELECT canonical_target, state FROM cron_delivery_outbox_v2"
            ).fetchall()
        )
    assert states == {
        "discord:chat-b": "failed",
        "telegram:chat-a:topic-1": "delivered",
    }


def test_weixin_multichunk_retry_never_replays_an_acked_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.delivery import DeliveryTransport
    from gateway.platforms.weixin import WeixinAdapter

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target = {"platform": "weixin", "chat_id": "owner", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler,
        "load_config",
        lambda: {"cron": {"wrap_response": False}},
    )

    pconfig = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"account_id": "test-account"},
    )
    gateway_config = GatewayConfig(platforms={Platform.WEIXIN: pconfig})
    adapter = WeixinAdapter(pconfig)
    adapter._send_session = object()
    adapter._token = "test-token"
    transport = DeliveryTransport(
        adapter=adapter,
        config=pconfig,
        transport_platform=Platform.WEIXIN,
    )
    monkeypatch.setattr(
        "gateway.config.load_gateway_config", lambda: gateway_config
    )
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: transport,
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_structured_transport_identity",
        lambda _target, _adapters: ("native", None, "a" * 64),
    )
    monkeypatch.setattr(
        scheduler,
        "_resolved_transport_identity_sha256",
        lambda *_args, **_kwargs: "a" * 64,
    )

    provider_calls: list[str] = []
    failed_once = False
    cooldown = 1534.4

    async def fake_send_text_chunk(**kwargs):
        nonlocal failed_once
        chunk = str(kwargs["chunk"])
        provider_calls.append(chunk)
        if len(provider_calls) == 2 and not failed_once:
            failed_once = True
            raise RuntimeError(
                "iLink sendmessage rate limited; cooldown active for 1534.4s"
            )

    monkeypatch.setattr(adapter, "_send_text_chunk", fake_send_text_chunk)
    monkeypatch.setattr(adapter, "_rate_limit_cooldown_remaining", lambda: cooldown)

    class ImmediateFuture:
        def __init__(self, coro):
            self._result = asyncio.run(coro)

        def result(self, timeout):
            assert timeout == 60
            return self._result

        def cancel(self):
            return False

    monkeypatch.setattr(
        "agent.async_utils.safe_schedule_threadsafe",
        lambda coro, _loop: ImmediateFuture(coro),
    )
    loop = SimpleNamespace(is_running=lambda: True)
    long_text = "A" * (adapter.MAX_MESSAGE_LENGTH + 501)

    error = scheduler._deliver_result(
        {"id": "job-weixin-chunks", "deliver": "weixin:owner"},
        long_text,
        adapters={Platform.WEIXIN: adapter},
        loop=loop,
        execution_id="exec-weixin-chunks",
    )
    assert error is not None and "rate limited" in error

    with sqlite3.connect(home / "state.db") as conn:
        rows = conn.execute(
            "SELECT state, payload_json, next_attempt_at "
            "FROM cron_delivery_outbox_v2 "
            "ORDER BY CAST(json_extract(payload_json, '$.unit_index') AS INTEGER)"
        ).fetchall()
        assert [state for state, _payload, _due in rows] == ["delivered", "failed"]
        failed_payload = json.loads(
            next(payload for state, payload, _due in rows if state == "failed")
        )
        assert failed_payload["chunk_index"] == 1
        assert failed_payload["chunk_total"] == 2
        failed_due = next(due for state, _payload, due in rows if state == "failed")
        assert failed_due >= 1534.0
        conn.execute(
            "UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0 WHERE state='failed'"
        )
        conn.execute("UPDATE cron_delivery_circuit_v2 SET blocked_until=0")
        conn.commit()

    cooldown = 0.0
    assert scheduler._recover_one_structured_delivery(
        adapters={Platform.WEIXIN: adapter}, loop=loop
    ) == 1

    assert len(provider_calls) == 3
    assert provider_calls.count(provider_calls[0]) == 1
    with sqlite3.connect(home / "state.db") as conn:
        assert conn.execute(
            "SELECT state, COUNT(*) FROM cron_delivery_outbox_v2 GROUP BY state"
        ).fetchall() == [("delivered", 2)]

    blocked_calls: list[str] = []

    async def reject_predecessor(**kwargs):
        blocked_calls.append(str(kwargs["chunk"]))
        raise RuntimeError("provider forbidden: definitive rejection")

    monkeypatch.setattr(adapter, "_send_text_chunk", reject_predecessor)
    blocked_error = scheduler._deliver_result(
        {"id": "job-weixin-blocked-successor", "deliver": "weixin:owner"},
        long_text,
        adapters={Platform.WEIXIN: adapter},
        loop=loop,
        execution_id="exec-weixin-blocked-successor",
    )
    assert blocked_error is not None and "forbidden" in blocked_error
    assert len(blocked_calls) == 1
    with sqlite3.connect(home / "state.db") as conn:
        blocked_states = conn.execute(
            "SELECT state FROM cron_delivery_outbox_v2 "
            "WHERE execution_id='exec-weixin-blocked-successor' "
            "ORDER BY CAST(json_extract(payload_json, '$.unit_index') AS INTEGER)"
        ).fetchall()
    assert blocked_states == [("abandoned",), ("pending",)]

    # Exercise the actual provider boundary rather than stubbing the adapter's
    # chunk helper: once `_send_message` starts, a generic response-loss error is
    # ACK-ambiguous and must never authorize adapter-local or outbox replay.
    monkeypatch.delattr(adapter, "_send_text_chunk")
    ambiguous_calls: list[str] = []

    async def accepted_then_response_lost(*_args, **kwargs):
        ambiguous_calls.append(str(kwargs["text"]))
        raise RuntimeError("provider response lost after acceptance")

    monkeypatch.setattr(
        "gateway.platforms.weixin._send_message",
        accepted_then_response_lost,
    )
    ambiguous_error = scheduler._deliver_result(
        {"id": "job-weixin-ack-unknown", "deliver": "weixin:owner"},
        "one atomic provider operation",
        adapters={Platform.WEIXIN: adapter},
        loop=loop,
        execution_id="exec-weixin-ack-unknown",
    )
    assert ambiguous_error is not None and ambiguous_error.startswith("ACK_UNKNOWN:")
    assert len(ambiguous_calls) == 1
    with sqlite3.connect(home / "state.db") as conn:
        assert conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2 "
            "WHERE execution_id='exec-weixin-ack-unknown'"
        ).fetchone() == ("unknown", 1)
    assert scheduler._recover_one_structured_delivery(
        adapters={Platform.WEIXIN: adapter}, loop=loop
    ) == 0
    assert len(ambiguous_calls) == 1


def test_weixin_provider_chunk_plan_exact_boundary_is_stable(monkeypatch) -> None:
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.delivery import DeliveryTransport
    from gateway.platforms.weixin import WeixinAdapter

    pconfig = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"account_id": "test-account"},
    )
    adapter = WeixinAdapter(pconfig)
    gateway_config = GatewayConfig(platforms={Platform.WEIXIN: pconfig})
    transport = DeliveryTransport(
        adapter=adapter,
        config=pconfig,
        transport_platform=Platform.WEIXIN,
    )
    monkeypatch.setattr(
        "gateway.config.load_gateway_config", lambda: gateway_config
    )
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: transport,
    )

    exact = scheduler._weixin_atomic_provider_chunks(
        "A" * adapter.MAX_MESSAGE_LENGTH,
        adapters={Platform.WEIXIN: adapter},
    )
    overflow = scheduler._weixin_atomic_provider_chunks(
        "A" * (adapter.MAX_MESSAGE_LENGTH + 1),
        adapters={Platform.WEIXIN: adapter},
    )
    overflow_again = scheduler._weixin_atomic_provider_chunks(
        "A" * (adapter.MAX_MESSAGE_LENGTH + 1),
        adapters={Platform.WEIXIN: adapter},
    )

    assert len(exact) == 1
    assert len(overflow) == 2
    assert overflow_again == overflow
    assert [hashlib.sha256(chunk.encode()).hexdigest() for chunk in overflow_again] == [
        hashlib.sha256(chunk.encode()).hexdigest() for chunk in overflow
    ]
    assert all(len(chunk) <= adapter.MAX_MESSAGE_LENGTH for chunk in overflow)


def test_slack_provider_chunk_plan_freezes_formatted_exact_chunks(monkeypatch) -> None:
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.delivery import DeliveryTransport
    from plugins.platforms.slack.adapter import SlackAdapter

    pconfig = PlatformConfig(enabled=True, token="test-token")
    adapter = SlackAdapter(pconfig)
    gateway_config = GatewayConfig(platforms={Platform.SLACK: pconfig})
    transport = DeliveryTransport(
        adapter=adapter,
        config=pconfig,
        transport_platform=Platform.SLACK,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: transport,
    )
    content = "**header**\n" + "A" * (adapter.MAX_MESSAGE_LENGTH + 100)

    chunks = scheduler._slack_atomic_provider_chunks(
        content, adapters={Platform.SLACK: adapter}
    )
    chunks_again = scheduler._slack_atomic_provider_chunks(
        content, adapters={Platform.SLACK: adapter}
    )

    assert chunks_again == chunks
    assert len(chunks) == 2
    assert all(chunk and len(chunk) <= adapter.MAX_MESSAGE_LENGTH for chunk in chunks)
    assert "*header*" in chunks[0]
    assert all("**header**" not in chunk for chunk in chunks)


def test_slack_standalone_retry_after_persists_through_scheduler_and_reopen(
    tmp_path: Path, monkeypatch
) -> None:
    import aiohttp

    from cron.durable_delivery import StructuredDeliveryOutbox
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    pconfig = PlatformConfig(enabled=True, token="token-a,token-b")
    config = GatewayConfig(platforms={Platform.SLACK: pconfig})
    target = {"platform": "slack", "chat_id": "C012AB3CD", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    discover_plugins()
    entry = platform_registry.get("slack")
    assert entry is not None and entry.standalone_sender_fn is not None

    provider_tokens = []

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

        def post(self, _url, *, headers, **_kwargs):
            provider_tokens.append(headers["Authorization"])
            return _Response()

    monkeypatch.setattr(aiohttp, "ClientSession", lambda **_kwargs: _Session())
    started_at = time.time()

    error = scheduler._deliver_result(
        {"id": "job-slack-rate-limit", "deliver": "slack:C012AB3CD"},
        "provider payload",
        execution_id="exec-slack-rate-limit",
    )

    assert error is not None and "ratelimited" in error
    assert provider_tokens == ["Bearer token-a"]
    with sqlite3.connect(home / "state.db") as conn:
        row = conn.execute(
            "SELECT state, attempts, next_attempt_at FROM cron_delivery_outbox_v2"
        ).fetchone()
        circuit = conn.execute(
            "SELECT platform, blocked_until FROM cron_delivery_circuit_v2"
        ).fetchone()
    assert row[0:2] == ("failed", 1)
    assert row[2] >= started_at + 1534.0
    assert circuit[0] == "slack"
    assert circuit[1] >= started_at + 1534.0

    reopened = StructuredDeliveryOutbox(
        home / "state.db", home / "cron" / "delivery-spool-v2"
    )
    assert reopened.claim_next(
        owner_pid=123,
        owner_started_at=45.0,
        now=started_at + 100.0,
    ) is None
    reopened.close()


def test_run_one_job_binds_delivery_to_the_durable_execution_id() -> None:
    job = {
        "id": "job-1",
        "name": "durable",
        "prompt": "work",
        "execution_id": "exec-boundary-1",
    }
    with patch("cron.scheduler.claim_dispatch", return_value=True), \
         patch("cron.scheduler.mark_execution_running"), \
         patch("cron.scheduler.finish_execution"), \
         patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"), \
         patch(
             "cron.scheduler.run_job",
             return_value=(True, "output", "final response", None),
         ), \
         patch("cron.scheduler.save_job_output", return_value="/tmp/output"), \
         patch("cron.scheduler._is_cron_silence_response", return_value=False), \
         patch("cron.scheduler._deliver_result", return_value=None) as deliver, \
         patch("cron.scheduler.mark_job_run"):
        assert scheduler.run_one_job(job) is True

    assert deliver.call_args.kwargs["execution_id"] == "exec-boundary-1"


def test_scheduler_spools_media_before_dispatch(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "report.pdf"
    source.write_bytes(b"report-v1")
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda _job: [{"platform": "telegram", "chat_id": "chat-a", "thread_id": None}],
    )

    calls = []

    def fake_legacy(_job, content, adapters=None, loop=None):
        calls.append(content)
        if source.exists():
            source.unlink()
        media, _cleaned = scheduler.BasePlatformAdapter.extract_media(content) \
            if hasattr(scheduler, "BasePlatformAdapter") else ([], content)
        # Import through the production dependency when it is not module-global.
        if "MEDIA:" in content and not media:
            from gateway.platforms.base import BasePlatformAdapter
            media, _cleaned = BasePlatformAdapter.extract_media(content)
        if media:
            assert len(media) == 1
            durable_path = Path(media[0][0])
            assert durable_path.parent == home / "cron" / "delivery-spool-v2"
            assert durable_path.read_bytes() == b"report-v1"
        _job["_durable_provider_message_id"] = "provider-ack-media"
        return None

    monkeypatch.setattr(scheduler, "_deliver_result_legacy", fake_legacy)
    result = scheduler._deliver_result(
        {"id": "job-1", "name": "media", "deliver": "telegram:chat-a"},
        f"report attached\nMEDIA:{source}",
        execution_id="exec-media-1",
    )
    assert result is None
    assert len(calls) == 1
    assert "report attached" in calls[0]
    assert calls[0].count("MEDIA:") == 1

    with sqlite3.connect(home / "state.db") as conn:
        rows = conn.execute(
            "SELECT payload_json, state FROM cron_delivery_outbox_v2 "
            "ORDER BY json_extract(payload_json, '$.unit_index')"
        ).fetchall()
    assert [state for _payload, state in rows] == ["delivered"]
    assert [
        len(__import__("json").loads(payload)["media_refs"])
        for payload, _state in rows
    ] == [1]


def test_each_media_is_an_independent_atomic_delivery_obligation(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda _job: [
            {"platform": "telegram", "chat_id": "chat-a", "thread_id": None}
        ],
    )
    calls = []

    def send_one(_job, content, adapters=None, loop=None):
        calls.append(content)
        if len(calls) == 2:
            return "definitive media rejection"
        _job["_durable_provider_message_id"] = "provider-ack-first-media"
        return None

    monkeypatch.setattr(scheduler, "_deliver_result_legacy", send_one)

    error = scheduler._deliver_result(
        {"id": "job-media-units", "deliver": "telegram:chat-a"},
        f"caption\nMEDIA:{first}\nMEDIA:{second}",
        execution_id="exec-media-units-1",
    )

    assert error == "definitive media rejection"
    assert len(calls) == 2
    assert "caption" in calls[0] and calls[0].count("MEDIA:") == 1
    assert "caption" not in calls[1] and calls[1].count("MEDIA:") == 1
    with sqlite3.connect(home / "state.db") as conn:
        rows = conn.execute(
            "SELECT payload_json, state FROM cron_delivery_outbox_v2 "
            "ORDER BY json_extract(payload_json, '$.unit_index')"
        ).fetchall()
    payloads = [__import__("json").loads(row[0]) for row in rows]
    assert [payload["kind"] for payload in payloads] == ["cron_media", "cron_media"]
    assert [len(payload["media_refs"]) for payload in payloads] == [1, 1]
    assert [row[1] for row in rows] == ["delivered", "failed"]


def test_live_media_send_result_failure_is_not_marked_delivered(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"media bytes")
    target = {"platform": "telegram", "chat_id": "chat-a", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})

    from gateway.config import Platform
    from gateway.delivery import DeliveryTransport
    from gateway.platforms.base import SendResult

    class Adapter:
        async def send_document(self, **_kwargs):
            return SendResult(success=False, error="upload failed")

    adapter = Adapter()
    platform_config = SimpleNamespace(enabled=True, extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    transport = SimpleNamespace(
        config=platform_config,
        adapter=adapter,
        is_relay=False,
        transport_platform=Platform.TELEGRAM,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: transport,
    )

    class ImmediateFuture:
        def result(self, timeout):
            assert timeout == 30
            return SendResult(success=False, error="upload failed")

        def cancel(self):
            return False

    def schedule(coro, _loop):
        coro.close()
        return ImmediateFuture()

    standalone_calls = []

    async def forbidden_standalone(*args, **kwargs):
        standalone_calls.append((args, kwargs))
        return {"success": True}

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", forbidden_standalone)
    loop = SimpleNamespace(is_running=lambda: True)

    error = scheduler._deliver_result(
        {"id": "job-media-fail", "deliver": "telegram:chat-a"},
        f"MEDIA:{source}",
        adapters={Platform.TELEGRAM: adapter},
        loop=loop,
        execution_id="exec-media-fail",
    )

    assert error is not None and "upload failed" in error
    assert standalone_calls == []
    with sqlite3.connect(home / "state.db") as conn:
        assert conn.execute(
            "SELECT state FROM cron_delivery_outbox_v2"
        ).fetchone()[0] == "failed"


def test_live_media_post_dispatch_exception_is_unknown_and_never_replayed(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "upload.pdf"
    source.write_bytes(b"media bytes")
    target = {"platform": "telegram", "chat_id": "chat-a", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )

    from gateway.config import Platform
    from gateway.delivery import DeliveryTransport
    from gateway.platforms.base import SendResult

    provider_boundaries = []

    class Adapter:
        async def send_document(self, **_kwargs):
            provider_boundaries.append("telegram:chat-a")
            return SendResult(success=True)

    adapter = Adapter()
    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    transport = DeliveryTransport(
        config=platform_config,
        adapter=adapter,
        transport_platform=Platform.TELEGRAM,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: transport,
    )

    class PostDispatchExceptionFuture:
        def result(self, timeout):
            assert timeout == 30
            raise RuntimeError("provider response lost after upload")

    def schedule(coro, _loop):
        asyncio.run(coro)
        return PostDispatchExceptionFuture()

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    loop = SimpleNamespace(is_running=lambda: True)
    adapters = {Platform.TELEGRAM: adapter}

    error = scheduler._deliver_result(
        {"id": "job-live-post-dispatch", "deliver": "telegram:chat-a"},
        f"caption\nMEDIA:{source}",
        adapters=adapters,
        loop=loop,
        execution_id="exec-live-post-dispatch",
    )
    with sqlite3.connect(home / "state.db") as conn:
        first_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()
        conn.execute("UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0")
        conn.commit()

    recovered = scheduler._recover_one_structured_delivery(adapters=adapters, loop=loop)
    with sqlite3.connect(home / "state.db") as conn:
        second_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert error is not None
    assert (first_row, recovered, second_row, len(provider_boundaries)) == (
        ("unknown", 1),
        0,
        ("unknown", 1),
        1,
    )


def test_live_text_post_dispatch_exception_is_unknown_and_never_replayed(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target = {"platform": "slack", "chat_id": "C1", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )
    monkeypatch.setattr(
        scheduler,
        "_slack_atomic_provider_chunks",
        lambda provider_content, _adapters=None: [provider_content],
    )

    from gateway.config import Platform
    from gateway.delivery import DeliveryTransport
    from gateway.platforms.base import SendResult

    provider_boundaries = []

    class Adapter:
        supports_inchannel_continuable = False

        async def send(self, _chat_id, _content, metadata=None):
            provider_boundaries.append("slack:C1")
            return SendResult(success=True)

    adapter = Adapter()
    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.SLACK: platform_config},
        get_home_channel=lambda _platform: None,
    )
    transport = DeliveryTransport(
        config=platform_config,
        adapter=adapter,
        transport_platform=Platform.SLACK,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: transport,
    )

    class PostDispatchExceptionFuture:
        def result(self, timeout):
            assert timeout == 60
            raise RuntimeError("provider response lost after send")

    def schedule(coro, _loop):
        asyncio.run(coro)
        return PostDispatchExceptionFuture()

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    loop = SimpleNamespace(is_running=lambda: True)
    adapters = {Platform.SLACK: adapter}

    error = scheduler._deliver_result(
        {"id": "job-live-text-post-dispatch", "deliver": "slack:C1"},
        "durable text",
        adapters=adapters,
        loop=loop,
        execution_id="exec-live-text-post-dispatch",
    )
    with sqlite3.connect(home / "state.db") as conn:
        first_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()
        conn.execute("UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0")
        conn.commit()

    recovered = scheduler._recover_one_structured_delivery(adapters=adapters, loop=loop)
    with sqlite3.connect(home / "state.db") as conn:
        second_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert error is not None
    assert (first_row, recovered, second_row, len(provider_boundaries)) == (
        ("unknown", 1),
        0,
        ("unknown", 1),
        1,
    )


def test_live_text_success_persists_exact_provider_message_id(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target = {"platform": "slack", "chat_id": "C1", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )
    monkeypatch.setattr(
        scheduler,
        "_slack_atomic_provider_chunks",
        lambda provider_content, _adapters=None: [provider_content],
    )

    from gateway.config import Platform
    from gateway.delivery import DeliveryTransport
    from gateway.platforms.base import SendResult

    provider_rows = []

    class Adapter:
        supports_inchannel_continuable = False

        async def send(self, _chat_id, _content, metadata=None):
            with sqlite3.connect(home / "state.db") as conn:
                provider_rows.append(
                    conn.execute(
                        "SELECT state, attempts, provider_message_id "
                        "FROM cron_delivery_outbox_v2"
                    ).fetchone()
                )
            return SendResult(
                success=True,
                message_id="slack-ts-1712345678.000001",
            )

    adapter = Adapter()
    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.SLACK: platform_config},
        get_home_channel=lambda _platform: None,
    )
    transport = DeliveryTransport(
        config=platform_config,
        adapter=adapter,
        transport_platform=Platform.SLACK,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: transport,
    )

    class CompletedFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout):
            assert timeout == 60
            return self._result

    def schedule(coro, _loop):
        return CompletedFuture(asyncio.run(coro))

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    loop = SimpleNamespace(is_running=lambda: True)

    error = scheduler._deliver_result(
        {"id": "job-live-text-receipt", "deliver": "slack:C1"},
        "durable text",
        adapters={Platform.SLACK: adapter},
        loop=loop,
        execution_id="exec-live-text-receipt",
    )

    with sqlite3.connect(home / "state.db") as conn:
        terminal_row = conn.execute(
            "SELECT state, attempts, provider_message_id "
            "FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert error is None
    assert (provider_rows, terminal_row) == (
        [("attempting", 1, None)],
        ("delivered", 1, "slack-ts-1712345678.000001"),
    )


def test_live_text_receiptless_success_is_unknown_and_never_replayed(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target = {"platform": "slack", "chat_id": "C1", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )
    monkeypatch.setattr(
        scheduler,
        "_slack_atomic_provider_chunks",
        lambda provider_content, _adapters=None: [provider_content],
    )

    from gateway.config import Platform
    from gateway.delivery import DeliveryTransport
    from gateway.platforms.base import SendResult

    provider_boundaries = []

    class Adapter:
        supports_inchannel_continuable = False

        async def send(self, _chat_id, _content, metadata=None):
            provider_boundaries.append("slack:C1")
            return SendResult(success=True)

    adapter = Adapter()
    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.SLACK: platform_config},
        get_home_channel=lambda _platform: None,
    )
    transport = DeliveryTransport(
        config=platform_config,
        adapter=adapter,
        transport_platform=Platform.SLACK,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: transport,
    )

    class CompletedFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout):
            assert timeout == 60
            return self._result

    def schedule(coro, _loop):
        return CompletedFuture(asyncio.run(coro))

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    loop = SimpleNamespace(is_running=lambda: True)
    adapters = {Platform.SLACK: adapter}

    error = scheduler._deliver_result(
        {"id": "job-live-text-receiptless", "deliver": "slack:C1"},
        "durable text",
        adapters=adapters,
        loop=loop,
        execution_id="exec-live-text-receiptless",
    )
    with sqlite3.connect(home / "state.db") as conn:
        first_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()
        conn.execute("UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0")
        conn.commit()

    recovered = scheduler._recover_one_structured_delivery(adapters=adapters, loop=loop)
    with sqlite3.connect(home / "state.db") as conn:
        second_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert error is not None
    assert (first_row, recovered, second_row, len(provider_boundaries)) == (
        ("unknown", 1),
        0,
        ("unknown", 1),
        1,
    )


def test_media_retry_never_resends_wrapper_text_after_media_rejection(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "report.pdf"
    source.write_bytes(b"media bytes")
    target = {"platform": "telegram", "chat_id": "chat-a", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": True}})

    from gateway.config import Platform
    from gateway.platforms.base import SendResult

    text_calls = []
    media_captions = []

    class Adapter:
        async def send(self, chat_id, content, metadata=None):
            text_calls.append((chat_id, content, metadata))
            return SendResult(success=True)

        async def send_document(self, *, caption=None, **_kwargs):
            media_captions.append(caption)
            if len(media_captions) == 1:
                return SendResult(success=False, error="definitive media rejection")
            return SendResult(success=True)

    adapter = Adapter()
    platform_config = SimpleNamespace(enabled=True, token="test-token", extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)

    class ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout):
            assert timeout in {30, 60}
            return self._result

        def cancel(self):
            return False

    def schedule(coro, _loop):
        return ImmediateFuture(asyncio.run(coro))

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    loop = SimpleNamespace(is_running=lambda: True)
    adapters = {Platform.TELEGRAM: adapter}

    error = scheduler._deliver_result(
        {"id": "job-atomic-media", "deliver": "telegram:chat-a"},
        f"caption\nMEDIA:{source}",
        adapters=adapters,
        loop=loop,
        execution_id="exec-atomic-media",
    )
    assert error is not None and "media rejection" in error
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute(
            "UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0 WHERE state='failed'"
        )
        conn.commit()

    assert scheduler._recover_one_structured_delivery(adapters=adapters, loop=loop) == 1

    assert text_calls == []
    assert len(media_captions) == 2
    assert media_captions[0] == media_captions[1]
    assert "caption" in media_captions[0]


def test_standalone_voice_rejection_has_one_provider_boundary_and_is_not_delivered(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "voice.ogg"
    source.write_bytes(b"voice bytes")
    target = {"platform": "telegram", "chat_id": "123", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})

    from gateway.config import Platform

    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("gateway.platforms.base.resolve_proxy_url", lambda *_a, **_k: None)

    provider_calls = []

    class BadRequest(RuntimeError):
        pass

    class FakeBot:
        def __init__(self, **_kwargs):
            pass

        async def send_message(self, **kwargs):
            provider_calls.append(("text", kwargs))
            return SimpleNamespace(message_id=1)

        async def send_voice(self, **kwargs):
            provider_calls.append(("voice", kwargs))
            raise BadRequest("definitive standalone voice rejection")

    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_module = SimpleNamespace(ParseMode=parse_mode)
    telegram_module = SimpleNamespace(
        Bot=FakeBot,
        MessageEntity=lambda **kwargs: SimpleNamespace(**kwargs),
        constants=constants_module,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_module)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_module)

    error = scheduler._deliver_result(
        {"id": "job-standalone-voice", "deliver": "telegram:123"},
        f"persisted voice caption\n[[audio_as_voice]]\nMEDIA:{source}",
        execution_id="exec-standalone-voice",
    )

    assert error is not None and "voice rejection" in error
    assert [kind for kind, _kwargs in provider_calls] == ["voice"]
    assert provider_calls[0][1]["caption"] == "persisted voice caption"
    with sqlite3.connect(home / "state.db") as conn:
        state = conn.execute("SELECT state FROM cron_delivery_outbox_v2").fetchone()[0]
    assert state == "failed"


def test_standalone_telegram_receiptless_send_is_unknown_and_never_replayed(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "voice.ogg"
    source.write_bytes(b"voice bytes")
    target = {"platform": "telegram", "chat_id": "123", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )

    from gateway.config import Platform

    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("gateway.platforms.base.resolve_proxy_url", lambda *_a, **_k: None)

    provider_calls = []

    class FakeBot:
        def __init__(self, **_kwargs):
            pass

        async def send_voice(self, **kwargs):
            provider_calls.append(("voice", kwargs))
            return SimpleNamespace(message_id=None)

    monkeypatch.setitem(sys.modules, "telegram", SimpleNamespace(Bot=FakeBot))

    error = scheduler._deliver_result(
        {"id": "job-standalone-receiptless", "deliver": "telegram:123"},
        f"caption\n[[audio_as_voice]]\nMEDIA:{source}",
        execution_id="exec-standalone-receiptless",
    )
    with sqlite3.connect(home / "state.db") as conn:
        first_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()
        conn.execute("UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0")
        conn.commit()

    recovered = scheduler._recover_one_structured_delivery(adapters=None, loop=None)
    with sqlite3.connect(home / "state.db") as conn:
        second_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert error is not None and error.startswith("ACK_UNKNOWN:")
    assert (first_row, recovered, second_row, len(provider_calls)) == (
        ("unknown", 1),
        0,
        ("unknown", 1),
        1,
    )


def test_standalone_media_provider_exception_is_unknown_and_never_replayed(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "voice.ogg"
    source.write_bytes(b"voice bytes")
    target = {"platform": "telegram", "chat_id": "123", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})

    from gateway.config import Platform

    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("gateway.platforms.base.resolve_proxy_url", lambda *_a, **_k: None)

    provider_calls = []

    class FakeBot:
        def __init__(self, **_kwargs):
            pass

        async def send_voice(self, **kwargs):
            provider_calls.append(("voice", kwargs))
            raise RuntimeError("provider response lost after upload")

    telegram_module = SimpleNamespace(Bot=FakeBot)
    monkeypatch.setitem(sys.modules, "telegram", telegram_module)

    error = scheduler._deliver_result(
        {"id": "job-standalone-ambiguous", "deliver": "telegram:123"},
        f"caption\n[[audio_as_voice]]\nMEDIA:{source}",
        execution_id="exec-standalone-ambiguous",
    )
    assert error is not None and "ACK_UNKNOWN" in error
    with sqlite3.connect(home / "state.db") as conn:
        first_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()
        conn.execute("UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0")
        conn.commit()

    recovered = scheduler._recover_one_structured_delivery(adapters=None, loop=None)
    with sqlite3.connect(home / "state.db") as conn:
        second_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert (first_row, recovered, second_row, len(provider_calls)) == (
        ("unknown", 1),
        0,
        ("unknown", 1),
        1,
    )


def test_fresh_thread_post_dispatch_timeout_is_unknown_and_never_replayed(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "voice.ogg"
    source.write_bytes(b"voice bytes")
    target = {"platform": "telegram", "chat_id": "123", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )

    from gateway.config import Platform

    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_args, **_kwargs: None
    )

    provider_boundaries = []
    real_asyncio_run = asyncio.run

    async def atomic_send(*_args, **_kwargs):
        provider_boundaries.append("telegram:123")
        return {"success": True}

    class PostDispatchTimeoutFuture:
        def result(self, timeout):
            assert timeout == 30
            raise TimeoutError()

    class InlineBoundaryPool:
        def submit(self, runner, coro):
            assert runner is scheduler.asyncio.run
            real_asyncio_run(coro)
            return PostDispatchTimeoutFuture()

        def shutdown(self, wait):
            assert wait is False

    def force_thread_fallback(coro):
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    monkeypatch.setattr(scheduler, "_send_atomic_media_standalone", atomic_send)
    monkeypatch.setattr(scheduler.asyncio, "run", force_thread_fallback)
    monkeypatch.setattr(
        scheduler.concurrent.futures,
        "ThreadPoolExecutor",
        lambda max_workers: InlineBoundaryPool(),
    )

    error = scheduler._deliver_result(
        {"id": "job-thread-timeout", "deliver": "telegram:123"},
        f"caption\n[[audio_as_voice]]\nMEDIA:{source}",
        execution_id="exec-thread-timeout",
    )
    with sqlite3.connect(home / "state.db") as conn:
        first_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()
        conn.execute("UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0")
        conn.commit()

    recovered = scheduler._recover_one_structured_delivery(adapters=None, loop=None)
    with sqlite3.connect(home / "state.db") as conn:
        second_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert error is not None
    assert (first_row, recovered, second_row, len(provider_boundaries)) == (
        ("unknown", 1),
        0,
        ("unknown", 1),
        1,
    )


@pytest.mark.parametrize(
    "provider_exception",
    [
        RuntimeError("provider response lost after send"),
        TimeoutError(),
    ],
    ids=["runtime-error", "timeout"],
)
def test_primary_standalone_post_dispatch_exception_is_unknown_without_replay(
    tmp_path: Path, monkeypatch, provider_exception: Exception
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target = {"platform": "slack", "chat_id": "C1", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )

    from gateway.config import Platform

    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.SLACK: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_args, **_kwargs: None
    )

    provider_boundaries = []

    async def standalone_send(*_args, **_kwargs):
        provider_boundaries.append("slack:C1")
        raise provider_exception

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", standalone_send)

    error = scheduler._deliver_result(
        {"id": "job-primary-standalone", "deliver": "slack:C1"},
        "durable text",
        execution_id=f"exec-primary-standalone-{type(provider_exception).__name__}",
    )
    with sqlite3.connect(home / "state.db") as conn:
        first_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()
        conn.execute("UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0")
        conn.commit()

    recovered = scheduler._recover_one_structured_delivery(adapters=None, loop=None)
    with sqlite3.connect(home / "state.db") as conn:
        second_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert error is not None
    assert (first_row, recovered, second_row, len(provider_boundaries)) == (
        ("unknown", 1),
        0,
        ("unknown", 1),
        1,
    )


def test_real_slack_standalone_text_response_loss_is_unknown_and_never_replayed(
    tmp_path: Path, monkeypatch
) -> None:
    """Keep ACK ambiguity intact through helper, wrappers, outbox, and recovery."""
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target = {"platform": "slack", "chat_id": "C1", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )

    from gateway.config import Platform
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins
    from plugins.platforms.slack import adapter as slack_adapter

    platform_config = SimpleNamespace(enabled=True, token="xoxb-test", extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.SLACK: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    discover_plugins()
    registered_sender = platform_registry.get("slack").standalone_sender_fn
    assert Path(inspect.getsourcefile(registered_sender)).resolve() == Path(
        slack_adapter.__file__
    ).resolve()
    monkeypatch.setitem(registered_sender.__globals__, "resolve_proxy_url", lambda: None)

    provider_calls = []

    class LostResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            raise RuntimeError("provider response lost after send")

    class FakeClientSession:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            provider_calls.append("chat.postMessage")
            return LostResponse()

    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", FakeClientSession)
    monkeypatch.setattr(aiohttp, "ClientTimeout", lambda *_args, **_kwargs: object())

    error = scheduler._deliver_result(
        {"id": "job-real-slack-loss", "deliver": "slack:C1"},
        "durable text",
        execution_id="exec-real-slack-loss",
    )
    with sqlite3.connect(home / "state.db") as conn:
        first_row = conn.execute(
            "SELECT state, attempts, provider_message_id FROM cron_delivery_outbox_v2"
        ).fetchone()
        conn.execute("UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0")
        conn.commit()

    recovered = scheduler._recover_one_structured_delivery(adapters=None, loop=None)
    with sqlite3.connect(home / "state.db") as conn:
        second_row = conn.execute(
            "SELECT state, attempts, provider_message_id FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert error is not None and error.startswith("ACK_UNKNOWN:")
    assert (first_row, recovered, second_row, provider_calls) == (
        ("unknown", 1, None),
        0,
        ("unknown", 1, None),
        ["chat.postMessage"],
    )


def test_real_slack_standalone_text_response_loss_preserves_nondurable_contract(
    monkeypatch,
) -> None:
    """Interactive/public standalone sends keep their existing error contract."""
    from plugins.platforms.slack import adapter as slack_adapter

    monkeypatch.setattr(slack_adapter, "resolve_proxy_url", lambda: None)

    class LostResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            raise RuntimeError("provider response lost after send")

    class FakeClientSession:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return LostResponse()

    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", FakeClientSession)
    monkeypatch.setattr(aiohttp, "ClientTimeout", lambda *_args, **_kwargs: object())

    result = asyncio.run(
        slack_adapter._standalone_send(
            SimpleNamespace(token="xoxb-test", extra={}),
            "C1",
            "interactive text",
        )
    )

    assert result == {"error": "Slack send failed: provider response lost after send"}


def test_real_slack_standalone_media_files_array_id_is_persisted(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "report.pdf"
    source.write_bytes(b"media bytes")
    target = {"platform": "slack", "chat_id": "C1", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler, "load_config", lambda: {"cron": {"wrap_response": False}}
    )

    from gateway.config import Platform
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins
    from plugins.platforms.slack import adapter as slack_adapter

    platform_config = SimpleNamespace(enabled=True, token="xoxb-test", extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.SLACK: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    discover_plugins()
    registered_sender = platform_registry.get("slack").standalone_sender_fn
    assert Path(inspect.getsourcefile(registered_sender)).resolve() == Path(
        slack_adapter.__file__
    ).resolve()
    monkeypatch.setitem(registered_sender.__globals__, "resolve_proxy_url", lambda: None)

    provider_calls = []

    class FakeAsyncWebClient:
        def __init__(self, **_kwargs):
            pass

        async def files_upload_v2(self, **kwargs):
            provider_calls.append(kwargs)
            return {"ok": True, "files": [{"id": "F-durable-123"}]}

    from types import ModuleType

    slack_sdk = ModuleType("slack_sdk")
    slack_web = ModuleType("slack_sdk.web")
    slack_async_client = ModuleType("slack_sdk.web.async_client")
    slack_async_client.AsyncWebClient = FakeAsyncWebClient
    slack_sdk.web = slack_web
    slack_web.async_client = slack_async_client
    monkeypatch.setitem(sys.modules, "slack_sdk", slack_sdk)
    monkeypatch.setitem(sys.modules, "slack_sdk.web", slack_web)
    monkeypatch.setitem(sys.modules, "slack_sdk.web.async_client", slack_async_client)

    error = scheduler._deliver_result(
        {"id": "job-real-slack-media", "deliver": "slack:C1"},
        f"exact caption\nMEDIA:{source}",
        execution_id="exec-real-slack-media",
    )

    with sqlite3.connect(home / "state.db") as conn:
        row = conn.execute(
            "SELECT state, attempts, provider_message_id FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert error is None
    assert row == ("delivered", 1, "F-durable-123")
    assert len(provider_calls) == 1
    assert provider_calls[0]["initial_comment"] == "exact caption"


def test_unsupported_atomic_media_is_unknown_and_never_replayed(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "report.pdf"
    source.write_bytes(b"media bytes")
    target = {"platform": "telegram", "chat_id": "123", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})

    from gateway.config import Platform

    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_args, **_kwargs: None
    )

    provider_calls = []

    class FakeBot:
        def __init__(self, **_kwargs):
            pass

        def __getattr__(self, method):
            async def provider_operation(**kwargs):
                provider_calls.append((method, kwargs))
                return SimpleNamespace(message_id=1)

            return provider_operation

    telegram_module = SimpleNamespace(Bot=FakeBot)
    monkeypatch.setitem(sys.modules, "telegram", telegram_module)

    error = scheduler._deliver_result(
        {"id": "job-over-limit-media", "deliver": "telegram:123"},
        f"{'x' * 1025}\nMEDIA:{source}",
        execution_id="exec-over-limit-media",
    )
    assert error is not None and "ACK_UNKNOWN" in error
    with sqlite3.connect(home / "state.db") as conn:
        first_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()
        conn.execute("UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0")
        conn.commit()

    recovered = scheduler._recover_one_structured_delivery(adapters=None, loop=None)
    with sqlite3.connect(home / "state.db") as conn:
        second_row = conn.execute(
            "SELECT state, attempts FROM cron_delivery_outbox_v2"
        ).fetchone()

    assert (first_row, recovered, second_row, provider_calls) == (
        ("unknown", 1),
        0,
        ("unknown", 1),
        [],
    )


@pytest.mark.parametrize("executor_outcome", ["success", "error"])
def test_thread_fallback_closes_unconsumed_atomic_media_coroutine(
    tmp_path: Path, monkeypatch, executor_outcome: str
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "voice.ogg"
    source.write_bytes(b"voice bytes")
    target = {"platform": "telegram", "chat_id": "123", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})

    from gateway.config import Platform

    platform_config = SimpleNamespace(enabled=True, token=object(), extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_args, **_kwargs: None
    )

    created = []

    async def atomic_send(*_args, **_kwargs):
        return {"success": True, "message_id": "provider-ack-atomic-media"}

    def atomic_factory(*args, **kwargs):
        coro = atomic_send(*args, **kwargs)
        created.append(coro)
        return coro

    class FakeFuture:
        def result(self, timeout):
            if executor_outcome == "error":
                raise ConnectionError("executor cancelled before coroutine start")
            return {"success": True, "message_id": "provider-ack-atomic-media"}

    class FakePool:
        def submit(self, _runner, coro):
            assert inspect.getcoroutinestate(coro) == inspect.CORO_CREATED
            return FakeFuture()

        def shutdown(self, wait):
            pass

    monkeypatch.setattr(scheduler, "_send_atomic_media_standalone", atomic_factory)
    monkeypatch.setattr(scheduler.asyncio, "run", lambda _coro: (_ for _ in ()).throw(RuntimeError("loop active")))
    monkeypatch.setattr(
        scheduler.concurrent.futures, "ThreadPoolExecutor", lambda max_workers: FakePool()
    )

    error = scheduler._deliver_result(
        {"id": "job-thread-coro-close", "deliver": "telegram:123"},
        f"caption\n[[audio_as_voice]]\nMEDIA:{source}",
        execution_id="exec-thread-coro-close",
    )

    if executor_outcome == "error":
        assert error is not None and "cancelled before coroutine start" in error
        expected_state = "failed"
    else:
        assert error is None
        expected_state = "delivered"
    with sqlite3.connect(home / "state.db") as conn:
        state = conn.execute(
            "SELECT state FROM cron_delivery_outbox_v2"
        ).fetchone()[0]
    assert state == expected_state
    assert len(created) == 2
    assert all(
        inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in created
    )


def test_ack_timeout_is_unknown_and_not_retryable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda _job: [{"platform": "telegram", "chat_id": "chat-a", "thread_id": None}],
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result_legacy",
        lambda *_args, **_kwargs: "ACK_UNKNOWN: provider confirmation timed out",
    )

    error = scheduler._deliver_result(
        {"id": "job-1", "deliver": "telegram:chat-a"},
        "ambiguous",
        execution_id="exec-timeout-1",
    )
    assert "ACK_UNKNOWN" in error
    with sqlite3.connect(tmp_path / "home" / "state.db") as conn:
        state = conn.execute("SELECT state FROM cron_delivery_outbox_v2").fetchone()[0]
    assert state == "unknown"


def test_post_dispatch_read_timeout_is_unknown_not_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda _job: [
            {"platform": "telegram", "chat_id": "chat-a", "thread_id": None}
        ],
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result_legacy",
        lambda *_args, **_kwargs: (
            "delivery failed: provider read timed out after request body was sent"
        ),
    )

    error = scheduler._deliver_result(
        {"id": "job-timeout", "deliver": "telegram:chat-a"},
        "ambiguous after dispatch",
        execution_id="exec-timeout-2",
    )

    assert error is not None and "timed out" in error
    with sqlite3.connect(home / "state.db") as conn:
        state, next_attempt_at = conn.execute(
            "SELECT state, next_attempt_at FROM cron_delivery_outbox_v2"
        ).fetchone()
    assert state == "unknown"
    assert next_attempt_at == 0


def test_retry_uses_persisted_provider_visible_wrapped_bytes_after_config_drift(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target = {"platform": "telegram", "chat_id": "chat-a", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])

    config_state = {"wrap_response": True}
    monkeypatch.setattr(
        scheduler,
        "load_config",
        lambda: {"cron": {"wrap_response": config_state["wrap_response"]}},
    )
    from gateway.config import Platform

    platform_config = SimpleNamespace(enabled=True, extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: None,
    )
    provider_contents = []

    async def fake_standalone(_platform, _config, _chat_id, content, **_kwargs):
        provider_contents.append(content)
        if len(provider_contents) == 1:
            return {"error": "temporary rejection"}
        return {"success": True}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", fake_standalone)
    job = {
        "id": "job-wrap",
        "name": "Snapshot Name",
        "deliver": "telegram:chat-a",
    }
    error = scheduler._deliver_result(
        job,
        "immutable payload",
        execution_id="exec-wrap-snapshot",
    )
    assert error is not None and "temporary rejection" in error
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute(
            "UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0 WHERE state='failed'"
        )
        conn.commit()

    config_state["wrap_response"] = False
    assert scheduler._recover_one_structured_delivery(adapters=None, loop=None) == 1

    expected = (
        "Cronjob Response: Snapshot Name\n"
        "(job_id: job-wrap)\n"
        "-------------\n\n"
        "immutable payload\n\n"
        "To stop or manage this job, send me a new message "
        "(e.g. \"stop reminder Snapshot Name\")."
    )
    assert provider_contents == [expected, expected]
    with sqlite3.connect(home / "state.db") as conn:
        payload = __import__("json").loads(
            conn.execute("SELECT payload_json FROM cron_delivery_outbox_v2").fetchone()[0]
        )
    assert payload["provider_content"] == expected
    assert payload["delivery_config_snapshot"] == {
        "job_id": "job-wrap",
        "task_name": "Snapshot Name",
        "wrap_response": True,
    }


def test_scheduler_rate_limit_opens_durable_circuit_before_claiming_next_target(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda _job: [
            {"platform": "telegram", "chat_id": "chat-a", "thread_id": None},
            {"platform": "telegram", "chat_id": "chat-b", "thread_id": None},
        ],
    )
    calls = []

    def rate_limited(job, _content, adapters=None, loop=None):
        calls.append(job["_durable_delivery_target"]["chat_id"])
        return "429 rate limited"

    monkeypatch.setattr(scheduler, "_deliver_result_legacy", rate_limited)

    error = scheduler._deliver_result(
        {"id": "job-rate-limit", "deliver": "telegram:ignored"},
        "paced payload",
        execution_id="exec-rate-limit-1",
    )

    assert error is not None and "429" in error
    assert calls == ["chat-a"]
    with sqlite3.connect(home / "state.db") as conn:
        rows = conn.execute(
            "SELECT canonical_target, state, attempts FROM cron_delivery_outbox_v2 "
            "ORDER BY canonical_target"
        ).fetchall()
        circuit = conn.execute(
            "SELECT platform, blocked_until FROM cron_delivery_circuit_v2"
        ).fetchone()
    assert rows == [
        ("telegram:chat-a", "failed", 1),
        ("telegram:chat-b", "pending", 0),
    ]
    assert circuit is not None
    assert circuit[0] == "telegram"
    assert circuit[1] > 0


def test_relay_transport_identity_is_durable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda _job: [{"platform": "slack", "chat_id": "C1", "thread_id": "T1"}],
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_structured_transport_identity",
        lambda _target, _adapters: (
            "relay",
            {"logical_platform": "slack"},
            "0" * 64,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "_slack_atomic_provider_chunks",
        lambda provider_content, _adapters=None: [provider_content],
    )
    monkeypatch.setattr(scheduler, "_deliver_result_legacy", lambda *_a, **_k: None)

    scheduler._deliver_result(
        {"id": "job-relay", "deliver": "slack:C1:T1"},
        "relay payload",
        adapters={"relay": object()},
        execution_id="exec-relay-1",
    )
    with sqlite3.connect(tmp_path / "home" / "state.db") as conn:
        payload = __import__("json").loads(
            conn.execute("SELECT payload_json FROM cron_delivery_outbox_v2").fetchone()[0]
        )
    assert payload["transport_kind"] == "relay"
    assert payload["relay_identity"] == {"logical_platform": "slack"}
    assert payload["thread_id"] == "T1"


def test_recovery_rejects_same_kind_provider_identity_drift_without_provider_call(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target = {"platform": "telegram", "chat_id": "chat-a", "thread_id": "topic-1"}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})

    from gateway.config import Platform

    platform_config = SimpleNamespace(
        enabled=True,
        token=object(),
        api_key=None,
        extra={"api_url": "https://endpoint-a.invalid", "account": "account-a"},
    )
    gateway_config = SimpleNamespace(
        platforms={Platform.TELEGRAM: platform_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: None,
    )
    provider_calls = []

    async def fake_standalone(*_args, **_kwargs):
        provider_calls.append((_args, _kwargs))
        return {"error": "definitive temporary rejection"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", fake_standalone)
    error = scheduler._deliver_result(
        {"id": "job-identity-drift", "deliver": "telegram:chat-a:topic-1"},
        "identity-bound payload",
        execution_id="exec-identity-drift",
    )
    assert error is not None
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute(
            "UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0 WHERE state='failed'"
        )
        persisted = conn.execute(
            "SELECT payload_json FROM cron_delivery_outbox_v2"
        ).fetchone()[0]
        conn.commit()

    platform_config.token=object()
    platform_config.extra = {
        "api_url": "https://endpoint-b.invalid",
        "account": "account-b",
    }
    assert scheduler._recover_one_structured_delivery(adapters=None, loop=None) == 1

    assert len(provider_calls) == 1
    assert "credential-a-secret" not in persisted
    assert "credential-b-secret" not in persisted
    payload = __import__("json").loads(persisted)
    assert len(payload["transport_identity_sha256"]) == 64
    with sqlite3.connect(home / "state.db") as conn:
        assert conn.execute(
            "SELECT state FROM cron_delivery_outbox_v2"
        ).fetchone()[0] == "abandoned"


def test_unresolved_durable_transport_fails_closed_without_a_provider_call(
    monkeypatch,
) -> None:
    target = {"platform": "telegram", "chat_id": "chat-a", "thread_id": None}
    calls = []
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: SimpleNamespace(platforms={}),
    )
    monkeypatch.setattr(
        "tools.send_message_tool._send_to_platform",
        lambda *_args, **_kwargs: calls.append((_args, _kwargs)),
    )
    job = {
        "id": "job-unresolved",
        "deliver": "telegram:chat-a",
        "_durable_delivery": True,
        "_durable_provider_content": "must not drift to a new transport",
        "_durable_delivery_target": target,
        "_durable_transport_kind": "unresolved",
        "_durable_relay_identity": None,
    }

    error = scheduler._deliver_result_legacy(job, "must not drift to a new transport")

    assert error is not None and "unresolved" in error
    assert calls == []


def test_recovery_refuses_to_switch_a_frozen_relay_obligation_to_native(
    monkeypatch,
) -> None:
    from gateway.config import Platform

    target = {"platform": "slack", "chat_id": "C1", "thread_id": "T1"}
    enabled = SimpleNamespace(enabled=True, extra={})
    config = SimpleNamespace(platforms={})
    current_native = SimpleNamespace(
        config=enabled,
        adapter=object(),
        is_relay=False,
        transport_platform=Platform.SLACK,
    )
    provider_calls = []
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: current_native,
    )
    monkeypatch.setattr(
        "tools.send_message_tool._send_to_platform",
        lambda *_args, **_kwargs: provider_calls.append((_args, _kwargs)),
    )
    job = {
        "id": "job-relay-drift",
        "deliver": "slack:C1:T1",
        "_durable_delivery": True,
        "_durable_provider_content": "relay-only retry",
        "_durable_delivery_target": target,
        "_durable_transport_kind": "relay",
        "_durable_transport_identity_sha256": (
            scheduler._resolved_transport_identity_sha256(
                target,
                logical=Platform.SLACK,
                config=config,
                transport=current_native,
                transport_kind="native",
            )
        ),
        "_durable_relay_identity": {"logical_platform": "slack"},
    }

    error = scheduler._deliver_result_legacy(job, "relay-only retry")

    assert error is not None and "transport changed" in error
    assert provider_calls == []


def test_disconnect_and_cancellation_errors_are_ack_ambiguous() -> None:
    for error in (
        "Remote end closed connection without response",
        "CancelledError",
        "network connection closed",
        "transport disconnected after dispatch",
        "",
    ):
        assert scheduler._structured_delivery_ack_is_ambiguous(error), error


def test_durable_delivery_without_frozen_thread_never_creates_dynamic_thread(
    monkeypatch,
) -> None:
    from gateway.config import Platform
    from gateway.delivery import DeliveryTransport
    from gateway.platforms.base import SendResult

    target = {"platform": "slack", "chat_id": "C1", "thread_id": None}

    class Adapter:
        supports_inchannel_continuable = False

        async def send(self, _chat_id, _content, metadata=None):
            return SendResult(success=True, message_id="1712345678.000001")

    adapter = Adapter()
    platform_config = SimpleNamespace(enabled=True, extra={})
    gateway_config = SimpleNamespace(
        platforms={Platform.SLACK: platform_config},
        get_home_channel=lambda _platform: None,
    )
    transport = DeliveryTransport(
        config=platform_config,
        adapter=adapter,
        transport_platform=Platform.SLACK,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", lambda *_a, **_k: transport
    )
    monkeypatch.setattr(
        scheduler, "_resolved_transport_identity_sha256", lambda *_a, **_k: "0" * 64
    )
    monkeypatch.setattr(
        scheduler,
        "_open_continuable_cron_thread",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("durable retry must not create an unfrozen provider thread")
        ),
    )
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr("gateway.mirror.mirror_to_session", lambda **_kwargs: False)

    class ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout):
            assert timeout == 60
            return self._result

        def cancel(self):
            return False

    def schedule(coro, _loop):
        return ImmediateFuture(asyncio.run(coro))

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    loop = SimpleNamespace(is_running=lambda: True)
    job = {
        "id": "job-durable-no-thread",
        "deliver": "slack:C1",
        "origin": {"platform": "slack", "chat_id": "C1", "user_id": "U1"},
        "attach_to_session": True,
        "_durable_delivery": True,
        "_durable_delivery_target": target,
        "_durable_transport_kind": "native",
        "_durable_transport_identity_sha256": "0" * 64,
        "_durable_relay_identity": None,
        "_durable_unit_kind": "cron_text",
        "_durable_provider_content": "durable brief",
        "_durable_delivery_config_snapshot": {
            "wrap_response": False,
            "task_name": "job-durable-no-thread",
            "job_id": "job-durable-no-thread",
        },
    }

    assert scheduler._deliver_result_legacy(
        job, "durable brief", adapters={Platform.SLACK: adapter}, loop=loop
    ) is None


def test_recovery_claim_dispatches_one_pending_row_and_records_ack(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from cron.durable_delivery import StructuredDeliveryOutbox

    outbox = StructuredDeliveryOutbox(home / "state.db", home / "cron" / "delivery-spool-v2")
    unit = {
        "schema_version": 2,
        "execution_id": "exec-recover-1",
        "job_id": "job-1",
        "target_index": 0,
        "unit_index": 0,
        "canonical_target": "telegram:chat-a",
        "logical_platform": "telegram",
        "chat_id": "chat-a",
        "thread_id": None,
        "transport_kind": "native",
        "transport_identity_sha256": "0" * 64,
        "relay_identity": None,
        "kind": "cron_result",
        "content": "recover me",
        "content_sha256": __import__("hashlib").sha256(b"recover me").hexdigest(),
        "provider_content": "recover me",
        "provider_content_sha256": __import__("hashlib").sha256(
            b"recover me"
        ).hexdigest(),
        "delivery_config_snapshot": {
            "wrap_response": False,
            "task_name": "job-1",
            "job_id": "job-1",
        },
        "media_ref": None,
        "media_refs": [],
        "continuation": {"mirror_enabled": False, "origin": None},
        "job_snapshot": {"id": "job-1", "deliver": "telegram:chat-a"},
        "target": {"platform": "telegram", "chat_id": "chat-a", "thread_id": None},
    }
    [record] = outbox.enqueue_batch([unit])
    outbox.close()
    calls = []

    def recover_success(job, content, adapters=None, loop=None):
        calls.append((job, content))
        job["_durable_provider_message_id"] = "provider-ack-recovery"
        return None

    monkeypatch.setattr(scheduler, "_deliver_result_legacy", recover_success)

    assert scheduler._recover_one_structured_delivery(adapters={}, loop=None) == 1
    assert calls[0][1] == "recover me"
    assert calls[0][0]["attach_to_session"] is False
    reopened = StructuredDeliveryOutbox(home / "state.db", home / "cron" / "delivery-spool-v2")
    terminal = reopened.get(record.obligation_id)
    assert (terminal.state, terminal.attempts, terminal.provider_message_id) == (
        "delivered",
        1,
        "provider-ack-recovery",
    )


def test_recovery_cancellation_reclassifies_unknown_and_releases_its_lease(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from cron.durable_delivery import StructuredDeliveryOutbox

    outbox = StructuredDeliveryOutbox(home / "state.db", home / "cron" / "delivery-spool-v2")
    unit = {
        "schema_version": 2,
        "execution_id": "exec-cancel-recovery",
        "job_id": "job-cancel",
        "target_index": 0,
        "unit_index": 0,
        "canonical_target": "telegram:chat-a",
        "logical_platform": "telegram",
        "chat_id": "chat-a",
        "thread_id": None,
        "transport_kind": "native",
        "transport_identity_sha256": "0" * 64,
        "relay_identity": None,
        "kind": "cron_text",
        "content": "cancel me",
        "content_sha256": __import__("hashlib").sha256(b"cancel me").hexdigest(),
        "provider_content": "cancel me",
        "provider_content_sha256": __import__("hashlib").sha256(
            b"cancel me"
        ).hexdigest(),
        "delivery_config_snapshot": {
            "wrap_response": False,
            "task_name": "job-cancel",
            "job_id": "job-cancel",
        },
        "media_ref": None,
        "media_refs": [],
        "continuation": {"mirror_enabled": False, "origin": None},
        "job_snapshot": {"id": "job-cancel", "deliver": "telegram:chat-a"},
        "target": {"platform": "telegram", "chat_id": "chat-a", "thread_id": None},
    }
    [record] = outbox.enqueue_batch([unit])
    outbox.close()

    def cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(scheduler, "_deliver_result_legacy", cancelled)

    with pytest.raises(asyncio.CancelledError):
        scheduler._recover_one_structured_delivery(adapters={}, loop=None)

    reopened = StructuredDeliveryOutbox(home / "state.db", home / "cron" / "delivery-spool-v2")
    assert reopened.get(record.obligation_id).state == "unknown"
    with sqlite3.connect(home / "state.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM cron_delivery_recovery_lease_v2"
        ).fetchone()[0] == 0


def test_original_scheduler_cancellation_reclassifies_unknown_and_releases_lease(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda _job: [
            {"platform": "telegram", "chat_id": "chat-a", "thread_id": None}
        ],
    )

    def cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(scheduler, "_deliver_result_legacy", cancelled)

    with pytest.raises(asyncio.CancelledError):
        scheduler._deliver_result(
            {"id": "job-cancel-original", "deliver": "telegram:chat-a"},
            "cancel original",
            execution_id="exec-cancel-original",
        )

    with sqlite3.connect(home / "state.db") as conn:
        state = conn.execute(
            "SELECT state FROM cron_delivery_outbox_v2"
        ).fetchone()[0]
        lease_count = conn.execute(
            "SELECT COUNT(*) FROM cron_delivery_recovery_lease_v2"
        ).fetchone()[0]
    assert state == "unknown"
    assert lease_count == 0


def test_durable_live_timeout_never_falls_back_when_future_cancel_reports_true(
    monkeypatch,
) -> None:
    """A wrapper future's cancellation result cannot prove no provider dispatch."""
    from gateway.config import Platform

    target = {"platform": "telegram", "chat_id": "chat-a", "thread_id": None}
    pconfig = SimpleNamespace(enabled=True, extra={})
    runtime_adapter = object()
    transport = SimpleNamespace(
        config=pconfig,
        adapter=runtime_adapter,
        is_relay=False,
        transport_platform=Platform.TELEGRAM,
    )
    config = SimpleNamespace(platforms={})
    loop = SimpleNamespace(is_running=lambda: True)
    standalone_calls = []

    class AmbiguousFuture:
        def result(self, timeout):
            assert timeout == 60
            raise TimeoutError()

        def cancel(self):
            # concurrent.futures.Future.cancel() is not a durable dispatch fence.
            return True

    def fake_schedule(coro, _loop):
        coro.close()
        return AmbiguousFuture()

    async def fake_standalone(*args, **kwargs):
        standalone_calls.append((args, kwargs))
        return {"success": True}

    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport",
        lambda *_args, **_kwargs: transport,
    )
    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", fake_schedule)
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", fake_standalone)

    error = scheduler._deliver_result_legacy(
        {
            "id": "job-ambiguous-cancel",
            "deliver": "telegram:chat-a",
            "_durable_delivery": True,
            "_durable_provider_content": "provider may already have accepted this",
            "_durable_delivery_target": target,
            "_durable_transport_kind": "native",
            "_durable_transport_identity_sha256": (
                scheduler._resolved_transport_identity_sha256(
                    target,
                    logical=Platform.TELEGRAM,
                    config=config,
                    transport=transport,
                    transport_kind="native",
                )
            ),
            "_durable_relay_identity": None,
        },
        "provider may already have accepted this",
        adapters={"telegram": runtime_adapter},
        loop=loop,
    )

    assert error is not None and "ACK_UNKNOWN" in error
    assert standalone_calls == []


def test_recovery_success_clears_correlated_last_delivery_error(
    tmp_path: Path, monkeypatch
) -> None:
    """A confirmed retry must repair the mutable job delivery status too."""
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(
        prompt="owner report",
        schedule="every 1h",
        name="durable recovery status",
        deliver="slack:C123",
    )
    execution_id = "exec-recovery-status-1"
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda _job: [
            {"platform": "slack", "chat_id": "C123", "thread_id": None}
        ],
    )
    monkeypatch.setattr(
        scheduler,
        "_slack_atomic_provider_chunks",
        lambda provider_content, _adapters=None: [provider_content],
    )

    monkeypatch.setattr(
        scheduler,
        "_deliver_result_legacy",
        lambda *_args, **_kwargs: "429 rate limited",
    )
    initial_error = scheduler._deliver_result(
        job,
        "daily owner report",
        execution_id=execution_id,
    )
    assert initial_error is not None
    mark_job_run(
        job["id"],
        True,
        delivery_error=initial_error,
        delivery_execution_id=execution_id,
    )
    assert get_job(job["id"])["last_delivery_error"] == initial_error

    with sqlite3.connect(home / "state.db") as conn:
        conn.execute(
            "UPDATE cron_delivery_outbox_v2 SET next_attempt_at=0 WHERE execution_id=?",
            (execution_id,),
        )
        conn.execute(
            "UPDATE cron_delivery_circuit_v2 SET blocked_until=0 WHERE platform='slack'"
        )
        conn.commit()

    def recovered(job_snapshot, *_args, **_kwargs):
        job_snapshot["_durable_provider_message_id"] = "slack-ack-recovered"
        return None

    monkeypatch.setattr(scheduler, "_deliver_result_legacy", recovered)
    assert scheduler._recover_one_structured_delivery() == 1

    updated = get_job(job["id"])
    assert updated["last_delivery_error"] is None
    assert updated["last_delivery_execution_id"] == execution_id
    with sqlite3.connect(home / "state.db") as conn:
        assert conn.execute(
            "SELECT state, attempts, provider_message_id "
            "FROM cron_delivery_outbox_v2 WHERE execution_id=?",
            (execution_id,),
        ).fetchone() == ("delivered", 2, "slack-ack-recovered")


def test_durable_send_metadata_captures_provider_retry_after() -> None:
    job = {"_durable_delivery": True}
    scheduler._record_durable_send_metadata(
        job,
        SimpleNamespace(retry_after=1534.4, error_kind="rate_limited"),
    )

    assert job["_durable_retry_after"] == pytest.approx(1534.4)
    assert job["_durable_error_kind"] == "rate_limited"


def test_live_rate_limit_honors_provider_retry_after(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda _job: [
            {"platform": "weixin", "chat_id": "owner", "thread_id": None}
        ],
    )

    monkeypatch.setattr(
        scheduler,
        "_weixin_atomic_provider_chunks",
        lambda provider_content, _adapters=None: [provider_content],
    )

    def rate_limited(job_snapshot, *_args, **_kwargs):
        job_snapshot["_durable_retry_after"] = 1534.4
        job_snapshot["_durable_error_kind"] = "rate_limited"
        return "iLink sendmessage rate limited; cooldown active for 1534.4s"

    monkeypatch.setattr(scheduler, "_deliver_result_legacy", rate_limited)
    before = scheduler.time.time()
    error = scheduler._deliver_result(
        {"id": "job-weixin-rate-limit", "deliver": "weixin:owner"},
        "owner report",
        execution_id="exec-weixin-rate-limit",
    )

    assert error is not None and "rate limited" in error
    with sqlite3.connect(home / "state.db") as conn:
        state, next_attempt_at = conn.execute(
            "SELECT state, next_attempt_at FROM cron_delivery_outbox_v2 "
            "WHERE execution_id=?",
            ("exec-weixin-rate-limit",),
        ).fetchone()
        blocked_until = conn.execute(
            "SELECT blocked_until FROM cron_delivery_circuit_v2 "
            "WHERE platform='weixin'"
        ).fetchone()[0]
    assert state == "failed"
    assert next_attempt_at >= before + 1534.0
    assert blocked_until >= before + 1534.0


def test_standalone_weixin_rate_limit_survives_store_restart(
    tmp_path: Path, monkeypatch
) -> None:
    from cron.durable_delivery import StructuredDeliveryOutbox
    from gateway.config import GatewayConfig, Platform, PlatformConfig

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target = {"platform": "weixin", "chat_id": "owner", "thread_id": None}
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [target])
    monkeypatch.setattr(
        scheduler,
        "load_config",
        lambda: {"cron": {"wrap_response": False}},
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_structured_transport_identity",
        lambda _target, _adapters: ("standalone", None, "b" * 64),
    )
    monkeypatch.setattr(
        scheduler,
        "_resolved_transport_identity_sha256",
        lambda *_args, **_kwargs: "b" * 64,
    )
    monkeypatch.setattr(
        scheduler,
        "_weixin_atomic_provider_chunks",
        lambda provider_content, _adapters=None: [provider_content],
    )

    pconfig = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={
            "account_id": "test-account",
            "rate_limit_circuit_open_seconds": 1534.4,
        },
    )
    gateway_config = GatewayConfig(platforms={Platform.WEIXIN: pconfig})
    monkeypatch.setattr(
        "gateway.config.load_gateway_config", lambda: gateway_config
    )

    provider_calls: list[str] = []

    async def rate_limited_provider(*_args, **kwargs):
        provider_calls.append(str(kwargs["text"]))
        return {
            "ret": -2,
            "errcode": -2,
            "errmsg": "frequency limit",
        }

    monkeypatch.setattr(
        "gateway.platforms.weixin._send_message",
        rate_limited_provider,
    )
    before = scheduler.time.time()
    error = scheduler._deliver_result(
        {"id": "job-weixin-standalone-rate", "deliver": "weixin:owner"},
        "standalone owner report",
        execution_id="exec-weixin-standalone-rate",
    )

    assert error is not None and "rate limited" in error
    assert len(provider_calls) == 1
    # Reopen the same durable store to prove the provider lower bound survives
    # process/store restart rather than living only in adapter memory.
    reopened = StructuredDeliveryOutbox(
        home / "state.db",
        home / "cron" / "delivery-spool-v2",
    )
    reopened.close()
    with sqlite3.connect(home / "state.db") as conn:
        state, next_attempt_at = conn.execute(
            "SELECT state, next_attempt_at FROM cron_delivery_outbox_v2 "
            "WHERE execution_id='exec-weixin-standalone-rate'"
        ).fetchone()
        blocked_until = conn.execute(
            "SELECT blocked_until FROM cron_delivery_circuit_v2 "
            "WHERE platform='weixin'"
        ).fetchone()[0]
    assert state == "failed"
    assert next_attempt_at >= before + 1534.0
    assert blocked_until >= before + 1534.0
