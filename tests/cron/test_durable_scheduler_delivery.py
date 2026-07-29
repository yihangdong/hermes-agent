import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cron import scheduler


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
        return None if "telegram" in job["deliver"] else "definitive rejection"

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
        return "definitive media rejection" if len(calls) == 2 else None

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
        token="credential-a-secret",
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

    platform_config.token = "credential-b-secret"
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
            return SendResult(success=True)

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
    monkeypatch.setattr(
        scheduler,
        "_deliver_result_legacy",
        lambda job, content, adapters=None, loop=None: calls.append((job, content)) or None,
    )

    assert scheduler._recover_one_structured_delivery(adapters={}, loop=None) == 1
    assert calls[0][1] == "recover me"
    assert calls[0][0]["attach_to_session"] is False
    reopened = StructuredDeliveryOutbox(home / "state.db", home / "cron" / "delivery-spool-v2")
    assert reopened.get(record.obligation_id).state == "delivered"


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
