import asyncio
import threading

import pytest

from gateway.run import GatewayRunner


def test_gateway_paced_recovery_runs_structured_cron_outbox_before_legacy_ledger(
    monkeypatch,
) -> None:
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._delivery_redelivery_not_before = 0.0
    calls = []

    monkeypatch.setattr(
        "cron.scheduler._recover_one_structured_delivery",
        lambda adapters, loop: calls.append((adapters, loop.is_running())) or 1,
    )
    monkeypatch.setattr("gateway.delivery_ledger.ledger_enabled", lambda: False)

    recovered = asyncio.run(GatewayRunner._redeliver_pending_obligations(runner))

    assert recovered == 1
    assert len(calls) == 1
    assert calls[0][0] is runner.adapters
    assert calls[0][1] is True


def test_one_structured_recovery_prevents_a_legacy_claim_in_the_same_paced_pass(
    monkeypatch,
) -> None:
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._delivery_redelivery_not_before = 0.0
    monkeypatch.setattr(
        "cron.scheduler._recover_one_structured_delivery",
        lambda _adapters, _loop: 1,
    )
    monkeypatch.setattr("gateway.delivery_ledger.ledger_enabled", lambda: True)

    def forbidden_legacy_claim(*_args, **_kwargs):
        raise AssertionError("one paced pass must not claim a second obligation")

    monkeypatch.setattr(
        "gateway.delivery_ledger.sweep_recoverable",
        forbidden_legacy_claim,
    )

    recovered = asyncio.run(GatewayRunner._redeliver_pending_obligations(runner))

    assert recovered == 1


def test_structured_recovery_exception_still_consumes_the_paced_pass(
    monkeypatch,
) -> None:
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._delivery_redelivery_not_before = 0.0

    def structured_failed_after_claim(_adapters, _loop):
        raise RuntimeError("post-claim failure")

    monkeypatch.setattr(
        "cron.scheduler._recover_one_structured_delivery",
        structured_failed_after_claim,
    )
    monkeypatch.setattr("gateway.delivery_ledger.ledger_enabled", lambda: True)

    legacy_claims = []

    def record_legacy_claim(*_args, **_kwargs):
        legacy_claims.append("claimed")
        return []

    monkeypatch.setattr(
        "gateway.delivery_ledger.sweep_recoverable",
        record_legacy_claim,
    )

    recovered = asyncio.run(GatewayRunner._redeliver_pending_obligations(runner))

    assert recovered == 0
    assert legacy_claims == []


@pytest.mark.asyncio
async def test_startup_recovery_is_joined_before_adapter_teardown(
    monkeypatch,
) -> None:
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._background_tasks = set()
    runner._delivery_redelivery_task = None
    runner._delivery_redelivery_not_before = 0.0
    started = threading.Event()
    release = threading.Event()
    events = []

    def blocked_recovery(_adapters, _loop):
        events.append("recovery-started")
        started.set()
        assert release.wait(5)
        events.append("recovery-finished")
        return 1

    monkeypatch.setattr(
        "cron.scheduler._recover_one_structured_delivery",
        blocked_recovery,
    )
    monkeypatch.setattr("gateway.delivery_ledger.ledger_enabled", lambda: False)

    startup = asyncio.create_task(runner._redeliver_pending_obligations())
    assert await asyncio.to_thread(started.wait, 2)
    quiesce = asyncio.create_task(runner._quiesce_delivery_redelivery_watcher())
    await asyncio.sleep(0.05)
    assert not quiesce.done()

    release.set()
    await quiesce
    events.append("adapter-teardown")
    assert await startup == 1
    assert events == [
        "recovery-started",
        "recovery-finished",
        "adapter-teardown",
    ]


@pytest.mark.asyncio
async def test_quiesce_joins_the_entire_startup_recovery_pass() -> None:
    runner = object.__new__(GatewayRunner)
    runner._background_tasks = set()
    runner._delivery_redelivery_task = None
    runner._delivery_recovery_workers = set()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def startup_recovery_pass():
        entered.set()
        await release.wait()

    startup = asyncio.create_task(startup_recovery_pass())
    runner._delivery_startup_recovery_task = startup
    await entered.wait()
    quiesce = asyncio.create_task(runner._quiesce_delivery_redelivery_watcher())
    await asyncio.sleep(0.05)
    assert not quiesce.done()

    release.set()
    await quiesce
    assert startup.done()
    assert runner._delivery_startup_recovery_task is None
