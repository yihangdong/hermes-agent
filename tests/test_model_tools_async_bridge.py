"""Regression tests for the _run_async() event-loop lifecycle.

These tests verify the fix for GitHub issue #2104:
  "Event loop is closed" after vision_analyze used as first call in session.

Root cause: asyncio.run() creates and *closes* a fresh event loop on every
call.  Cached httpx/AsyncOpenAI clients that were bound to the now-dead loop
would crash with RuntimeError("Event loop is closed") when garbage-collected.

The fix replaces asyncio.run() with a persistent event loop in _run_async().
"""

import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_current_loop():
    """Return the running event loop from inside a coroutine."""
    return asyncio.get_event_loop()


async def _create_and_return_transport():
    """Simulate an async client creating a transport on the current loop.

    Returns a simple asyncio.Future bound to the running loop so we can
    later check whether the loop is still alive.
    """
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    fut.set_result("ok")
    return loop, fut


def _run_bounded_python(script, *, timeout=2.0):
    """Run an isolated regression child with bounded TERM-to-KILL cleanup."""
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-W", "error", "-c", script],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=0.25)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout, stderr = proc.communicate(timeout=0.25)
    return proc.returncode, stdout, stderr, timed_out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunAsyncLoopLifecycle:
    """Verify _run_async() keeps the event loop alive after returning."""

    def test_loop_not_closed_after_run_async(self):
        """The loop used by _run_async must still be open after the call."""
        from model_tools import _run_async

        loop = _run_async(_get_current_loop())

        assert not loop.is_closed(), (
            "_run_async() closed the event loop — cached async clients will "
            "crash with 'Event loop is closed' on GC (issue #2104)"
        )

    def test_same_loop_reused_across_calls(self):
        """Consecutive _run_async calls should reuse the same loop."""
        from model_tools import _run_async

        loop1 = _run_async(_get_current_loop())
        loop2 = _run_async(_get_current_loop())

        assert loop1 is loop2, (
            "_run_async() created a new loop on the second call — cached "
            "async clients from the first call would be orphaned"
        )

    def test_cached_transport_survives_between_calls(self):
        """A transport/future created in call 1 must be valid in call 2."""
        from model_tools import _run_async

        loop, fut = _run_async(_create_and_return_transport())

        assert not loop.is_closed()
        assert fut.result() == "ok"

        loop2 = _run_async(_get_current_loop())
        assert loop2 is loop, "Loop changed between calls"
        assert not loop.is_closed(), "Loop closed before second call"

    def test_explicit_shutdown_is_idempotent_and_next_call_recreates_loop(self):
        import model_tools

        loop = model_tools._run_async(_get_current_loop())
        model_tools._shutdown_persistent_tool_loops()
        model_tools._shutdown_persistent_tool_loops()

        assert loop.is_closed()
        assert model_tools._tool_loop is None

        replacement = model_tools._run_async(_get_current_loop())
        assert replacement is not loop
        assert not replacement.is_closed()

    def test_process_exit_closes_persistent_loop_without_resource_warning(self):
        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                "-W",
                "error",
                "-c",
                (
                    "import asyncio, model_tools; "
                    "model_tools._run_async(asyncio.sleep(0))"
                ),
            ],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )

        assert proc.returncode == 0, proc.stderr
        assert "ResourceWarning" not in proc.stderr
        assert "unclosed event loop" not in proc.stderr

    def test_shutdown_rejects_new_loops_until_registry_is_closed(self, monkeypatch):
        """A getter racing shutdown must fail, never escape its snapshot."""
        import model_tools

        loop = model_tools._get_tool_loop()
        entered_close = threading.Event()
        release_close = threading.Event()
        real_close = model_tools._close_persistent_loop

        def _blocking_close(target):
            entered_close.set()
            assert release_close.wait(5), "test did not release shutdown"
            real_close(target)

        monkeypatch.setattr(model_tools, "_close_persistent_loop", _blocking_close)
        shutdown_errors = []
        shutdown = threading.Thread(
            target=lambda: _capture_exception(
                model_tools._shutdown_persistent_tool_loops,
                shutdown_errors,
            )
        )
        shutdown.start()
        assert entered_close.wait(5), "shutdown never owned the registry"

        with pytest.raises(RuntimeError, match="shutting down"):
            model_tools._get_tool_loop()

        worker_errors = []
        worker = threading.Thread(
            target=lambda: _capture_exception(model_tools._get_worker_loop, worker_errors)
        )
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
        assert len(worker_errors) == 1
        assert isinstance(worker_errors[0], RuntimeError)

        release_close.set()
        shutdown.join(5)
        assert not shutdown.is_alive()
        assert shutdown_errors == []
        assert loop.is_closed()
        assert model_tools._tool_loop is None
        assert model_tools._worker_loops == set()

    def test_run_async_closes_unstarted_coroutine_when_shutdown_gate_rejects(
        self, monkeypatch
    ):
        """The bridge retains no ownership of a coroutine it cannot schedule."""
        import model_tools

        model_tools._get_tool_loop()
        close_entered = threading.Event()
        release_close = threading.Event()
        real_close = model_tools._close_persistent_loop

        def _barrier_close(loop):
            close_entered.set()
            assert release_close.wait(2), "test did not release shutdown"
            return real_close(loop)

        monkeypatch.setattr(model_tools, "_close_persistent_loop", _barrier_close)
        shutdown_errors = []
        shutdown = threading.Thread(
            target=lambda: _capture_exception(
                model_tools._shutdown_persistent_tool_loops,
                shutdown_errors,
            )
        )
        shutdown.start()
        assert close_entered.wait(2), "shutdown never closed the gate"

        coro = _get_current_loop()
        try:
            with pytest.raises(RuntimeError, match="shutting down"):
                model_tools._run_async(coro)
            production_closed_coro = coro.cr_frame is None
        finally:
            if coro.cr_frame is not None:
                coro.close()
            release_close.set()
            shutdown.join(2)

        assert not shutdown.is_alive()
        assert shutdown_errors == []
        assert production_closed_coro

    def test_shutdown_waits_for_active_worker_lease_then_closes_it(self):
        """An active worker loop remains registered until its run completes."""
        from concurrent.futures import ThreadPoolExecutor
        import time
        import model_tools

        started = threading.Event()
        release = threading.Event()

        async def _held_run():
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            return asyncio.get_running_loop()

        with ThreadPoolExecutor(max_workers=1) as pool:
            active = pool.submit(model_tools._run_async, _held_run())
            assert started.wait(5)
            shutdown_errors = []
            shutdown = threading.Thread(
                target=lambda: _capture_exception(
                    model_tools._shutdown_persistent_tool_loops,
                    shutdown_errors,
                )
            )
            shutdown.start()
            deadline = time.monotonic() + 5
            while not model_tools._loop_shutdown_in_progress and time.monotonic() < deadline:
                time.sleep(0.01)
            assert model_tools._loop_shutdown_in_progress
            assert shutdown.is_alive(), "shutdown detached an active loop"
            with pytest.raises(RuntimeError, match="shutting down"):
                model_tools._get_tool_loop()
            release.set()
            worker_loop = active.result(timeout=5)
            shutdown.join(5)

        assert not shutdown.is_alive()
        assert shutdown_errors == []
        assert worker_loop.is_closed()
        assert model_tools._active_persistent_loop_runs == 0
        assert model_tools._worker_loops == set()

    def test_shutdown_from_active_lease_owner_fails_fast(self):
        """A coroutine cannot wait for the persistent lease it currently owns."""
        script = """
import model_tools

async def invoke_shutdown_from_owner():
    try:
        model_tools._shutdown_persistent_tool_loops()
    except RuntimeError as exc:
        assert "own active persistent loop lease" in str(exc), str(exc)
    else:
        raise AssertionError("same-owner shutdown unexpectedly succeeded")

model_tools._run_async(invoke_shutdown_from_owner())
assert model_tools._active_persistent_loop_runs == 0
model_tools._shutdown_persistent_tool_loops()
print("SELF_SHUTDOWN_FAIL_FAST_OK")
"""
        returncode, stdout, stderr, timed_out = _run_bounded_python(script)

        assert not timed_out, "same-owner shutdown deadlocked on its own lease"
        assert returncode == 0, stderr
        assert stdout.strip() == "SELF_SHUTDOWN_FAIL_FAST_OK"
        assert "ResourceWarning" not in stderr
        assert "Task was destroyed" not in stderr

    def test_shutdown_preserves_unleased_running_loop_reference(self):
        """A foreign running private loop is rejected, not silently forgotten."""
        import model_tools

        loop = model_tools._get_tool_loop()
        started = threading.Event()

        def _run_forever():
            asyncio.set_event_loop(loop)
            started.set()
            loop.run_forever()

        owner = threading.Thread(target=_run_forever)
        owner.start()
        assert started.wait(5)
        try:
            with pytest.raises(RuntimeError, match="still running"):
                model_tools._shutdown_persistent_tool_loops()
            assert model_tools._tool_loop is loop
            assert not loop.is_closed()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            owner.join(5)
        assert not owner.is_alive()

        model_tools._shutdown_persistent_tool_loops()
        assert loop.is_closed()
        assert model_tools._tool_loop is None

    def test_shutdown_keeps_reference_if_loop_starts_during_close(self, monkeypatch):
        """Close/running TOCTOU must fail without detaching the live loop."""
        import model_tools

        loop = model_tools._get_tool_loop()
        close_phase = threading.Event()
        release_close = threading.Event()
        owner_started = threading.Event()
        real_close = loop.close

        def _barrier_close():
            close_phase.set()
            assert release_close.wait(2), "test did not release close phase"
            return real_close()

        monkeypatch.setattr(loop, "close", _barrier_close)
        errors = []
        shutdown = threading.Thread(
            target=lambda: _capture_exception(
                model_tools._shutdown_persistent_tool_loops,
                errors,
            )
        )
        shutdown.start()
        assert close_phase.wait(2), "shutdown never reached close phase"

        def _foreign_owner():
            asyncio.set_event_loop(loop)
            owner_started.set()
            loop.run_forever()

        owner = threading.Thread(target=_foreign_owner)
        owner.start()
        assert owner_started.wait(2)
        release_close.set()
        shutdown.join(2)
        assert not shutdown.is_alive(), "strict shutdown hung in close/running race"
        retained = model_tools._tool_loop is loop
        still_open = not loop.is_closed()

        loop.call_soon_threadsafe(loop.stop)
        owner.join(2)
        assert not owner.is_alive()
        monkeypatch.setattr(loop, "close", real_close)
        if model_tools._tool_loop is loop:
            model_tools._shutdown_persistent_tool_loops()
        elif not loop.is_closed():
            real_close()

        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "running" in str(errors[0])
        assert retained
        assert still_open

    def test_concurrent_shutdown_waiters_share_close_failure_generation(self, monkeypatch):
        """A close failure is published to every waiter in that generation."""
        import model_tools

        loop = model_tools._get_tool_loop()
        close_entered = threading.Event()
        release_close = threading.Event()
        waiter_bound = threading.Event()
        real_close = loop.close
        real_wait = model_tools._loop_lifecycle.wait
        waiter_thread = None

        def _failing_close():
            close_entered.set()
            assert release_close.wait(2), "test did not release injected close"
            raise OSError("injected loop close failure")

        def _observed_wait(timeout=None):
            if threading.current_thread() is waiter_thread:
                waiter_bound.set()
            return real_wait(timeout)

        monkeypatch.setattr(loop, "close", _failing_close)
        monkeypatch.setattr(model_tools._loop_lifecycle, "wait", _observed_wait)
        outcomes = {}

        def _invoke(name):
            try:
                model_tools._shutdown_persistent_tool_loops()
            except Exception as exc:
                outcomes[name] = exc

        owner = threading.Thread(target=_invoke, args=("owner",))
        owner.start()
        assert close_entered.wait(2), "owner never attempted close"
        waiter_thread = threading.Thread(target=_invoke, args=("waiter",))
        waiter_thread.start()
        assert waiter_bound.wait(2), "second shutdown did not bind to generation"
        release_close.set()
        owner.join(2)
        waiter_thread.join(2)
        assert not owner.is_alive()
        assert not waiter_thread.is_alive()

        retained = model_tools._tool_loop is loop
        still_open = not loop.is_closed()
        monkeypatch.setattr(loop, "close", real_close)
        model_tools._shutdown_persistent_tool_loops()

        assert set(outcomes) == {"owner", "waiter"}
        assert isinstance(outcomes["owner"], OSError)
        assert outcomes["waiter"] is outcomes["owner"]
        assert "injected loop close failure" in str(outcomes["owner"])
        assert retained
        assert still_open
        assert loop.is_closed()
        assert model_tools._tool_loop is None

    def test_shutdown_cancels_and_drains_pending_tasks(self):
        import model_tools

        cancelled = threading.Event()

        async def _pending_forever():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def _create_pending_task():
            task = asyncio.create_task(_pending_forever())
            await asyncio.sleep(0)
            return task

        task = model_tools._run_async(_create_pending_task())
        loop = task.get_loop()
        assert not task.done()

        model_tools._shutdown_persistent_tool_loops()

        assert cancelled.is_set()
        assert task.cancelled()
        assert loop.is_closed()

    def test_strict_shutdown_bounds_noncooperative_cancel_and_reports_failure(self):
        """Strict shutdown force-closes but reports a task that ignores cancel."""
        script = """
import asyncio
import gc
import time
import model_tools

async def ignores_cancel():
    while True:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            continue

async def create_stubborn_task():
    task = asyncio.create_task(ignores_cancel())
    await asyncio.sleep(0)
    return task

task = model_tools._run_async(create_stubborn_task())
loop = task.get_loop()
started = time.monotonic()
try:
    model_tools._shutdown_persistent_tool_loops()
except RuntimeError as exc:
    assert "did not quiesce" in str(exc), str(exc)
else:
    raise AssertionError("strict shutdown hid a noncooperative task")
assert time.monotonic() - started < 1.0
assert loop.is_closed()
assert model_tools._tool_loop is None
assert model_tools._active_persistent_loop_runs == 0
del task
gc.collect()
print("BOUNDED_STRICT_SHUTDOWN_OK")
"""
        returncode, stdout, stderr, timed_out = _run_bounded_python(script)

        assert not timed_out, "strict shutdown hung on cancellation suppression"
        assert returncode == 0, stderr
        assert stdout.strip() == "BOUNDED_STRICT_SHUTDOWN_OK"
        assert "Task was destroyed" not in stderr
        assert "was never awaited" not in stderr
        assert "ResourceWarning" not in stderr
        assert "unclosed event loop" not in stderr

    def test_atexit_bounds_noncooperative_cancel_without_warning_tail(self):
        """Atexit uses the bounded transaction but never raises or hangs."""
        script = """
import asyncio
import model_tools

async def ignores_cancel():
    while True:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            continue

async def leave_pending():
    task = asyncio.create_task(ignores_cancel())
    await asyncio.sleep(0)
    return task

stubborn_task = model_tools._run_async(leave_pending())
print("ATEXIT_BOUNDED_SHUTDOWN_OK")
"""
        returncode, stdout, stderr, timed_out = _run_bounded_python(script)

        assert not timed_out, "atexit hung on cancellation suppression"
        assert returncode == 0, stderr
        assert stdout.strip() == "ATEXIT_BOUNDED_SHUTDOWN_OK"
        assert "Task was destroyed" not in stderr
        assert "was never awaited" not in stderr
        assert "ResourceWarning" not in stderr
        assert "unclosed event loop" not in stderr
        assert "Exception ignored in atexit" not in stderr

    def test_process_exit_closes_worker_loop_without_resource_warning(self):
        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                "-W",
                "error",
                "-c",
                (
                    "import asyncio, concurrent.futures, model_tools; "
                    "pool=concurrent.futures.ThreadPoolExecutor(max_workers=1); "
                    "pool.submit(model_tools._run_async, asyncio.sleep(0)).result(); "
                    "pool.shutdown()"
                ),
            ],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )

        assert proc.returncode == 0, proc.stderr
        assert "ResourceWarning" not in proc.stderr
        assert "unclosed event loop" not in proc.stderr


def _capture_exception(fn, errors):
    try:
        fn()
    except Exception as exc:
        errors.append(exc)


class TestRunAsyncWorkerThread:
    """Verify worker threads get persistent per-thread loops (delegate_task fix)."""

    def test_worker_thread_loop_not_closed(self):
        """A worker thread's loop must stay open after _run_async returns,
        so cached httpx/AsyncOpenAI clients don't crash on GC."""
        from concurrent.futures import ThreadPoolExecutor
        from model_tools import _run_async

        def _run_on_worker():
            loop = _run_async(_get_current_loop())
            still_open = not loop.is_closed()
            return loop, still_open

        with ThreadPoolExecutor(max_workers=1) as pool:
            loop, still_open = pool.submit(_run_on_worker).result()

        assert still_open, (
            "Worker thread's event loop was closed after _run_async — "
            "cached async clients will crash with 'Event loop is closed'"
        )

    def test_worker_thread_reuses_loop_across_calls(self):
        """Multiple _run_async calls on the same worker thread should
        reuse the same persistent loop (not create-and-destroy each time)."""
        from concurrent.futures import ThreadPoolExecutor
        from model_tools import _run_async

        def _run_twice_on_worker():
            loop1 = _run_async(_get_current_loop())
            loop2 = _run_async(_get_current_loop())
            return loop1, loop2

        with ThreadPoolExecutor(max_workers=1) as pool:
            loop1, loop2 = pool.submit(_run_twice_on_worker).result()

        assert loop1 is loop2, (
            "Worker thread created different loops for consecutive calls — "
            "cached clients from the first call would be orphaned"
        )
        assert not loop1.is_closed()

    def test_parallel_workers_get_separate_loops(self):
        """Different worker threads must get their own loops to avoid
        contention (the original reason for the worker-thread branch)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from model_tools import _run_async

        barrier = threading.Barrier(3, timeout=5)

        def _get_loop_id():
            # Use a barrier to force all 3 threads to be alive simultaneously,
            # ensuring the ThreadPoolExecutor actually uses 3 distinct threads.
            loop = _run_async(_get_current_loop())
            barrier.wait()
            return id(loop), not loop.is_closed(), threading.current_thread().ident

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_get_loop_id) for _ in range(3)]
            results = [f.result() for f in as_completed(futures)]

        loop_ids = {r[0] for r in results}
        thread_ids = {r[2] for r in results}
        all_open = all(r[1] for r in results)

        assert all_open, "At least one worker thread's loop was closed"
        # The barrier guarantees 3 distinct threads were used
        assert len(thread_ids) == 3, f"Expected 3 threads, got {len(thread_ids)}"
        # Each thread should have its own loop
        assert len(loop_ids) == 3, (
            f"Expected 3 distinct loops for 3 parallel workers, "
            f"got {len(loop_ids)} — workers may be contending on a shared loop"
        )

    def test_worker_loop_separate_from_main_loop(self):
        """Worker thread loops must be different from the main thread's
        persistent loop to avoid cross-thread contention."""
        from concurrent.futures import ThreadPoolExecutor
        from model_tools import _run_async, _get_tool_loop

        main_loop = _get_tool_loop()

        def _get_worker_loop_id():
            loop = _run_async(_get_current_loop())
            return id(loop)

        with ThreadPoolExecutor(max_workers=1) as pool:
            worker_loop_id = pool.submit(_get_worker_loop_id).result()

        assert worker_loop_id != id(main_loop), (
            "Worker thread used the main thread's loop — this would cause "
            "cross-thread contention on the event loop"
        )


class TestRunAsyncWithRunningLoop:
    """When a loop is already running, _run_async falls back to a thread."""

    @pytest.mark.asyncio
    async def test_run_async_from_async_context(self):
        """_run_async should still work when called from inside an
        already-running event loop (gateway / Atropos path)."""
        from model_tools import _run_async

        async def _simple():
            return 42

        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_async, _simple()
        )
        assert result == 42

    @pytest.mark.asyncio
    async def test_timeout_uses_nonblocking_executor_shutdown(self, monkeypatch):
        """A timeout in the running-loop branch must not block the caller.

        If shutdown ever waits for a stuck worker, a tool coroutine that
        ignores (or can't observe) cancellation would hang the whole agent.
        Guard: the caller must raise TimeoutError and pool.shutdown must be
        called with wait=False. The worker's own event loop handles cleanup
        (cancellation is scheduled via call_soon_threadsafe before the
        caller returns).
        """
        import concurrent.futures
        from model_tools import _run_async

        events = {
            "result_timeout": None,
            "shutdown_calls": [],
            "submitted_fn": None,
        }

        class TimeoutFuture:
            def result(self, timeout=None):
                events["result_timeout"] = timeout
                raise concurrent.futures.TimeoutError()

            def cancel(self):
                return True

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.shutdown(wait=True)
                return False

            def submit(self, fn, *args, **kwargs):
                # Record which function got submitted -- should be the
                # in-function worker wrapper, not bare asyncio.run, so we
                # know _run_async is using a loop it owns and can cancel.
                events["submitted_fn"] = getattr(fn, "__name__", repr(fn))
                return TimeoutFuture()

            def shutdown(self, wait=True, cancel_futures=False):
                events["shutdown_calls"].append((wait, cancel_futures))

        async def _never_finishes():
            await asyncio.sleep(999)

        monkeypatch.setattr(
            concurrent.futures,
            "ThreadPoolExecutor",
            FakeExecutor,
        )

        coro = _never_finishes()
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                _run_async(coro)
        finally:
            # FakeExecutor intentionally never invokes the submitted worker;
            # therefore the test, not production, retains ownership.
            coro.close()

        assert events["result_timeout"] == 300
        # The worker wrapper creates its own event loop so _run_async can
        # cancel the task on timeout — this must NOT be bare asyncio.run.
        assert events["submitted_fn"] != "run", (
            "_run_async submitted asyncio.run directly — it must submit a "
            "worker wrapper that owns the event loop so timeouts can cancel "
            "the task"
        )
        # Critical: shutdown must NOT wait. If wait=True, a stuck coroutine
        # would freeze the caller (converts a thread leak into a hang).
        assert events["shutdown_calls"], "shutdown was never called"
        for wait, _cancel in events["shutdown_calls"]:
            assert wait is False, (
                f"shutdown called with wait={wait} — a stuck tool coroutine "
                f"would hang the caller indefinitely"
            )

    @pytest.mark.asyncio
    async def test_timeout_cancels_coroutine_in_worker_loop(self, monkeypatch):
        """On timeout, the worker's event loop must receive a cancel request
        so the coroutine stops and the thread exits — not leaked.

        Before the fix, future.cancel() on a running ThreadPoolExecutor
        future is a no-op, so the worker thread kept running the coroutine
        to completion (leaking one thread per tool-timeout).
        """
        from model_tools import _run_async

        # Shrink the 300s internal timeout by patching future.result.
        # We do this surgically: let everything else run for real so the
        # worker loop actually exists and can observe cancellation.
        import concurrent.futures as _cf

        real_pool_cls = _cf.ThreadPoolExecutor

        class FastTimeoutPool(real_pool_cls):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)

        # Patch future.result to time out after 1s instead of 300s.
        real_result = _cf.Future.result

        def fast_result(self, timeout=None):
            return real_result(self, timeout=1.0 if timeout == 300 else timeout)

        monkeypatch.setattr(_cf.Future, "result", fast_result)

        cancel_observed = threading.Event()

        async def _slow_cancellable():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancel_observed.set()
                raise

        import time as _time
        t0 = _time.time()
        with pytest.raises(_cf.TimeoutError):
            _run_async(_slow_cancellable())
        elapsed = _time.time() - t0

        # Caller must return fast (no hang waiting for the coro).
        assert elapsed < 3.0, (
            f"_run_async blocked caller for {elapsed:.1f}s — should return "
            f"on timeout regardless of whether the coroutine has finished"
        )

        # Worker thread must cancel the task (not leak).
        deadline = _time.time() + 5
        while not cancel_observed.is_set() and _time.time() < deadline:
            _time.sleep(0.05)
        assert cancel_observed.is_set(), (
            "Coroutine never received CancelledError — worker thread leaked "
            "(ThreadPoolExecutor.cancel() is a no-op on a running future; "
            "_run_async must cancel the task inside its worker loop)"
        )


# ---------------------------------------------------------------------------
# Integration: full vision_analyze dispatch chain
# ---------------------------------------------------------------------------

def _mock_vision_response():
    """Build a fake LLM response matching async_call_llm's return shape."""
    message = SimpleNamespace(content="A cat sitting on a chair.")
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/vision", usage=None)


class TestVisionDispatchLoopSafety:
    """Simulate the full registry.dispatch('vision_analyze') chain and
    verify the event loop stays alive afterwards — the exact scenario
    from issue #2104."""

    def test_vision_dispatch_keeps_loop_alive(self, tmp_path):
        """After dispatching vision_analyze via the registry, the event
        loop must remain open so cached async clients don't crash on GC."""
        from model_tools import _get_tool_loop
        from tools.registry import registry

        fake_response = _mock_vision_response()

        with (
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=fake_response,
            ),
            patch(
                "tools.vision_tools._download_image",
                new_callable=AsyncMock,
                side_effect=lambda url, dest, **kw: _write_fake_image(dest),
            ),
            patch(
                "tools.vision_tools._validate_image_url_async",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/jpeg;base64,abc",
            ),
        ):
            result_json = registry.dispatch(
                "vision_analyze",
                {"image_url": "https://example.com/cat.png", "question": "What is this?"},
            )

        result = json.loads(result_json)
        assert result.get("success") is True, f"dispatch failed: {result}"
        assert "cat" in result.get("analysis", "").lower()

        loop = _get_tool_loop()
        assert not loop.is_closed(), (
            "Event loop closed after vision_analyze dispatch — cached async "
            "clients will crash with 'Event loop is closed' (issue #2104)"
        )

    def test_two_consecutive_vision_dispatches(self, tmp_path):
        """Two back-to-back vision_analyze dispatches must both succeed
        and share the same loop (simulates 'first call fails, second
        works' from the issue report)."""
        from model_tools import _get_tool_loop
        from tools.registry import registry

        fake_response = _mock_vision_response()

        with (
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                return_value=fake_response,
            ),
            patch(
                "tools.vision_tools._download_image",
                new_callable=AsyncMock,
                side_effect=lambda url, dest, **kw: _write_fake_image(dest),
            ),
            patch(
                "tools.vision_tools._validate_image_url_async",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/jpeg;base64,abc",
            ),
        ):
            args = {"image_url": "https://example.com/cat.png", "question": "Describe"}

            r1 = json.loads(registry.dispatch("vision_analyze", args))
            loop_after_first = _get_tool_loop()

            r2 = json.loads(registry.dispatch("vision_analyze", args))
            loop_after_second = _get_tool_loop()

        assert r1.get("success") is True
        assert r2.get("success") is True
        assert loop_after_first is loop_after_second, "Loop changed between dispatches"
        assert not loop_after_second.is_closed()


def _write_fake_image(dest):
    """Write minimal bytes so vision_analyze_tool thinks download succeeded."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"\xff\xd8\xff" + b"\x00" * 16)
    return dest
