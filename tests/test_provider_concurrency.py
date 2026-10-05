"""Tests for cross-process provider call coordination."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from llm_council.providers import concurrency
from llm_council.providers.concurrency import acquire_provider_call_lease, provider_call_slot


def _hold_provider_lock(
    lock_dir: str, provider_name: str, ready_queue: multiprocessing.Queue
) -> None:
    os.environ["LLM_COUNCIL_LOCK_DIR"] = lock_dir
    lease = acquire_provider_call_lease(provider_name, timeout_seconds=1.0)
    ready_queue.put("locked")
    try:
        time.sleep(0.35)
    finally:
        lease.release()


@pytest.mark.skipif(not hasattr(multiprocessing, "Process"), reason="multiprocessing unavailable")
def test_acquire_provider_call_lease_times_out_when_another_process_holds_slot(tmp_path: Path):
    """A second process should not be able to enter the same provider slot immediately."""

    ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing
    )
    ready_queue = ctx.Queue()
    process = ctx.Process(
        target=_hold_provider_lock,
        args=(str(tmp_path), "openai", ready_queue),
    )
    process.start()

    try:
        assert ready_queue.get(timeout=2) == "locked"
        os.environ["LLM_COUNCIL_LOCK_DIR"] = str(tmp_path)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="provider slot: openai"):
            acquire_provider_call_lease(
                "openai",
                timeout_seconds=0.1,
                poll_interval_seconds=0.02,
            )
        assert time.monotonic() - started >= 0.09
    finally:
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)


def test_acquire_provider_call_lease_noops_when_locks_disabled(monkeypatch: pytest.MonkeyPatch):
    """Disabling locks should yield an immediate no-op lease."""

    monkeypatch.setenv("LLM_COUNCIL_DISABLE_PROVIDER_LOCKS", "1")
    lease = acquire_provider_call_lease("claude", timeout_seconds=0.1)
    try:
        assert lease.wait_ms == 0.0
        assert lease.fd is None
        assert lease.lock_path is None
    finally:
        lease.release()


@pytest.fixture
def isolated_locks(tmp_path, monkeypatch):
    if concurrency.fcntl is None:
        pytest.skip("POSIX locks required")
    monkeypatch.setenv("LLM_COUNCIL_LOCK_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_COUNCIL_DISABLE_PROVIDER_LOCKS", raising=False)


@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("timeout_seconds", [None, 0.25])
async def test_cancelled_waiter_cannot_later_leak_slot(
    isolated_locks, monkeypatch, repeated, timeout_seconds
):
    held = acquire_provider_call_lease("cancel")
    attempted = threading.Event()
    finished = threading.Event()
    leases = []
    real_acquire = acquire_provider_call_lease

    def observed_acquire(*args, **kwargs):
        attempted.set()
        try:
            lease = real_acquire(*args, **kwargs)
            leases.append(lease)
            return lease
        finally:
            finished.set()

    async def waiter():
        async with provider_call_slot("cancel", timeout_seconds=timeout_seconds):
            pytest.fail("cancelled waiter entered slot")

    monkeypatch.setattr(concurrency, "acquire_provider_call_lease", observed_acquire)
    task = asyncio.create_task(waiter())
    try:
        assert await asyncio.to_thread(attempted.wait, 1)
        task.cancel()
        if repeated:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.15)
        held.release()
        assert await asyncio.to_thread(finished.wait, 1)
        # Any cancelled worker has now returned. No orphan lease may own this file.
        lease = real_acquire("cancel", timeout_seconds=0.05, poll_interval_seconds=0.005)
        lease.release()
    finally:
        held.release()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(finished.wait, 1)
        for lease in leases:
            lease.release()


async def test_cancellation_concurrent_with_acquire_releases_slot(isolated_locks, monkeypatch):
    real_acquire = acquire_provider_call_lease
    loop = asyncio.get_running_loop()
    leases = []
    acquired = threading.Event()

    def cancel_on_acquire(*args, **kwargs):
        lease = real_acquire(*args, **kwargs)
        leases.append(lease)
        loop.call_soon_threadsafe(task.cancel)
        acquired.set()
        return lease

    async def waiter():
        async with provider_call_slot("race", timeout_seconds=0.2):
            await asyncio.sleep(1)

    monkeypatch.setattr(concurrency, "acquire_provider_call_lease", cancel_on_acquire)
    task = asyncio.create_task(waiter())
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert acquired.is_set()
        lease = real_acquire("race", timeout_seconds=0.05, poll_interval_seconds=0.005)
        lease.release()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for lease in leases:
            lease.release()


async def test_exhausted_wait_is_bounded_by_timeout_not_poll_interval(isolated_locks):
    held = acquire_provider_call_lease("exhausted")
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="provider slot: exhausted"):
            async with provider_call_slot(
                "exhausted", timeout_seconds=0.04, poll_interval_seconds=0.25
            ):
                pytest.fail("exhausted waiter entered slot")
        assert time.monotonic() - started < 0.15
    finally:
        held.release()
    async with provider_call_slot("exhausted", timeout_seconds=0.1) as waited_ms:
        assert waited_ms >= 0


async def test_cancelled_slot_body_releases_synchronously(isolated_locks):
    entered = asyncio.Event()

    async def waiter():
        async with provider_call_slot("body", timeout_seconds=0.1):
            entered.set()
            await asyncio.sleep(10)

    task = asyncio.create_task(waiter())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    async with provider_call_slot("body", timeout_seconds=0.1):
        pass


async def test_waiter_acquires_after_release_and_reports_wait(isolated_locks):
    held = acquire_provider_call_lease("wait")

    async def waiter():
        async with provider_call_slot("wait", timeout_seconds=0.5) as waited_ms:
            return waited_ms

    task = asyncio.create_task(waiter())
    try:
        await asyncio.sleep(0.03)
        assert not task.done()
        held.release()
        assert await asyncio.wait_for(task, timeout=1) >= 20
    finally:
        held.release()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
