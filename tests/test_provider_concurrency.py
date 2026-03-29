"""Tests for cross-process provider call coordination."""

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from llm_council.providers.concurrency import acquire_provider_call_lease


def _hold_provider_lock(lock_dir: str, provider_name: str, ready_queue: multiprocessing.Queue) -> None:
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

    ctx = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else multiprocessing
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
