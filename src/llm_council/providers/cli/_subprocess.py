"""Shared subprocess helpers for CLI-backed providers."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal


async def terminate_process_tree(
    proc: asyncio.subprocess.Process, grace_seconds: float = 1.0
) -> None:
    """Terminate a CLI subprocess and any children it spawned.

    CLI wrappers such as Node-based agents can leave child processes running after
    the parent receives a kill signal. Starting subprocesses in their own session
    lets us terminate the whole process group and settle bounded council runs
    promptly instead of leaking until the outer case timeout fires.
    """

    if proc.returncode is None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows fallback
                proc.kill()
        except ProcessLookupError:
            pass

    try:
        await asyncio.wait_for(proc.communicate(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.communicate()
