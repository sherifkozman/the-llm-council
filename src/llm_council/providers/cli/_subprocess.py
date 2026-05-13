"""Shared subprocess helpers for CLI-backed providers."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import signal


async def write_stdin_and_close(writer: object, text: str) -> None:
    """Write text to a subprocess stdin stream and close it."""

    data = text.encode("utf-8")
    result = writer.write(data)  # type: ignore[attr-defined]
    if inspect.isawaitable(result):
        await result

    drain = writer.drain()  # type: ignore[attr-defined]
    if inspect.isawaitable(drain):
        await drain

    closed = writer.close()  # type: ignore[attr-defined]
    if inspect.isawaitable(closed):
        await closed


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
