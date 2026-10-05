"""Shared subprocess helpers for CLI-backed providers."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import signal

logger = logging.getLogger(__name__)


async def terminate_process_tree(
    proc: asyncio.subprocess.Process, grace_seconds: float = 1.0
) -> None:
    """Kill an owned process group and wait a finite time for the child to be reaped.

    POSIX callers must spawn with ``start_new_session=True`` and retain ownership
    of that group, even after the leader exits. Escaped groups are not contained;
    Windows supports only direct-child termination. Readers remain caller-owned:
    this helper neither drains nor closes pipes, and never calls ``communicate``.

    Cancellation is deferred until this cleanup allowance expires or cleanup
    finishes, then re-raised. On failure, raise the cleanup error (or chain it as
    the cause of cancellation). Success covers the child/group, not pipe EOF or
    reader completion; callers must bound and settle their own reader tasks.
    """
    if not math.isfinite(grace_seconds):
        raise ValueError("Subprocess cleanup allowance must be finite")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, grace_seconds)
    cancelled: asyncio.CancelledError | None = None
    group_owned = hasattr(os, "killpg")

    try:
        try:
            if group_owned:
                # Never signal the caller's process group, including a bad PID 0.
                if proc.pid <= 0 or proc.pid == os.getpgrp():
                    raise ValueError("Subprocess cleanup requires an owned process group")
                os.killpg(proc.pid, signal.SIGKILL)
            elif proc.returncode is None:  # pragma: no cover - Windows fallback
                proc.kill()
        except ProcessLookupError:
            pass

        while True:
            group_alive = False
            if group_owned:
                try:
                    os.killpg(proc.pid, 0)
                    group_alive = True
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # EPERM is not proof of absence; macOS can report it while
                    # an exiting group is still present. Retain the deadline.
                    group_alive = True
            # asyncio's child watcher sets returncode after reaping. Polling it
            # avoids waiting for pipe EOF on runtimes whose wait() also needs EOF.
            if proc.returncode is not None and not group_alive:
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"Subprocess cleanup incomplete for PID {proc.pid}")
            try:
                await asyncio.sleep(min(0.01, remaining))
            except asyncio.CancelledError as exc:
                if cancelled is None:
                    cancelled = exc
    except Exception as exc:
        if cancelled is not None:
            # Python 3.10 can lose a CancelledError's cause at a Task boundary.
            logger.error(
                "Subprocess cleanup failed during cancellation for PID %s: %s", proc.pid, exc
            )
            raise cancelled from exc
        raise

    if cancelled is not None:
        raise cancelled
