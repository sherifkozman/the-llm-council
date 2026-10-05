"""Real, harmless process groups exercising bounded provider cleanup."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from llm_council.providers.cli import _subprocess

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")


async def _until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("Test condition did not settle")
        await asyncio.sleep(0.005)


def _group_exists(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@asynccontextmanager
async def _tree(tmp_path, *, exited_leader=False, escaped=False):
    marker = tmp_path / "child-ready"
    child_code = (
        "import os, pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent_code = (
        "import subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], "
        f"start_new_session={escaped!r}); " + ("sys.exit(0)" if exited_leader else "child.wait()")
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        parent_code,
        child_code,
        str(marker),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        await _until(marker.exists)
        if exited_leader:
            await _until(lambda: proc.returncode is not None)
        yield proc, int(marker.read_text())
    finally:
        # Only the group created above and its explicitly recorded escaped child.
        if escaped and marker.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(marker.read_text()), signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        # Harness-only transport close keeps failed assertions bounded too.
        proc._transport.close()
        await asyncio.wait_for(proc.wait(), timeout=2)
        await _until(lambda: not _group_exists(proc.pid))


@pytest.mark.parametrize("exited_leader", [False, True])
async def test_terminates_descendant_holding_pipes(tmp_path, exited_leader):
    async with _tree(tmp_path, exited_leader=exited_leader) as (proc, _):
        await asyncio.wait_for(
            _subprocess.terminate_process_tree(proc, grace_seconds=0.3), timeout=0.8
        )
        assert proc.returncode is not None
        await _until(lambda: not _group_exists(proc.pid))


async def test_cleanup_does_not_compete_with_existing_pipe_readers(tmp_path):
    async with _tree(tmp_path) as (proc, _):
        reader = asyncio.create_task(proc.communicate())
        await asyncio.sleep(0)
        try:
            await _subprocess.terminate_process_tree(proc, grace_seconds=0.3)
            assert await asyncio.wait_for(reader, timeout=1) == (b"", b"")
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(reader, timeout=1)


async def test_already_settled_child_is_safe():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "pass",
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(proc.communicate(), timeout=2)
    await _subprocess.terminate_process_tree(proc, grace_seconds=0.1)
    assert proc.returncode == 0


async def test_escaped_group_is_not_claimed_or_killed(tmp_path):
    async with _tree(tmp_path, escaped=True) as (proc, escaped_pid):
        started = time.monotonic()
        # A pipe inherited by an escaped child must never cause an unbounded retry.
        cleanup = asyncio.create_task(_subprocess.terminate_process_tree(proc, grace_seconds=0.1))
        try:
            done, _ = await asyncio.wait({cleanup}, timeout=0.5)
            assert cleanup in done, "cleanup exceeded its finite allowance"
            cleanup.result()
            assert time.monotonic() - started < 0.5
            assert proc.returncode is not None
            assert os.getpgid(escaped_pid) == escaped_pid
        finally:
            cleanup.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(cleanup, timeout=1)


async def test_failed_group_cleanup_is_reported(tmp_path, monkeypatch):
    async with _tree(tmp_path) as (proc, _):
        real_killpg = os.killpg

        def refuse_signal(pid, sig):
            if pid == proc.pid and sig == signal.SIGKILL:
                return
            return real_killpg(pid, sig)

        with monkeypatch.context() as patch:
            patch.setattr(_subprocess.os, "killpg", refuse_signal)
            with pytest.raises(TimeoutError, match="cleanup"):
                await asyncio.wait_for(
                    _subprocess.terminate_process_tree(proc, grace_seconds=0.05), timeout=0.5
                )


@pytest.mark.parametrize("repeated", [False, True])
async def test_cleanup_finishes_before_preserving_cancellation(tmp_path, monkeypatch, repeated):
    async with _tree(tmp_path) as (proc, _):
        real_killpg = os.killpg
        loop = asyncio.get_running_loop()
        signalled = asyncio.Event()

        def delayed_signal(pid, sig):
            if pid == proc.pid and sig == signal.SIGKILL:
                loop.call_later(0.06, real_killpg, pid, sig)
                signalled.set()
                return
            return real_killpg(pid, sig)

        with monkeypatch.context() as patch:
            patch.setattr(_subprocess.os, "killpg", delayed_signal)
            cleanup = asyncio.create_task(
                _subprocess.terminate_process_tree(proc, grace_seconds=0.3)
            )
            await asyncio.wait_for(signalled.wait(), timeout=1)
            cleanup.cancel("first")
            if repeated:
                await asyncio.sleep(0.01)
                cleanup.cancel("second")
            try:
                done, _ = await asyncio.wait({cleanup}, timeout=0.5)
                assert cleanup in done
                with pytest.raises(asyncio.CancelledError):
                    await cleanup
                assert proc.returncode is not None, "cancellation abandoned reaping"
                assert not _group_exists(proc.pid), "cancellation abandoned the owned group"
            finally:
                # Let the injected timer finish before the harness can kill this group.
                await asyncio.sleep(0.07)


@pytest.mark.parametrize("repeated", [False, True])
async def test_cancellation_keeps_cleanup_failure_as_cause(tmp_path, monkeypatch, caplog, repeated):
    async with _tree(tmp_path) as (proc, _):
        real_killpg = os.killpg

        def refuse_signal(pid, sig):
            if pid == proc.pid and sig == signal.SIGKILL:
                return
            return real_killpg(pid, sig)

        with monkeypatch.context() as patch:
            patch.setattr(_subprocess.os, "killpg", refuse_signal)
            failures = []

            async def run_cleanup():
                try:
                    await _subprocess.terminate_process_tree(proc, grace_seconds=0.05)
                except asyncio.CancelledError as exc:
                    # Python 3.10 may discard the cause at the Task boundary.
                    failures.append(exc.__cause__)
                    raise

            cleanup = asyncio.create_task(run_cleanup())
            await asyncio.sleep(0)
            cleanup.cancel()
            if repeated:
                await asyncio.sleep(0.01)
                cleanup.cancel()
            done, _ = await asyncio.wait({cleanup}, timeout=0.5)
            assert cleanup in done
            with pytest.raises(asyncio.CancelledError):
                await cleanup
            assert len(failures) == 1
            assert isinstance(failures[0], TimeoutError)
            assert "cleanup" in caplog.text and "cancellation" in caplog.text


async def test_group_probe_permission_error_does_not_abandon_reaping(tmp_path, monkeypatch):
    async with _tree(tmp_path) as (proc, _):
        real_killpg = os.killpg
        blocked_until = time.monotonic() + 0.04

        def transient_permission(pid, sig):
            if pid == proc.pid and sig == 0 and time.monotonic() < blocked_until:
                raise PermissionError("group is not yet observable")
            return real_killpg(pid, sig)

        with monkeypatch.context() as patch:
            patch.setattr(_subprocess.os, "killpg", transient_permission)
            await asyncio.wait_for(
                _subprocess.terminate_process_tree(proc, grace_seconds=0.3), timeout=0.5
            )
        assert proc.returncode is not None
        assert not _group_exists(proc.pid)


@pytest.mark.parametrize("allowance", [float("inf"), float("nan")])
async def test_cleanup_rejects_nonfinite_allowance(allowance):
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "pass", start_new_session=True
    )
    await asyncio.wait_for(proc.wait(), timeout=2)
    with pytest.raises(ValueError, match="finite"):
        await _subprocess.terminate_process_tree(proc, grace_seconds=allowance)


@pytest.mark.parametrize("pid", [0, os.getpgrp()])
async def test_cleanup_refuses_callers_process_group(pid):
    proc = SimpleNamespace(pid=pid, returncode=None)
    with pytest.raises(ValueError, match="owned process group"):
        await _subprocess.terminate_process_tree(proc, grace_seconds=0.1)


async def test_cleanup_does_not_signal_an_unrelated_process_group(tmp_path):
    unrelated = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)", start_new_session=True
    )
    try:
        async with _tree(tmp_path) as (proc, _):
            await asyncio.wait_for(
                _subprocess.terminate_process_tree(proc, grace_seconds=0.3), timeout=0.5
            )
            assert unrelated.returncode is None
            assert os.getpgid(unrelated.pid) == unrelated.pid
    finally:
        with contextlib.suppress(ProcessLookupError):
            unrelated.kill()
        await asyncio.wait_for(unrelated.wait(), timeout=2)
