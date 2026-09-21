"""
Codex CLI provider adapter.

Wraps the ``codex`` CLI for non-interactive generation via ``codex exec``.
Useful for agent-to-agent delegation and environments where CLI auth
is available but API keys may not be.

SECURITY NOTE: Uses asyncio.create_subprocess_exec with argument lists,
which is safe from shell injection (equivalent to execFile in Node.js).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shlex
import shutil
import signal
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from llm_council.providers.base import (
    DoctorResult,
    ErrorType,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
    classify_error,
    get_billing_help_url,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4"
# SECURITY: Least-privilege defaults - read-only sandbox, no auto-approve
# Override via CODEX_CLI_FLAGS env var or default_flags param for agentic mode
DEFAULT_FLAGS = "--sandbox read-only --skip-git-repo-check"
_CODEX_SUFFIX = "-codex"

# Unsafe modes that require explicit opt-in
_UNSAFE_FLAGS = {"--full-auto", "--sandbox workspace-write", "--approval-mode yolo"}
_UNSAFE_WARNING = (
    "WARNING: Codex CLI is running with permissive flags that allow local file/env access. "
    "This is unsafe with untrusted inputs. Ensure you trust the task source."
)

# Codex CLI panics under an aggressively stripped environment when launched
# from nested council subprocesses. Preserve the ambient runtime and strip only
# telemetry variables that can destabilize or leak outer-session tracing.
_ENV_DENYLIST_PREFIXES = (
    "OTEL_",
    "LANGSMITH_",
    "LANGCHAIN_",
)

# Optional fast-fail: seconds after `turn.started` without any answer before the
# call is abandoned. Unset (the default) means the request timeout, so only the
# deadline ends a silent turn. `codex exec --json` prints nothing while the model
# reasons, so silence is not a stall: under the former fixed 45 s window every one
# of the 11 "stalled" turns seen on 2026-09-18 was healthy and went on to finish
# with a full answer after the adapter had given up on it.
_STALL_SECONDS_ENV = "LLM_COUNCIL_CODEX_STALL_SECONDS"

# An isolated CODEX_HOME does not load the user's config.toml, so Codex sends no effort
# at all: measured on 0.154.0, `-m gpt-5.6-sol` under an isolated home reports
# `reasoning effort: none`, and `xhigh` with the real profile. The default here is what
# that profile asks for, so isolating changes where Codex writes without quietly
# changing how hard it thinks.
_REASONING_EFFORT_ENV = "LLM_COUNCIL_CODEX_REASONING_EFFORT"
_DEFAULT_REASONING_EFFORT = "xhigh"

# Windows only. After `turn.completed` Codex needs a few seconds to shut down by itself
# (measured: launcher gone after 3.3 s, the whole tree after 4.5 s), so a finished turn
# gets up to this long for its whole job to empty before the tree is ended; the wait is
# over as soon as it does. Deadlines, stalls and cancellation never wait.
# The original reason was that Codex wrote to the user's REAL profile, which CODEX_HOME
# now prevents. It is kept because it is nearly free and still the difference between
# 0 and 3 of 18 processes being force-killed mid-write to the throwaway home - a killed
# writer there leaves a locked file that the cleanup then has to retry around.
_NATURAL_EXIT_GRACE_SECONDS = 10.0

_CREATE_SUSPENDED = 0x00000004
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS = 1


def _extract_usage_payload(stdout_text: str) -> dict[str, int] | None:
    """Extract usage stats from Codex JSONL output."""

    for line in reversed(stdout_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("type") != "turn.completed":
            continue
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt_tokens = int(usage.get("input_tokens", 0) or 0) + int(
            usage.get("cached_input_tokens", 0) or 0
        )
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return None


def _prepare_schema_for_codex(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform a JSON schema for Codex structured output strictness."""

    result: dict[str, Any] = {}

    for key, value in schema.items():
        if key == "$schema":
            continue
        if key == "additionalProperties":
            continue

        if key == "properties" and isinstance(value, dict):
            result[key] = {
                prop_name: _prepare_schema_for_codex(prop_schema)
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "object"
                else (
                    {
                        **prop_schema,
                        "items": _prepare_schema_for_codex(prop_schema["items"]),
                    }
                    if isinstance(prop_schema, dict)
                    and prop_schema.get("type") == "array"
                    and isinstance(prop_schema.get("items"), dict)
                    and prop_schema["items"].get("type") == "object"
                    else prop_schema
                )
                for prop_name, prop_schema in value.items()
            }
            result["required"] = list(value.keys())
            result["additionalProperties"] = False
        elif key == "required":
            continue
        elif isinstance(value, dict) and value.get("type") == "object":
            result[key] = _prepare_schema_for_codex(value)
        else:
            result[key] = value

    if schema.get("type") == "object" and "additionalProperties" not in result:
        result["additionalProperties"] = False

    return result


def _extract_agent_message(stdout_text: str) -> str:
    """Extract the last agent message from Codex JSONL output."""

    for line in reversed(stdout_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("type") != "item.completed":
            continue
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str):
            return text
    return ""


def _extract_error_message(stdout_text: str) -> str:
    """Extract a Codex error payload from JSONL stdout when stderr is empty.

    Recognizes both ``type: "error"`` events (synchronous client-side errors,
    e.g. an invalid JSON schema) and ``type: "turn.failed"`` events (server-side
    rejections such as an unsupported model). The latter arrives with exit
    code 0, so callers must consult this before assuming success.
    """

    for line in reversed(stdout_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event_type = payload.get("type")
        if event_type == "error":
            message = payload.get("message")
            if isinstance(message, str) and message:
                return message
        elif event_type == "turn.failed":
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                message = error_payload.get("message")
                if isinstance(message, str) and message:
                    return message
    return ""


@dataclass
class _LiveCodexState:
    """Incremental subprocess state for Codex CLI calls."""

    stdout_parts: list[str] = field(default_factory=list)
    stderr_parts: list[str] = field(default_factory=list)
    agent_message: str = ""
    error_message: str = ""
    usage: dict[str, int] | None = None
    saw_turn_started: bool = False
    saw_turn_completed: bool = False
    # `turn.failed` is a REJECTED turn, not a finished one. It sets saw_turn_completed
    # so the loop stops waiting, which would otherwise make it look like a normal end of
    # turn; this flag keeps the two apart for the handler after the loop.
    saw_turn_failed: bool = False
    turn_started_at: float | None = None


def _ingest_codex_stdout_line(line: str, state: _LiveCodexState) -> None:
    """Update live state from a single Codex JSONL stdout line."""

    state.stdout_parts.append(line)
    stripped = line.strip()
    if not stripped:
        return

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return

    if not isinstance(payload, dict):
        return

    event_type = payload.get("type")
    if event_type == "turn.started":
        state.saw_turn_started = True
    elif event_type == "item.completed":
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str) and text:
                state.agent_message = text
    elif event_type == "turn.completed":
        state.saw_turn_completed = True
        usage = payload.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("input_tokens", 0) or 0) + int(
                usage.get("cached_input_tokens", 0) or 0
            )
            completion_tokens = int(usage.get("output_tokens", 0) or 0)
            state.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
    elif event_type == "turn.failed":
        # Codex emits turn.failed with exit code 0 when the server rejects the
        # request -- e.g. a model this CLI version does not support. Capture the
        # message so the post-loop handler surfaces it instead of an empty success.
        state.saw_turn_completed = True
        state.saw_turn_failed = True
        error_payload = payload.get("error")
        if isinstance(error_payload, dict):
            message = error_payload.get("message")
            if isinstance(message, str) and message:
                state.error_message = message
    elif event_type == "error":
        message = payload.get("message")
        if isinstance(message, str):
            state.error_message = message


async def _read_codex_stdout(stream: asyncio.StreamReader | None, state: _LiveCodexState) -> None:
    """Consume Codex stdout incrementally."""

    if stream is None:
        return

    while True:
        line = await stream.readline()
        if not line:
            return
        _ingest_codex_stdout_line(line.decode("utf-8", errors="replace"), state)
        now = asyncio.get_running_loop().time()
        if state.saw_turn_started and state.turn_started_at is None:
            state.turn_started_at = now


async def _read_codex_stderr(stream: asyncio.StreamReader | None, state: _LiveCodexState) -> None:
    """Consume Codex stderr incrementally."""

    if stream is None:
        return

    while True:
        line = await stream.readline()
        if not line:
            return
        state.stderr_parts.append(line.decode("utf-8", errors="replace"))


async def _drain_reader_tasks(*tasks: asyncio.Task[None]) -> None:
    """Wait briefly for stream reader tasks to finish, then cancel if needed."""

    pending = [task for task in tasks if task is not None]
    if not pending:
        return

    try:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=1.0)
    except asyncio.TimeoutError:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def _own_process_handle(proc: asyncio.subprocess.Process) -> int | None:
    """Return the OS handle asyncio already holds for a real child, else None.

    Never OpenProcess(pid): a pid may belong to somebody else by the time it is
    used (and unit tests hand in fakes with made-up pids). The handle inside the
    Popen object can only ever be our own child.
    """

    transport = getattr(proc, "_transport", None)
    if not isinstance(transport, asyncio.SubprocessTransport):
        return None
    handle = getattr(transport.get_extra_info("subprocess"), "_handle", None)
    return int(handle) if isinstance(handle, int) else None


def _resume_process(handle: int) -> int:
    """Windows: let a process that was created suspended run. Returns an NTSTATUS (0 = ok).

    `NtResumeProcess` is undocumented but long stable. The documented route,
    `ResumeThread`, needs the primary thread handle, which Popen closes at once.
    """

    if sys.platform != "win32":
        return 0
    import ctypes
    from ctypes import wintypes

    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    return int(ntdll.NtResumeProcess(handle))


def _adopt_into_job(proc: asyncio.subprocess.Process) -> int | None:
    """Windows: confine a child that was created SUSPENDED to a kill-on-close Job Object.

    `proc.kill()` ends only the `codex.CMD` shim; node and codex.exe live on, keep
    spending quota and keep the isolated HOME open so it can never be removed. A
    parent-pid walk (`taskkill /T`) is no answer either: after pid reuse it adopts
    unrelated processes. A job has neither problem - whatever a member spawns is
    born inside the job, and because this child has not run yet, nothing can have
    escaped.

    Returns the job handle; None on other platforms and for test doubles, where
    nothing was created suspended. A real child never leaves here suspended, and
    never runs outside a job: if anything at all goes wrong it is ended and the
    failure is raised (as RuntimeError, unless it is an interrupt), because Codex
    outside a job IS the leak this prevents.
    """

    if sys.platform != "win32" or not isinstance(proc, asyncio.subprocess.Process):
        return None

    job: int | None = None
    try:
        import ctypes
        from ctypes import wintypes

        handle = _own_process_handle(proc)
        if handle is None:
            raise RuntimeError("this Python does not expose the child's process handle")

        class _BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimits),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ) or not kernel32.AssignProcessToJobObject(job, handle):
            raise ctypes.WinError(ctypes.get_last_error())

        status = _resume_process(handle)
        if status != 0:
            raise RuntimeError(f"NtResumeProcess failed (NTSTATUS 0x{status & 0xFFFFFFFF:08x})")
        return job
    except BaseException as exc:
        # The child has not run yet. Whatever happened, it must neither stay suspended
        # nor be let loose outside a job, and the job handle must not leak.
        with contextlib.suppress(Exception):
            _close_job(job)  # kill-on-close ends the child if it got as far as the job
        with contextlib.suppress(Exception):
            proc.kill()
        if isinstance(exc, Exception):
            raise RuntimeError(
                f"Codex CLI: could not confine the child to a job object ({exc}); it was "
                "ended rather than run where it could not be stopped."
            ) from exc
        raise


def _terminate_job(job: int | None) -> None:
    """Windows: end every process in the job at once."""

    if sys.platform != "win32" or not job:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    if not kernel32.TerminateJobObject(job, 1):
        # Not fatal: closing the handle ends the job as well.
        logger.warning("Codex CLI: TerminateJobObject failed (%s)", ctypes.get_last_error())


def _job_is_active(job: int) -> bool:
    """Windows: is any process of the job still alive? True when that cannot be told."""

    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class _Accounting(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    info = _Accounting()
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
        ]
        answered = kernel32.QueryInformationJobObject(
            job,
            _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        )
    except OSError:
        return True
    return bool(info.ActiveProcesses) if answered else True


def _close_job(job: int | None) -> None:
    """Windows: release the job handle.

    Kill-on-close ends whatever is still in the job, and the same happens when a
    council process dies with the handle open, so it cannot leave Codex behind.
    """

    if sys.platform != "win32" or not job:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    if not kernel32.CloseHandle(job):
        logger.warning("Codex CLI: closing the job handle failed (%s)", ctypes.get_last_error())


# The ChatGPT sign-in Codex keeps in CODEX_HOME. Its refresh token ROTATES: every refresh issues a
# new one and the old one is dead from then on. A council call must therefore SHARE the user's
# file and never copy it - a refresh into a throwaway copy dies with the copy and leaves the real
# file holding a spent token, which signs the user out at their next refresh ("your refresh token
# was already used"). Codex saves by truncating and rewriting the file in place and signs out with
# remove_file (codex-rs FileAuthStorage), so a HARD LINK shares every refresh, while a sign-out in
# the call's home drops only that entry. Not a symlink: a save through a symlink whose target was
# signed out elsewhere would create the real file again and undo that sign-out. This module never
# writes the real file itself: Codex and the desktop app write it without a lock
# (openai/codex#10332), so any write of ours could race theirs and put back a spent token.
_SIGN_IN_FILE = "auth.json"
# Copied as before and never shared: nothing shows how Codex writes or rotates it, and a council
# call runs without the MCP servers it belongs to.
_COPIED_FILES = (".credentials.json",)


@dataclass
class _SharedSignIn:
    real: Path
    isolated: Path
    identity: tuple[int, int]  # (st_dev, st_ino) of the file both names shared
    digest: str  # sha256 of its content when it was shared


def _file_identity(path: Path) -> tuple[int, int] | None:
    """(st_dev, st_ino), or None where the filesystem cannot tell files apart (st_ino 0)."""

    st = path.stat()
    return (st.st_dev, st.st_ino) if st.st_ino else None


def _resolved_identity(path: Path) -> tuple[int, int] | None:
    """_file_identity, or None when the name does not resolve to a file that can be stat'ed."""

    try:
        return _file_identity(path)
    except OSError:
        return None


def _name_exists(path: Path) -> bool:
    """Whether the name is there, even as a link that no longer resolves.

    Only a missing name is False: Codex signs out with remove_file, which removes the name.
    Any other failure to look it up propagates rather than passing for a sign-out.
    """

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _share_sign_in(source: Path, target: Path) -> tuple[tuple[int, int], str]:
    """Hard-link `target` to the user's real sign-in, or refuse. Never a symlink or a copy."""

    try:
        os.link(source, target)
    except OSError as exc:
        raise RuntimeError(
            f"Codex CLI: cannot share your sign-in with the isolated home ({exc}). Refusing to "
            "copy it: Codex rotates its refresh token, and a refresh into a copy would sign "
            "you out of Codex everywhere else."
        ) from exc
    identity = _file_identity(target)
    if identity is None or identity != _file_identity(source):
        target.unlink(missing_ok=True)
        raise RuntimeError(
            "Codex CLI: this filesystem cannot confirm that the isolated sign-in is your real "
            "one; refusing to run Codex on it."
        )
    return identity, hashlib.sha256(target.read_bytes()).hexdigest()


def _is_valid_json(data: bytes) -> bool:
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


def _keep_aside(real: Path, data: bytes) -> Path:
    """Store `data` next to `real` under a new, unique, owner-only name. Never overwrites."""

    fd, name = tempfile.mkstemp(prefix=f"{real.name}.council-", suffix=".json", dir=real.parent)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return Path(name)


def _check_shared_sign_in(shared: _SharedSignIn) -> None:
    """After the call, report what the two names show - never repair the real file.

    - Both names still share one file: if it no longer parses on two reads, a Codex ended in
      the middle of its in-place save cut it short; by then its refresh token has rotated, so
      the only fix is to sign in again.
    - They no longer share one file (either side was replaced), or that cannot be confirmed
      (the real name no longer resolves, e.g. a dangling link), and the call's side holds a
      save the call made: that sign-in may exist nowhere else, so it is kept aside and
      reported.
    - A name is gone: a sign-out - in the call's home, or elsewhere, which wins; nothing is
      recreated or kept. Only a missing name counts; any other failure is reported.
    """

    real, isolated = shared.real, shared.isolated
    try:
        if not _name_exists(isolated) or not _name_exists(real):
            return
        still_shared = (
            _resolved_identity(isolated) == shared.identity
            and _resolved_identity(real) == shared.identity
        )
        if still_shared:
            if not _is_valid_json(real.read_bytes()) and not _is_valid_json(real.read_bytes()):
                logger.warning(
                    "Codex CLI: a Codex process ended while saving your sign-in and %s is now "
                    "unreadable. Sign in again with `codex login`.",
                    real,
                )
            return
        data = isolated.read_bytes()
        if hashlib.sha256(data).hexdigest() == shared.digest:
            return
        try:
            kept = _keep_aside(real, data)
        except OSError as exc:
            logger.warning(
                "Codex CLI: Codex saved a sign-in during a council call that %s may not hold, "
                "and it could not be kept (%s), so it is lost with the call's home. If Codex "
                "asks you to sign in, run `codex login`.",
                real,
                exc,
            )
            return
        logger.warning(
            "Codex CLI: Codex saved a sign-in during a council call that %s may not hold: the "
            "two no longer share one file, or that could not be confirmed. If Codex asks you "
            "to sign in, run `codex login`. The call's version was kept as %s; delete it once "
            "you no longer need it.",
            real,
            kept,
        )
    except OSError as exc:
        logger.warning(
            "Codex CLI: could not check the sign-in %s shared with a council call (%s). If "
            "Codex asks you to sign in, run `codex login`.",
            real,
            exc,
        )


def _remove_call_files(cli_home: str | None, temp_paths: tuple[str, ...]) -> None:
    """Remove what one call left on disk: its temp files and its isolated home.

    Blocking: `generate` runs it in a thread of its own. Ending a job is not a
    barrier: Windows releases the dead processes' handles a moment later - the prompt
    file is the child's stdin - hence up to 2 s of retries. In a thread so that the
    event loop does not wait for the disk, and a thread of its own rather than the
    executor's so that nothing can cancel it while it is still queued: once started
    it finishes, however often the caller is cancelled.
    """

    left: list[str] = []
    for attempts_left in range(9, -1, -1):
        for temp_path in temp_paths:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)
        if cli_home:
            shutil.rmtree(cli_home, ignore_errors=True)
        left = [path for path in (*temp_paths, cli_home) if path and os.path.exists(path)]
        if not left:
            return
        if attempts_left:
            time.sleep(0.2)
    logger.warning("Codex CLI: could not remove %s", ", ".join(left))


async def _terminate_live_process(
    proc: asyncio.subprocess.Process, grace_seconds: float = 1.0
) -> None:
    """Terminate a live subprocess without re-reading already-consumed streams."""

    if proc.returncode is None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows fallback
                proc.kill()
        except ProcessLookupError:
            pass

    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


class CodexCLIProvider(ProviderAdapter):
    """Codex CLI provider adapter."""

    name: ClassVar[str] = "codex"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=False,
        tool_use=False,
        structured_output=True,
        multimodal=False,
        max_tokens=4096,
    )

    def __init__(
        self,
        cli_path: str | None = None,
        default_model: str | None = None,
        default_flags: str | None = None,
        timeout: int = 120,
    ) -> None:
        self._cli_path = cli_path or shutil.which("codex")
        self._default_model = default_model or DEFAULT_MODEL
        self._default_flags = default_flags or DEFAULT_FLAGS
        self._timeout = timeout
        self._login_status_checked = False
        self._login_status_cache: str | None = None
        # isolated home -> the real sign-in it shares, for the post-call check
        self._shared_sign_ins: dict[str, _SharedSignIn] = {}

    def _build_command(
        self,
        *,
        model: str,
        output_last_message_path: str | None = None,
        output_schema_path: str | None = None,
    ) -> list[str]:
        """Build the CLI command as argument list (safe from injection)."""
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found.")

        cmd = [self._cli_path, "exec"]
        cmd.extend(shlex.split(self._default_flags))
        cmd.extend(["--json", "--color", "never"])
        cmd.extend(["-m", model])
        effort = self._reasoning_effort()
        if effort:
            cmd.extend(["-c", f"model_reasoning_effort={effort}"])
        if output_schema_path:
            cmd.extend(["--output-schema", output_schema_path])
        if output_last_message_path:
            cmd.extend(["-o", output_last_message_path])

        # prompt is fed via stdin in generate(), NOT argv. Windows
        # CreateProcess caps the command line at ~32KB; schema+prompt exceed it.
        return cmd

    def _reasoning_effort(self) -> str:
        """The reasoning effort to ask for, or "" to leave it to Codex.

        An isolated `CODEX_HOME` has no `config.toml`, so without this the CLI reports
        `reasoning effort: none` and the profile's `xhigh` is silently lost. Set
        `LLM_COUNCIL_CODEX_REASONING_EFFORT` to override, or to empty to opt out.
        Codex does NOT check the value - `bogusvalue` reaches its banner unchanged - so
        anything that is not a bare word is dropped here instead of reaching the CLI.
        """

        raw = os.environ.get(_REASONING_EFFORT_ENV)
        effort = _DEFAULT_REASONING_EFFORT if raw is None else raw.strip()
        if not effort:
            return ""
        return effort if effort.replace("_", "").isalnum() else _DEFAULT_REASONING_EFFORT

    def _check_unsafe_flags(self) -> None:
        """Emit warning if using unsafe permissive flags."""
        flags_str = self._default_flags
        for unsafe_flag in _UNSAFE_FLAGS:
            if unsafe_flag in flags_str:
                warnings.warn(_UNSAFE_WARNING, UserWarning, stacklevel=3)
                break

    def _get_subprocess_env(self) -> dict[str, str]:
        """Get a Codex-safe subprocess environment."""

        return {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(_ENV_DENYLIST_PREFIXES)
        }

    def _copy_isolated_runtime_state(self, codex_dir: Path) -> None:
        """Share the user's Codex sign-in with the isolated home, and nothing else.

        Required on every platform: `CODEX_HOME` points Codex here, and with no
        credentials it answers `401 Unauthorized: Missing bearer` without falling back
        (measured 2026-09-20). auth.json is hard-linked, so a rotated refresh token lands
        in the user's real file the moment Codex writes it; where that is impossible the
        call is refused rather than run on a copy.
        """

        source_dir = Path.home() / ".codex"
        sign_in = source_dir / _SIGN_IN_FILE
        if sign_in.exists():
            isolated = codex_dir / _SIGN_IN_FILE
            identity, digest = _share_sign_in(sign_in, isolated)
            self._shared_sign_ins[str(codex_dir.parent)] = _SharedSignIn(
                sign_in, isolated, identity, digest
            )
        for filename in _COPIED_FILES:
            source = source_dir / filename
            if not source.exists():
                continue
            try:
                shutil.copy2(source, codex_dir / filename)
            except OSError as exc:
                logger.warning("Codex CLI: could not copy %s: %s", filename, exc)

    def _create_isolated_cli_home(self) -> str:
        """Create an isolated HOME so nested Codex runs do not inherit tools/plugins."""

        base_dir = Path.home() / ".codex" / ".tmp"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            cli_home = Path(tempfile.mkdtemp(prefix="llm-council-codex-home-", dir=base_dir))
        except OSError:
            cli_home = Path(tempfile.mkdtemp(prefix="llm-council-codex-home-"))
        codex_dir = cli_home / ".codex"
        codex_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._copy_isolated_runtime_state(codex_dir)
        except BaseException:
            shutil.rmtree(cli_home, ignore_errors=True)
            raise
        return str(cli_home)

    def _request_timeout(self, request: GenerateRequest) -> float:
        """Return the effective timeout for this request."""

        return (
            float(request.timeout_seconds) if request.timeout_seconds is not None else self._timeout
        )

    def _stall_after_turn_started_seconds(self, request_timeout: float) -> float:
        """Return how long a turn may go without any answer before it is abandoned.

        The request timeout unless `LLM_COUNCIL_CODEX_STALL_SECONDS` asks for an
        earlier fast-fail: silence after `turn.started` is what reasoning looks like.
        """

        raw = os.environ.get(_STALL_SECONDS_ENV, "").strip()
        try:
            configured = float(raw) if raw else 0.0
        except ValueError:
            configured = 0.0
        return min(configured, request_timeout) if configured > 0 else request_timeout

    async def _login_status_text(self) -> str | None:
        """Return cached Codex login status output when available."""

        if self._login_status_checked:
            return self._login_status_cache
        self._login_status_checked = True

        if not self._cli_path:
            return None

        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli_path,
                "login",
                "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_subprocess_env(),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        except Exception:  # pragma: no cover - defensive path
            return None

        output = stdout.decode("utf-8", errors="replace").strip()
        error_output = stderr.decode("utf-8", errors="replace").strip()
        status_text = output or error_output or f"CLI returned exit code {proc.returncode}"
        self._login_status_cache = status_text
        return self._login_status_cache

    async def _resolve_model(self, request: GenerateRequest) -> str:
        """Normalize incompatible `*-codex` model names for local ChatGPT auth."""

        model = request.model or self._default_model
        if not (model.startswith("gpt-") and model.endswith(_CODEX_SUFFIX)):
            return model

        status_text = await self._login_status_text()
        if status_text and "logged in using chatgpt" in status_text.lower():
            compat_model = model[: -len(_CODEX_SUFFIX)]
            logger.warning(
                "Codex CLI model %s is not supported for ChatGPT-authenticated sessions; "
                "using %s instead.",
                model,
                compat_model,
            )
            return compat_model
        return model

    def _error_details(self, stderr_text: str) -> str:
        """Collapse stderr to the lines most useful for classification."""

        lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
        if not lines:
            return stderr_text.strip()

        error_lines = [line for line in lines if line.startswith("ERROR:")]
        if error_lines:
            return "\n".join(error_lines[-3:])

        return "\n".join(lines[-8:])

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate using safe subprocess with argument list."""
        if request.stream:
            raise NotImplementedError("Streaming not supported for CLI")

        self._check_unsafe_flags()
        model = await self._resolve_model(request)
        cli_home: str | None = None
        output_path: str | None = None
        schema_path: str | None = None
        stdin_path: str | None = None
        job: int | None = None

        try:
            cli_home = self._create_isolated_cli_home()
            output_fd, output_path = tempfile.mkstemp(
                prefix="llm-council-codex-last-message-", suffix=".txt"
            )
            os.close(output_fd)
            if request.structured_output:
                schema_fd, schema_path = tempfile.mkstemp(
                    prefix="llm-council-codex-schema-", suffix=".json"
                )
                with os.fdopen(schema_fd, "w", encoding="utf-8") as schema_file:
                    json.dump(
                        _prepare_schema_for_codex(dict(request.structured_output.json_schema)),
                        schema_file,
                    )
            prompt_text = ""
            if request.messages:
                prompt_text = "\n\n".join(m.content for m in request.messages if m.role == "user")
            elif request.prompt:
                prompt_text = request.prompt
            if not prompt_text:
                raise ValueError("Either 'messages' or 'prompt' must be provided")
            stdin_fd, stdin_path = tempfile.mkstemp(
                prefix="llm-council-codex-prompt-", suffix=".txt"
            )
            with os.fdopen(stdin_fd, "w", encoding="utf-8") as stdin_file:
                stdin_file.write(prompt_text)
            cmd = self._build_command(
                model=model,
                output_last_message_path=output_path,
                output_schema_path=schema_path,
            )
            env = self._get_subprocess_env()
            # HOME is what Codex uses on POSIX; on Windows it ignores it entirely, which
            # is why the isolation silently did nothing there. CODEX_HOME is the
            # documented variable for both config and auth (`codex exec --help`:
            # "--ignore-user-config ... auth still uses CODEX_HOME") and is what keeps a
            # nested run out of the user's real profile - which on this machine is the
            # live desktop app's 19 GB directory, memories and goals included.
            env["HOME"] = cli_home
            env["CODEX_HOME"] = str(Path(cli_home) / ".codex")
            # Safe: uses argument list, no shell; minimal environment.
            # The prompt is handed over as the child's stdin: the child gets its
            # own dup of the descriptor, so the parent's handle closes here.
            spawn_options: dict[str, Any] = {"start_new_session": True}
            if sys.platform == "win32":
                # Born suspended so that it is inside the job before it can spawn.
                spawn_options["creationflags"] = _CREATE_SUSPENDED
            with open(stdin_path, "rb") as stdin_fh:
                proc = await asyncio.create_subprocess_exec(
                    cmd[0],
                    *cmd[1:],
                    stdin=stdin_fh,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    **spawn_options,
                )
            try:
                job = _adopt_into_job(proc)
            except BaseException:
                # It was ended there. Make sure of it rather than assume, then reap it
                # so that the transport is not left to __del__.
                if isinstance(proc, asyncio.subprocess.Process):
                    with contextlib.suppress(Exception):
                        proc.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                raise

            async def stop_codex() -> None:
                _terminate_job(job)
                await _terminate_live_process(proc)

            timeout = self._request_timeout(request)
            if (
                not isinstance(proc.stdout, asyncio.StreamReader)
                or not isinstance(proc.stderr, asyncio.StreamReader)
                or not hasattr(proc, "wait")
            ):
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    await stop_codex()
                    raise RuntimeError(
                        f"Codex CLI timed out after {timeout}s. "
                        "Consider increasing timeout or simplifying the task."
                    )

                stdout_text = stdout.decode("utf-8", errors="replace")
                stderr_text = stderr.decode("utf-8", errors="replace")
                # turn.failed is reported with exit code 0, so check stdout too.
                batch_error = _extract_error_message(stdout_text)
                if proc.returncode != 0 or batch_error:
                    error_text = stderr_text or batch_error
                    error_details = self._error_details(error_text)
                    error_type = classify_error(error_details or stderr_text, proc.returncode or 0)

                    if error_type == ErrorType.BILLING:
                        billing_url = get_billing_help_url("codex")
                        raise RuntimeError(
                            f"BILLING ERROR: OpenAI credits exhausted. "
                            f"Add credits at {billing_url}\n"
                            f"Details: {error_details}"
                        )
                    elif error_type == ErrorType.AUTH:
                        raise RuntimeError(
                            f"AUTH ERROR: Invalid or missing API key. "
                            f"Check OPENAI_API_KEY environment variable.\n"
                            f"Details: {error_details}"
                        )
                    elif error_type == ErrorType.MODEL_UNAVAILABLE:
                        raise RuntimeError(f"MODEL UNAVAILABLE: {error_details}")
                    elif error_type == ErrorType.RATE_LIMIT:
                        raise RuntimeError(
                            f"RATE LIMIT: Too many requests. Wait and retry.\nDetails: {error_details}"
                        )
                    else:
                        raise RuntimeError(f"CLI failed ({error_type.value}): {error_details}")

                output = ""
                with contextlib.suppress(OSError):
                    if output_path:
                        output = Path(output_path).read_text(encoding="utf-8")
                if not output:
                    output = _extract_agent_message(stdout_text)
                if not output.strip():
                    logger.warning("Codex CLI returned success but empty output")

                return GenerateResponse(
                    text=output,
                    content=output,
                    usage=_extract_usage_payload(stdout_text),
                    raw={"stdout": stdout_text},
                )

            state = _LiveCodexState()
            stdout_task = asyncio.create_task(_read_codex_stdout(proc.stdout, state))
            stderr_task = asyncio.create_task(_read_codex_stderr(proc.stderr, state))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            stall_window = self._stall_after_turn_started_seconds(timeout)
            completion_started_at: float | None = None
            terminated_after_output = False
            output = ""

            while True:
                file_output_present = False
                with contextlib.suppress(OSError):
                    if output_path:
                        file_output = Path(output_path).read_text(encoding="utf-8")
                        if file_output:
                            output = file_output
                            file_output_present = True

                # Track the LATEST agent_message, not the first. `output` used to be
                # assigned only while falsy, which latched onto a pre-tool preamble and
                # ignored the real answer that arrived after it. The `-o` file still wins
                # whenever it exists, since it is written once at end of turn.
                if not file_output_present and state.agent_message:
                    output = state.agent_message

                now = loop.time()
                # Only a real end-of-turn signal may arm the termination timer.
                # An agent_message on its own is NOT one: current Codex models
                # emit a short "here is my plan" message before running tools,
                # so arming on it killed the process ~1s later and returned that
                # preamble as the answer. `-o` is written at turn end, so a
                # non-empty output file is an equally valid completion signal.
                if completion_started_at is None and (
                    state.saw_turn_completed or file_output_present
                ):
                    completion_started_at = now

                if proc.returncode is not None:
                    break

                if completion_started_at is not None and (
                    state.saw_turn_completed or now - completion_started_at >= 1.0
                ):
                    terminated_after_output = True
                    break

                # The deadline first: when both limits are reached in the same pass, the
                # caller must get the timeout, which is classified and retried differently.
                if now >= deadline:
                    await stop_codex()
                    await _drain_reader_tasks(stdout_task, stderr_task)
                    raise RuntimeError(
                        f"Codex CLI timed out after {timeout}s. "
                        "Consider increasing timeout or simplifying the task."
                    )

                # Reachable only with an explicit, shorter window: by default the window is
                # the timeout, the turn starts after the deadline clock, and the deadline
                # above is therefore always reached first.
                if (
                    state.turn_started_at is not None
                    and not output
                    and now - state.turn_started_at >= stall_window
                ):
                    await stop_codex()
                    await _drain_reader_tasks(stdout_task, stderr_task)
                    # The wording is a contract: `classify_error` matches substrings such
                    # as "timeout" and "500". This must stay UNKNOWN (one retry), so it
                    # names the setting and quotes neither the word nor the number.
                    raise RuntimeError(
                        "Codex CLI stalled after turn.started: no answer within the window "
                        f"set by {_STALL_SECONDS_ENV}. Codex prints nothing while it "
                        "reasons, so its silence says nothing about why."
                    )

                await asyncio.sleep(0.05)

            if proc.returncode is not None:
                # The launcher went first: read what it said, or the end of the turn may
                # not be known yet and its descendants would get no grace below.
                await _drain_reader_tasks(stdout_task, stderr_task)
            if (
                job is not None
                and not state.saw_turn_failed  # a rejected turn has no answer to protect
                and (completion_started_at is not None or state.saw_turn_completed)
            ):
                # See _NATURAL_EXIT_GRACE_SECONDS; never past the deadline. The whole
                # job, not the launcher: a descendant may still be writing.
                grace_ends = min(loop.time() + _NATURAL_EXIT_GRACE_SECONDS, deadline)
                while loop.time() < grace_ends and _job_is_active(job):
                    await asyncio.sleep(0.05)
            if terminated_after_output:
                await stop_codex()
            await _drain_reader_tasks(stdout_task, stderr_task)

            stdout_text = "".join(state.stdout_parts)
            stderr_text = "".join(state.stderr_parts)
            # turn.failed arrives with exit code 0, so a non-zero return code is
            # not the only failure signal -- state.error_message covers it.
            # `terminated_after_output` means "we stopped it after it had answered", so a
            # non-zero exit is ours and not a failure. A REJECTED turn is different: it
            # also sets the completion signal, but there is no answer, and returning it as
            # an empty success loses the seat silently (CLAUDE.md's 'silent seat loss').
            # `turn.failed` is itself the failure. Keying off the message alone left the
            # hole this change exists to close: a rejection with a missing, empty or
            # differently shaped `error` payload set no message, so `failed` stayed False
            # and the empty success escaped anyway.
            # Only `turn.failed` is terminal here; a bare `error` event can precede a turn
            # that still succeeds, which is why it alone does not end the call.
            failed = state.saw_turn_failed or proc.returncode != 0 or bool(state.error_message)
            if failed and (state.saw_turn_failed or not terminated_after_output):
                error_text = (
                    stderr_text
                    or state.error_message
                    or _extract_error_message(stdout_text)
                    or (
                        "Codex reported turn.failed without an error message"
                        if state.saw_turn_failed
                        else ""
                    )
                )
                error_details = self._error_details(error_text)
                error_type = classify_error(error_details or stderr_text, proc.returncode or 0)

                # Provide actionable error messages based on error type
                if error_type == ErrorType.BILLING:
                    billing_url = get_billing_help_url("codex")
                    raise RuntimeError(
                        f"BILLING ERROR: OpenAI credits exhausted. "
                        f"Add credits at {billing_url}\n"
                        f"Details: {error_details}"
                    )
                elif error_type == ErrorType.AUTH:
                    raise RuntimeError(
                        f"AUTH ERROR: Invalid or missing API key. "
                        f"Check OPENAI_API_KEY environment variable.\n"
                        f"Details: {error_details}"
                    )
                elif error_type == ErrorType.MODEL_UNAVAILABLE:
                    raise RuntimeError(f"MODEL UNAVAILABLE: {error_details}")
                elif error_type == ErrorType.RATE_LIMIT:
                    raise RuntimeError(
                        f"RATE LIMIT: Too many requests. Wait and retry.\nDetails: {error_details}"
                    )
                else:
                    raise RuntimeError(f"CLI failed ({error_type.value}): {error_details}")

            if not output:
                output = _extract_agent_message(stdout_text)

            # Warn if output is empty on success
            if not output.strip():
                logger.warning("Codex CLI returned success but empty output")

            return GenerateResponse(
                text=output,
                content=output,
                usage=state.usage or _extract_usage_payload(stdout_text),
                raw={"stdout": stdout_text},
            )
        finally:
            # First, so that nothing is left holding the files removed below. Nothing here
            # may stop the rest of the cleanup.
            with contextlib.suppress(Exception):
                _close_job(job)
            # Only now that no Codex is left running, and before its home is removed.
            if cli_home:
                shared = self._shared_sign_ins.pop(cli_home, None)
                if shared is not None:
                    try:
                        _check_shared_sign_in(shared)
                    except Exception:
                        logger.exception("Codex CLI: the shared sign-in check failed")
            temp_paths = tuple(p for p in (output_path, schema_path, stdin_path) if p)
            if cli_home or temp_paths:
                remover = threading.Thread(
                    target=_remove_call_files,
                    args=(cli_home, temp_paths),
                    name="codex-home-removal",
                    daemon=False,  # it must finish even if the interpreter is on its way out
                )
                try:
                    remover.start()
                except RuntimeError:  # no new threads (interpreter shutting down): do it here
                    _remove_call_files(cli_home, temp_paths)
                # Normally a few milliseconds. Bounded, so that a stubborn file does not
                # hold the caller past its own deadline: the thread carries on alone.
                give_up_waiting_at = time.monotonic() + 1.0
                while remover.is_alive() and time.monotonic() < give_up_waiting_at:
                    await asyncio.sleep(0.02)

    async def supports(self, capability: str) -> bool:
        return self.supports_capability(capability)

    async def doctor(self) -> DoctorResult:
        if not self._cli_path:
            return DoctorResult(ok=False, message="CLI not found")

        try:
            status_text = await self._login_status_text()
        except asyncio.TimeoutError:
            return DoctorResult(ok=False, message="CLI login status check timed out")
        except Exception as exc:  # pragma: no cover - defensive path
            return DoctorResult(ok=False, message=f"CLI login status check failed: {exc}")

        status_text = status_text or "CLI login status check failed"
        lowered = status_text.lower()

        if "not logged in" in lowered or "logged out" in lowered:
            return DoctorResult(ok=False, message=status_text)
        if "logged in" in lowered:
            return DoctorResult(ok=True, message=status_text)
        return DoctorResult(ok=False, message=status_text)


def _register() -> None:
    from llm_council.providers.registry import get_registry

    with contextlib.suppress(ValueError):
        get_registry().register_provider("codex", CodexCLIProvider)


_register()
