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
import json
import logging
import os
import shlex
import signal
import shutil
import tempfile
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
    """Extract a Codex error payload from JSONL stdout when stderr is empty."""

    for line in reversed(stdout_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("type") != "error":
            continue
        message = payload.get("message")
        if isinstance(message, str):
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
    turn_started_at: float | None = None
    last_stdout_at: float | None = None


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
    elif event_type == "error":
        message = payload.get("message")
        if isinstance(message, str):
            state.error_message = message


async def _read_codex_stdout(
    stream: asyncio.StreamReader | None, state: _LiveCodexState
) -> None:
    """Consume Codex stdout incrementally."""

    if stream is None:
        return

    while True:
        line = await stream.readline()
        if not line:
            return
        _ingest_codex_stdout_line(line.decode("utf-8", errors="replace"), state)
        now = asyncio.get_running_loop().time()
        state.last_stdout_at = now
        if state.saw_turn_started and state.turn_started_at is None:
            state.turn_started_at = now


async def _read_codex_stderr(
    stream: asyncio.StreamReader | None, state: _LiveCodexState
) -> None:
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

    def _build_command(
        self,
        request: GenerateRequest,
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
        if output_schema_path:
            cmd.extend(["--output-schema", output_schema_path])
        if output_last_message_path:
            cmd.extend(["-o", output_last_message_path])

        prompt = ""
        if request.messages:
            parts = [m.content for m in request.messages if m.role == "user"]
            prompt = "\n\n".join(parts)
        elif request.prompt:
            prompt = request.prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        cmd.append(prompt)
        return cmd

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
        """Copy only the auth material needed for isolated Codex subprocesses."""

        source_dir = Path.home() / ".codex"
        for filename in ("auth.json", ".credentials.json"):
            source = source_dir / filename
            target = codex_dir / filename
            if source.exists():
                with contextlib.suppress(OSError):
                    shutil.copy2(source, target)

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
        self._copy_isolated_runtime_state(codex_dir)
        return str(cli_home)

    def _request_timeout(self, request: GenerateRequest) -> float:
        """Return the effective timeout for this request."""

        return (
            float(request.timeout_seconds) if request.timeout_seconds is not None else self._timeout
        )

    def _stall_after_turn_started_seconds(self, request_timeout: float) -> float:
        """Return the max silent interval after `turn.started` before fast-failing."""

        return min(45.0, max(15.0, request_timeout * 0.33))

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
                        _prepare_schema_for_codex(
                            dict(request.structured_output.json_schema)
                        ),
                        schema_file,
                    )
            cmd = self._build_command(
                request,
                model=model,
                output_last_message_path=output_path,
                output_schema_path=schema_path,
            )
            env = self._get_subprocess_env()
            env["HOME"] = cli_home
            # Safe: uses argument list, no shell; minimal environment
            proc = await asyncio.create_subprocess_exec(
                cmd[0],
                *cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )

            timeout = self._request_timeout(request)
            if (
                not isinstance(proc.stdout, asyncio.StreamReader)
                or not isinstance(proc.stderr, asyncio.StreamReader)
                or not hasattr(proc, "wait")
            ):
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    await _terminate_live_process(proc)
                    raise RuntimeError(
                        f"Codex CLI timed out after {timeout}s. "
                        "Consider increasing timeout or simplifying the task."
                    )

                stdout_text = stdout.decode("utf-8", errors="replace")
                stderr_text = stderr.decode("utf-8", errors="replace")
                if proc.returncode != 0:
                    error_text = stderr_text or _extract_error_message(stdout_text)
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
            completion_started_at: float | None = None
            terminated_after_output = False
            output = ""

            while True:
                with contextlib.suppress(OSError):
                    if output_path:
                        file_output = Path(output_path).read_text(encoding="utf-8")
                        if file_output:
                            output = file_output

                if not output and state.agent_message:
                    output = state.agent_message

                now = loop.time()
                if output and completion_started_at is None:
                    completion_started_at = now

                if proc.returncode is not None:
                    break

                if completion_started_at is not None and (
                    state.saw_turn_completed or now - completion_started_at >= 1.0
                ):
                    terminated_after_output = True
                    await _terminate_live_process(proc)
                    break

                if (
                    state.turn_started_at is not None
                    and not output
                    and now - state.turn_started_at
                    >= self._stall_after_turn_started_seconds(timeout)
                ):
                    await _terminate_live_process(proc)
                    await _drain_reader_tasks(stdout_task, stderr_task)
                    raise RuntimeError(
                        "Codex CLI stalled after turn.started without producing any answer. "
                        "This looks like a nested `codex exec` failure rather than a normal "
                        "slow response."
                    )

                if now >= deadline:
                    await _terminate_live_process(proc)
                    await _drain_reader_tasks(stdout_task, stderr_task)
                    raise RuntimeError(
                        f"Codex CLI timed out after {timeout}s. "
                        "Consider increasing timeout or simplifying the task."
                    )

                await asyncio.sleep(0.05)

            await _drain_reader_tasks(stdout_task, stderr_task)

            stdout_text = "".join(state.stdout_parts)
            stderr_text = "".join(state.stderr_parts)
            if proc.returncode != 0 and not terminated_after_output:
                error_text = stderr_text or _extract_error_message(stdout_text)
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
            for temp_path in (output_path, schema_path):
                if temp_path:
                    with contextlib.suppress(OSError):
                        os.unlink(temp_path)
            if cli_home:
                shutil.rmtree(cli_home, ignore_errors=True)

    async def supports(self, capability: str) -> bool:
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

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
