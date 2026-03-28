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
import logging
import os
import shlex
import shutil
import warnings
from collections.abc import AsyncIterator
from typing import ClassVar

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
from llm_council.providers.cli._subprocess import terminate_process_tree

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

# Minimal environment allowlist for subprocess
_ENV_ALLOWLIST = {
    "PATH",
    "HOME",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "TERM",
    "LANG",
    "LC_ALL",
}


class CodexCLIProvider(ProviderAdapter):
    """Codex CLI provider adapter."""

    name: ClassVar[str] = "codex"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=False,
        tool_use=False,
        structured_output=False,
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

    def _build_command(self, request: GenerateRequest, *, model: str) -> list[str]:
        """Build the CLI command as argument list (safe from injection)."""
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found.")

        cmd = [self._cli_path, "exec"]
        cmd.extend(shlex.split(self._default_flags))
        cmd.extend(["-m", model])

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

    def _get_minimal_env(self) -> dict[str, str]:
        """Get minimal environment with only allowlisted variables."""
        return {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}

    def _request_timeout(self, request: GenerateRequest) -> float:
        """Return the effective timeout for this request."""

        return (
            float(request.timeout_seconds) if request.timeout_seconds is not None else self._timeout
        )

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
                env=self._get_minimal_env(),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        except Exception:  # pragma: no cover - defensive path
            return None

        output = stdout.decode("utf-8", errors="replace").strip()
        error_output = stderr.decode("utf-8", errors="replace").strip()
        status_text = output or error_output or f"CLI returned exit code {proc.returncode}"
        if proc.returncode == 0:
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
        cmd = self._build_command(request, model=model)

        # Safe: uses argument list, no shell; minimal environment
        proc = await asyncio.create_subprocess_exec(
            cmd[0],
            *cmd[1:],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._get_minimal_env(),
            start_new_session=True,
        )

        timeout = self._request_timeout(request)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await terminate_process_tree(proc)
            raise RuntimeError(
                f"Codex CLI timed out after {timeout}s. "
                "Consider increasing timeout or simplifying the task."
            )

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")
            error_details = self._error_details(stderr_text)
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

        output = stdout.decode("utf-8", errors="replace")

        # Warn if output is empty on success
        if not output.strip():
            logger.warning("Codex CLI returned success but empty output")

        return GenerateResponse(text=output, content=output)

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
        return DoctorResult(ok=True, message=f"CLI available ({status_text})")


def _register() -> None:
    from llm_council.providers.registry import get_registry

    with contextlib.suppress(ValueError):
        get_registry().register_provider("codex", CodexCLIProvider)


_register()
