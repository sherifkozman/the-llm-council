"""
Codex CLI provider adapter.

DEPRECATED: Use the 'openai' provider with direct API access instead.
This CLI adapter will be removed in v1.0.

Invokes the OpenAI Codex CLI tool via subprocess.
This is an experimental provider - prefer API-based providers for reliability.

SECURITY NOTE: Uses asyncio.create_subprocess_exec with argument lists,
which is safe from shell injection (equivalent to execFile in Node.js).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import warnings
from collections.abc import AsyncIterator
from typing import ClassVar

logger = logging.getLogger(__name__)

import contextlib

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

DEFAULT_MODEL = "gpt-5.2-codex"
# SECURITY: Least-privilege defaults - read-only sandbox, no auto-approve
# Override via CODEX_CLI_FLAGS env var or default_flags param for agentic mode
DEFAULT_FLAGS = "--sandbox read-only --skip-git-repo-check"

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

    name: ClassVar[str] = "codex-cli"
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

    def _build_command(self, request: GenerateRequest) -> list[str]:
        """Build the CLI command as argument list (safe from injection)."""
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found.")

        cmd = [self._cli_path, "exec"]
        cmd.extend(shlex.split(self._default_flags))
        cmd.extend(["-m", request.model or self._default_model])

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

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate using safe subprocess with argument list."""
        if request.stream:
            raise NotImplementedError("Streaming not supported for CLI")

        self._check_unsafe_flags()
        cmd = self._build_command(request)

        # Safe: uses argument list, no shell; minimal environment
        proc = await asyncio.create_subprocess_exec(
            cmd[0],
            *cmd[1:],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._get_minimal_env(),
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise RuntimeError(
                f"Codex CLI timed out after {self._timeout}s. "
                "Consider increasing timeout or simplifying the task."
            )

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")
            error_type = classify_error(stderr_text, proc.returncode or 0)

            # Provide actionable error messages based on error type
            if error_type == ErrorType.BILLING:
                billing_url = get_billing_help_url("codex")
                raise RuntimeError(
                    f"BILLING ERROR: OpenAI credits exhausted. "
                    f"Add credits at {billing_url}\n"
                    f"Details: {stderr_text[:200]}"
                )
            elif error_type == ErrorType.AUTH:
                raise RuntimeError(
                    f"AUTH ERROR: Invalid or missing API key. "
                    f"Check OPENAI_API_KEY environment variable.\n"
                    f"Details: {stderr_text[:200]}"
                )
            elif error_type == ErrorType.RATE_LIMIT:
                raise RuntimeError(
                    f"RATE LIMIT: Too many requests. Wait and retry.\nDetails: {stderr_text[:200]}"
                )
            else:
                raise RuntimeError(f"CLI failed ({error_type.value}): {stderr_text}")

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
        return DoctorResult(ok=True, message="CLI available")


def _register() -> None:
    from llm_council.providers.registry import get_registry

    with contextlib.suppress(ValueError):
        get_registry().register_provider("codex-cli", CodexCLIProvider)


_register()
