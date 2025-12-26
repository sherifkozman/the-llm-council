"""
Gemini CLI provider adapter.

DEPRECATED: Use the 'google' provider with direct API access instead.
This CLI adapter will be removed in v1.0.

SECURITY NOTE: Uses asyncio.create_subprocess_exec with argument lists,
which is safe from shell injection (equivalent to execFile in Node.js).
"""

from __future__ import annotations

import asyncio
import logging
import os
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

DEFAULT_MODEL = "gemini-2.0-flash-exp"
# SECURITY: Least-privilege defaults - require approval for actions
# Override via default_flags param for agentic mode
DEFAULT_APPROVAL_MODE = "confirm"

# Unsafe modes that require explicit opt-in
_UNSAFE_MODES = {"yolo", "auto"}
_UNSAFE_WARNING = (
    "WARNING: Gemini CLI is running with permissive approval mode that allows "
    "local file/env access without confirmation. This is unsafe with untrusted inputs."
)

# Minimal environment allowlist for subprocess
_ENV_ALLOWLIST = {
    "PATH",
    "HOME",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "TERM",
    "LANG",
    "LC_ALL",
}


class GeminiCLIProvider(ProviderAdapter):
    """Gemini CLI provider adapter."""

    name: ClassVar[str] = "gemini-cli"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=False,
        tool_use=False,
        structured_output=False,
        multimodal=False,
        max_tokens=8192,
    )

    def __init__(
        self,
        cli_path: str | None = None,
        default_model: str | None = None,
        approval_mode: str | None = None,
        timeout: int = 120,
    ) -> None:
        self._cli_path = cli_path or shutil.which("gemini")
        self._default_model = default_model or DEFAULT_MODEL
        self._approval_mode = approval_mode or DEFAULT_APPROVAL_MODE
        self._timeout = timeout

    def _check_unsafe_mode(self) -> None:
        """Emit warning if using unsafe permissive approval mode."""
        if self._approval_mode in _UNSAFE_MODES:
            warnings.warn(_UNSAFE_WARNING, UserWarning, stacklevel=3)

    def _get_minimal_env(self) -> dict[str, str]:
        """Get minimal environment with only allowlisted variables."""
        return {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}

    def _build_command(self, request: GenerateRequest) -> list[str]:
        """Build CLI command as argument list (safe from injection)."""
        if not self._cli_path:
            raise RuntimeError("Gemini CLI not found.")

        cmd = [self._cli_path, "--approval-mode", self._approval_mode]
        cmd.extend(["-m", request.model or self._default_model])

        prompt = ""
        if request.messages:
            parts = [m.content for m in request.messages if m.role == "user"]
            prompt = "\n\n".join(parts)
        elif request.prompt:
            prompt = request.prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' required")

        cmd.append(prompt)
        return cmd

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate using safe subprocess with argument list."""
        if request.stream:
            raise NotImplementedError("Streaming not supported for CLI")

        self._check_unsafe_mode()
        cmd = self._build_command(request)

        # Use minimal environment to reduce secret exposure
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
                f"Gemini CLI timed out after {self._timeout}s. "
                "Consider increasing timeout or simplifying the task."
            )

        output = stdout.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")
            error_type = classify_error(stderr_text, proc.returncode or 0)

            # Provide actionable error messages based on error type
            if error_type == ErrorType.BILLING:
                billing_url = get_billing_help_url("gemini")
                raise RuntimeError(
                    f"BILLING ERROR: Google API credits exhausted. "
                    f"Check billing at {billing_url}\n"
                    f"Details: {stderr_text[:200]}"
                )
            elif error_type == ErrorType.AUTH:
                raise RuntimeError(
                    f"AUTH ERROR: Invalid or missing API key. "
                    f"Check GEMINI_API_KEY environment variable.\n"
                    f"Details: {stderr_text[:200]}"
                )
            elif error_type == ErrorType.RATE_LIMIT:
                raise RuntimeError(
                    f"RATE LIMIT: Too many requests. Wait and retry.\nDetails: {stderr_text[:200]}"
                )
            else:
                raise RuntimeError(f"CLI failed ({error_type.value}): {stderr_text}")

        # Warn if output is empty on success - don't use stderr as output
        # to avoid leaking CLI logs into conversation flow
        if not output.strip():
            logger.warning("Gemini CLI returned success but empty output")
            output = ""

        return GenerateResponse(text=output, content=output)

    async def supports(self, capability: str) -> bool:
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

    async def doctor(self) -> DoctorResult:
        if not self._cli_path:
            return DoctorResult(ok=False, message="CLI not found")
        if not os.environ.get("GEMINI_API_KEY"):
            return DoctorResult(ok=False, message="GEMINI_API_KEY not set")
        return DoctorResult(ok=True, message="CLI available")


def _register() -> None:
    from llm_council.providers.registry import get_registry

    with contextlib.suppress(ValueError):
        get_registry().register_provider("gemini-cli", GeminiCLIProvider)


_register()
