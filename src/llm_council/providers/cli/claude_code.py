"""
Claude Code CLI provider adapter.

Wraps the ``claude`` CLI (Claude Code) for non-interactive generation.
Uses --bare mode to skip hooks/LSP/plugins and prevent recursion
when council is invoked from within Claude Code.

SECURITY NOTE: Uses asyncio.create_subprocess_exec with argument lists,
which is safe from shell injection (equivalent to execFile in Node.js).
No shell is spawned; arguments are passed as a list directly to the binary.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator
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
from llm_council.providers.cli._subprocess import terminate_process_tree

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sonnet"

# Minimal environment allowlist for subprocess
_ENV_ALLOWLIST = {
    "PATH",
    "HOME",
    "ANTHROPIC_API_KEY",
    "TERM",
    "LANG",
    "LC_ALL",
}


def _extract_text_payload(payload: Any) -> str:
    """Extract text from Claude Code JSON envelopes."""

    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str):
            return text
        for key in ("result", "response", "message", "content"):
            nested = _extract_text_payload(payload.get(key))
            if nested:
                return nested
        return ""
    if isinstance(payload, list):
        parts: list[str] = []
        seen: set[str] = set()
        for item in payload:
            text = _extract_text_payload(item).strip()
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
        return "\n".join(parts)
    return ""


def _normalize_usage(raw_usage: Any) -> dict[str, int] | None:
    """Normalize Claude Code usage payloads to council's usage shape."""

    if not isinstance(raw_usage, dict):
        return None
    input_tokens = int(raw_usage.get("input_tokens", 0) or 0) + int(
        raw_usage.get("cache_read_input_tokens", 0) or 0
    )
    output_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _extract_usage_payload(payload: Any) -> dict[str, int] | None:
    """Find a Claude Code usage block in nested JSON payloads."""

    if isinstance(payload, dict):
        usage = _normalize_usage(payload.get("usage"))
        if usage is not None:
            return usage
        for key in ("result", "response", "message", "content"):
            nested = _extract_usage_payload(payload.get(key))
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in reversed(payload):
            nested = _extract_usage_payload(item)
            if nested is not None:
                return nested
    return None


class ClaudeCodeCLIProvider(ProviderAdapter):
    """Claude Code CLI provider adapter.

    Invokes the ``claude`` CLI in non-interactive mode (-p) with JSON output.
    Uses --bare to skip hooks, LSP, and plugin loading for speed and to
    prevent infinite recursion when council runs inside Claude Code.

    Security: All subprocess calls use asyncio.create_subprocess_exec with
    argument lists (no shell). Environment is restricted to an allowlist.
    """

    name: ClassVar[str] = "claude"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=False,
        tool_use=False,
        structured_output=False,
        multimodal=False,
        max_tokens=32000,
    )

    def __init__(
        self,
        cli_path: str | None = None,
        default_model: str | None = None,
        timeout: int = 120,
    ) -> None:
        self._cli_path = cli_path or shutil.which("claude")
        self._default_model = default_model or DEFAULT_MODEL
        self._timeout = timeout

    def _build_command(self, request: GenerateRequest) -> list[str]:
        """Build the CLI command as an argument list (no shell, safe from injection)."""
        if not self._cli_path:
            raise RuntimeError(
                "Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code"
            )

        cmd = [
            self._cli_path,
            "-p",  # non-interactive print mode
            "--output-format",
            "json",
            "--bare",  # skip hooks/LSP/plugins — prevents recursion
            "--model",
            request.model or self._default_model,
        ]

        # Build prompt from messages or prompt field
        prompt = ""
        if request.messages:
            system_parts = [m.content for m in request.messages if m.role == "system"]
            user_parts = [m.content for m in request.messages if m.role == "user"]
            if system_parts:
                cmd.extend(["--system-prompt", "\n\n".join(str(p) for p in system_parts)])
            prompt = "\n\n".join(str(p) for p in user_parts)
        elif request.prompt:
            prompt = request.prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        cmd.append(prompt)
        return cmd

    def _get_minimal_env(self) -> dict[str, str]:
        """Get minimal environment with only allowlisted variables."""
        return {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}

    def _request_timeout(self, request: GenerateRequest) -> float:
        """Return the effective timeout for this request."""

        return (
            float(request.timeout_seconds) if request.timeout_seconds is not None else self._timeout
        )

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate using subprocess with argument list (no shell)."""
        if request.stream:
            raise NotImplementedError("Streaming not supported for CLI providers")

        cmd = self._build_command(request)

        # Safe: uses argument list via create_subprocess_exec, no shell spawned
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
                f"Claude Code CLI timed out after {timeout}s. "
                "Consider increasing timeout or simplifying the task."
            )

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            error_type = classify_error(stderr_text, proc.returncode or 0)
            if error_type == ErrorType.BILLING:
                billing_url = get_billing_help_url("claude-code")
                raise RuntimeError(
                    f"BILLING ERROR: Anthropic credits exhausted. "
                    f"Check billing at {billing_url}\n"
                    f"Details: {stderr_text[:200]}"
                )
            elif error_type == ErrorType.AUTH:
                raise RuntimeError(
                    "AUTH ERROR: Invalid or missing credentials. "
                    "Run 'claude login' or check ANTHROPIC_API_KEY.\n"
                    f"Details: {stderr_text[:200]}"
                )
            elif error_type == ErrorType.RATE_LIMIT:
                raise RuntimeError(
                    f"RATE LIMIT: Too many requests. Wait and retry.\nDetails: {stderr_text[:200]}"
                )
            else:
                raise RuntimeError(f"CLI failed ({error_type.value}): {stderr_text}")

        # Parse JSON output from --output-format json.
        output = ""
        usage = None
        try:
            data = json.loads(stdout_text)
            if isinstance(data, dict) and data.get("is_error"):
                error_text = _extract_text_payload(data.get("result")) or "Unknown error"
                raise RuntimeError(f"Claude Code error: {error_text}")
            output = _extract_text_payload(data)
            usage = _extract_usage_payload(data)
        except json.JSONDecodeError:
            # Fall back to raw text if JSON parsing fails
            output = stdout_text
            logger.warning("Claude Code CLI returned non-JSON output, using raw text")

        if not output.strip():
            logger.warning("Claude Code CLI returned success but empty output")

        return GenerateResponse(
            text=output,
            content=output,
            usage=usage,
            raw={"stdout": stdout_text, "stderr": stderr_text},
        )

    async def supports(self, capability: str) -> bool:
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

    async def doctor(self) -> DoctorResult:
        if not self._cli_path:
            return DoctorResult(
                ok=False,
                message="Claude Code CLI not found"
                " (install: npm install -g @anthropic-ai/claude-code)",
            )

        # Verify it responds — uses argument list, no shell
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            version = stdout.decode("utf-8", errors="replace").strip()
            if proc.returncode == 0 and version:
                return DoctorResult(ok=True, message=f"Claude Code v{version}")
            return DoctorResult(ok=False, message=f"CLI returned exit code {proc.returncode}")
        except asyncio.TimeoutError:
            return DoctorResult(ok=False, message="CLI version check timed out")
        except Exception as e:
            return DoctorResult(ok=False, message=f"CLI check failed: {e}")


def _register() -> None:
    from llm_council.providers.registry import get_registry

    with contextlib.suppress(ValueError):
        get_registry().register_provider("claude", ClaudeCodeCLIProvider)


_register()
