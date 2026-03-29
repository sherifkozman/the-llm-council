"""
Gemini CLI provider adapter.

Wraps the ``gemini`` CLI for non-interactive generation.
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
import shutil
import tempfile
import warnings
from collections.abc import AsyncIterator
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
from llm_council.providers.cli._subprocess import terminate_process_tree

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3-flash-preview"
# SECURITY: Least-privilege defaults - require approval for actions
# Older adapter configs used "confirm"/"auto"; normalize them to the
# current CLI vocabulary to stay backward compatible.
DEFAULT_APPROVAL_MODE = "default"
_APPROVAL_MODE_ALIASES = {
    "confirm": "default",
    "auto": "yolo",
}
_VALID_APPROVAL_MODES = {"default", "auto_edit", "yolo", "plan"}

# Unsafe modes that require explicit opt-in
_UNSAFE_MODES = {"yolo"}
_UNSAFE_WARNING = (
    "WARNING: Gemini CLI is running with permissive approval mode that allows "
    "local file/env access without confirmation. This is unsafe with untrusted inputs."
)

# Minimal environment allowlist for subprocess
_COMMON_ENV_ALLOWLIST = {
    "PATH",
    "HOME",
    "TERM",
    "LANG",
    "LC_ALL",
    "TMPDIR",
}
_GEMINI_API_ENV_ALLOWLIST = {"GEMINI_API_KEY"}
_VERTEX_ENV_ALLOWLIST = {
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_PROJECT_ID",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_GENAI_USE_GCA",
    "GEMINI_CLI_USE_COMPUTE_ADC",
}
_OAUTH_ENV_ALLOWLIST = {
    "GOOGLE_GENAI_USE_GCA",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GEMINI_CLI_USE_COMPUTE_ADC",
}
_KNOWN_AUTH_ENV_ALLOWLIST = _GEMINI_API_ENV_ALLOWLIST | _VERTEX_ENV_ALLOWLIST
_AUTH_TYPE_VERTEX = "vertex-ai"
_AUTH_TYPE_GEMINI_API_KEY = "gemini-api-key"
_AUTH_TYPE_LOGIN_WITH_GOOGLE = "oauth-personal"
_AUTH_TYPE_COMPUTE_ADC = "compute-default-credentials"


def _extract_text_payload(payload: Any) -> str:
    """Extract text from Gemini CLI JSON payloads."""

    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str):
            return text
        for key in ("response", "content", "message", "result"):
            nested = _extract_text_payload(payload.get(key))
            if nested:
                return nested
        return ""
    if isinstance(payload, list):
        parts = [part for item in payload if (part := _extract_text_payload(item).strip())]
        return "\n".join(parts)
    return ""


def _normalize_usage(raw_stats: Any) -> dict[str, int] | None:
    """Normalize Gemini CLI stats payloads to council's usage shape."""

    if not isinstance(raw_stats, dict):
        return None
    prompt_tokens = int(
        raw_stats.get("inputTokenCount", raw_stats.get("input_tokens", 0)) or 0
    )
    completion_tokens = int(
        raw_stats.get("outputTokenCount", raw_stats.get("output_tokens", 0)) or 0
    )
    total_tokens = int(
        raw_stats.get("totalTokenCount", raw_stats.get("total_tokens", 0))
        or (prompt_tokens + completion_tokens)
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _format_gemini_error(error_payload: Any) -> str:
    """Format Gemini CLI JSON errors into a stable string."""

    if isinstance(error_payload, str):
        return error_payload
    if isinstance(error_payload, dict):
        if error_payload.get("message"):
            return str(error_payload["message"])
        return json.dumps(error_payload)
    return str(error_payload)


def _extract_json_payload(stdout_text: str) -> Any:
    """Parse the last JSON payload from Gemini stdout, tolerating warning prelude."""

    stripped = stdout_text.strip()
    if not stripped:
        raise json.JSONDecodeError("Empty Gemini CLI stdout", stdout_text, 0)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    lines = stripped.splitlines()
    for index in range(len(lines)):
        candidate = "\n".join(lines[index:]).strip()
        if not candidate or candidate[0] not in "[{":
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No JSON payload found in Gemini CLI stdout", stdout_text, 0)


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
        self._approval_mode = self._normalize_approval_mode(approval_mode)
        self._timeout = timeout

    def _normalize_approval_mode(self, approval_mode: str | None) -> str:
        """Map legacy aliases to current Gemini CLI approval modes."""

        mode = (approval_mode or DEFAULT_APPROVAL_MODE).strip().lower()
        mode = _APPROVAL_MODE_ALIASES.get(mode, mode)
        if mode not in _VALID_APPROVAL_MODES:
            valid = ", ".join(sorted(_VALID_APPROVAL_MODES))
            raise ValueError(
                f"Unsupported Gemini approval mode '{approval_mode}'. Use one of: {valid}"
            )
        return mode

    def _check_unsafe_mode(self) -> None:
        """Emit warning if using unsafe permissive approval mode."""
        if self._approval_mode in _UNSAFE_MODES:
            warnings.warn(_UNSAFE_WARNING, UserWarning, stacklevel=3)

    def _selected_auth_type(self, auth_settings: dict[str, Any] | None) -> str | None:
        """Return the configured Gemini auth type, if present."""

        if not isinstance(auth_settings, dict):
            return None
        selected_type = auth_settings.get("selectedType")
        if isinstance(selected_type, str) and selected_type.strip():
            return selected_type.strip()
        return None

    def _get_minimal_env(self, auth_settings: dict[str, Any] | None) -> dict[str, str]:
        """Get a minimal environment that preserves the configured Gemini auth mode."""

        env = {k: v for k, v in os.environ.items() if k in _COMMON_ENV_ALLOWLIST}
        selected_type = self._selected_auth_type(auth_settings)

        if selected_type == _AUTH_TYPE_VERTEX:
            for key in _VERTEX_ENV_ALLOWLIST:
                value = os.environ.get(key)
                if value:
                    env[key] = value
            has_vertex_project = bool(env.get("GOOGLE_CLOUD_PROJECT") or env.get("GOOGLE_CLOUD_PROJECT_ID"))
            has_vertex_location = bool(env.get("GOOGLE_CLOUD_LOCATION"))
            # Gemini CLI's Vertex path will prefer GOOGLE_API_KEY over ADC/project
            # config. Drop the API key when full Vertex project config is available
            # so we preserve the user's selected auth mode instead of silently
            # switching to Vertex express mode.
            if has_vertex_project and has_vertex_location:
                env.pop("GOOGLE_API_KEY", None)
            return env

        if selected_type == _AUTH_TYPE_GEMINI_API_KEY:
            for key in _GEMINI_API_ENV_ALLOWLIST:
                value = os.environ.get(key)
                if value:
                    env[key] = value
            return env

        if selected_type in {_AUTH_TYPE_LOGIN_WITH_GOOGLE, _AUTH_TYPE_COMPUTE_ADC}:
            for key in _OAUTH_ENV_ALLOWLIST:
                value = os.environ.get(key)
                if value:
                    env[key] = value
            return env

        # Fallback for unset/legacy auth config: preserve known auth env so the
        # subprocess behaves like the current shell.
        for key in _KNOWN_AUTH_ENV_ALLOWLIST:
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def _load_existing_auth_settings(self) -> dict[str, Any] | None:
        """Load the user's existing Gemini auth settings so subprocess isolation preserves auth."""

        settings_path = Path.home() / ".gemini" / "settings.json"
        with contextlib.suppress(OSError, json.JSONDecodeError):
            payload = json.loads(settings_path.read_text(encoding="utf-8"))
            security = payload.get("security")
            if isinstance(security, dict):
                auth = security.get("auth")
                if isinstance(auth, dict):
                    return dict(auth)
        return None

    def _copy_isolated_runtime_state(self, gemini_dir: Path) -> None:
        """Copy the minimal Gemini runtime state needed for isolated subprocesses."""

        source_dir = Path.home() / ".gemini"
        for filename in ("projects.json", "google_accounts.json"):
            source = source_dir / filename
            target = gemini_dir / filename
            if source.exists():
                with contextlib.suppress(OSError):
                    shutil.copy2(source, target)

    def _create_isolated_cli_home(self) -> tuple[str, dict[str, Any] | None]:
        """Create a temporary Gemini CLI home with extensions disabled for this subprocess."""

        cli_home = tempfile.mkdtemp(prefix="llm-council-gemini-home-")
        gemini_dir = Path(cli_home) / ".gemini"
        gemini_dir.mkdir(parents=True, exist_ok=True)
        settings_payload: dict[str, Any] = {
            "admin": {
                "extensions": {"enabled": False},
                "mcp": {"enabled": False},
            }
        }
        existing_auth = self._load_existing_auth_settings()
        if existing_auth:
            settings_payload["security"] = {"auth": existing_auth}
        (gemini_dir / "settings.json").write_text(
            json.dumps(settings_payload),
            encoding="utf-8",
        )
        self._copy_isolated_runtime_state(gemini_dir)
        return cli_home, existing_auth

    def _request_timeout(self, request: GenerateRequest) -> float:
        """Return the effective timeout for this request."""

        return (
            float(request.timeout_seconds) if request.timeout_seconds is not None else self._timeout
        )

    def _build_command(self, request: GenerateRequest) -> list[str]:
        """Build CLI command as argument list (safe from injection)."""
        if not self._cli_path:
            raise RuntimeError("Gemini CLI not found.")

        prompt = ""
        if request.messages:
            parts = [m.content for m in request.messages if m.role == "user"]
            prompt = "\n\n".join(parts)
        elif request.prompt:
            prompt = request.prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' required")

        cmd = [self._cli_path, "-p", prompt]
        cmd.extend(["--approval-mode", self._approval_mode])
        cmd.extend(["-m", request.model or self._default_model])
        cmd.extend(["--output-format", "json"])
        return cmd

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate using safe subprocess with argument list."""
        if request.stream:
            raise NotImplementedError("Streaming not supported for CLI")

        self._check_unsafe_mode()
        cmd = self._build_command(request)
        cli_home, auth_settings = self._create_isolated_cli_home()
        env = self._get_minimal_env(auth_settings)
        env["GEMINI_CLI_HOME"] = cli_home

        # Use minimal environment to reduce secret exposure
        try:
            proc = await asyncio.create_subprocess_exec(
                cmd[0],
                *cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )

            timeout = self._request_timeout(request)
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                await terminate_process_tree(proc)
                raise RuntimeError(
                    f"Gemini CLI timed out after {timeout}s. "
                    "Consider increasing timeout or simplifying the task."
                )

            stdout_text = stdout.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="replace")
                error_type = classify_error(stderr_text, proc.returncode or 0)

                # Provide actionable error messages based on error type
                if error_type == ErrorType.BILLING:
                    billing_url = get_billing_help_url("gemini-cli")
                    raise RuntimeError(
                        f"BILLING ERROR: Google API credits exhausted. "
                        f"Check billing at {billing_url}\n"
                        f"Details: {stderr_text[:200]}"
                    )
                elif error_type == ErrorType.AUTH:
                    raise RuntimeError(
                        "AUTH ERROR: Gemini CLI authentication failed. "
                        "Check the auth method configured in ~/.gemini/settings.json "
                        "and the matching environment variables.\n"
                        f"Details: {stderr_text[:200]}"
                    )
                elif error_type == ErrorType.RATE_LIMIT:
                    raise RuntimeError(
                        f"RATE LIMIT: Too many requests. Wait and retry.\nDetails: {stderr_text[:200]}"
                    )
                else:
                    raise RuntimeError(f"CLI failed ({error_type.value}): {stderr_text}")

            output = ""
            usage = None
            try:
                payload = _extract_json_payload(stdout_text)
                if isinstance(payload, dict) and payload.get("error"):
                    raise RuntimeError(f"Gemini CLI error: {_format_gemini_error(payload['error'])}")
                output = _extract_text_payload(payload)
                if isinstance(payload, dict):
                    usage = _normalize_usage(payload.get("stats"))
            except json.JSONDecodeError:
                output = stdout_text
                logger.warning("Gemini CLI returned non-JSON output, using raw text")

            # Warn if output is empty on success - don't use stderr as output
            # to avoid leaking CLI logs into conversation flow
            if not output.strip():
                logger.warning("Gemini CLI returned success but empty output")
                output = ""

            return GenerateResponse(
                text=output,
                content=output,
                usage=usage,
                raw={"stdout": stdout_text},
            )
        finally:
            shutil.rmtree(cli_home, ignore_errors=True)

    async def supports(self, capability: str) -> bool:
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

    async def doctor(self) -> DoctorResult:
        if not self._cli_path:
            return DoctorResult(ok=False, message="CLI not found")
        auth_settings = self._load_existing_auth_settings()
        selected_type = self._selected_auth_type(auth_settings)

        if selected_type == _AUTH_TYPE_VERTEX:
            has_vertex_project = bool(
                os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
            )
            has_vertex_location = bool(os.environ.get("GOOGLE_CLOUD_LOCATION"))
            has_google_api_key = bool(os.environ.get("GOOGLE_API_KEY"))
            if (has_vertex_project and has_vertex_location) or has_google_api_key:
                return DoctorResult(ok=True, message="CLI available (auth: vertex-ai)")
            return DoctorResult(
                ok=False,
                message=(
                    "Gemini is configured for vertex-ai but is missing either "
                    "GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION or GOOGLE_API_KEY"
                ),
            )

        if selected_type == _AUTH_TYPE_GEMINI_API_KEY:
            if os.environ.get("GEMINI_API_KEY"):
                return DoctorResult(ok=True, message="CLI available (auth: gemini-api-key)")
            return DoctorResult(
                ok=False,
                message="Gemini is configured for gemini-api-key but GEMINI_API_KEY is not set",
            )

        if selected_type in {_AUTH_TYPE_LOGIN_WITH_GOOGLE, _AUTH_TYPE_COMPUTE_ADC}:
            return DoctorResult(ok=True, message=f"CLI available (auth: {selected_type})")

        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return DoctorResult(ok=True, message="CLI available")
        return DoctorResult(
            ok=False,
            message=(
                "No Gemini auth detected. Configure Gemini CLI auth in ~/.gemini/settings.json "
                "or set GEMINI_API_KEY / GOOGLE_API_KEY"
            ),
        )


def _register() -> None:
    from llm_council.providers.registry import get_registry

    with contextlib.suppress(ValueError):
        get_registry().register_provider("gemini-cli", GeminiCLIProvider)


_register()
