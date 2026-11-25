"""Opt-in Antigravity CLI adapter with isolated ADC and bounded headless requests."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar, cast

from jsonschema import Draft202012Validator

from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
    cache_read_reconciles,
    token_counts_present,
    token_usage_reconciles,
)
from llm_council.providers.cli._subprocess import terminate_process_tree
from llm_council.providers.registry import ANTIGRAVITY_PROVIDER

_DEFAULT_MODEL = "gemini-4-argon"
_DENIED_TOOL_RULES = (
    "command(*)",
    "read_file(*)",
    "write_file(*)",
    "read_url(*)",
    "execute_url(*)",
    "mcp(*)",
)
_ISOLATED_SETTINGS = json.dumps({"permissions": {"deny": _DENIED_TOOL_RULES}})
_TERMINAL_STATES = {"SUCCESS", "ERROR", "CANCELED", "INTERRUPTED", "INVALID", "WAITING", "RUNNING"}


def _adc_path() -> Path | None:
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    appdata = os.environ.get("APPDATA") if os.name == "nt" else None
    config = Path(appdata) if appdata else Path.home() / ".config"
    path = (
        Path(explicit).expanduser()
        if explicit
        else config / "gcloud/application_default_credentials.json"
    )
    return path.resolve() if path.is_file() else None


def _read_headless_result(stdout: bytes, returncode: int | None) -> dict[str, object]:
    results = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (UnicodeError, json.JSONDecodeError):
            raise RuntimeError("Antigravity CLI emitted invalid NDJSON") from None
        if isinstance(event, dict) and event.get("event") == "result":
            result = event.get("result")
            if isinstance(result, dict):
                results.append(result)
    if len(results) != 1:
        raise RuntimeError(f"Antigravity CLI returned no unique result (exit {returncode})")
    result = results[0]
    status = result.get("status")
    if returncode != 0 or status != "SUCCESS" or result.get("error"):
        safe_status = (
            status if isinstance(status, str) and status in _TERMINAL_STATES else "UNKNOWN"
        )
        raise RuntimeError(f"Antigravity CLI generation failed ({safe_status}, exit {returncode})")
    return result


def _normalized_usage(raw: object) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    counts = tuple(
        raw.get(k) for k in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens")
    )
    if not token_counts_present(*counts):
        return None
    prompt, output, thinking, total = cast("tuple[int, int, int, int]", counts)
    # agy's output_tokens includes thinking. Council stores it separately.
    if not token_usage_reconciles(prompt, output - thinking, total, thinking):
        return None
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": output - thinking,
        "reasoning_tokens": thinking,
        "total_tokens": total,
    }
    if "cache_read_tokens" in raw:
        cached = raw["cache_read_tokens"]
        if (
            not isinstance(cached, int)
            or not token_counts_present(cached)
            or not cache_read_reconciles(prompt, cached)
        ):
            return None
        usage["cache_read_tokens"] = cached
    return usage


class AntigravityCLIProvider(ProviderAdapter):
    """Wrap an already installed agy. Council only routes supplementary drafts here."""

    name: ClassVar[str] = ANTIGRAVITY_PROVIDER
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(structured_output=True)

    def __init__(
        self, cli_path: str | None = None, default_model: str | None = None, timeout: int = 120
    ) -> None:
        self._cli_path = cli_path or shutil.which("agy")
        self._default_model = default_model or _DEFAULT_MODEL
        self._timeout = timeout

    async def supports(self, capability: str) -> bool:
        return self.supports_capability(capability)

    async def doctor(self) -> DoctorResult:
        if not self._cli_path:
            return DoctorResult(ok=False, message="Antigravity CLI (agy) not found")
        if _adc_path() is None:
            return DoctorResult(ok=False, message="No Google ADC file for Antigravity CLI")
        return DoctorResult(ok=True, message="Antigravity CLI and Google ADC file available")

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if not self._cli_path:
            raise RuntimeError("Antigravity CLI (agy) not found")
        adc = _adc_path()
        if adc is None:
            raise RuntimeError("No Google ADC file for Antigravity CLI")
        if request.stream or request.tools:
            raise NotImplementedError("Antigravity adapter supports one bounded text request")

        prompt = (
            "\n\n".join(f"{m.role}:\n{m.content}" for m in request.messages)
            if request.messages
            else request.prompt
        )
        event = json.dumps({"event": "user", "message": {"content": prompt}}) + "\n"
        timeout = request.timeout_seconds or self._timeout
        model = request.model or self._default_model
        command = [
            self._cli_path,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--model",
            model,
            "--mode",
            "plan",
            "--sandbox",
            "--disable-slash-commands",
            "--settings",
            _ISOLATED_SETTINGS,
            "--print-timeout",
            f"{timeout:.6f}s",
        ]
        if request.reasoning and request.reasoning.enabled:
            effort = request.reasoning.effort or request.reasoning.thinking_level
            if effort is not None and effort in {"low", "medium", "high", "xhigh", "max"}:
                command.extend(["--effort", effort])
        if request.structured_output:
            command.extend(["--json-schema", json.dumps(request.structured_output.json_schema)])

        with tempfile.TemporaryDirectory(prefix="llm-council-agy-") as scratch:
            env = {
                name: os.environ[name]
                for name in ("GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_QUOTA_PROJECT", "SystemRoot")
                if os.environ.get(name)
            }
            env.update(
                HOME=scratch,
                USERPROFILE=scratch,
                XDG_CONFIG_HOME=scratch,
                XDG_DATA_HOME=scratch,
                XDG_CACHE_HOME=scratch,
                APPDATA=scratch,
                LOCALAPPDATA=scratch,
                TMPDIR=scratch,
                TEMP=scratch,
                TMP=scratch,
                PATH=os.defpath,
                CI="true",
                AGY_ADC_AUTH="true",
                GOOGLE_APPLICATION_CREDENTIALS=str(adc),
            )
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=scratch,
                env=env,
                start_new_session=os.name != "nt",
            )
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(event.encode()), timeout)
            except asyncio.TimeoutError:
                await terminate_process_tree(process)
                raise TimeoutError(f"Antigravity CLI timed out after {timeout:g}s") from None
            except asyncio.CancelledError:
                await terminate_process_tree(process)
                raise

        result = _read_headless_result(stdout, process.returncode)
        content = (
            result.get("structured_output") if request.structured_output else result.get("response")
        )
        if request.structured_output:
            if "structured_output" not in result or not Draft202012Validator(
                request.structured_output.json_schema
            ).is_valid(content):
                raise RuntimeError("Antigravity CLI result did not match the requested schema")
            text = json.dumps(content)
        elif isinstance(content, str) and content.strip():
            text = content
        else:
            raise RuntimeError("Antigravity CLI returned empty text")
        usage = _normalized_usage(result.get("usage"))
        return GenerateResponse(
            text=text,
            content=content,
            usage=usage,
            usage_verified=usage is not None,
            usage_invalid="usage" in result and usage is None,
        )
