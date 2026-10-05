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
import importlib
import json
import logging
import os
import re
import shlex
import shutil
import sys
import tempfile
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

from llm_council.providers.base import (
    DoctorResult,
    ErrorType,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
    classify_error,
)
from llm_council.providers.cli._subprocess import terminate_process_tree

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4"
# Generation only: native tools are separately disabled and catalog-validated.
DEFAULT_FLAGS = "--sandbox read-only --skip-git-repo-check"
_VERIFIED_VERSION = "codex-cli 0.149.1"
_CLEANUP_SECONDS = 1.0
_GENERATION_CONFIG = {
    "features.shell_tool": False,
    "features.view_image": False,
    "features.image_generation": False,
    "features.multi_agent": False,
    "features.apps": False,
    "features.plugins": False,
    "features.hooks": False,
    "web_search": "disabled",
    "tools.update_plan.enabled": False,
    "tools.experimental_request_user_input.enabled": False,
}

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
_ENV_SESSION_KEYS = {"CODEX_THREAD_ID", "CODEX_SESSION_ID", "CODEX_INTERNAL_ORIGINATOR_OVERRIDE"}


@dataclass
class _CodexRuntime:
    env: dict[str, str]
    auth: bytes | None
    category: str
    config: dict[str, str]

    def diagnostics_env(self) -> dict[str, str]:
        values = dict(self.env)
        if self.auth:
            auth = json.loads(self.auth)
            secrets = [auth.get("OPENAI_API_KEY"), *(auth.get("tokens") or {}).values()]
            values.update(
                {
                    f"AUTH_SECRET_{i}": value
                    for i, value in enumerate(secrets)
                    if isinstance(value, str)
                }
            )
        return values


def _config_args(config: dict[str, Any]) -> list[str]:
    return [arg for key, value in config.items() for arg in ("-c", f"{key}={json.dumps(value)}")]


def _cleanup_home(home: str | Path) -> None:
    original = sys.exc_info()[1]
    try:
        shutil.rmtree(home)
    except OSError as exc:
        logger.error("Codex cleanup incomplete: temporary home remains at %s", home)
        if original is not None:
            raise original from exc
        raise RuntimeError(f"Codex cleanup incomplete: temporary home remains at {home}") from exc


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
    saw_turn_completed: bool = False
    saw_event: bool = False
    protocol_error: str = ""


def _ingest_codex_stdout_line(line: str, state: _LiveCodexState) -> None:
    """Update live state from a single Codex JSONL stdout line."""

    state.stdout_parts.append(line)
    stripped = line.strip()
    if not stripped:
        return

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        if state.saw_event or stripped.startswith(("{", "[")):
            state.protocol_error = "malformed or truncated JSONL"
        return

    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        state.protocol_error = "malformed JSONL event"
        return

    state.saw_event = True
    event_type = payload.get("type")
    if event_type == "turn.started":
        state.saw_turn_completed = False
        state.agent_message = ""
    elif event_type == "item.completed":
        if state.saw_turn_completed:
            state.protocol_error = "item after terminal event"
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str):
                state.agent_message = text
            else:
                state.protocol_error = "malformed agent message"
    elif event_type == "turn.completed":
        state.saw_turn_completed = True
        usage = payload.get("usage")
        if isinstance(usage, dict):
            try:
                prompt_tokens = int(usage.get("input_tokens", 0) or 0) + int(
                    usage.get("cached_input_tokens", 0) or 0
                )
                completion_tokens = int(usage.get("output_tokens", 0) or 0)
            except (ValueError, TypeError):
                state.protocol_error = "malformed usage"
                return
            state.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
    elif event_type == "turn.failed":
        # Codex emits turn.failed with exit code 0 when the server rejects the
        # request -- e.g. a model this CLI version does not support. Capture the
        # message so the post-loop handler surfaces it instead of an empty success.
        state.saw_turn_completed = False
        state.error_message = "Codex turn.failed"
        error_payload = payload.get("error")
        if isinstance(error_payload, dict):
            message = error_payload.get("message")
            if isinstance(message, str) and message:
                state.error_message = message
    elif event_type == "error":
        message = payload.get("message")
        state.error_message = message if isinstance(message, str) and message else "Codex error"


async def _read_codex_stdout(stream: asyncio.StreamReader | None, state: _LiveCodexState) -> None:
    """Consume Codex stdout incrementally."""

    if stream is None:
        return

    pending = b""
    while chunk := await stream.read(65536):
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            _ingest_codex_stdout_line(line.decode("utf-8") + "\n", state)
    if pending:
        _ingest_codex_stdout_line(pending.decode("utf-8"), state)


async def _read_codex_stderr(stream: asyncio.StreamReader | None, state: _LiveCodexState) -> None:
    """Consume Codex stderr incrementally."""

    if stream is None:
        return

    while chunk := await stream.read(65536):
        state.stderr_parts.append(chunk.decode("utf-8", errors="replace"))


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
        if output_schema_path:
            cmd.extend(["--output-schema", output_schema_path])
        if output_last_message_path:
            cmd.extend(["-o", output_last_message_path])

        # prompt is fed via stdin in generate(), NOT argv. Windows
        # CreateProcess caps the command line at ~32KB; schema+prompt exceed it.
        return cmd

    def _check_unsafe_flags(self) -> None:
        """Emit warning if using unsafe permissive flags."""
        flags_str = self._default_flags
        for unsafe_flag in _UNSAFE_FLAGS:
            if unsafe_flag in flags_str:
                warnings.warn(_UNSAFE_WARNING, UserWarning, stacklevel=3)
                break

    def _get_subprocess_env(self) -> dict[str, str]:
        """Keep routing/auth identity, not outer session or telemetry identity."""
        return {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(_ENV_DENYLIST_PREFIXES) and key not in _ENV_SESSION_KEYS
        }

    def _resolve_runtime(self) -> _CodexRuntime:
        """Snapshot one supported identity/route; never consult a fallback home."""
        env = self._get_subprocess_env()
        source = Path(env.get("CODEX_HOME") or (Path.home() / ".codex")).resolve()
        config: dict[str, Any] = {}
        config_path = source / "config.toml"
        if config_path.exists():
            try:
                # Python 3.10 has no TOML parser. Do not guess at native configuration.
                parser = importlib.import_module(
                    "tomllib" if sys.version_info >= (3, 11) else "tomli"
                )
            except ImportError as exc:
                raise ValueError(
                    "Codex unsupported auth/route config: TOML parser unavailable"
                ) from exc
            try:
                config = parser.loads(config_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValueError("Codex unsupported auth/route configuration") from exc
        providers = config.get("model_providers", {})
        if (
            config.get("model_provider", "openai") != "openai"
            or not isinstance(providers, dict)
            or "openai" in providers
        ):
            raise ValueError("Codex unsupported configured model provider route")
        if "profile" in config:
            raise ValueError("Codex unsupported auth/route profile")
        if "chatgpt_base_url" in config:
            raise ValueError(
                "Codex unsupported auth route: chatgpt_base_url does not route generation"
            )
        if config.get("cli_auth_credentials_store", "file") != "file":
            raise ValueError("Codex unsupported auth storage; only explicit file auth is supported")
        unsupported_env = (
            "CODEX_ACCESS_TOKEN",
            "CODEX_AUTH_TOKEN",
            "OPENAI_ORG_ID",
            "OPENAI_ORGANIZATION",
            "OPENAI_PROJECT_ID",
            "OPENAI_API_BASE",
            "AZURE_OPENAI_ENDPOINT",
        )
        if any(env.get(key) for key in unsupported_env):
            raise ValueError("Codex unsupported ambient auth/route source")
        # Native .credentials.json is an MCP OAuth store, not model auth. Do not read/copy it.
        auth_path = source / "auth.json"
        # Verified 0.149.1 precedence: explicit Codex key, then selected file.
        # OPENAI_API_KEY belongs to other providers and is not Codex CLI auth.
        key = env.get("CODEX_API_KEY")
        auth = None
        category = "missing"
        account = None
        if key:
            if not key.strip():
                raise ValueError("Codex unsupported empty ambient auth key")
            auth = json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": key}).encode()
            category = "apikey"
        elif auth_path.exists():
            try:
                auth = auth_path.read_bytes()
                data = json.loads(auth)
            except (ValueError, OSError) as exc:
                raise ValueError("Codex unsupported or unreadable auth file") from exc
            if not isinstance(data, dict) or set(data) - {
                "auth_mode",
                "OPENAI_API_KEY",
                "tokens",
                "last_refresh",
            }:
                raise ValueError("Codex unsupported auth file shape")
            key, tokens = data.get("OPENAI_API_KEY"), data.get("tokens")
            if (
                isinstance(key, str)
                and key.strip()
                and tokens is None
                and data.get("auth_mode") in (None, "apikey")
            ):
                category = "apikey"
            elif (
                key is None
                and isinstance(tokens, dict)
                and data.get("auth_mode") in (None, "chatgpt")
            ):
                fields = {"id_token", "access_token", "refresh_token", "account_id"}
                if set(tokens) != fields or not all(
                    isinstance(tokens[x], str) and tokens[x] for x in fields
                ):
                    raise ValueError("Codex unsupported ChatGPT auth token shape")
                category = "chatgpt"
                account = tokens["account_id"]
            else:
                raise ValueError("Codex unsupported or ambiguous auth file identity")
        native = {"cli_auth_credentials_store": "file", "model_provider": "openai"}
        method = config.get("forced_login_method")
        if method is not None:
            if (
                method not in ("api", "chatgpt")
                or category != {"api": "apikey", "chatgpt": "chatgpt"}[method]
            ):
                raise ValueError("Codex auth does not satisfy forced login method")
            native["forced_login_method"] = method
        workspace = config.get("forced_chatgpt_workspace_id")
        if workspace is not None:
            if not isinstance(workspace, str) or category != "chatgpt" or workspace != account:
                raise ValueError("Codex auth does not satisfy forced workspace")
            native["forced_chatgpt_workspace_id"] = workspace
        ambient_url = env.pop("OPENAI_BASE_URL", None)
        if ambient_url and config.get("openai_base_url") not in (None, ambient_url):
            raise ValueError("Codex ambiguous OpenAI route sources")
        value = ambient_url or config.get("openai_base_url")
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("Codex unsupported route URL")
            url = urlsplit(value)
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError("Codex unsupported route URL")
            if category not in ("apikey", "chatgpt"):
                raise ValueError("Codex route requires a supported auth identity")
            native["openai_base_url"] = value
        for key in ("CODEX_API_KEY", "OPENAI_API_KEY"):
            env.pop(key, None)
        return _CodexRuntime(env, auth, category, native)

    def _create_isolated_cli_home(self, runtime: _CodexRuntime | None = None) -> str:
        runtime = runtime or self._resolve_runtime()
        cli_home = Path(tempfile.mkdtemp(prefix="llm-council-codex-home-"))
        try:
            codex_dir = cli_home / ".codex"
            codex_dir.mkdir()
            if runtime.auth is not None:
                try:
                    auth_path = codex_dir / "auth.json"
                    auth_path.touch(mode=0o600)
                    auth_path.write_bytes(runtime.auth)
                except OSError as exc:
                    raise RuntimeError(
                        "Codex auth copy failed; refusing another auth source"
                    ) from exc
            (cli_home / "work").mkdir()
            return str(cli_home)
        except BaseException:
            _cleanup_home(cli_home)
            raise

    def _isolated_env(self, cli_home: str, runtime: _CodexRuntime) -> dict[str, str]:
        env = dict(runtime.env)
        env.update(HOME=cli_home, CODEX_HOME=str(Path(cli_home) / ".codex"))
        env["XDG_CONFIG_HOME"] = str(Path(cli_home) / ".config")
        return env

    def _request_timeout(self, request: GenerateRequest) -> float:
        return (
            float(request.timeout_seconds) if request.timeout_seconds is not None else self._timeout
        )

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        return remaining

    async def _run_cli(
        self,
        cmd: list[str],
        *,
        env: dict[str, str],
        cwd: str,
        deadline: float,
        stdin: Any = None,
        events: bool = False,
        diagnostics_env: dict[str, str] | None = None,
    ) -> tuple[int, _LiveCodexState]:
        """Own spawn, one set of pipe readers and finite cancellation-safe cleanup."""
        try:
            self._remaining(deadline)
        except asyncio.TimeoutError:
            raise RuntimeError("Codex CLI timed out before subprocess spawn")
        proc = None
        readers: list[asyncio.Future[Any]] = []
        state = _LiveCodexState()
        cancellation: asyncio.CancelledError | None = None
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *cmd,
                stdin=stdin,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
                start_new_session=True,
            )
        )
        try:
            proc = await asyncio.wait_for(asyncio.shield(spawn), self._remaining(deadline))
            if isinstance(proc.stdout, asyncio.StreamReader) and isinstance(
                proc.stderr, asyncio.StreamReader
            ):
                if events:
                    stdout_reader = _read_codex_stdout(proc.stdout, state)
                else:
                    stream = proc.stdout

                    async def read_probe() -> None:
                        data = await stream.read()
                        state.stdout_parts.append(data.decode("utf-8", errors="replace"))

                    stdout_reader = read_probe()
                readers = [
                    asyncio.create_task(stdout_reader),
                    asyncio.create_task(_read_codex_stderr(proc.stderr, state)),
                    asyncio.create_task(proc.wait()),
                ]
                joined = asyncio.gather(*readers)
                readers.append(joined)
                await asyncio.wait_for(joined, self._remaining(deadline))
            else:
                # Test doubles and alternate asyncio process implementations.
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), self._remaining(deadline)
                )
                if events:
                    for line in stdout.decode("utf-8").splitlines(keepends=True):
                        _ingest_codex_stdout_line(line, state)
                else:
                    state.stdout_parts.append(stdout.decode("utf-8", errors="replace"))
                state.stderr_parts.append(stderr.decode("utf-8", errors="replace"))
            if proc.returncode is None:
                raise RuntimeError("Codex CLI exited without a return code")
            return proc.returncode, state
        except asyncio.TimeoutError:
            if events and state.error_message:
                self._raise_cli_error(
                    state, proc.returncode if proc else -1, diagnostics_env or env
                )
            raise RuntimeError("Codex CLI timed out within the request deadline")
        except asyncio.CancelledError as exc:
            cancellation = exc
            raise
        finally:

            async def cleanup() -> None:
                nonlocal proc
                try:
                    if proc is None:
                        # A cancelled spawn can already have created an OS process.
                        proc = await asyncio.wait_for(asyncio.shield(spawn), _CLEANUP_SECONDS)
                    await terminate_process_tree(proc, _CLEANUP_SECONDS)
                except asyncio.TimeoutError as exc:
                    spawn.cancel()
                    raise RuntimeError("Codex cleanup incomplete: process did not settle") from exc
                finally:
                    for reader in readers:
                        if not reader.done():
                            reader.cancel()
                    if readers:
                        done, pending = await asyncio.wait(readers, timeout=_CLEANUP_SECONDS)
                        for task in done:
                            if not task.cancelled():
                                task.exception()
                        if pending:
                            raise RuntimeError("Codex cleanup incomplete: readers did not settle")

            cleanup_task = asyncio.create_task(cleanup())
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
                except Exception:
                    break
            if cancellation is not None:
                cleanup_error = None if cleanup_task.cancelled() else cleanup_task.exception()
                if cleanup_error:
                    logger.error("Codex cleanup incomplete during cancellation")
                    raise cancellation from cleanup_error
                raise cancellation
            cleanup_task.result()

    async def _verify_native_contract(
        self, *, env: dict[str, str], cwd: str, deadline: float
    ) -> None:
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found")
        code, state = await self._run_cli(
            [self._cli_path, "--version"],
            env=env,
            cwd=cwd,
            deadline=deadline,
        )
        if code != 0 or "".join(state.stdout_parts).strip() != _VERIFIED_VERSION:
            raise RuntimeError(
                f"Codex unsupported or unverified CLI version; native contract requires {_VERIFIED_VERSION}"
            )

    async def _login_status_text(
        self,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        deadline: float | None = None,
    ) -> str | None:
        if not self._cli_path:
            return None
        cli_home = None
        native: dict[str, str] = {}
        diagnostics_env = env
        if deadline is None:
            deadline = asyncio.get_running_loop().time() + 5
        try:
            if env is None:
                runtime = self._resolve_runtime()
                cli_home = self._create_isolated_cli_home(runtime)
                env = self._isolated_env(cli_home, runtime)
                cwd = str(Path(cli_home) / "work")
                native = runtime.config
                diagnostics_env = runtime.diagnostics_env()
                await self._verify_native_contract(env=env, cwd=cwd, deadline=deadline)
            if cwd is None:
                raise ValueError("Codex login status requires an isolated working directory")
            code, state = await self._run_cli(
                [self._cli_path, *_config_args(native), "login", "status"],
                env=env,
                cwd=cwd,
                deadline=deadline,
            )
            text = "".join(state.stdout_parts).strip() or "".join(state.stderr_parts).strip()
            if code != 0 and "logged in" in text.lower() and "not logged in" not in text.lower():
                return f"CLI login status failed (exit {code})"
            return self._redact(text, diagnostics_env or env) or f"CLI returned exit code {code}"
        finally:
            if cli_home:
                _cleanup_home(cli_home)

    async def _generation_catalog(
        self,
        model: str,
        effort: str | None,
        *,
        env: dict[str, str],
        cwd: str,
        deadline: float,
        root: Path,
    ) -> str:
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found")
        code, state = await self._run_cli(
            [self._cli_path, "debug", "models", "--bundled"],
            env=env,
            cwd=cwd,
            deadline=deadline,
        )
        if code:
            raise ValueError("Codex bundled model catalog discovery failed")
        try:
            catalog = json.loads("".join(state.stdout_parts))
        except ValueError as exc:
            raise ValueError("Codex unsupported bundled model catalog JSON") from exc
        if (
            not isinstance(catalog, dict)
            or set(catalog) != {"models"}
            or not isinstance(catalog["models"], list)
        ):
            raise ValueError("Codex unsupported bundled model catalog shape")
        models = catalog["models"]
        slugs = [entry.get("slug") for entry in models if isinstance(entry, dict)]
        if (
            len(slugs) != len(models)
            or any(not isinstance(slug, str) or not slug for slug in slugs)
            or len(set(slugs)) != len(slugs)
        ):
            raise ValueError("Codex unsupported bundled model catalog entries")
        if model not in slugs:
            raise ValueError(f"Codex unsupported model in bundled catalog: {model}")
        selected = models[slugs.index(model)]
        if (
            "apply_patch_tool_type" not in selected
            or selected["apply_patch_tool_type"] not in (None, "freeform", "function")
            or selected.get("experimental_supported_tools") != []
        ):
            raise ValueError("Codex unsupported model catalog tool shape")
        levels = selected.get("supported_reasoning_levels")
        if (
            not isinstance(levels, list)
            or not levels
            or any(
                not isinstance(level, dict) or not isinstance(level.get("effort"), str)
                for level in levels
            )
        ):
            raise ValueError("Codex unsupported model catalog reasoning shape")
        if effort is not None and effort not in {level["effort"] for level in levels}:
            raise ValueError(f"Codex unsupported reasoning effort {effort} for model {model}")
        # Preserve all bundled metadata and model identities; only remove this tool.
        selected["apply_patch_tool_type"] = None
        path = root / "models.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        return str(path)

    def _validate_generation_flags(self) -> None:
        flags = iter(shlex.split(self._default_flags))
        for flag in flags:
            if flag == "--skip-git-repo-check":
                continue
            if flag == "--sandbox" and next(flags, None) == "read-only":
                continue
            if flag in ("-c", "--config"):
                value = next(flags, "")
                if value.startswith("developer_instructions="):
                    continue
            raise ValueError("Codex unsupported flags for generation-only auth/route isolation")

    @staticmethod
    def _request_text(request: GenerateRequest) -> tuple[str, str]:
        systems: list[str] = []
        users: list[str] = []
        if request.messages:
            for message in request.messages:
                if not isinstance(message.content, str):
                    raise ValueError("Codex unsupported non-text message content")
                if message.role == "system" and not users:
                    systems.append(message.content)
                elif message.role == "user":
                    users.append(message.content)
                else:
                    raise ValueError(
                        "Codex unsupported history: only leading system then user messages"
                    )
            prompt = "\n\n".join(users)
        else:
            prompt = request.prompt or ""
        if not prompt.strip():
            raise ValueError("Codex requires non-empty user input")
        return "\n\n".join(systems), prompt

    @staticmethod
    def _reasoning_effort(request: GenerateRequest) -> str | None:
        reasoning = request.reasoning
        if reasoning is None:
            return None
        if reasoning.budget_tokens is not None or reasoning.thinking_level is not None:
            raise ValueError("Codex unsupported reasoning budget or thinking_level")
        if not reasoning.enabled:
            if reasoning.effort not in (None, "none"):
                raise ValueError("Codex contradictory disabled reasoning and effort")
            return "none"
        if reasoning.effort is None:
            raise ValueError("Codex reasoning requires an explicit supported effort")
        return reasoning.effort

    @staticmethod
    def _redact(text: str, env: dict[str, str]) -> str:
        for key, value in env.items():
            if value and re.search(r"KEY|TOKEN|SECRET|PASSWORD", key):
                text = text.replace(value, "[REDACTED]")
        text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
        return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)

    def _error_details(self, stderr_text: str) -> str:
        lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
        error_lines = [line for line in lines if line.startswith("ERROR:")]
        return "\n".join(error_lines[-3:] or lines[-8:])

    def _raise_cli_error(
        self, state: _LiveCodexState, code: int | None, env: dict[str, str]
    ) -> None:
        details = self._redact(
            "\n".join(
                filter(
                    None,
                    [
                        self._error_details("".join(state.stderr_parts)),
                        state.error_message,
                    ],
                )
            ),
            env,
        )
        error_type = classify_error(details, code or 0)
        labels = {
            ErrorType.BILLING: "BILLING ERROR",
            ErrorType.AUTH: "AUTH ERROR",
            ErrorType.MODEL_UNAVAILABLE: "MODEL UNAVAILABLE",
            ErrorType.RATE_LIMIT: "RATE LIMIT",
        }
        label = labels.get(error_type, f"CLI failed ({error_type.value})")
        raise RuntimeError(f"{label}: {details or f'exit {code}'}")

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        if request.stream:
            raise NotImplementedError("Streaming not supported for CLI")
        deadline = asyncio.get_running_loop().time() + self._request_timeout(request)
        instructions, prompt = self._request_text(request)
        effort = self._reasoning_effort(request)
        self._check_unsafe_flags()
        self._validate_generation_flags()
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found")
        runtime = self._resolve_runtime()
        if runtime.category == "missing":
            raise ValueError("Codex has no supported auth in the selected root or CODEX_API_KEY")
        cli_home = self._create_isolated_cli_home(runtime)
        try:
            env = self._isolated_env(cli_home, runtime)
            cwd = str(Path(cli_home) / "work")
            await self._verify_native_contract(env=env, cwd=cwd, deadline=deadline)
            model = request.model or self._default_model
            root = Path(cli_home)
            catalog_path = await self._generation_catalog(
                model,
                effort,
                env=env,
                cwd=cwd,
                deadline=deadline,
                root=root,
            )
            output_path = root / "last-message.txt"
            schema_path = None
            if request.structured_output:
                schema_path = root / "schema.json"
                schema_path.write_text(
                    json.dumps(
                        _prepare_schema_for_codex(dict(request.structured_output.json_schema))
                    ),
                    encoding="utf-8",
                )
            stdin_path = root / "prompt.txt"
            stdin_path.write_text(prompt, encoding="utf-8")
            cmd = self._build_command(
                model=model,
                output_last_message_path=str(output_path),
                output_schema_path=str(schema_path) if schema_path else None,
            )
            cmd.extend(["--ignore-user-config", "--ignore-rules", "--ephemeral", "--strict-config"])
            cmd.extend(_config_args(runtime.config))
            cmd.extend(_config_args(_GENERATION_CONFIG))
            cmd.extend(["-c", f"model_catalog_json={json.dumps(catalog_path)}"])
            if instructions:
                instruction_path = root / "instructions.txt"
                instruction_path.write_text(instructions, encoding="utf-8")
                cmd.extend(["-c", f"model_instructions_file={json.dumps(str(instruction_path))}"])
            if effort is not None:
                cmd.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
            with stdin_path.open("rb") as stdin:
                code, state = await self._run_cli(
                    cmd,
                    env=env,
                    cwd=cwd,
                    deadline=deadline,
                    stdin=stdin,
                    events=True,
                    diagnostics_env=runtime.diagnostics_env(),
                )
            if code != 0 or state.error_message:
                self._raise_cli_error(state, code, runtime.diagnostics_env())
            if state.protocol_error:
                raise RuntimeError(f"Codex CLI {state.protocol_error}")
            if not state.saw_turn_completed:
                raise RuntimeError("Codex CLI missing terminal turn.completed")
            output = (
                output_path.read_text(encoding="utf-8")
                if output_path.exists()
                else state.agent_message
            )
            if not output.strip():
                raise RuntimeError("Codex CLI empty final output")
            return GenerateResponse(
                text=output,
                content=output,
                usage=state.usage,
                raw={
                    "stdout": "".join(state.stdout_parts),
                    "native_version": _VERIFIED_VERSION,
                    "model": model,
                    "auth_category": runtime.category,
                    "reasoning": {
                        "effort": effort,
                        "status": "mapped" if effort else "uncontrolled",
                    },
                },
            )
        finally:
            _cleanup_home(cli_home)

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
