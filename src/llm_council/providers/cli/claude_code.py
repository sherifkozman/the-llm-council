"""Generation-only Claude Code adapter for the versioned native JSON contract.

Flags reduce inherited execution surfaces; they are not a filesystem/network
sandbox. Safe mode retains native context and managed enterprise policy.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, ClassVar

from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
    classify_error,
)
from llm_council.providers.cli._compatibility import check_verified_version
from llm_council.providers.cli._subprocess import terminate_process_tree

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "sonnet"
# The compiler imports this tested baseline; runtime metadata reports the observed version.
_VERIFIED_VERSION = "2.1.288 (Claude Code)"
_VERIFIED_VERSIONS = (
    _VERIFIED_VERSION,
    "2.1.289 (Claude Code)",
    "2.1.290 (Claude Code)",
    "2.1.291 (Claude Code)",
    "2.1.292 (Claude Code)",
)
_CLEANUP_SECONDS = 1.0

_ENV_ALLOWLIST = {
    "PATH",
    "HOME",
    "TERM",
    "LANG",
    "LC_ALL",
    "USER",
    "LOGNAME",
    "USERNAME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "CLAUDE_CONFIG_DIR",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_VERTEX",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "CLOUD_ML_REGION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "CLOUDSDK_CONFIG",
    "ANTHROPIC_VERTEX_BASE_URL",
}
_SECRET_KEYS = {"ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"}


@contextlib.contextmanager
def _scratch_directory() -> Iterator[str]:
    scratch = tempfile.mkdtemp(prefix="llm-council-claude-cwd-")
    try:
        yield scratch
    finally:
        original = sys.exc_info()[1]
        try:
            shutil.rmtree(scratch)
        except OSError as exc:
            logger.error("Claude cleanup incomplete: temporary directory remains at %s", scratch)
            if original is not None:
                raise original from exc
            raise RuntimeError("Claude cleanup incomplete: temporary directory remains") from exc


def _normalize_usage(raw_usage: Any) -> dict[str, int] | None:
    if not isinstance(raw_usage, dict):
        return None
    try:
        input_tokens = int(raw_usage.get("input_tokens", 0) or 0) + int(
            raw_usage.get("cache_read_input_tokens", 0) or 0
        )
        output_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Claude Code invalid terminal usage") from exc
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class ClaudeCodeCLIProvider(ProviderAdapter):
    """Use stdin for one user message and the native system instruction channel."""

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

    @staticmethod
    def _request_text(request: GenerateRequest) -> tuple[str, str]:
        systems: list[str] = []
        user: str | None = None
        # Nonempty messages win; an empty list falls back to prompt.
        if request.messages:
            for message in request.messages:
                if not isinstance(message.content, str) or message.name or message.model_extra:
                    raise ValueError("Claude unsupported non-text or attributed message")
                if message.role == "system" and user is None:
                    systems.append(message.content)
                elif message.role == "user" and user is None:
                    user = message.content
                else:
                    raise ValueError(
                        "Claude supports only leading system instructions and one user message"
                    )
        else:
            user = request.prompt
        if user is None or not user.strip():
            raise ValueError("Claude requires non-empty user input")
        return "\n\n".join(systems), user

    @staticmethod
    def _prompt_text(request: GenerateRequest) -> str:
        return ClaudeCodeCLIProvider._request_text(request)[1]

    def _reasoning_effort(self, request: GenerateRequest) -> str | None:
        reasoning = request.reasoning
        if reasoning is None:
            return None
        if (
            not reasoning.enabled
            or reasoning.effort not in ("high", "low")
            or reasoning.budget_tokens is not None
            or reasoning.thinking_level is not None
            or (request.model or self._default_model) != "claude-opus-5"
        ):
            raise ValueError(
                "Claude unsupported reasoning control: only high/low effort on claude-opus-5 "
                "is native-proved; thinking off is unsupported"
            )
        return reasoning.effort

    @staticmethod
    def _resolve_auth(environment: dict[str, str]) -> str:
        for key in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_FOUNDRY"):
            if environment.get(key, "").lower() not in ("", "0", "false"):
                raise ValueError(f"Claude unsupported backend selector: {key}")
        vertex = environment.get("CLAUDE_CODE_USE_VERTEX", "")
        if vertex not in ("", "0", "false", "1"):
            raise ValueError("Claude unsupported backend selector value for CLAUDE_CODE_USE_VERTEX")
        if vertex == "1":
            return "vertex"
        if environment.get("ANTHROPIC_AUTH_TOKEN") or environment.get("ANTHROPIC_CUSTOM_HEADERS"):
            raise ValueError("Claude unsupported gateway auth source")
        has_api_key = bool(environment.get("ANTHROPIC_API_KEY"))
        token = bool(environment.get("CLAUDE_CODE_OAUTH_TOKEN"))
        if has_api_key and token:
            raise ValueError("Claude contradictory API key and OAuth auth sources")
        if has_api_key:
            return "api_key"
        if token:
            return "oauth_token"
        if environment.get("ANTHROPIC_BASE_URL"):
            raise ValueError("Claude unsupported subscription gateway auth source")
        return "subscription"

    @staticmethod
    def _assert_unmanaged_environment(environment: dict[str, str]) -> None:
        # Presence is enough to reject: never interpret, disable, or shadow policy.
        # Source paths are pinned by the 2.1.288 native implementation. Remote
        # server-only policy and changes after this snapshot remain unverified.
        if sys.platform not in ("darwin", "linux"):
            raise RuntimeError(
                "Claude unsupported isolation: managed-policy discovery unverified on this platform"
            )
        import pwd

        try:
            account = pwd.getpwuid(os.getuid())
            home = Path(environment.get("HOME") or account.pw_dir)
            config = Path(environment.get("CLAUDE_CONFIG_DIR") or home / ".claude")
            if not home.is_absolute() or not config.is_absolute():
                raise RuntimeError("Claude unsupported isolation: relative auth discovery root")
            paths = [
                config / name
                for name in (
                    "managed-settings.json",
                    "managed-settings.d",
                    "managed-mcp.json",
                    "remote-settings.json",
                )
            ]
            if sys.platform == "darwin":
                paths.extend(
                    [
                        Path("/Library/Application Support/ClaudeCode"),
                        Path("/Library/Managed Preferences/com.anthropic.claudecode.plist"),
                        Path(
                            f"/Library/Managed Preferences/{account.pw_name}/com.anthropic.claudecode.plist"
                        ),
                    ]
                )
            else:
                paths.append(Path("/etc/claude-code"))
            for path in paths:
                try:
                    path.lstat()
                except FileNotFoundError:
                    continue
                raise RuntimeError(
                    "Claude unsupported isolation: managed-policy configuration present"
                )
        except (OSError, KeyError) as exc:
            raise RuntimeError(
                "Claude unsupported isolation: managed-policy state could not be verified"
            ) from exc

    def _get_minimal_env(self, environment: dict[str, str] | None = None) -> dict[str, str]:
        source = os.environ if environment is None else environment
        return {
            key: value
            for key, value in source.items()
            if key in _ENV_ALLOWLIST or re.fullmatch(r"VERTEX_REGION_CLAUDE_[A-Z0-9_]+", key)
        }

    def _isolated_env(
        self, source: str, environment: dict[str, str], scratch: str
    ) -> dict[str, str]:
        env = self._get_minimal_env(environment)
        if source == "vertex":
            # Explicit backend intent wins over unrelated direct-provider keys.
            for key in (*_SECRET_KEYS, "ANTHROPIC_BASE_URL"):
                env.pop(key, None)
        else:
            for key in list(env):
                if key.startswith(("VERTEX_REGION_", "ANTHROPIC_VERTEX_")) or key in {
                    "CLAUDE_CODE_USE_VERTEX",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "GOOGLE_CLOUD_PROJECT",
                    "CLOUD_ML_REGION",
                    "CLOUDSDK_CONFIG",
                }:
                    env.pop(key, None)
        if source in ("api_key", "oauth_token"):
            env["HOME"] = scratch
        if source != "subscription":
            env["CLAUDE_CONFIG_DIR"] = scratch
        # Subscription keeps HOME/config-root and USER: a different config root
        # selects a different native keychain service, even with the same USER.
        env.update(
            {
                "TMPDIR": scratch,
                "DISABLE_AUTOUPDATER": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        )
        return env

    def _build_command(
        self, request: GenerateRequest, *, auth_source: str | None = None
    ) -> list[str]:
        if not self._cli_path:
            raise RuntimeError("Claude Code CLI not found")
        system, _ = self._request_text(request)
        effort = self._reasoning_effort(request)
        source = auth_source or self._resolve_auth(dict(os.environ))
        cmd = [
            self._cli_path,
            "-p",
            "--output-format",
            "json",
            "--model",
            request.model or self._default_model,
            "--setting-sources",
            "",
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--disable-slash-commands",
            "--settings",
            '{"disableAllHooks":true,"autoMemoryEnabled":false}',
            "--no-chrome",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
            "--bare" if source in ("api_key", "vertex") else "--safe-mode",
        ]
        if system:
            cmd.extend(["--system-prompt", system])
        if effort:
            cmd.extend(["--effort", effort])
        return cmd

    def _request_timeout(self, request: GenerateRequest) -> float:
        return (
            float(request.timeout_seconds) if request.timeout_seconds is not None else self._timeout
        )

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RuntimeError("Claude Code timed out within the request deadline")
        return remaining

    async def _run_cli(
        self,
        cmd: list[str],
        *,
        env: dict[str, str],
        cwd: str,
        deadline: float,
        input_bytes: bytes | None = None,
    ) -> tuple[int | None, bytes, bytes]:
        self._remaining(deadline)
        proc = None
        reader = None
        feeder = None
        completion = None
        cancellation: asyncio.CancelledError | None = None
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
                start_new_session=True,
            )
        )
        try:
            spawn_done, _ = await asyncio.wait({spawn}, timeout=self._remaining(deadline))
            if spawn not in spawn_done:
                raise asyncio.TimeoutError
            proc = spawn.result()
            remaining = self._remaining(deadline)
            owned_proc = proc

            async def communicate_before_deadline() -> tuple[bytes, bytes]:
                self._remaining(deadline)
                return await owned_proc.communicate()

            async def feed_stdin_before_deadline() -> None:
                stdin = owned_proc.stdin
                if stdin is None:
                    raise RuntimeError("Claude Code stdin pipe unavailable")
                try:
                    if input_bytes is not None:
                        # No await between the deadline guard and the actual write.
                        self._remaining(deadline)
                        stdin.write(input_bytes)
                        await stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    stdin.close()

            reader = asyncio.create_task(communicate_before_deadline())
            feeder = asyncio.create_task(feed_stdin_before_deadline())
            completion = asyncio.gather(reader, feeder)
            completion_done, _ = await asyncio.wait({completion}, timeout=remaining)
            if completion not in completion_done:
                raise asyncio.TimeoutError
            (stdout, stderr), _ = completion.result()
            return proc.returncode, stdout, stderr
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Claude Code timed out within the request deadline") from exc
        except asyncio.CancelledError as exc:
            cancellation = exc
            raise
        finally:

            async def cleanup() -> None:
                nonlocal proc
                try:
                    if proc is None:
                        try:
                            proc = await asyncio.wait_for(asyncio.shield(spawn), _CLEANUP_SECONDS)
                        except asyncio.TimeoutError as exc:
                            spawn.cancel()
                            _, pending_spawn = await asyncio.wait({spawn}, timeout=_CLEANUP_SECONDS)
                            if pending_spawn:
                                raise RuntimeError(
                                    "Claude cleanup incomplete: spawn did not settle"
                                ) from exc
                            if spawn.cancelled():
                                raise RuntimeError(
                                    "Claude cleanup incomplete: spawn ownership unknown"
                                ) from exc
                            proc = spawn.result()
                    # Successful leaders can leave live descendants.
                    await terminate_process_tree(proc, grace_seconds=_CLEANUP_SECONDS)
                except (asyncio.TimeoutError, TimeoutError) as exc:
                    raise RuntimeError("Claude cleanup incomplete: process did not settle") from exc
                finally:
                    communication_tasks: set[asyncio.Future[Any]] = {
                        task for task in (reader, feeder) if task is not None
                    }
                    if communication_tasks:
                        for task in communication_tasks:
                            if not task.done():
                                task.cancel()
                        if completion is not None:
                            communication_tasks.add(completion)
                        done, pending_readers = await asyncio.wait(
                            communication_tasks, timeout=_CLEANUP_SECONDS
                        )
                        for task in done:
                            if not task.cancelled():
                                task.exception()
                        if pending_readers:
                            raise RuntimeError(
                                "Claude cleanup incomplete: communication did not settle"
                            )

            # Repeated outer cancellation cannot abandon the finite cleanup owner.
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
                error = None if cleanup_task.cancelled() else cleanup_task.exception()
                if error:
                    logger.error("Claude cleanup incomplete during cancellation")
                    raise cancellation from error
                raise cancellation
            cleanup_task.result()

    @staticmethod
    def _redact(text: str, env: dict[str, str]) -> str:
        for key in _SECRET_KEYS:
            secret = env.get(key)
            if secret:
                text = text.replace(secret, "[REDACTED]")
                text = text.replace(json.dumps(secret)[1:-1], "[REDACTED]")
        return re.sub(
            r"(?i)((?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token)[\"']?\s*[:=]\s*[\"']?(?:bearer\s+)?)[^\s\"',;}]+",
            r"\1[REDACTED]",
            text,
        )

    def _failure(
        self, reason: str, stdout: str, stderr: str, code: int | None, env: dict[str, str]
    ) -> RuntimeError:
        details = self._redact(f"stderr: {stderr}\nstdout: {stdout}", env)
        category = classify_error(details, code if code is not None else -1)
        return RuntimeError(f"Claude Code {reason} ({category.value}, exit {code}): {details}")

    async def _verify_native_contract(
        self, *, env: dict[str, str], cwd: str, deadline: float
    ) -> str:
        if not self._cli_path:
            raise RuntimeError("Claude Code CLI not found")
        code, stdout, stderr = await self._run_cli(
            [self._cli_path, "--version"],
            env=env,
            cwd=cwd,
            deadline=deadline,
        )
        version = stdout.decode("utf-8", errors="replace").strip()
        try:
            check_verified_version(version, _VERIFIED_VERSIONS, path=self._cli_path, code=code)
        except RuntimeError as exc:
            raise self._failure(
                self._redact(str(exc), env),
                version,
                stderr.decode("utf-8", errors="replace"),
                code,
                env,
            ) from None
        return version

    def _parse_response(
        self, code: int | None, stdout: bytes, stderr: bytes, env: dict[str, str]
    ) -> GenerateResponse:
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        def fail(reason: str) -> RuntimeError:
            return self._failure(reason, stdout_text, stderr_text, code, env)

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON field")
                result[key] = value
            return result

        def reject_constant(_value: str) -> None:
            raise ValueError("Non-JSON numeric constant")

        try:
            data = json.loads(
                stdout.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
        except (ValueError, UnicodeError) as exc:
            raise fail("invalid or truncated terminal JSON") from exc
        events = [data] if isinstance(data, dict) else data
        if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
            raise fail("unsupported terminal JSON envelope")
        terminals = [event for event in events if event.get("type") == "result"]
        if len(terminals) != 1:
            raise fail("requires exactly one terminal result")
        result = terminals[0]
        if events[-1] is not result:
            raise fail("unexpected event after terminal result")
        if code != 0 or result.get("is_error") is not False or result.get("subtype") != "success":
            raise fail("terminal result failed")
        if "num_turns" in result and (
            type(result["num_turns"]) is not int or result["num_turns"] != 1
        ):
            raise fail(
                "unsupported multi-turn result: num_turns must be integer 1; "
                "possible omitted continuation"
            )
        text = result.get("result")
        if not isinstance(text, str) or not text.strip():
            raise fail("empty or invalid final result")
        return GenerateResponse(
            text=text,
            content=text,
            usage=_normalize_usage(result.get("usage")),
            raw={
                "stdout": self._redact(stdout_text, env),
                "stderr": self._redact(stderr_text, env),
            },
        )

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        deadline = asyncio.get_running_loop().time() + self._request_timeout(request)
        if request.stream:
            raise NotImplementedError("Streaming not supported for CLI providers")
        prompt = self._prompt_text(request).encode("utf-8")
        environment = dict(os.environ)
        source = self._resolve_auth(environment)
        cmd = self._build_command(request, auth_source=source)
        self._assert_unmanaged_environment(environment)
        with _scratch_directory() as scratch:
            cwd = str(Path(scratch) / "work")
            Path(cwd).mkdir()
            env = self._isolated_env(source, environment, scratch)
            version = await self._verify_native_contract(env=env, cwd=cwd, deadline=deadline)
            self._assert_unmanaged_environment(environment)
            self._assert_unmanaged_environment(env)
            code, stdout, stderr = await self._run_cli(
                cmd,
                env=env,
                cwd=cwd,
                deadline=deadline,
                input_bytes=prompt,
            )
            try:
                response = self._parse_response(code, stdout, stderr, env)
            except RuntimeError as exc:
                raise RuntimeError(f"{self._cli_path!r} ({version}): {exc}") from exc
            effort = self._reasoning_effort(request)
            raw = response.raw if isinstance(response.raw, dict) else {}
            raw.update(
                {
                    "auth_source": source,
                    "cli_version": version,
                    "cli_path": self._cli_path,
                    "reasoning": {
                        "status": "native" if effort else "uncontrolled",
                        "effort": effort,
                    },
                }
            )
            response.raw = raw
            return response

    async def supports(self, capability: str) -> bool:
        return self.supports_capability(capability)

    async def doctor(self) -> DoctorResult:
        deadline = asyncio.get_running_loop().time() + 5
        try:
            environment = dict(os.environ)
            source = self._resolve_auth(environment)
            self._assert_unmanaged_environment(environment)
            with _scratch_directory() as scratch:
                env = self._isolated_env(source, environment, scratch)
                version = await self._verify_native_contract(
                    env=env, cwd=scratch, deadline=deadline
                )
            return DoctorResult(
                ok=True,
                message=f"Claude Code {version}; authentication not verified",
                details={
                    "auth_source": source,
                    "authentication": "not_verified",
                    "cli_version": version,
                    "cli_path": self._cli_path,
                },
            )
        except Exception as exc:
            return DoctorResult(
                ok=False,
                message=f"Claude CLI check failed: {self._redact(str(exc), dict(os.environ))}",
            )


def _register() -> None:
    from llm_council.providers.registry import get_registry

    with contextlib.suppress(ValueError):
        get_registry().register_provider("claude", ClaudeCodeCLIProvider)


_register()
