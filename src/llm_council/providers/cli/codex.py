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
import importlib
import json
import logging
import os
import platform
import re
import shlex
import shutil
import stat
import sys
import tempfile
import warnings
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

import certifi

from llm_council.providers.base import (
    DoctorResult,
    ErrorType,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
    classify_error,
)
from llm_council.providers.cli._compatibility import (
    native_identity,
    selected_cli_path,
)
from llm_council.providers.cli._subprocess import terminate_process_tree

logger = logging.getLogger(__name__)

# Generation only: native tools are separately disabled and catalog-validated.
DEFAULT_FLAGS = "--sandbox read-only --skip-git-repo-check"
_CLEANUP_SECONDS = 1.0
_MAX_PROBE_OUTPUT_BYTES = 8 * 1024 * 1024
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
_CREDENTIAL_ENV_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?keys?|secrets?|passwords?|credentials?|authorization|cookies?|"
    r"(?:access|refresh|auth|bearer|id|session)[_-]?tokens?|key|token)$|^AUTH_SECRET_\d+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ValidatedCodexRecords:
    records: list[dict[str, Any]]
    issued: datetime
    expires: datetime

    def current_at(self, now: datetime) -> bool:
        return self.issued <= now < self.expires


def _validated_compatibility_records(
    payload: bytes, now: datetime
) -> _ValidatedCodexRecords | None:
    """Validate a complete bounded document; validation does not authorize execution."""
    if len(payload) > 256 * 1024 or now.tzinfo is None or now.utcoffset() is None:
        return None

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        document = json.loads(payload, object_pairs_hook=unique_pairs)
        if (
            not isinstance(document, dict)
            or set(document) != {"schema", "issued_at", "expires_at", "records"}
            or type(document["schema"]) is not int
            or document["schema"] not in (1, 2)
            or not isinstance(document["records"], list)
            or len(document["records"]) > 256
        ):
            return None
        issued = datetime.strptime(document["issued_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        expires = datetime.strptime(document["expires_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        if not (issued <= now < expires and expires - issued <= timedelta(days=90)):
            return None
        fields = {
            "version",
            "os",
            "arch",
            "model",
            "auth",
            "endpoint",
            "reasoning",
            "output",
            "catalog_sha256",
            "files",
            "executables",
        }
        if document["schema"] == 2:
            context_fields = {"model", "auth", "endpoint", "reasoning", "output"}
            build_fields = (fields - context_fields) | {"profile", "contexts"}
            expanded: list[dict[str, Any]] = []
            for build in document["records"]:
                if not isinstance(build, dict) or set(build) != build_fields:
                    return None
                contexts = build["contexts"]
                if not isinstance(contexts, list) or not 1 <= len(contexts) <= 1024:
                    return None
                if len(expanded) + len(contexts) > 4096:
                    return None
                common = {key: value for key, value in build.items() if key != "contexts"}
                for context in contexts:
                    if not isinstance(context, dict) or set(context) != context_fields:
                        return None
                    expanded.append({**common, **context})
            document["records"] = expanded
        validated_inventories: set[tuple[int, int]] = set()
        for record in document["records"]:
            if not isinstance(record, dict) or set(record) != fields | {"profile"}:
                return None
            if record["profile"] not in ("generation-v1", "generation-legacy-v1"):
                return None
            if any(
                not isinstance(record[key], str) or not record[key] or len(record[key]) > 128
                for key in fields - {"files", "executables"}
            ):
                return None
            if not re.fullmatch(r"codex-cli [0-9]+\.[0-9]+\.[0-9]+", record["version"]):
                return None
            if not re.fullmatch(r"[0-9a-f]{64}", record["catalog_sha256"]):
                return None
            if (
                record["os"] not in ("darwin", "linux", "win32")
                or record["arch"] not in ("arm64", "x86_64")
                or (record["auth"], record["endpoint"])
                not in (("apikey", "openai"), ("apikey", "custom"), ("chatgpt", "chatgpt"))
                or record["reasoning"]
                not in ("default", "none", "low", "medium", "high", "xhigh", "max", "ultra")
                or record["output"] not in ("text", "json_schema")
            ):
                return None
            files = record["files"]
            executables = record["executables"]
            inventory_key = (id(files), id(executables))
            if inventory_key in validated_inventories:
                continue
            if (
                not isinstance(files, dict)
                or not (1 <= len(files) <= 64)
                or "bin/codex" not in files
                or (record["os"] == "win32" and "bin/codex.exe" in files)
            ):
                return None
            if (
                not isinstance(executables, list)
                or not all(isinstance(name, str) for name in executables)
                or executables != sorted(set(executables))
                or "bin/codex" not in executables
                or not set(executables) <= set(files)
            ):
                return None
            for name, digest in files.items():
                if (
                    not isinstance(name, str)
                    or len(name) > 240
                    or any(char in name for char in ("\x00", "\\", ":"))
                ):
                    return None
                path = PurePosixPath(name)
                if (
                    path.is_absolute()
                    or any(part in (".", "..") for part in name.split("/"))
                    or str(path) != name
                    or not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                ):
                    return None
            validated_inventories.add(inventory_key)
        return _ValidatedCodexRecords(document["records"], issued, expires)
    except (ValueError, TypeError, UnicodeDecodeError, OverflowError, RecursionError):
        return None


def _compatibility_candidates(
    payload: bytes,
    criteria: dict[str, str],
    now: datetime,
    *,
    clock: Callable[[], datetime] | None = None,
) -> list[dict[str, Any]]:
    """Select possible records; copy and full identity matching must follow."""
    fields = {"os", "arch", "model", "auth", "endpoint", "reasoning", "output"}
    if set(criteria) != fields:
        return []
    records = _validated_compatibility_records(payload, now)
    if records is None:
        return []
    matches = [
        record for record in records.records if all(record[key] == criteria[key] for key in fields)
    ]
    if clock is not None and not records.current_at(clock()):
        return []
    return matches


def _inventory_sha256(files: dict[str, str], executables: list[str]) -> str:
    canonical = json.dumps(
        {"files": files, "executables": executables}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _match_snapshot_identity(
    payload: bytes,
    criteria: dict[str, str],
    executable_sha256: str,
    version: str,
    catalog: bytes,
    inventory_sha256: str,
    now: datetime,
    *,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any] | None:
    """Match observations to one current record; snapshot ownership remains mandatory."""
    if len(catalog) > _MAX_PROBE_OUTPUT_BYTES:
        return None
    fields = {"os", "arch", "model", "auth", "endpoint", "reasoning", "output"}
    if set(criteria) != fields:
        return None
    validated = _validated_compatibility_records(payload, now)
    if validated is None:
        return None
    catalog_sha256 = hashlib.sha256(catalog).hexdigest()
    matches = [
        record
        for record in validated.records
        if all(record[key] == criteria[key] for key in fields)
        and record["files"]["bin/codex"] == executable_sha256
        and record["version"] == version
        and record["catalog_sha256"] == catalog_sha256
        and _inventory_sha256(record["files"], record["executables"]) == inventory_sha256
    ]
    if clock is not None and not validated.current_at(clock()):
        return None
    return matches[0] if len(matches) == 1 else None


def _native_executable_matches(path: Path, system: str, architecture: str) -> bool:
    """Reject wrappers and executables for a different recorded platform."""
    try:
        if path.is_symlink():
            return False
        fd = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_BINARY", 0),
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return False
            header = os.read(fd, 64)
            if system == "win32":
                if len(header) != 64 or header[:2] != b"MZ":
                    return False
                offset = int.from_bytes(header[60:64], "little")
                if not 64 <= offset <= 1024 * 1024:
                    return False
                os.lseek(fd, offset, os.SEEK_SET)
                pe = os.read(fd, 26)
                machine = {"x86_64": 0x8664, "arm64": 0xAA64}.get(architecture)
                flags = int.from_bytes(pe[22:24], "little")
                return (
                    len(pe) == 26
                    and pe[:4] == b"PE\0\0"
                    and int.from_bytes(pe[4:6], "little") == machine
                    and pe[24:26] == b"\x0b\x02"
                    and bool(flags & 2)
                    and not bool(flags & 0x2000)
                )
        finally:
            os.close(fd)
    except OSError:
        return False
    if system == "darwin" and len(header) >= 16 and header[:4] == b"\xcf\xfa\xed\xfe":
        cpu = int.from_bytes(header[4:8], "little")
        file_type = int.from_bytes(header[12:16], "little")
        return file_type == 2 and cpu == {"arm64": 0x100000C, "x86_64": 0x1000007}.get(architecture)
    if system == "linux" and len(header) >= 20 and header[:5] == b"\x7fELF\x02":
        if header[5] == 1:
            byte_order: Literal["little", "big"] = "little"
        elif header[5] == 2:
            byte_order = "big"
        else:
            return False
        if header[6] != 1:
            return False
        file_type = int.from_bytes(header[16:18], byte_order)
        cpu = int.from_bytes(header[18:20], byte_order)
        return file_type in (2, 3) and cpu == {"arm64": 183, "x86_64": 62}.get(architecture)
    return False


def _resolve_codex_executable(selected: Path, system: str, architecture: str) -> Path:
    """Resolve standard npm/pnpm entrypoints without running JavaScript or a shell.

    Package metadata only locates the binary. The record-selected snapshot still
    verifies every runtime file before any native process can run.
    """
    selected = selected.resolve(strict=True)
    if not selected.is_file():
        raise RuntimeError("Codex selected launcher is not a regular file")
    package: Path | None = None
    if selected.name == "codex.js" and selected.parent.name == "bin":
        package = selected.parent.parent
    elif (
        selected.name in ("codex", "codex.cmd", "codex.ps1")
        and selected.stat().st_size <= 64 * 1024
    ):
        text = selected.read_text(encoding="utf-8", errors="replace").replace("\\", "/")
        for relative in (
            "../@openai/codex/bin/codex.js",
            "node_modules/@openai/codex/bin/codex.js",
        ):
            if relative in text:
                entry = (selected.parent / relative).resolve(strict=True)
                package = entry.parent.parent
                break
    if package is None:
        return selected

    def package_metadata(path: Path) -> Any:
        if not path.is_file():
            raise RuntimeError("Codex package metadata is not a regular file")
        with path.open("rb") as file:
            data = file.read(64 * 1024 + 1)
        if len(data) > 64 * 1024:
            raise RuntimeError("Codex package metadata exceeds size limit")
        return json.loads(data)

    metadata = package_metadata(package / "package.json")
    if (
        not isinstance(metadata, dict)
        or metadata.get("name") != "@openai/codex"
        or not isinstance(metadata.get("bin"), dict)
        or metadata["bin"].get("codex")
        not in (
            "bin/codex.js",
            "./bin/codex.js",
        )
    ):
        raise RuntimeError("Codex launcher does not identify the expected package")
    targets = {
        ("darwin", "arm64"): ("darwin-arm64", "aarch64-apple-darwin"),
        ("darwin", "x86_64"): ("darwin-x64", "x86_64-apple-darwin"),
        ("linux", "arm64"): ("linux-arm64", "aarch64-unknown-linux-musl"),
        ("linux", "x86_64"): ("linux-x64", "x86_64-unknown-linux-musl"),
        ("win32", "arm64"): ("win32-arm64", "aarch64-pc-windows-msvc"),
        ("win32", "x86_64"): ("win32-x64", "x86_64-pc-windows-msvc"),
    }
    suffix, target = targets[(system, architecture)]
    name = f"@openai/codex-{suffix}"
    executable = "codex.exe" if system == "win32" else "codex"
    for modules in (
        package / "node_modules",
        *(parent for parent in package.parents if parent.name == "node_modules"),
    ):
        platform_package = modules / name
        if (platform_package / "package.json").is_file():
            info = package_metadata(platform_package / "package.json")
            dependencies = metadata.get("optionalDependencies", {})
            alias_matches = (
                isinstance(info, dict)
                and info.get("name") == "@openai/codex"
                and isinstance(info.get("version"), str)
                and isinstance(dependencies, dict)
                and dependencies.get(name) == f"npm:@openai/codex@{info['version']}"
            )
            if not isinstance(info, dict) or (info.get("name") != name and not alias_matches):
                raise RuntimeError("Codex platform package has an unexpected identity")
            return (platform_package / "vendor" / target / "bin" / executable).resolve(strict=True)
    return (package / "vendor" / target / "bin" / executable).resolve(strict=True)


class _PreSpawnRejectedError(RuntimeError):
    """A pre-spawn guard rejected the request before a child was created."""


@dataclass
class _SnapshotCleanup:
    loop: asyncio.AbstractEventLoop
    settled: bool = True
    closed: bool = False


@dataclass
class _CodexSnapshot:
    executable: Path
    cleanup_confirmed: _SnapshotCleanup
    executable_sha256: str | None = None
    inventory_sha256: str | None = None


def _snapshot_has_current_candidate(
    payload: bytes,
    criteria: dict[str, str],
    snapshot: _CodexSnapshot,
    *,
    allow_pending: bool = False,
) -> bool:
    if (
        snapshot.executable_sha256 is None
        or snapshot.inventory_sha256 is None
        or snapshot.cleanup_confirmed.closed
        or (not snapshot.cleanup_confirmed.settled and not allow_pending)
    ):
        return False
    fields = {"os", "arch", "model", "auth", "endpoint", "reasoning", "output"}
    if set(criteria) != fields:
        return False
    validated = _validated_compatibility_records(payload, datetime.now(timezone.utc))
    if validated is None:
        return False
    matched = any(
        all(record[key] == criteria[key] for key in fields)
        and record["files"]["bin/codex"] == snapshot.executable_sha256
        and _inventory_sha256(record["files"], record["executables"]) == snapshot.inventory_sha256
        for record in validated.records
    )
    return matched and validated.current_at(datetime.now(timezone.utc))


@dataclass
class _CodexAdmission:
    snapshot: _CodexSnapshot
    version: str
    catalog: bytes
    payload: bytes
    criteria: dict[str, str]
    record: dict[str, Any]

    def is_current(self, *, allow_pending: bool = False) -> bool:
        if (
            self.snapshot.executable_sha256 is None
            or self.snapshot.inventory_sha256 is None
            or self.snapshot.cleanup_confirmed.closed
            or (not self.snapshot.cleanup_confirmed.settled and not allow_pending)
        ):
            return False
        try:
            if self.snapshot.cleanup_confirmed.loop is not asyncio.get_running_loop():
                return False
        except RuntimeError:
            return False
        return (
            _match_snapshot_identity(
                self.payload,
                self.criteria,
                self.snapshot.executable_sha256,
                self.version,
                self.catalog,
                self.snapshot.inventory_sha256,
                datetime.now(timezone.utc),
                clock=lambda: datetime.now(timezone.utc),
            )
            == self.record
        )


@dataclass
class _CodexRuntime:
    env: dict[str, str]
    auth: bytes | None
    category: str
    config: dict[str, str]
    configured_model: Any = None

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


def _cleanup_home(home: str | Path, original: BaseException | None = None) -> None:
    def remove_readonly(function: Callable[..., Any], path: str, error: Any) -> None:
        if sys.platform != "win32" or not isinstance(error[1], PermissionError):
            raise error[1]
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        function(path)

    try:
        if sys.platform == "win32":
            shutil.rmtree(home, onerror=remove_readonly)
        else:
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
    elif event_type in ("item.started", "item.updated", "item.completed"):
        if state.saw_turn_completed:
            state.protocol_error = "item after terminal event"
            return
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") not in ("agent_message", "reasoning"):
            state.protocol_error = "unexpected generation item"
            return
        if event_type == "item.completed" and item.get("type") == "agent_message":
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
    elif event_type != "thread.started":
        state.protocol_error = "unexpected generation event"


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
            if state.protocol_error:
                raise RuntimeError(f"Codex CLI {state.protocol_error}")
    if pending:
        _ingest_codex_stdout_line(pending.decode("utf-8"), state)
        if state.protocol_error:
            raise RuntimeError(f"Codex CLI {state.protocol_error}")


async def _read_probe_stream(stream: asyncio.StreamReader, parts: list[str], label: str) -> None:
    """Read a bounded probe stream before decoding it as UTF-8."""
    data = bytearray()
    while chunk := await stream.read(65536):
        if len(data) + len(chunk) > _MAX_PROBE_OUTPUT_BYTES:
            raise RuntimeError(f"Codex CLI probe {label} exceeded output limit")
        data.extend(chunk)
    parts.append(data.decode("utf-8", errors="replace"))


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
        self._cli_path = selected_cli_path(cli_path, "codex")
        self._default_model = default_model
        self._default_flags = default_flags or DEFAULT_FLAGS
        self._timeout = timeout

    def _model_for_request(self, request: GenerateRequest, runtime: _CodexRuntime) -> str:
        model = request.model or self._default_model or runtime.configured_model
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Codex: select a model in Council or in your Codex configuration")
        return model

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

    @staticmethod
    def _read_native_config(env: dict[str, str]) -> dict[str, Any]:
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
        return config

    def _resolve_runtime(self) -> _CodexRuntime:
        """Snapshot one supported identity/route; never consult a fallback home."""
        env = self._get_subprocess_env()
        source = Path(env.get("CODEX_HOME") or (Path.home() / ".codex")).resolve()
        config = self._read_native_config(env)
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
        return _CodexRuntime(env, auth, category, native, config.get("model"))

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
        except BaseException as exc:
            _cleanup_home(cli_home, exc)
            raise

    @staticmethod
    def _record_environment(home: str) -> dict[str, str]:
        """Fixed environment shared by record-backed probes and generation."""
        env = {
            "PATH": os.defpath,
            "HOME": home,
            "CODEX_HOME": str(Path(home) / ".codex"),
            "XDG_CONFIG_HOME": str(Path(home) / ".config"),
            "TMPDIR": home,
            "TMP": home,
            "TEMP": home,
            "LANG": "C",
            "LC_ALL": "C",
        }
        if sys.platform == "win32":
            # Windows needs its OS loader path; user configuration remains isolated.
            if os.environ.get("SYSTEMROOT"):
                env["SystemRoot"] = os.environ["SYSTEMROOT"]
            env.update(
                {
                    "USERPROFILE": home,
                    "APPDATA": str(Path(home) / "AppData" / "Roaming"),
                    "LOCALAPPDATA": str(Path(home) / "AppData" / "Local"),
                }
            )
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

    def _require_remaining(self, deadline: float) -> None:
        try:
            self._remaining(deadline)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Codex CLI timed out before subprocess spawn") from exc

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
        cleanup_confirmed: _SnapshotCleanup | None = None,
        pre_spawn_guard: Callable[[], bool] | None = None,
    ) -> tuple[int, _LiveCodexState]:
        """Own spawn, one set of pipe readers and finite cancellation-safe cleanup."""
        if cleanup_confirmed is not None:
            if cleanup_confirmed.closed:
                raise RuntimeError("Codex snapshot is closed")
            if cleanup_confirmed.loop is not asyncio.get_running_loop():
                raise RuntimeError("Codex snapshot is bound to a different event loop")
            if not cleanup_confirmed.settled:
                raise RuntimeError("Codex snapshot has an active child or incomplete cleanup")
        try:
            self._remaining(deadline)
        except asyncio.TimeoutError:
            raise RuntimeError("Codex CLI timed out before subprocess spawn")
        proc = None
        readers: list[asyncio.Future[Any]] = []
        state = _LiveCodexState()
        cancellation: asyncio.CancelledError | None = None
        if cleanup_confirmed is not None:
            cleanup_confirmed.settled = False

        async def spawn_checked() -> asyncio.subprocess.Process:
            def check_deadline() -> None:
                try:
                    self._require_remaining(deadline)
                except RuntimeError as exc:
                    raise _PreSpawnRejectedError(str(exc)) from exc

            check_deadline()
            try:
                if pre_spawn_guard is not None and not pre_spawn_guard():
                    raise _PreSpawnRejectedError(
                        "Codex compatibility record expired before subprocess spawn"
                    )
            except _PreSpawnRejectedError:
                raise
            except Exception as exc:
                raise _PreSpawnRejectedError("Codex pre-spawn check failed") from exc
            check_deadline()
            return await asyncio.create_subprocess_exec(
                *cmd,
                stdin=stdin,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
                start_new_session=True,
            )

        spawn = asyncio.create_task(spawn_checked())
        try:
            # Python 3.10 wait_for() can swallow cancellation racing completion.
            done, _ = await asyncio.wait({spawn}, timeout=self._remaining(deadline))
            if not done:
                raise asyncio.TimeoutError
            proc = spawn.result()
            if isinstance(proc.stdout, asyncio.StreamReader) and isinstance(
                proc.stderr, asyncio.StreamReader
            ):
                if events:
                    stdout_reader = _read_codex_stdout(proc.stdout, state)
                else:
                    stdout_reader = _read_probe_stream(proc.stdout, state.stdout_parts, "stdout")
                stderr_reader = (
                    _read_codex_stderr(proc.stderr, state)
                    if events
                    else _read_probe_stream(proc.stderr, state.stderr_parts, "stderr")
                )
                readers = [
                    asyncio.create_task(stdout_reader),
                    asyncio.create_task(stderr_reader),
                    asyncio.create_task(proc.wait()),
                ]
                joined = asyncio.gather(*readers)
                readers.append(joined)
                joined_done, _ = await asyncio.wait({joined}, timeout=self._remaining(deadline))
                if not joined_done:
                    raise asyncio.TimeoutError
                joined.result()
            else:
                if not events:
                    raise RuntimeError("Codex CLI probe requires bounded stream readers")
                # Test doubles and alternate asyncio process implementations.
                communicate = asyncio.create_task(proc.communicate())
                readers.append(communicate)
                communicate_done, _ = await asyncio.wait(
                    {communicate}, timeout=self._remaining(deadline)
                )
                if not communicate_done:
                    raise asyncio.TimeoutError
                stdout, stderr = communicate.result()
                for line in stdout.decode("utf-8").splitlines(keepends=True):
                    _ingest_codex_stdout_line(line, state)
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
                        if (
                            spawn.done()
                            and not spawn.cancelled()
                            and isinstance(
                                spawn.exception(),
                                (FileNotFoundError, PermissionError, _PreSpawnRejectedError),
                            )
                        ):
                            return  # An exec failure did not create a child.
                        try:
                            proc = await asyncio.wait_for(asyncio.shield(spawn), _CLEANUP_SECONDS)
                        except (FileNotFoundError, PermissionError, _PreSpawnRejectedError):
                            return  # The pending spawn definitively created no child.
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
                if cleanup_confirmed is not None and not cleanup_task.cancelled():
                    cleanup_confirmed.settled = True
                raise cancellation
            cleanup_task.result()
            if cleanup_confirmed is not None:
                cleanup_confirmed.settled = True

    @contextlib.asynccontextmanager
    async def _snapshot_verified_codex(
        self,
        source: Path,
        candidates: list[dict[str, Any]],
        deadline: float,
        *,
        source_entrypoint: str = "bin/codex",
    ) -> AsyncIterator[_CodexSnapshot]:
        """Own a hash-verified private runtime until the request finishes."""
        self._require_remaining(deadline)
        snapshot = Path(tempfile.mkdtemp(prefix="llm-council-codex-snapshot-"))
        cleanup_confirmed = _SnapshotCleanup(asyncio.get_running_loop())
        operation_error: BaseException | None = None
        try:
            worker = Path(__file__).with_name("_codex_snapshot_worker.py")
            with tempfile.TemporaryFile() as request:
                inventory: dict[str, Any] = {
                    "candidates": candidates,
                    "source_entrypoint": source_entrypoint,
                }
                # Keep the exact serialized input owned by this operation, even if the
                # caller later mutates its original candidate dictionaries.
                inventory = json.loads(json.dumps(inventory))
                request.write(
                    json.dumps(
                        {"source": str(source), "destination": str(snapshot), **inventory}
                    ).encode("utf-8")
                )
                request.seek(0)
                code, state = await self._run_cli(
                    [sys.executable, "-I", str(worker)],
                    env={
                        key: value
                        for key, value in self._record_environment(str(snapshot)).items()
                        if key in ("HOME", "TMPDIR", "PATH", "SystemRoot")
                    },
                    cwd=str(snapshot),
                    deadline=deadline,
                    stdin=request,
                    cleanup_confirmed=cleanup_confirmed,
                )
            if not cleanup_confirmed.settled:
                raise RuntimeError("Codex snapshot worker cleanup incomplete")
            if code != 0:
                raise RuntimeError(
                    "Codex runtime snapshot did not match a compatibility record for this "
                    "build, model, authentication, reasoning and output context"
                )
            digest = "".join(state.stdout_parts).strip()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise RuntimeError("Codex snapshot worker returned an invalid executable digest")
            matches = [
                candidate
                for candidate in inventory["candidates"]
                if candidate["files"]["bin/codex"] == digest
            ]
            if not matches or any(candidate != matches[0] for candidate in matches[1:]):
                raise RuntimeError("Codex snapshot worker returned an ambiguous inventory")
            inventory_digest = _inventory_sha256(matches[0]["files"], matches[0]["executables"])
            self._require_remaining(deadline)
            yield _CodexSnapshot(
                snapshot / "bin" / ("codex.exe" if sys.platform == "win32" else "codex"),
                cleanup_confirmed,
                digest,
                inventory_digest,
            )
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            cleanup_confirmed.closed = True
            if cleanup_confirmed.settled:
                _cleanup_home(snapshot, operation_error)
            else:
                logger.error(
                    "Codex snapshot retained because child cleanup was not confirmed: %s", snapshot
                )
                if operation_error is None:
                    raise RuntimeError(f"Codex snapshot cleanup incomplete; retained at {snapshot}")

    @contextlib.asynccontextmanager
    async def _snapshot_compatible_codex(
        self, candidates: list[dict[str, Any]], deadline: float
    ) -> AsyncIterator[_CodexSnapshot]:
        """Snapshot the selected CLI package before executing any Codex process."""
        self._require_remaining(deadline)
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found")
        try:
            architecture = {
                "arm64": "arm64",
                "aarch64": "arm64",
                "x86_64": "x86_64",
                "amd64": "x86_64",
            }.get(platform.machine().lower(), "")
            selected = _resolve_codex_executable(Path(self._cli_path), sys.platform, architecture)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            raise RuntimeError("Codex selected executable could not be resolved") from exc
        if selected.parent.name == "bin":
            source, source_entrypoint = selected.parent.parent, f"bin/{selected.name}"
        else:
            source, source_entrypoint = selected.parent, selected.name
        inventories = [
            {"files": record["files"], "executables": record["executables"]}
            for record in candidates
        ]
        async with self._snapshot_verified_codex(
            source, inventories, deadline, source_entrypoint=source_entrypoint
        ) as snapshot:
            yield snapshot

    async def _fetch_compatibility_record(self, *, deadline: float) -> bytes | None:
        """Fetch only the fixed public record, using a credential-free child."""
        self._require_remaining(deadline)
        worker = Path(__file__).with_name("_codex_record_worker.py")
        fetch_deadline = min(deadline, asyncio.get_running_loop().time() + 5.0)
        code, state = await self._run_cli(
            [sys.executable, "-I", str(worker), certifi.where()],
            env={
                key: value
                for key, value in self._record_environment(tempfile.gettempdir()).items()
                if key in ("PATH", "SystemRoot")
            },
            cwd=tempfile.gettempdir(),
            deadline=fetch_deadline,
        )
        if code != 0:
            return None
        data = "".join(state.stdout_parts).encode("utf-8")
        return data if len(data) <= 256 * 1024 else None

    @contextlib.asynccontextmanager
    async def _admit_recorded_codex(
        self,
        runtime: _CodexRuntime,
        request: GenerateRequest,
        *,
        deadline: float,
        identity: dict[str, str | None],
    ) -> AsyncIterator[_CodexAdmission]:
        """Inspect a record-selected snapshot with no model credentials."""
        self._require_remaining(deadline)
        system = {"darwin": "darwin", "linux": "linux", "win32": "win32"}.get(sys.platform)
        architecture = {
            "arm64": "arm64",
            "aarch64": "arm64",
            "x86_64": "x86_64",
            "amd64": "x86_64",
        }.get(platform.machine().lower())
        endpoint = {"apikey": "openai", "chatgpt": "chatgpt"}.get(runtime.category)
        if system is None or architecture is None or endpoint is None:
            raise RuntimeError("Codex runtime context is not covered by compatibility records")
        if "openai_base_url" in runtime.config:
            if runtime.category != "apikey":
                raise RuntimeError("Codex custom provider route is not qualified for this auth")
            endpoint = "custom"
        criteria = {
            "os": system,
            "arch": architecture,
            "model": self._model_for_request(request, runtime),
            "auth": runtime.category,
            "endpoint": endpoint,
            "reasoning": self._reasoning_effort(request) or "default",
            "output": "json_schema" if request.structured_output else "text",
        }
        payload = await self._fetch_compatibility_record(deadline=deadline)
        if payload is None:
            raise RuntimeError("Codex compatibility record unavailable")
        validated_payload: bytes = payload
        candidates = _compatibility_candidates(
            payload, criteria, datetime.now(timezone.utc), clock=lambda: datetime.now(timezone.utc)
        )
        if not candidates:
            raise RuntimeError("Codex request has no current matching compatibility record")
        async with self._snapshot_compatible_codex(candidates, deadline) as snapshot:
            if not _native_executable_matches(snapshot.executable, system, architecture):
                raise RuntimeError(
                    "Codex snapshot executable format or architecture is unsupported"
                )
            probe_runtime = _CodexRuntime(runtime.env, None, runtime.category, runtime.config)
            home = self._create_isolated_cli_home(probe_runtime)
            operation_error: BaseException | None = None
            try:
                env = self._record_environment(home)
                cwd = str(Path(home) / "work")
                if not _snapshot_has_current_candidate(payload, criteria, snapshot):
                    raise RuntimeError("Codex compatibility record expired before version probe")
                code, version_state = await self._run_cli(
                    [str(snapshot.executable), "--version"],
                    env=env,
                    cwd=cwd,
                    deadline=deadline,
                    cleanup_confirmed=snapshot.cleanup_confirmed,
                    pre_spawn_guard=lambda: _snapshot_has_current_candidate(
                        validated_payload, criteria, snapshot, allow_pending=True
                    ),
                )
                version = "".join(version_state.stdout_parts).strip()
                if code != 0:
                    raise RuntimeError("Codex snapshot version probe failed")
                if not _snapshot_has_current_candidate(payload, criteria, snapshot):
                    raise RuntimeError("Codex compatibility record expired before catalog probe")
                code, catalog_state = await self._run_cli(
                    [str(snapshot.executable), "debug", "models", "--bundled"],
                    env=env,
                    cwd=cwd,
                    deadline=deadline,
                    cleanup_confirmed=snapshot.cleanup_confirmed,
                    pre_spawn_guard=lambda: _snapshot_has_current_candidate(
                        validated_payload, criteria, snapshot, allow_pending=True
                    ),
                )
                catalog_text = "".join(catalog_state.stdout_parts)
                if code != 0 or "\ufffd" in catalog_text:
                    raise RuntimeError("Codex snapshot catalog probe failed")
                catalog = catalog_text.encode("utf-8")
                if snapshot.executable_sha256 is None or snapshot.inventory_sha256 is None:
                    raise RuntimeError("Codex snapshot identity is unavailable")
                record = _match_snapshot_identity(
                    payload,
                    criteria,
                    snapshot.executable_sha256,
                    version,
                    catalog,
                    snapshot.inventory_sha256,
                    datetime.now(timezone.utc),
                    clock=lambda: datetime.now(timezone.utc),
                )
                if record is None:
                    raise RuntimeError(
                        "Codex snapshot does not match a current compatibility record"
                    )
                identity["cli_version"] = version
            except BaseException as exc:
                operation_error = exc
                raise
            finally:
                if snapshot.cleanup_confirmed.settled:
                    _cleanup_home(home, operation_error)
                else:
                    logger.error(
                        "Codex probe home retained because child cleanup was not confirmed: %s",
                        home,
                    )
                    if operation_error is None:
                        raise RuntimeError(f"Codex probe cleanup incomplete; retained at {home}")
            yield _CodexAdmission(snapshot, version, catalog, payload, criteria, record)

    @staticmethod
    def _write_generation_catalog(
        catalog_text: str,
        model: str,
        effort: str | None,
        *,
        root: Path,
        profile: str | None = None,
    ) -> str:
        if profile not in ("generation-v1", "generation-legacy-v1"):
            raise ValueError("Codex unsupported compatibility profile")
        try:
            catalog = json.loads(catalog_text)
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
        if profile == "generation-v1":
            if (
                selected.get("experimental_supported_tools")
                not in ([], ["send_user_message_async", "clock"])
                or selected.get("tool_mode") not in (None, "direct", "code_mode_only")
                or selected.get("multi_agent_version") not in (None, "disabled", "v2")
            ):
                raise ValueError("Codex unsupported model catalog tool shape")
            # These model-level controls can override feature flags.
            selected["experimental_supported_tools"] = []
            if selected.get("tool_mode") is not None:
                selected["tool_mode"] = "direct"
            if selected.get("multi_agent_version") is not None:
                selected["multi_agent_version"] = "disabled"
        elif profile == "generation-legacy-v1":
            if selected.get("tool_mode") not in (
                None,
                "direct",
                "code_mode",
                "code_mode_only",
            ) or selected.get("multi_agent_version") not in (None, "disabled", "v1", "v2"):
                raise ValueError("Codex unsupported model catalog tool shape")
            if selected.get("tool_mode") is not None:
                selected["tool_mode"] = "direct"
            if selected.get("multi_agent_version") is not None:
                selected["multi_agent_version"] = "disabled"
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
        # Preserve model identity, wire format and reasoning; remove only native tools.
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
        # Keep conservative long-secret protection without treating short feature flags as secrets.
        secrets = {
            value
            for key, value in env.items()
            if value
            and (
                _CREDENTIAL_ENV_KEY.search(key)
                or (len(value) >= 4 and re.search(r"KEY|TOKEN|SECRET|PASSWORD", key, re.IGNORECASE))
            )
        }
        for value in sorted(secrets, key=len, reverse=True):
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
        identity = native_identity(self._cli_path)
        try:
            response = await self._generate(request, identity)
        except BaseException as exc:
            # Include validation, spawn, timeout, cancellation and cleanup failures.
            exc.__dict__["native_identity"] = dict(identity)
            raise
        response.raw = {**(response.raw or {}), "native_identity": dict(identity)}
        return response

    async def _execute_generation(
        self,
        request: GenerateRequest,
        runtime: _CodexRuntime,
        *,
        root: Path,
        env: dict[str, str],
        cwd: str,
        deadline: float,
        version: str,
        model: str,
        effort: str | None,
        instructions: str,
        prompt: str,
        catalog_path: str,
        admission: _CodexAdmission,
    ) -> GenerateResponse:
        output_path = root / "last-message.txt"
        schema_path = None
        if request.structured_output:
            schema_path = root / "schema.json"
            schema_path.write_text(
                json.dumps(_prepare_schema_for_codex(dict(request.structured_output.json_schema))),
                encoding="utf-8",
            )
        stdin_path = root / "prompt.txt"
        stdin_path.write_text(prompt, encoding="utf-8")
        cmd = self._build_command(
            model=model,
            output_last_message_path=str(output_path),
            output_schema_path=str(schema_path) if schema_path else None,
        )
        cmd[0] = str(admission.snapshot.executable)
        cmd.extend(["--ignore-user-config", "--ignore-rules", "--ephemeral", "--strict-config"])
        cmd.extend(_config_args(runtime.config))
        cmd.extend(_config_args(_GENERATION_CONFIG))
        if admission.record["profile"] == "generation-v1":
            cmd.extend(["-c", "features.goals=false"])
        cmd.extend(["-c", f"model_catalog_json={json.dumps(catalog_path)}"])
        if instructions:
            instruction_path = root / "instructions.txt"
            instruction_path.write_text(instructions, encoding="utf-8")
            cmd.extend(["-c", f"model_instructions_file={json.dumps(str(instruction_path))}"])
        if effort is not None:
            cmd.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
        with stdin_path.open("rb") as stdin:
            if not admission.is_current():
                raise RuntimeError("Codex compatibility record expired before generation")
            run_options: dict[str, Any] = {
                "cleanup_confirmed": admission.snapshot.cleanup_confirmed,
                "pre_spawn_guard": lambda: admission.is_current(allow_pending=True),
            }
            code, state = await self._run_cli(
                cmd,
                env=env,
                cwd=cwd,
                deadline=deadline,
                stdin=stdin,
                events=True,
                diagnostics_env=runtime.diagnostics_env(),
                **run_options,
            )
        if code != 0 or state.error_message:
            self._raise_cli_error(state, code, runtime.diagnostics_env())
        if state.protocol_error:
            raise RuntimeError(f"Codex CLI {state.protocol_error}")
        if not state.saw_turn_completed:
            raise RuntimeError("Codex CLI missing terminal turn.completed")
        output = (
            output_path.read_text(encoding="utf-8") if output_path.exists() else state.agent_message
        )
        if not output.strip():
            raise RuntimeError("Codex CLI empty final output")
        return GenerateResponse(
            text=output,
            content=output,
            usage=state.usage,
            raw={
                "stdout": "".join(state.stdout_parts),
                "native_version": version,
                "cli_path": self._cli_path,
                "model": model,
                "auth_category": runtime.category,
                "reasoning": {
                    "effort": effort,
                    "status": "mapped" if effort else "uncontrolled",
                },
            },
        )

    async def _generate(
        self, request: GenerateRequest, identity: dict[str, str | None]
    ) -> GenerateResponse:
        """Generate from one record-admitted snapshot and its observed catalog."""
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
        model = self._model_for_request(request, runtime)
        if runtime.category == "missing":
            raise ValueError("Codex has no supported auth in the selected root or CODEX_API_KEY")
        async with self._admit_recorded_codex(
            runtime, request, deadline=deadline, identity=identity
        ) as admission:
            home = self._create_isolated_cli_home(runtime)
            operation_error: BaseException | None = None
            try:
                root = Path(home)
                catalog_path = self._write_generation_catalog(
                    admission.catalog.decode("utf-8"),
                    model,
                    effort,
                    root=root,
                    profile=admission.record["profile"],
                )
                return await self._execute_generation(
                    request,
                    runtime,
                    root=root,
                    env=self._record_environment(home),
                    cwd=str(root / "work"),
                    deadline=deadline,
                    version=admission.version,
                    model=model,
                    effort=effort,
                    instructions=instructions,
                    prompt=prompt,
                    catalog_path=catalog_path,
                    admission=admission,
                )
            except BaseException as exc:
                operation_error = exc
                raise
            finally:
                if admission.snapshot.cleanup_confirmed.settled:
                    _cleanup_home(home, operation_error)
                else:
                    logger.error(
                        "Codex request home retained because child cleanup was not confirmed: %s",
                        home,
                    )
                    if operation_error is None:
                        raise RuntimeError(f"Codex request cleanup incomplete; retained at {home}")

    async def _login_status_recorded(self, native_details: dict[str, str]) -> str:
        """Check default-model readiness with the same record and snapshot as generation."""
        deadline = asyncio.get_running_loop().time() + min(float(self._timeout), 30.0)
        self._check_unsafe_flags()
        self._validate_generation_flags()
        request = GenerateRequest(prompt="doctor")
        runtime = self._resolve_runtime()
        model = self._model_for_request(request, runtime)
        if runtime.category == "missing":
            raise ValueError("Codex has no supported auth in the selected root or CODEX_API_KEY")
        identity: dict[str, str | None] = {"cli_version": None}
        async with self._admit_recorded_codex(
            runtime, request, deadline=deadline, identity=identity
        ) as admission:
            home = self._create_isolated_cli_home(runtime)
            operation_error: BaseException | None = None
            try:
                root = Path(home)
                self._write_generation_catalog(
                    admission.catalog.decode("utf-8"),
                    model,
                    None,
                    root=root,
                    profile=admission.record["profile"],
                )
                if not admission.is_current():
                    raise RuntimeError("Codex compatibility record expired before login check")
                code, state = await self._run_cli(
                    [
                        str(admission.snapshot.executable),
                        *_config_args(runtime.config),
                        "login",
                        "status",
                    ],
                    env=self._record_environment(home),
                    cwd=str(root / "work"),
                    deadline=deadline,
                    cleanup_confirmed=admission.snapshot.cleanup_confirmed,
                    pre_spawn_guard=lambda: admission.is_current(allow_pending=True),
                )
                if code != 0:
                    return f"CLI login status failed (exit {code})"
                text = "\n".join(("".join(state.stdout_parts), "".join(state.stderr_parts)))
                native_details.update(cli_path=self._cli_path or "", cli_version=admission.version)
                lowered = text.lower()
                if "not logged in" in lowered or "logged out" in lowered:
                    return "CLI is not logged in"
                api = bool(
                    re.search(
                        r"(?im)^(?:API key configured(?:[ \t]*[-:]|[ \t]*$)|"
                        r"Logged in using (?:an? )?API key\b)",
                        text,
                    )
                )
                chatgpt = bool(re.search(r"(?im)^Logged in using ChatGPT\b", text))
                if runtime.category == "apikey" and api and not chatgpt:
                    return "API key configured"
                if runtime.category == "chatgpt" and chatgpt and not api:
                    return "Logged in using ChatGPT"
                return "CLI login status does not confirm the selected authentication"
            except BaseException as exc:
                operation_error = exc
                raise
            finally:
                if admission.snapshot.cleanup_confirmed.settled:
                    _cleanup_home(home, operation_error)
                else:
                    logger.error(
                        "Codex doctor home retained because child cleanup was not confirmed: %s",
                        home,
                    )
                    if operation_error is None:
                        raise RuntimeError(f"Codex doctor cleanup incomplete; retained at {home}")

    async def supports(self, capability: str) -> bool:
        return self.supports_capability(capability)

    async def doctor(self) -> DoctorResult:
        if not self._cli_path:
            return DoctorResult(ok=False, message="CLI not found")

        native_details: dict[str, str] = {}
        try:
            status_text = await self._login_status_recorded(native_details)
        except asyncio.TimeoutError:
            return DoctorResult(ok=False, message="CLI login status check timed out")
        except Exception as exc:  # pragma: no cover - defensive path
            return DoctorResult(ok=False, message=f"CLI login status check failed: {exc}")

        status_text = status_text or "CLI login status check failed"
        lowered = status_text.lower()
        api_key_configured = lowered == "api key configured"
        if native_details:
            status_text = (
                f"{native_details['cli_path']} ({native_details['cli_version']}): {status_text}"
            )

        if "not logged in" in lowered or "logged out" in lowered:
            return DoctorResult(ok=False, message=status_text, details=native_details or None)
        if "logged in" in lowered or api_key_configured:
            return DoctorResult(ok=True, message=status_text, details=native_details or None)
        return DoctorResult(ok=False, message=status_text, details=native_details or None)


def _register() -> None:
    from llm_council.providers.registry import get_registry

    with contextlib.suppress(ValueError):
        get_registry().register_provider("codex", CodexCLIProvider)


_register()
