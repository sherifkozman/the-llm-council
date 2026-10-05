"""Claude 2.1.288 contract regressions; no real Claude/auth/provider calls."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_council.providers.base import (
    ErrorType,
    GenerateRequest,
    Message,
    ReasoningConfig,
    classify_error,
)
from llm_council.providers.cli import claude_code as claude


def terminal(**changes):
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "FINAL",
        "usage": {"input_tokens": 11, "output_tokens": 7},
        **changes,
    }


class FakeProcess:
    def __init__(self, stdout, stderr=b"", code=0):
        self.stdout_bytes = stdout
        self.stderr_bytes = stderr
        self.code = code
        self.returncode = None
        self.input = None
        self.cleaned = False
        self.stdin = SimpleNamespace(
            write=self._write_input, drain=self._drain_input, close=lambda: None
        )

    def _write_input(self, data):
        self.input = data

    async def _drain_input(self):
        pass

    async def communicate(self, input=None):
        if input is not None:
            self.stdin.write(input)
        self.returncode = self.code
        return self.stdout_bytes, self.stderr_bytes


@pytest.fixture(autouse=True)
def synthetic_environment():
    with patch.dict(
        os.environ,
        {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "USER": "dummy-user"},
        clear=True,
    ):
        yield


@pytest.fixture
def fake_cli(monkeypatch):
    class CLI:
        stdout = json.dumps(terminal()).encode()
        stderr = b""
        code = 0
        version = b"2.1.288 (Claude Code)\n"

        def __init__(self):
            self.calls = []

        async def spawn(self, *cmd, **kwargs):
            proc = (
                FakeProcess(self.version)
                if "--version" in cmd
                else FakeProcess(self.stdout, self.stderr, self.code)
            )
            self.calls.append((cmd, kwargs, proc))
            return proc

    async def cleanup(proc, grace_seconds=1.0):
        proc.cleaned = True

    fake = CLI()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.spawn)
    monkeypatch.setattr(claude, "terminate_process_tree", cleanup)
    return fake


@pytest.fixture
def provider():
    return claude.ClaudeCodeCLIProvider(cli_path="/synthetic/claude")


@pytest.mark.parametrize("as_list", [False, True])
async def test_only_terminal_text_and_usage_are_returned(provider, fake_cli, as_list):
    payload = terminal()
    if as_list:
        payload = [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"text": "EARLIER"}]}},
            terminal(),
        ]
    fake_cli.stdout = json.dumps(payload).encode()
    response = await provider.generate(GenerateRequest(prompt="hello"))
    assert response.text == "FINAL"
    assert response.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


@pytest.mark.parametrize("turn_fields", [{}, {"num_turns": 1}])
@pytest.mark.parametrize("as_list", [False, True])
def test_single_turn_or_absent_turn_count_remains_valid(provider, turn_fields, as_list):
    payload = terminal(**turn_fields)
    response = provider._parse_response(
        0, json.dumps([payload] if as_list else payload).encode(), b"", {}
    )
    assert response.text == "FINAL"


@pytest.mark.parametrize("num_turns", [2, 3, 0, -1, True, False, "1", 1.0, None, [], {}])
@pytest.mark.parametrize("as_list", [False, True])
def test_unsupported_turn_count_rejects_successful_final_text(provider, num_turns, as_list):
    payload = terminal(num_turns=num_turns, result="FINAL_TURN_ONLY")
    if as_list:
        payload = [
            {"type": "assistant", "message": {"content": [{"text": "EARLIER_TEXT"}]}},
            payload,
        ]
    with pytest.raises(RuntimeError, match="unsupported multi-turn.*possible omitted continuation"):
        provider._parse_response(0, json.dumps(payload).encode(), b"", {})


def test_single_turn_count_does_not_override_terminal_error(provider):
    payload = terminal(num_turns=1, is_error=True)
    with pytest.raises(RuntimeError, match="terminal result failed"):
        provider._parse_response(0, json.dumps(payload).encode(), b"", {})


@pytest.mark.parametrize("code", [0, 1])
async def test_terminal_error_overrides_partial_text(provider, fake_cli, code):
    fake_cli.code = code
    fake_cli.stdout = json.dumps(
        [
            {"type": "assistant", "message": {"content": [{"text": "PARTIAL"}]}},
            terminal(is_error=True, result="SYNTHETIC_AFTER_PARTIAL", terminal_reason="api_error"),
        ]
    ).encode()
    with pytest.raises(RuntimeError, match="SYNTHETIC_AFTER_PARTIAL"):
        await provider.generate(GenerateRequest(prompt="hello"))


@pytest.mark.parametrize(
    "payload",
    [
        b"raw answer",
        b'{"type":"result",',
        b"",
        b"null",
        b'"answer"',
        b'[{"type":"assistant","message":{"content":[{"text":"PARTIAL"}]}}]',
        json.dumps([terminal(), terminal(result="CONFLICT")]).encode(),
        json.dumps([terminal(), terminal()]).encode(),
        json.dumps([terminal(), "invalid event"]).encode(),
        json.dumps(terminal(result="   ")).encode(),
        json.dumps(terminal(result={"text": "nested"})).encode(),
        json.dumps(terminal(subtype="error_max_turns")).encode(),
        json.dumps(terminal(is_error=0)).encode(),
        b'{"type":"result","result":"missing required fields"}',
    ],
)
async def test_invalid_completion_is_never_success(provider, fake_cli, payload):
    fake_cli.stdout = payload
    with pytest.raises(RuntimeError):
        await provider.generate(GenerateRequest(prompt="hello"))


async def test_exit_failure_preserves_both_redacted_error_channels(provider, fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-private-key")
    fake_cli.code = 1
    fake_cli.stderr = b"stderr-diagnostic synthetic-private-key"
    fake_cli.stdout = json.dumps(
        terminal(is_error=True, result="Not logged in: synthetic-private-key")
    ).encode()
    with pytest.raises(RuntimeError) as caught:
        await provider.generate(GenerateRequest(prompt="hello"))
    text = str(caught.value)
    assert "stderr-diagnostic" in text
    assert "Not logged in" in text
    assert "synthetic-private-key" not in text
    assert classify_error(text) == ErrorType.AUTH


def test_not_logged_in_is_auth_error():
    assert classify_error("Not logged in. Please run /login", 1) == ErrorType.AUTH


@pytest.mark.parametrize(
    "messages",
    [
        [Message(role="system", content="system-only")],
        [Message(role="user", content=" ")],
        [Message(role="assistant", content="lost"), Message(role="user", content="hi")],
        [Message(role="tool", content="lost"), Message(role="user", content="hi")],
        [Message(role="user", content="hi"), Message(role="system", content="late")],
        [Message(role="user", content="hi"), Message(role="user", content="second")],
        [Message(role="user", content=[{"type": "text", "text": "hi"}])],
        [Message(role="user", content="hi", name="cannot-preserve")],
    ],
)
async def test_unsupported_inputs_rejected_before_any_spawn(provider, fake_cli, messages):
    with pytest.raises(ValueError):
        await provider.generate(
            GenerateRequest(messages=messages, prompt="must not replace history")
        )
    assert not fake_cli.calls


@pytest.mark.parametrize(
    "messages,expected",
    [
        (None, "PROMPT"),
        ([], "PROMPT"),
        ([Message(role="system", content="SYSTEM"), Message(role="user", content="USER")], "USER"),
    ],
)
async def test_messages_precedence_and_native_channels(provider, fake_cli, messages, expected):
    await provider.generate(GenerateRequest(prompt="PROMPT", messages=messages))
    cmd, kwargs, proc = fake_cli.calls[-1]
    assert proc.input == expected.encode()
    assert "PROMPT" not in cmd and "USER" not in cmd
    if messages:
        assert cmd[cmd.index("--system-prompt") + 1] == "SYSTEM"
    assert kwargs["start_new_session"] is True
    assert not Path(kwargs["cwd"]).exists()


async def test_large_unicode_and_quoted_instructions_stay_on_stdin(provider, fake_cli):
    text = "BEGIN\n" + "x" * 71680 + "\nEND\u03bb <system>untrusted</system>"
    await provider.generate(
        GenerateRequest(
            messages=[Message(role="system", content="SYSTEM"), Message(role="user", content=text)]
        )
    )
    cmd, _, proc = fake_cli.calls[-1]
    assert proc.input == text.encode("utf-8")
    assert text not in cmd


@pytest.mark.parametrize(
    "auth,branch,source",
    [
        ({"ANTHROPIC_API_KEY": "dummy-api"}, "--bare", "api_key"),
        ({"CLAUDE_CODE_OAUTH_TOKEN": "dummy-oauth"}, "--safe-mode", "oauth_token"),
        ({}, "--safe-mode", "subscription"),
        (
            {
                "CLAUDE_CODE_USE_VERTEX": "1",
                "ANTHROPIC_VERTEX_PROJECT_ID": "dummy-project",
                "CLOUD_ML_REGION": "us-east5",
                "ANTHROPIC_API_KEY": "unrelated-direct-key",
            },
            "--bare",
            "vertex",
        ),
    ],
)
async def test_all_auth_routes_get_common_generation_only_flags(
    provider, fake_cli, monkeypatch, auth, branch, source
):
    for key, value in auth.items():
        monkeypatch.setenv(key, value)
    for key in [
        "CLAUDECODE",
        "CLAUDE_CODE_SIMPLE",
        "CLAUDE_CODE_EFFORT_LEVEL",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.setenv(key, "1")
    response = await provider.generate(GenerateRequest(prompt="hi"))
    cmd, kwargs, _ = fake_cli.calls[-1]
    assert branch in cmd
    assert ("--safe-mode" in cmd) != ("--bare" in cmd)
    for flag, expected in [
        ("--setting-sources", ""),
        ("--tools", ""),
        ("--mcp-config", '{"mcpServers":{}}'),
        ("--permission-mode", "dontAsk"),
        ("--permission-prompts", "none"),
    ]:
        assert cmd[cmd.index(flag) + 1] == expected
    for flag in [
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-chrome",
        "--no-session-persistence",
    ]:
        assert flag in cmd
    assert json.loads(cmd[cmd.index("--settings") + 1]) == {
        "disableAllHooks": True,
        "autoMemoryEnabled": False,
    }
    env = kwargs["env"]
    assert (
        not {
            "CLAUDECODE",
            "CLAUDE_CODE_SIMPLE",
            "CLAUDE_CODE_EFFORT_LEVEL",
            "CLAUDE_CODE_SKIP_VERTEX_AUTH",
            "OPENAI_API_KEY",
        }
        & env.keys()
    )
    if source == "vertex":
        assert "ANTHROPIC_API_KEY" not in env
        assert env["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert response.raw["auth_source"] == source


@pytest.mark.parametrize(
    "auth",
    [
        {"ANTHROPIC_API_KEY": "api", "CLAUDE_CODE_OAUTH_TOKEN": "oauth"},
        {"CLAUDE_CODE_USE_BEDROCK": "1"},
        {"CLAUDE_CODE_USE_FOUNDRY": "1"},
        {"ANTHROPIC_AUTH_TOKEN": "unsupported-gateway"},
        {"CLAUDE_CODE_USE_VERTEX": "maybe"},
    ],
)
async def test_conflicting_or_unsupported_auth_never_spawns(provider, fake_cli, monkeypatch, auth):
    for key, value in auth.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="auth|backend|source"):
        await provider.generate(GenerateRequest(prompt="hi"))
    assert not fake_cli.calls


@pytest.mark.parametrize(
    "version", [b"2.1.287 (Claude Code)", b"2.1.289 (Claude Code)", b"garbage"]
)
async def test_unverified_version_never_generates(provider, fake_cli, version):
    fake_cli.version = version
    with pytest.raises(RuntimeError, match="version"):
        await provider.generate(GenerateRequest(prompt="hi"))
    assert len(fake_cli.calls) == 1
    assert "--version" in fake_cli.calls[0][0]


@pytest.mark.parametrize(
    "reasoning,effort",
    [
        (None, None),
        (ReasoningConfig(enabled=True, effort="high"), "high"),
        (ReasoningConfig(enabled=True, effort="low"), "low"),
    ],
)
async def test_proved_effort_mapping(provider, fake_cli, reasoning, effort):
    response = await provider.generate(
        GenerateRequest(prompt="hi", model="claude-opus-5", reasoning=reasoning)
    )
    cmd = fake_cli.calls[-1][0]
    if effort:
        assert cmd[cmd.index("--effort") + 1] == effort
    else:
        assert "--effort" not in cmd
    assert response.raw["reasoning"] == {
        "status": "native" if effort else "uncontrolled",
        "effort": effort,
    }


@pytest.mark.parametrize(
    "model,reasoning",
    [
        ("claude-opus-5", ReasoningConfig(enabled=False)),
        ("claude-opus-5", ReasoningConfig(enabled=True, effort="none")),
        ("claude-opus-5", ReasoningConfig(enabled=True, effort="medium")),
        ("claude-opus-5", ReasoningConfig(enabled=True, budget_tokens=2048)),
        ("claude-haiku-4-5-20251001", ReasoningConfig(enabled=True, effort="high")),
    ],
)
async def test_unproved_reasoning_is_not_silently_accepted(provider, fake_cli, model, reasoning):
    with pytest.raises(ValueError, match="reasoning|effort"):
        await provider.generate(GenerateRequest(prompt="hi", model=model, reasoning=reasoning))
    assert not fake_cli.calls


@pytest.mark.parametrize("source", ["subscription", "vertex", "api_key", "oauth_token"])
async def test_preserves_selected_auth_discovery_and_routes(
    provider, fake_cli, monkeypatch, source
):
    monkeypatch.setenv("HOME", "/selected/home")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/selected/claude-root")
    monkeypatch.setenv("LOGNAME", "selected-account")
    if source == "vertex":
        monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/selected/adc.json")
        monkeypatch.setenv("CLOUDSDK_CONFIG", "/selected/gcloud")
        monkeypatch.setenv("ANTHROPIC_VERTEX_BASE_URL", "https://synthetic.invalid")
        monkeypatch.setenv("VERTEX_REGION_CLAUDE_OPUS_5", "us-east5")
    elif source == "api_key":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://synthetic.invalid")
    elif source == "oauth_token":
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy")
    await provider.generate(GenerateRequest(prompt="hi"))
    env = fake_cli.calls[-1][1]["env"]
    assert env == fake_cli.calls[0][1]["env"]
    if source in ("subscription", "vertex"):
        assert env["HOME"] == "/selected/home"
    else:
        assert env["HOME"] != "/selected/home"
    if source == "subscription":
        assert env["CLAUDE_CONFIG_DIR"] == "/selected/claude-root"
        assert env["USER"] == "dummy-user" and env["LOGNAME"] == "selected-account"
    if source == "vertex":
        assert env["CLOUDSDK_CONFIG"] == "/selected/gcloud"
        assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/selected/adc.json"
        assert env["ANTHROPIC_VERTEX_BASE_URL"] == "https://synthetic.invalid"
        assert env["VERTEX_REGION_CLAUDE_OPUS_5"] == "us-east5"
    if source == "api_key":
        assert env["ANTHROPIC_BASE_URL"] == "https://synthetic.invalid"


async def test_auth_snapshot_cannot_switch_between_probe_and_generation(
    provider, fake_cli, monkeypatch
):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy-oauth")
    spawn = fake_cli.spawn

    async def change_parent(*cmd, **kwargs):
        proc = await spawn(*cmd, **kwargs)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "late-api")
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", change_parent)
    await provider.generate(GenerateRequest(prompt="hi"))
    cmd, kwargs, _ = fake_cli.calls[-1]
    assert "--safe-mode" in cmd
    assert kwargs["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "dummy-oauth"
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]


async def test_doctor_reports_source_and_unverified_auth_without_generating(
    provider, fake_cli, monkeypatch
):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy-oauth")
    result = await provider.doctor()
    assert result.ok
    assert result.details["auth_source"] == "oauth_token"
    assert result.details["authentication"] == "not_verified"
    assert len(fake_cli.calls) == 1 and "--version" in fake_cli.calls[0][0]
    assert all(proc.cleaned for _, _, proc in fake_cli.calls)
    assert not Path(fake_cli.calls[0][1]["cwd"]).exists()


@pytest.mark.parametrize("phase", ["spawn", "version", "generation"])
async def test_deadline_covers_each_phase_and_cleans_owned_processes(
    provider, fake_cli, monkeypatch, phase
):
    spawn = fake_cli.spawn

    async def slow_spawn(*cmd, **kwargs):
        proc = await spawn(*cmd, **kwargs)
        if phase == "spawn":
            await asyncio.sleep(0.04)
        if (phase == "version" and "--version" in cmd) or (phase == "generation" and "-p" in cmd):
            communicate = proc.communicate

            async def delayed(input=None):
                await asyncio.sleep(0.04)
                return await communicate(input)

            proc.communicate = delayed
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_spawn)
    with pytest.raises(RuntimeError, match="timed out|deadline"):
        await provider.generate(GenerateRequest(prompt="hi", timeout_seconds=0.01))
    assert all(proc.cleaned for _, _, proc in fake_cli.calls)
    if phase != "generation":
        assert not any("-p" in cmd for cmd, _, _ in fake_cli.calls)


async def test_budget_is_not_reset_after_version(provider, fake_cli, monkeypatch):
    spawn = fake_cli.spawn

    async def slow_spawn(*cmd, **kwargs):
        proc = await spawn(*cmd, **kwargs)
        communicate = proc.communicate

        async def delayed(input=None):
            await asyncio.sleep(0.04)
            return await communicate(input)

        proc.communicate = delayed
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_spawn)
    with pytest.raises(RuntimeError, match="timed out|deadline"):
        await provider.generate(GenerateRequest(prompt="hi", timeout_seconds=0.06))


@pytest.mark.parametrize("phase", ["spawn", "version", "generation", "cleanup"])
async def test_external_repeated_cancellation_reaps_before_reraising(
    provider, fake_cli, monkeypatch, phase
):
    entered = asyncio.Event()
    spawn = fake_cli.spawn

    async def controlled_spawn(*cmd, **kwargs):
        proc = await spawn(*cmd, **kwargs)
        if phase == "spawn":
            entered.set()
            await asyncio.sleep(0.04)
        if (phase == "version" and "--version" in cmd) or (phase == "generation" and "-p" in cmd):

            async def stalled(input=None):
                entered.set()
                await asyncio.sleep(60)

            proc.communicate = stalled
        return proc

    async def delayed_cleanup(proc, grace_seconds=1.0):
        if phase == "cleanup":
            entered.set()
        await asyncio.sleep(0.04)
        proc.cleaned = True

    monkeypatch.setattr(asyncio, "create_subprocess_exec", controlled_spawn)
    monkeypatch.setattr(claude, "terminate_process_tree", delayed_cleanup)
    task = asyncio.create_task(provider.generate(GenerateRequest(prompt="hi")))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert all(proc.cleaned for _, _, proc in fake_cli.calls)
        assert all(not Path(kwargs["cwd"]).exists() for _, kwargs, _ in fake_cli.calls)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_cleanup_failure_cannot_be_success(provider, fake_cli, monkeypatch):
    async def failed_cleanup(proc, grace_seconds=1.0):
        raise TimeoutError("synthetic cleanup incomplete")

    monkeypatch.setattr(claude, "terminate_process_tree", failed_cleanup)
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        await provider.generate(GenerateRequest(prompt="hi"))


async def test_completed_processes_are_also_cleaned(provider, fake_cli):
    await provider.generate(GenerateRequest(prompt="hi"))
    assert all(proc.cleaned for _, _, proc in fake_cli.calls)


@pytest.mark.parametrize(
    "spawn_time,reader_time", [(1.0, None), (2.0, None), (0.0, 1.0), (0.0, 2.0)]
)
async def test_expired_spawn_or_reader_sends_zero_task_bytes(
    provider, fake_cli, monkeypatch, spawn_time, reader_time
):
    clock = [0.0]
    spawn = fake_cli.spawn
    create_task = asyncio.create_task
    scheduled = []

    async def timed_spawn(*cmd, **kwargs):
        proc = await spawn(*cmd, **kwargs)
        clock[0] = spawn_time
        return proc

    def delayed_reader(coro):
        if len(scheduled) == 1 and reader_time is not None:
            clock[0] = reader_time
        task = create_task(coro)
        scheduled.append(task)
        return task

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", timed_spawn)
    monkeypatch.setattr(asyncio, "create_task", delayed_reader)
    with pytest.raises(RuntimeError, match="deadline"):
        await provider._run_cli(
            ["/synthetic/claude", "-p"],
            env={},
            cwd="/nonexistent",
            deadline=1.0,
            input_bytes=b"MUST_NOT_SEND",
        )
    assert all(proc.input is None and proc.cleaned for _, _, proc in fake_cli.calls)
    assert all(task.done() for task in scheduled)


@pytest.mark.parametrize("operation", ["generate", "doctor"])
async def test_temp_cleanup_error_preserves_cancellation(
    provider, fake_cli, monkeypatch, caplog, operation
):
    entered = asyncio.Event()
    spawn = fake_cli.spawn
    original_rmtree = claude.shutil.rmtree
    scratch = []

    async def stalled_spawn(*cmd, **kwargs):
        proc = await spawn(*cmd, **kwargs)
        scratch.append(Path(kwargs["cwd"]))

        async def stalled(input=None):
            entered.set()
            await asyncio.sleep(60)

        proc.communicate = stalled
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", stalled_spawn)
    task = asyncio.create_task(
        provider.generate(GenerateRequest(prompt="hi"))
        if operation == "generate"
        else provider.doctor()
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)

        def denied(*args, **kwargs):
            raise PermissionError("synthetic temp cleanup denied")

        monkeypatch.setattr(claude.shutil, "rmtree", denied)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "cleanup incomplete" in caplog.text
    finally:
        monkeypatch.setattr(claude.shutil, "rmtree", original_rmtree)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for path in scratch:
            root = path.parent if path.name == "work" else path
            original_rmtree(root, ignore_errors=True)


@pytest.mark.parametrize(
    "name",
    ["managed-settings.json", "managed-settings.d", "managed-mcp.json", "remote-settings.json"],
)
async def test_managed_config_is_rejected_before_any_native_execution(
    provider, fake_cli, monkeypatch, tmp_path, name
):
    root = tmp_path / "selected-auth-root"
    root.mkdir()
    path = root / name
    if name.endswith(".d"):
        path.mkdir()
    else:
        path.write_text('{"hooks":{"SessionStart":[]}}', encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(root))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy-oauth")
    with pytest.raises(RuntimeError, match="managed|isolation"):
        await provider.generate(GenerateRequest(prompt="hi"))
    assert not fake_cli.calls


async def test_unknown_policy_filesystem_state_fails_closed(provider, fake_cli, monkeypatch):
    def denied(_path):
        raise PermissionError("synthetic managed policy stat denied")

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(RuntimeError, match="managed|isolation"):
        await provider.generate(GenerateRequest(prompt="hi"))
    assert not fake_cli.calls


@pytest.mark.parametrize(
    "payload",
    [
        b'{"type":"result","subtype":"success","is_error":true,"is_error":false,"result":"FINAL"}',
        b'{"type":"result","subtype":"success","is_error":false,"result":"FINAL","usage":NaN}',
    ],
)
async def test_ambiguous_or_non_json_terminal_is_rejected(provider, fake_cli, payload):
    fake_cli.stdout = payload
    with pytest.raises(RuntimeError, match="JSON"):
        await provider.generate(GenerateRequest(prompt="hi"))


async def test_exit_nonzero_cannot_succeed_even_with_valid_terminal(provider, fake_cli):
    fake_cli.code = 1
    with pytest.raises(RuntimeError, match="exit 1"):
        await provider.generate(GenerateRequest(prompt="hi"))


async def test_terminal_usage_never_falls_back_to_assistant(provider, fake_cli):
    fake_cli.stdout = json.dumps(
        [
            {"type": "assistant", "usage": {"input_tokens": 999}},
            terminal(usage=None),
        ]
    ).encode()
    response = await provider.generate(GenerateRequest(prompt="hi"))
    assert response.usage is None


async def test_events_after_terminal_are_not_accepted(provider, fake_cli):
    fake_cli.stdout = json.dumps([terminal(), {"type": "error", "message": "LATE_ERROR"}]).encode()
    with pytest.raises(RuntimeError, match="terminal"):
        await provider.generate(GenerateRequest(prompt="hi"))


async def test_structured_unknown_auth_headers_are_redacted(provider, fake_cli):
    fake_cli.stdout = json.dumps(
        terminal(
            is_error=True,
            result="Not logged in",
            error={
                "Authorization": "Bearer synthetic-private-bearer",
                "api_key": "synthetic-private-key",
            },
        )
    ).encode()
    fake_cli.stderr = b"Authorization: Bearer synthetic-private-stderr"
    with pytest.raises(RuntimeError) as caught:
        await provider.generate(GenerateRequest(prompt="hi"))
    assert "synthetic-private" not in str(caught.value)
    assert "Not logged in" in str(caught.value)


async def test_delayed_final_beyond_45_seconds_uses_remaining_deadline(
    provider, fake_cli, monkeypatch
):
    clock = [0.0]
    timeouts = []
    wait_for = asyncio.wait_for
    spawn = fake_cli.spawn

    async def timed_spawn(*cmd, **kwargs):
        proc = await spawn(*cmd, **kwargs)
        communicate = proc.communicate

        async def delayed(input=None):
            clock[0] += 2 if "--version" in cmd else 50
            return await communicate(input)

        proc.communicate = delayed
        return proc

    async def record_wait(awaitable, timeout):
        timeouts.append(timeout)
        return await wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", timed_spawn)
    monkeypatch.setattr(asyncio, "wait_for", record_wait)
    result = await provider.generate(GenerateRequest(prompt="hi", timeout_seconds=60))
    assert result.text == "FINAL"
    assert clock[0] == 52
    assert timeouts == [60, 60, 58, 58]


@pytest.fixture
def local_process_cli(monkeypatch):
    """Run Python stand-ins only, never the installed native executable."""
    real_spawn = asyncio.create_subprocess_exec
    processes = []
    config = {"body": "print(json.dumps(result))", "delay_spawn": False}
    script_prefix = """
import json, os, subprocess, sys, time
if "--version" in sys.argv:
    print("2.1.288 (Claude Code)")
    raise SystemExit(0)
prompt = sys.stdin.buffer.read().decode("utf-8")
result = {"type":"result", "subtype":"success", "is_error":False, "result":"FINAL"}
"""

    async def spawn(*cmd, **kwargs):
        assert cmd[0] == "/synthetic/claude"
        proc = await real_spawn(
            sys.executable, "-c", script_prefix + config["body"], *cmd[1:], **kwargs
        )
        processes.append(proc)
        if config["delay_spawn"]:
            await asyncio.sleep(0.1)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    yield config, processes
    for proc in processes:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)


def assert_groups_gone(processes):
    for proc in processes:
        assert proc.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(proc.pid, 0)


async def test_real_delayed_stdin_feeder_writes_zero_task_bytes(provider, monkeypatch, tmp_path):
    loop = asyncio.get_running_loop()
    baseline = asyncio.all_tasks()
    deadline = loop.time() + 0.5
    payload = b"MUST_NOT_SEND_AFTER_DEADLINE\n"
    writes = []
    entered_before_deadline = []
    processes = []
    real_spawn = asyncio.create_subprocess_exec

    async def observed_spawn(*cmd, **kwargs):
        proc = await real_spawn(*cmd, **kwargs)
        processes.append(proc)
        write = proc.stdin.write
        communicate = proc.communicate

        def record_write(data):
            write(data)
            writes.append({"bytes": len(data), "after_deadline": loop.time() >= deadline})

        async def delayed_communication(input=None):
            entered_before_deadline.append(loop.time() < deadline)
            # Delay the real pipe feeder, not a fake communicate implementation.
            # Other ready callbacks can impose this same event-loop delay.
            time.sleep(max(0, deadline - loop.time()) + 0.05)
            return await communicate(input=input)

        monkeypatch.setattr(proc.stdin, "write", record_write)
        monkeypatch.setattr(proc, "communicate", delayed_communication)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", observed_spawn)
    started = loop.time()
    try:
        with pytest.raises(RuntimeError, match="deadline"):
            await provider._run_cli(
                [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
                env={"PATH": "/usr/bin:/bin"},
                cwd=str(tmp_path),
                deadline=deadline,
                input_bytes=payload,
            )
        assert entered_before_deadline == [True]
        assert_groups_gone(processes)
        await asyncio.sleep(0)
        assert not asyncio.all_tasks() - baseline
        assert loop.time() - started < 2
        assert writes == [], f"Actual pipe writes: {writes}; payload bytes: {len(payload)}"
    finally:
        for proc in processes:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await asyncio.wait_for(proc.wait(), 1)


async def test_real_pipe_drains_output_while_feeding_and_delivers_eof(provider, tmp_path):
    payload = b"x" * (1024 * 1024)
    script = """
import sys
sys.stdout.buffer.write(b'o' * (1024 * 1024))
sys.stdout.buffer.flush()
sys.stderr.buffer.write(b'e' * (1024 * 1024))
sys.stderr.buffer.flush()
data = sys.stdin.buffer.read()
sys.stdout.buffer.write(b'\\nEOF:' + str(len(data)).encode())
"""
    code, stdout, stderr = await provider._run_cli(
        [sys.executable, "-c", script],
        env={"PATH": "/usr/bin:/bin"},
        cwd=str(tmp_path),
        deadline=asyncio.get_running_loop().time() + 3,
        input_bytes=payload,
    )
    assert code == 0
    assert stdout == b"o" * (1024 * 1024) + b"\nEOF:1048576"
    assert stderr == b"e" * (1024 * 1024)


async def test_real_local_pipe_preserves_large_unicode_input(provider, local_process_cli):
    config, processes = local_process_cli
    config["body"] = """
result["result"] = json.dumps({"prompt":prompt, "argv":sys.argv[1:], "cwd":os.getcwd()})
print(json.dumps(result))
"""
    text = "BEGIN\n" + "x" * 71680 + "\nEND\u03bb <system>untrusted</system>"
    result = await provider.generate(
        GenerateRequest(
            messages=[
                Message(role="system", content="SYSTEM"),
                Message(role="user", content=text),
            ]
        )
    )
    capture = json.loads(result.text)
    assert capture["prompt"] == text
    assert capture["argv"][capture["argv"].index("--system-prompt") + 1] == "SYSTEM"
    assert text not in capture["argv"]
    assert not Path(capture["cwd"]).exists()
    assert_groups_gone(processes)


@pytest.mark.parametrize("holding_pipes", [False, True])
async def test_real_descendant_cleanup_after_leader_exit(
    provider, local_process_cli, holding_pipes
):
    config, processes = local_process_cli
    pipe_args = "" if holding_pipes else ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
    config["body"] = f"""
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]{pipe_args})
print(json.dumps(result), flush=True)
os._exit(0)
"""
    if holding_pipes:
        with pytest.raises(RuntimeError, match="timed out"):
            await provider.generate(GenerateRequest(prompt="hi", timeout_seconds=0.4))
    else:
        result = await provider.generate(GenerateRequest(prompt="hi", timeout_seconds=2))
        assert result.text == "FINAL"
    assert_groups_gone(processes)


async def test_real_terminal_before_process_exit_is_not_completion(provider, local_process_cli):
    config, processes = local_process_cli
    config["body"] = "print(json.dumps(result), flush=True)\ntime.sleep(60)"
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        await provider.generate(GenerateRequest(prompt="hi", timeout_seconds=0.3))
    assert time.monotonic() - started < 2
    assert_groups_gone(processes)


@pytest.mark.parametrize("during_spawn", [False, True])
async def test_real_process_cancellation_is_finite_and_reaps(
    provider, local_process_cli, during_spawn
):
    config, processes = local_process_cli
    config["delay_spawn"] = during_spawn
    config["body"] = "time.sleep(60)"
    task = asyncio.create_task(provider.generate(GenerateRequest(prompt="hi")))
    try:

        async def until_launched():
            while len(processes) < (1 if during_spawn else 2):
                await asyncio.sleep(0.005)

        await asyncio.wait_for(until_launched(), 2)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert_groups_gone(processes)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
