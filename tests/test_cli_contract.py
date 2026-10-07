"""Generic CLI consumers, using synthetic results and local subprocesses only."""

import importlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from unittest.mock import AsyncMock, patch

import pytest
from rich.text import Text
from typer.testing import CliRunner

from llm_council.cli.main import app
from llm_council.engine.orchestrator import CouncilResult

cli = importlib.import_module("llm_council.cli.main")


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    original_defaults = cli._load_config_defaults
    monkeypatch.setattr(cli, "_load_config_defaults", lambda: {})
    monkeypatch.setattr(cli, "_load_provider_configs", lambda: {})
    return original_defaults


@pytest.mark.parametrize("output_args", [["--json"], ["--json", "--output"], ["--output"]])
def test_missing_explicit_config_keeps_failed_result_contract(
    tmp_path, monkeypatch, isolated_config, output_args
):
    monkeypatch.setattr(cli, "_load_config_defaults", isolated_config)
    monkeypatch.setattr(cli.Path, "home", lambda: tmp_path / "home")
    output = tmp_path / "result.json"
    args = [*output_args, str(output)] if "--output" in output_args else output_args
    missing = tmp_path / "missing-config.yaml"
    with patch("llm_council.Council") as council:
        invoked = CliRunner().invoke(
            app, ["--config", str(missing), "run", "planner", "task", *args]
        )
    assert invoked.exit_code == 1
    if "--output" in output_args:
        assert invoked.stdout == ""
        payload = json.loads(output.read_text())
    else:
        payload = json.loads(invoked.stdout)
    assert payload["success"] is False
    assert payload["execution_status"] == "failed"
    assert "Config file not found" in payload["error"]
    assert str(missing) in payload["error"]
    council.assert_not_called()


@pytest.mark.parametrize(
    "success,plan,status,code",
    [
        (True, {}, "completed", 0),
        (True, {"degraded_output": {"source": "draft"}}, "degraded", 0),
        (False, {}, "failed", 1),
    ],
)
def test_json_result_and_exact_exit(success, plan, status, code):
    result = CouncilResult(
        success=success, output={"verdict": "request_changes"}, execution_plan=plan
    )
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=result)
        invoked = CliRunner().invoke(app, ["--quiet", "run", "critic", "review", "--json"])
    assert invoked.exit_code == code
    assert json.loads(invoked.stdout)["execution_status"] == status
    assert invoked.stderr == ""


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--input", "missing-task.txt"],
        ["task", "--files", "missing-reference.txt"],
        ["task", "--schema", "missing-schema.json"],
        ["task", "--route"],
        ["task", "--timeout", "0"],
        ["task", "--format", "invalid"],
    ],
)
def test_post_parse_errors_are_failed_json_before_provider_calls(args):
    with patch("llm_council.Council") as council:
        invoked = CliRunner().invoke(app, ["run", "planner", *args, "--json"])
    assert invoked.exit_code == 1
    payload = json.loads(invoked.stdout)
    assert payload["success"] is False
    assert payload["execution_status"] == "failed"
    assert payload["error"]
    council.assert_not_called()


def test_parser_error_remains_stderr_exit_two():
    invoked = CliRunner().invoke(app, ["run", "planner", "task", "--json", "--unknown"])
    assert invoked.exit_code == 2
    assert invoked.stdout == ""
    assert "No such option: --unknown" in Text.from_ansi(invoked.stderr).plain


def test_json_dry_run_never_loads_credentials_or_models():
    with (
        patch("llm_council.Council") as council,
        patch.object(cli, "_load_provider_configs", side_effect=AssertionError("credential load")),
    ):
        invoked = CliRunner().invoke(app, ["run", "planner", "task", "--json", "--dry-run"])
    assert invoked.exit_code == 0
    payload = json.loads(invoked.stdout)
    assert payload["kind"] == "plan"
    assert payload["execution_status"] == "not_executed"
    assert "success" not in payload
    council.assert_not_called()


@pytest.mark.parametrize("as_json", [False, True])
def test_output_is_atomic_json_and_stdout_empty(tmp_path, as_json):
    output = tmp_path / "result.json"
    output.write_text("old result")
    replaced = []
    original = cli.os.replace

    def replace(source, destination):
        assert output.read_text() == "old result"
        replaced.append(json.loads(source.read_text()))
        original(source, destination)

    with patch("llm_council.Council") as council, patch.object(cli.os, "replace", replace):
        council.return_value.run = AsyncMock(return_value=CouncilResult(success=True))
        invoked = CliRunner().invoke(
            app,
            ["run", "planner", "task", "--output", str(output), *(["--json"] if as_json else [])],
        )
    assert invoked.exit_code == 0
    assert invoked.stdout == ""
    assert len(replaced) == 1
    assert json.loads(output.read_text())["execution_status"] == "completed"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]


@pytest.mark.parametrize("format_args", [["--format", "markdown"], ["--format", "md"], []])
def test_markdown_output_file_is_atomic_and_stdout_empty(tmp_path, monkeypatch, format_args):
    monkeypatch.setattr(cli, "_load_config_defaults", lambda: {"output_format": "markdown"})
    output = tmp_path / "result.md"
    output.write_text("old result")
    expected = (
        "# Council Result: COMPLETED\n\n## Output\n\n```json\n"
        '{\n  "result": "ship it"\n}\n```\n\n## Metrics\n\n'
        "- Duration: 0ms\n- Synthesis attempts: 1\n- Providers: openrouter\n"
    )
    replacements = []
    original = cli.os.replace

    def replace(source, destination):
        assert source.parent == output.parent
        assert output.read_text() == "old result"
        assert source.read_text() == expected
        replacements.append(destination)
        original(source, destination)

    with patch("llm_council.Council") as council, patch.object(cli.os, "replace", replace):
        council.return_value.run = AsyncMock(
            return_value=CouncilResult(success=True, output={"result": "ship it"})
        )
        invoked = CliRunner().invoke(
            app, ["run", "router", "task", *format_args, "--output", str(output)]
        )
    assert invoked.exit_code == 0
    assert invoked.stdout == ""
    assert output.read_text() == expected
    assert replacements == [output]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.md"]


def test_invalid_output_fails_before_provider_calls(tmp_path):
    output = tmp_path / "directory"
    output.mkdir()
    with patch("llm_council.Council") as council:
        invoked = CliRunner().invoke(
            app, ["run", "planner", "task", "--output", str(output), "--json"]
        )
    assert invoked.exit_code == 1
    assert invoked.stdout == ""
    assert invoked.stderr
    council.assert_not_called()


def test_failure_replaces_stale_result_with_failed_envelope(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"success": true}')
    with patch("llm_council.Council") as council:
        invoked = CliRunner().invoke(app, ["run", "planner", "--json", "--output", str(output)])
    assert invoked.exit_code == 1
    assert invoked.stdout == ""
    assert json.loads(output.read_text())["execution_status"] == "failed"
    council.assert_not_called()


@pytest.mark.parametrize("length,expected", [(50_000, "completed"), (50_001, "degraded")])
def test_reference_cap_retains_exact_boundary_and_reports_lost_evidence(tmp_path, length, expected):
    reference = tmp_path / "evidence.txt"
    reference.write_text("x" * (length - 1) + "Z")
    captured = []

    async def run(**kwargs):
        config = captured[0]
        return CouncilResult(
            success=True,
            critique="review",
            execution_plan={"context_preparation": config.context_metadata},
        )

    with patch("llm_council.Council") as council:

        def build(*, config):
            captured.append(config)
            return type("StubCouncil", (), {"run": staticmethod(run)})()

        council.side_effect = build
        invoked = CliRunner().invoke(
            app, ["run", "planner", "task", "--files", str(reference), "--json"]
        )
    assert invoked.exit_code == 0
    payload = json.loads(invoked.stdout)
    assert payload["execution_status"] == expected
    entry = captured[0].context_metadata["files"][0]
    assert entry["truncated"] is (length > 50_000)
    assert entry["retained_chars"] == 50_000
    assert ("Z" in captured[0].system_context) is (length == 50_000)


@pytest.mark.parametrize("extra", [False, True])
def test_total_cap_cannot_silently_skip_explicit_file(tmp_path, extra):
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.txt"
        path.write_text("x" * 49_999 + "Z")
        paths.extend(["--files", str(path)])
    if extra:
        last = tmp_path / "decisive.txt"
        last.write_text("late decisive evidence")
        paths.extend(["--files", str(last)])
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=CouncilResult(success=True))
        invoked = CliRunner().invoke(app, ["run", "planner", "task", "--json", *paths])
    assert invoked.exit_code == (1 if extra else 0)
    assert json.loads(invoked.stdout)["execution_status"] == ("failed" if extra else "completed")
    if extra:
        council.assert_not_called()


def test_exception_uses_full_envelope():
    with patch("llm_council.Council", side_effect=RuntimeError("synthetic failure")):
        invoked = CliRunner().invoke(app, ["run", "planner", "task", "--json"])
    assert invoked.exit_code == 1
    payload = json.loads(invoked.stdout)
    assert payload["execution_status"] == "failed"
    assert payload["error"] == "synthetic failure"


STUB_PROGRAM = r"""
import asyncio, json, os, sys
from pathlib import Path
import llm_council.cli.main as cli
from llm_council.providers.base import ProviderAdapter, GenerateResponse, DoctorResult
from llm_council.providers.registry import get_registry

class Stub(ProviderAdapter):
    name = "contract-stub"
    async def generate(self, request):
        action = os.environ.get("STUB_ACTION", "signal")
        if action == "failure":
            raise RuntimeError("synthetic stub failure")
        if action == "success":
            return GenerateResponse(text='{"ok": true}')
        child = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(60)")
        Path(os.environ["READY"]).write_text(str(child.pid))
        try:
            await asyncio.Event().wait()
        finally:
            child.terminate()
            await child.wait()
            Path(os.environ["CLEANED"]).write_text("reaped")
    async def doctor(self):
        return DoctorResult(ok=True)
    async def supports(self, capability):
        return False

get_registry().register_provider("contract-stub", Stub)
cli._load_config_defaults = lambda: {"providers": ["contract-stub"], "max_retries": 1, "enable_degradation": False}
cli._load_provider_configs = lambda: {}
cli.app()
"""


def stub_command(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object"}')
    return [
        sys.executable,
        "-c",
        STUB_PROGRAM,
        "run",
        "planner",
        "synthetic task",
        "--json",
        "--schema",
        str(schema),
    ]


def stub_env(tmp_path, action="signal"):
    return {
        **os.environ,
        "STUB_ACTION": action,
        "COUNCIL_HOME": str(tmp_path / "store"),
        "READY": str(tmp_path / "ready"),
        "CLEANED": str(tmp_path / "cleaned"),
    }


@pytest.mark.parametrize("signum,expected", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
@pytest.mark.parametrize("use_output", [False, True])
def test_real_signal_cancels_stub_and_finalizes_ledger(tmp_path, signum, expected, use_output):
    command = stub_command(tmp_path)
    output = tmp_path / "result.json"
    if use_output:
        command += ["--output", str(output)]
    process = subprocess.Popen(
        command, env=stub_env(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        ready = tmp_path / "ready"
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), process.communicate(timeout=2)
        child_pid = int(ready.read_text())
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == expected, stderr
        payload = json.loads(output.read_text() if use_output else stdout)
        if use_output:
            assert stdout == ""
        assert payload["success"] is False
        assert payload["execution_status"] == "cancelled"
        assert payload["run_id"]
        assert (tmp_path / "cleaned").read_text() == "reaped"
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        with sqlite3.connect(tmp_path / "store" / "ledger.db") as connection:
            row = connection.execute(
                "SELECT status FROM runs WHERE run_id=?", (payload["run_id"],)
            ).fetchone()
        assert row == ("cancelled",)
    finally:
        if process.poll() is None:
            process.kill()
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.communicate(timeout=5)


def test_shell_set_e_consumer_can_read_failed_result_once(tmp_path):
    output = tmp_path / "result.json"
    process = subprocess.run(
        [
            "/bin/sh",
            "-c",
            'set -e; rc=0; "$@" || rc=$?; printf "%s" "$rc" >&2; exit "$rc"',
            "consumer",
            *stub_command(tmp_path),
            "--output",
            str(output),
        ],
        env=stub_env(tmp_path, "failure"),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 1
    assert process.stdout == ""
    assert process.stderr.endswith("1")
    assert json.loads(output.read_text())["execution_status"] == "failed"
    assert json.loads(output.read_text())["run_id"]


def test_check_true_consumer_retains_failed_json(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as caught:
        subprocess.run(
            stub_command(tmp_path),
            env=stub_env(tmp_path, "failure"),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    assert caught.value.returncode == 1
    assert json.loads(caught.value.stdout)["execution_status"] == "failed"
    assert json.loads(caught.value.stdout)["run_id"]


def test_broken_stdout_pipe_exits_one_without_traceback(tmp_path):
    process = subprocess.Popen(
        stub_command(tmp_path),
        env=stub_env(tmp_path, "success"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.stdout.close()
    process.wait(timeout=10)
    stderr = process.stderr.read().decode()
    process.stderr.close()
    assert process.returncode == 1
    assert "Traceback" not in stderr


def test_signal_handlers_are_restored_after_run():
    originals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=CouncilResult(success=True))
        invoked = CliRunner().invoke(app, ["run", "planner", "task", "--json"])
    assert invoked.exit_code == 0
    assert {sig: signal.getsignal(sig) for sig in originals} == originals


@pytest.mark.parametrize("length,code,status", [(100_000, 0, "completed"), (100_001, 1, "failed")])
def test_task_character_cap_is_exact(length, code, status):
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=CouncilResult(success=True))
        invoked = CliRunner().invoke(
            app, ["run", "planner", "--input", "-", "--json"], input="x" * (length - 1) + "Z"
        )
    assert invoked.exit_code == code
    assert json.loads(invoked.stdout)["execution_status"] == status
    if code:
        council.assert_not_called()
    else:
        assert council.return_value.run.await_args.kwargs["task"].endswith("Z")


def test_unwritable_output_fails_before_calls(tmp_path):
    parent = tmp_path / "locked"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        with patch("llm_council.Council") as council:
            invoked = CliRunner().invoke(
                app, ["run", "planner", "task", "--json", "--output", str(parent / "result.json")]
            )
        assert invoked.exit_code == 1
        assert invoked.stdout == ""
        assert "Error" in invoked.stderr
        council.assert_not_called()
    finally:
        parent.chmod(0o700)


def test_output_replace_failure_is_nonzero_and_never_accepts_stale_result(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"success": true, "output": "stale"}')
    with (
        patch("llm_council.Council") as council,
        patch.object(cli.os, "replace", side_effect=OSError("synthetic disk failure")),
    ):
        council.return_value.run = AsyncMock(return_value=CouncilResult(success=True))
        invoked = CliRunner().invoke(
            app, ["run", "planner", "task", "--json", "--output", str(output)]
        )
    assert invoked.exit_code == 1
    assert invoked.stdout == ""
    assert "synthetic disk failure" in invoked.stderr
    assert json.loads(output.read_text())["output"] == "stale"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.json"]


def test_broken_pipe_on_error_envelope_is_handled(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-c", STUB_PROGRAM, "run", "planner", "--json"],
        env=stub_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.stdout.close()
    process.wait(timeout=10)
    stderr = process.stderr.read().decode()
    process.stderr.close()
    assert process.returncode == 1
    assert "Traceback" not in stderr


@pytest.mark.parametrize("action", ["slow-spawn", "slow-cleanup"])
def test_signal_during_stub_startup_or_cleanup_is_bounded(tmp_path, action):
    script = STUB_PROGRAM.replace(
        "child = await asyncio.create_subprocess_exec",
        'if action == "slow-spawn":\n            Path(os.environ["READY"]).write_text("starting")\n            await asyncio.sleep(60)\n        child = await asyncio.create_subprocess_exec',
    ).replace(
        "child.terminate()",
        'child.terminate()\n            if action == "slow-cleanup":\n                await asyncio.sleep(60)',
    )
    command = stub_command(tmp_path)
    command[2] = script
    process = subprocess.Popen(
        command,
        env=stub_env(tmp_path, action),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not (tmp_path / "ready").exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert (tmp_path / "ready").exists(), process.communicate(timeout=2)
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        stdout, _ = process.communicate(timeout=8)
        assert process.returncode == 143
        assert time.monotonic() - started < 7
        payload = json.loads(stdout)
        assert payload["execution_status"] == "cancelled"
        if action == "slow-cleanup":
            assert "cleanup incomplete" in payload["error"]
        else:
            assert payload["run_id"]
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


@pytest.mark.parametrize(
    "first_signal,second_signal,expected_exit",
    [(signal.SIGINT, signal.SIGTERM, 143), (signal.SIGTERM, signal.SIGINT, 130)],
)
def test_actual_second_signal_during_cleanup_exits_immediately(
    tmp_path, first_signal, second_signal, expected_exit
):
    script = STUB_PROGRAM.replace(
        'Path(os.environ["CLEANED"]).write_text("reaped")',
        'Path(os.environ["CLEANED"]).write_text("cleanup-entered")\n'
        "            await asyncio.sleep(60)",
    )
    command = stub_command(tmp_path)
    command[2] = script
    output = tmp_path / "result.json"
    command += ["--output", str(output)]
    process = subprocess.Popen(
        command, env=stub_env(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    child_pid = None
    try:
        ready = tmp_path / "ready"
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), process.communicate(timeout=2)
        child_pid = int(ready.read_text())
        process.send_signal(first_signal)
        cleanup = tmp_path / "cleaned"
        deadline = time.monotonic() + 2
        while not cleanup.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cleanup.read_text() == "cleanup-entered"
        assert process.poll() is None
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        started = time.monotonic()
        process.send_signal(second_signal)
        stdout, stderr = process.communicate(timeout=2)
        assert process.returncode == expected_exit
        assert time.monotonic() - started < 2
        assert stdout == ""
        assert stderr == "Second signal: cleanup incomplete.\n"
        assert not output.exists()
    finally:
        if process.poll() is None:
            process.kill()
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.communicate(timeout=5)
