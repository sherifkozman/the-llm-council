"""Execute shipped caller examples offline, rather than asserting their text."""

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "skills/council/references/invocation-contract.md"


@pytest.fixture(params=["python", "sh"])
def run_example(request, tmp_path):
    language = request.param
    snippet = re.search(rf"^```{language}\n(.*?)^```", CONTRACT.read_text(), re.M | re.S)
    assert snippet is not None
    script = tmp_path / f"caller.{language}"
    script.write_text(snippet.group(1))
    binary = tmp_path / "council"
    binary.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import json, os, pathlib, sys
        if sys.argv[1:] == ["--version"]:
            print(os.environ["TEST_VERSION"])
            raise SystemExit(0)
        pathlib.Path(os.environ["TEST_CALL_LOG"]).write_text(json.dumps(sys.argv[1:]))
        result = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
        result.write_text(os.environ["TEST_PAYLOAD"])
        raise SystemExit(int(os.environ["TEST_EXIT"]))
    """)
    )
    binary.chmod(0o700)
    task = tmp_path / "task.txt"
    task.write_text("Synthetic task only")
    calls = tmp_path / "calls.json"

    def run(version, *, status="completed", success=True, code=0):
        env = dict(
            os.environ,
            COUNCIL_BIN=str(binary),
            COUNCIL_WORKDIR=str(tmp_path),
            COUNCIL_TASK_FILE=str(task),
            COUNCIL_REFERENCE_FILE=str(task),
            COUNCIL_PROVIDERS="mock",
            COUNCIL_MODELS="synthetic",
            TMPDIR=str(tmp_path),
            TEST_VERSION=version,
            TEST_CALL_LOG=str(calls),
            TEST_EXIT=str(code),
            TEST_PAYLOAD=json.dumps(
                {"execution_status": status, "success": success, "run_id": "synthetic-run"}
            ),
        )
        interpreter = sys.executable if language == "python" else "sh"
        result = subprocess.run(
            [interpreter, str(script)],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result, calls

    return run


@pytest.mark.parametrize("version", ["LLM Council v0.8.1", "LLM Council v0.8.2rc1", "unknown"])
def test_examples_reject_old_or_unverified_versions_before_run(run_example, version):
    result, calls = run_example(version)
    assert result.returncode != 0
    assert not calls.exists()


@pytest.mark.parametrize("version", ["LLM Council v0.8.2", "LLM Council v0.9.0"])
def test_examples_accept_supported_version_and_read_result(run_example, version):
    result, calls = run_example(version)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["execution_status"] == "completed"
    assert summary["run_id"] == "synthetic-run"
    argv = json.loads(calls.read_text())
    assert argv[argv.index("--reasoning-profile") + 1] == "default"
    assert argv[argv.index("--runtime-profile") + 1] == "default"


@pytest.mark.parametrize(
    "status,success,code",
    [
        ("degraded", True, 0),
        ("failed", False, 1),
        ("cancelled", False, 143),
    ],
)
def test_examples_read_terminal_results_even_on_nonzero_exit(run_example, status, success, code):
    result, _ = run_example("LLM Council v0.8.2", status=status, success=success, code=code)
    assert result.returncode == code, result.stderr
    assert json.loads(result.stdout)["execution_status"] == status


def test_examples_reject_result_exit_disagreement(run_example):
    result, _ = run_example("LLM Council v0.8.2", code=1)
    assert result.returncode != 0
    assert "Result and process outcome disagree" in result.stderr
