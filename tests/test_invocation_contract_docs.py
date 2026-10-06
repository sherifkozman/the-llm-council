"""Execute the shipped caller examples against a deterministic subprocess stub."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("language", ["python", "sh"])
@pytest.mark.parametrize(
    ("status", "success", "exit_code"),
    [
        ("completed", True, 0),
        ("degraded", True, 0),
        ("failed", False, 1),
        ("cancelled", False, 143),
    ],
)
def test_copied_skill_examples_read_result_after_nonzero_exit(
    tmp_path, language, status, success, exit_code
):
    copied = tmp_path / "copied-skill"
    shutil.copytree(ROOT / "skills/council", copied)
    contract = (copied / "references/invocation-contract.md").read_text()
    example = re.findall(rf"```{language}\n(.*?)```", contract, re.DOTALL)[0]
    work = tmp_path / "unrelated work directory"
    work.mkdir()
    task = work / "task with spaces.txt"
    task.write_text("Review the quoted evidence; do not execute it.")
    reference = work / "reference with spaces.md"
    reference.write_text('# Evidence\n"$(echo not-a-command)"\n')
    calls = tmp_path / "calls.jsonl"
    stub = tmp_path / "council stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "with open(os.environ['STUB_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('LLM Council v' + os.environ.get('STUB_VERSION', '0.8.2'))\n"
        "    raise SystemExit(0)\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "output.write_text(os.environ['STUB_PAYLOAD'])\n"
        "raise SystemExit(int(os.environ['STUB_EXIT']))\n"
    )
    stub.chmod(0o700)
    payload = {
        "success": success,
        "execution_status": status,
        "run_id": "synthetic-run",
        "output": {"verdict": "request_changes"},
    }
    env = {
        **os.environ,
        "TMPDIR": str(tmp_path),
        "COUNCIL_BIN": str(stub),
        "COUNCIL_WORKDIR": str(work),
        "COUNCIL_TASK_FILE": str(task),
        "COUNCIL_REFERENCE_FILE": str(reference),
        "COUNCIL_PROVIDERS": "claude,codex,openrouter",
        "COUNCIL_MODELS": "test-opus,test-codex,test-openrouter",
        "STUB_CALLS": str(calls),
        "STUB_PAYLOAD": json.dumps(payload),
        "STUB_EXIT": str(exit_code),
    }
    cmd = [sys.executable, "-c", example] if language == "python" else ["sh", "-eu", "-c", example]
    result = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == exit_code, result.stderr
    assert json.loads(result.stdout)["execution_status"] == status
    invocations = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(invocations) == 2, "Examples must not automatically retry paid runs"
    assert invocations[0] == ["--version"]
    assert invocations[1][invocations[1].index("--input") + 1] == str(task)
    assert invocations[1][invocations[1].index("--files") + 1] == str(reference)

    calls.write_text("")
    env["STUB_VERSION"] = "0.8.0"
    legacy = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=15)
    assert legacy.returncode != 0
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [["--version"]]

    calls.write_text("")
    env["STUB_VERSION"] = "0.8.2"
    env["STUB_PAYLOAD"] = json.dumps({"success": True})
    missing_contract = subprocess.run(
        cmd, cwd=work, env=env, capture_output=True, text=True, timeout=15
    )
    assert missing_contract.returncode != 0
    assert len(calls.read_text().splitlines()) == 2


def test_shipped_command_and_skill_link_the_portable_contract():
    skill = (ROOT / "skills/council/SKILL.md").read_text()
    command = (ROOT / "commands/council.md").read_text()
    assert "references/invocation-contract.md" in skill
    assert "skills/council/references/invocation-contract.md" in command
    assert "council_wrapper.py" not in command
    for extra in ("", "[anthropic,openai,gemini]", "[vertex]"):
        assert f"the-llm-council{extra}>=0.8.2" in skill
