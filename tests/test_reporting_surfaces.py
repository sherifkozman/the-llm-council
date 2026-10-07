"""Offline regressions for reporting and artifact occurrence identity."""

import importlib
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from llm_council import __version__
from llm_council.engine.orchestrator import CouncilResult, runtime_identity
from llm_council.storage.artifacts import ArtifactStore, ArtifactType

cli = importlib.import_module("llm_council.cli.main")


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    monkeypatch.setattr(cli, "_load_config_defaults", lambda: {})
    monkeypatch.setattr(cli, "_load_provider_configs", lambda: {})


def degraded_result():
    return CouncilResult(
        success=True,
        output={"verdict": "approve"},
        drafts={"a": "draft", "b": ""},
        critique="",
        provider_errors={"b": "Synthetic provider failure"},
        validation_errors=["Synthetic fallback validation"],
        degradation_report={"fallbacks_used": ["draft"]},
        execution_plan={
            "selected_providers": ["a", "b"],
            "required_phases": ["draft", "critique", "synthesis"],
            "persistence": {"errors": [{"operation": "complete_run", "error_type": "OSError"}]},
            "context_preparation": {
                "coverage": [
                    {
                        "path": "synthetic.txt",
                        "source_chars": 100,
                        "delivered_source_chars": 50,
                        "complete": False,
                        "selection_applied": True,
                    }
                ]
            },
            "warnings": ["Synthetic evidence loss"],
        },
    )


@pytest.mark.parametrize("args", [["--json"], ["--format", "markdown"], []])
def test_degraded_diagnostics_visible_without_verbose(args):
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=degraded_result())
        invoked = CliRunner().invoke(cli.app, ["run", "critic", "task", *args])
    assert invoked.exit_code == 0
    text = invoked.stdout
    assert "degraded" in text.lower()
    assert "Council Result: SUCCESS" not in text
    for detail in (
        "Synthetic provider failure",
        "Synthetic fallback validation",
        "fallbacks_used",
        "complete_run",
        "OSError",
        "delivered_source_chars",
        "Synthetic evidence loss",
    ):
        assert detail in text


@pytest.mark.parametrize("status", ["completed", "degraded", "failed", "cancelled"])
def test_markdown_status_is_execution_status_not_success(status):
    result = CouncilResult(success=True).model_copy(update={"execution_status": status})
    text = cli._render_result_markdown(result, include_metrics=False)
    assert text.startswith(f"# Council Result: {status.upper()}")


@pytest.mark.parametrize(
    "status,success,code",
    [
        ("completed", True, 0),
        ("degraded", True, 0),
        ("failed", False, 1),
        ("cancelled", False, 130),
    ],
)
@pytest.mark.parametrize("args", [["--json"], ["--format", "markdown"], []])
def test_status_parity_preserves_exit_policy(status, success, code, args):
    result = CouncilResult(success=success).model_copy(
        update={
            "execution_status": status,
            "signal_number": 2 if status == "cancelled" else None,
        }
    )
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=result)
        invoked = CliRunner().invoke(cli.app, ["run", "critic", "task", *args])
    assert invoked.exit_code == code
    if "--json" in args:
        assert json.loads(invoked.stdout)["execution_status"] == status
    else:
        assert f"Council Result: {status.upper()}" in invoked.stdout


@pytest.mark.parametrize("output_file", [False, True])
def test_early_failure_respects_markdown_format_and_redacts(monkeypatch, tmp_path, output_file):
    monkeypatch.setenv("SYNTHETIC_API_KEY", "test-private-credential-value")
    target = tmp_path / "report.md"
    args = ["--output", str(target)] if output_file else []
    with patch(
        "llm_council.Council", side_effect=ValueError("Failure test-private-credential-value")
    ):
        invoked = CliRunner().invoke(
            cli.app, ["run", "critic", "task", "--format", "markdown", *args]
        )
    assert invoked.exit_code == 1
    text = target.read_text() if output_file else invoked.stdout
    assert "# Council Result: FAILED" in text
    assert "Failure" in text
    assert "test-private-credential-value" not in text + invoked.stderr


def test_runtime_identity_on_failed_json_before_provider_calls():
    invoked = CliRunner().invoke(cli.app, ["run", "critic", "--json"])
    assert invoked.exit_code == 1
    identity = json.loads(invoked.stdout)["execution_plan"]["runtime_identity"]
    assert identity == runtime_identity()
    assert identity["council_version"] == __version__


def test_runtime_identity_reports_actual_import_and_interpreter():
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=CouncilResult(success=True))
        invoked = CliRunner().invoke(cli.app, ["run", "critic", "task", "--json"])
    assert invoked.exit_code == 0
    identity = json.loads(invoked.stdout)["execution_plan"]["runtime_identity"]
    assert identity == runtime_identity()
    assert identity["council_version"] == __version__
    assert identity["python_executable"] == sys.executable
    assert Path(identity["package_path"]).resolve() == Path(cli.__file__).resolve().parents[1]
    assert "PATH" not in identity and "environment" not in identity and "argv" not in identity


@pytest.mark.parametrize("args", [["--json"], ["--format", "markdown"], ["--verbose"]])
def test_diagnostic_secrets_are_redacted_on_all_surfaces(monkeypatch, args):
    monkeypatch.setenv("SYNTHETIC_API_KEY", "test-private-credential-value")
    result = degraded_result()
    result.provider_errors = {
        "b": "Failure test-private-credential-value Authorization: Bearer hidden-token"
    }
    result.execution_plan["persistence"]["errors"][0]["token"] = "another-private-value"
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=result)
        invoked = CliRunner().invoke(cli.app, ["run", "critic", "task", *args])
    assert invoked.exit_code == 0
    assert "test-private-credential-value" not in invoked.output
    assert "hidden-token" not in invoked.output
    assert "another-private-value" not in invoked.output
    assert "Failure" in invoked.output


@pytest.mark.parametrize(
    "message",
    [
        'Failure {"api_key": "synthetic-private-value"}',
        "Failure Authorization: Basic synthetic-private-value",
        "Failure password='synthetic-private-value'",
    ],
)
def test_credential_literals_in_diagnostic_strings_are_redacted(message):
    redacted = cli._safe_diagnostic(message)
    assert "synthetic-private-value" not in redacted
    assert "Failure" in redacted


def test_diagnostic_redaction_preserves_token_metrics_and_limits(monkeypatch):
    monkeypatch.setenv("MAX_TOKENS", "16384")
    monkeypatch.setenv("KEYCHAIN", "1")
    metrics = {
        "tokens": 900,
        "max_tokens": 16384,
        "estimated_input_tokens": 1200,
        "budget_tokens": 32000,
        "input_tokens": 1100,
        "output_tokens": 400,
        "total_tokens": 1500,
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "summary_tokens": 0,
        "input_token": 7,
        "KEYCHAIN": 1,
        "key_count": 2,
        "note": "max_tokens=16384; estimated_input_tokens=1200; KEYCHAIN=1",
    }
    assert cli._safe_diagnostic({"limits": [metrics]}) == {"limits": [metrics]}
    encoded = "HTTP400:" + json.dumps(metrics)
    assert cli._safe_diagnostic(encoded) == encoded


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "openai_api_key",
        "token",
        "access_token",
        "refresh_token",
        "service_auth_token",
        "client_secret",
        "password",
        "Authorization",
        "Cookie",
        "credentials",
        "AWS_SECRET_ACCESS_KEY",
        "PRIVATE_KEY",
    ],
)
def test_diagnostic_credential_fields_still_redacted(key):
    assert cli._safe_diagnostic({key: "synthetic-private-value"}) == {key: "[REDACTED]"}
    for encoded in (
        "HTTP400:" + json.dumps({key: "synthetic-private-value"}),
        f"Failure {key}='synthetic-private-value'",
    ):
        assert "synthetic-private-value" not in cli._safe_diagnostic(encoded)


@pytest.mark.parametrize("key", ["AWS_SECRET_ACCESS_KEY", "PRIVATE_KEY"])
def test_diagnostic_redacts_secret_key_environment_values(monkeypatch, key):
    monkeypatch.setenv(key, "SYNTHETIC_PRIVATE_KEY_329")
    assert cli._safe_diagnostic("Failure SYNTHETIC_PRIVATE_KEY_329") == "Failure [REDACTED]"


@pytest.mark.parametrize("args", [["--json"], ["--format", "markdown"], []])
def test_http_error_string_redacts_prefixed_credentials_without_environment(args):
    result = degraded_result()
    result.provider_errors = {
        "b": 'HTTP400:{"access_token":"SYNTHETIC_SECRET_127",'
        '"refresh_token":"SYNTHETIC_SECRET_128","client_secret":"SYNTHETIC_SECRET_129",'
        '"max_tokens":4096,"estimated_input_tokens":1200}'
    }
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=result)
        invoked = CliRunner().invoke(cli.app, ["run", "critic", "task", *args])
    assert invoked.exit_code == 0
    assert "SYNTHETIC_SECRET_" not in invoked.output
    assert "HTTP400" in invoked.output
    assert "4096" in invoked.output and "1200" in invoked.output


def test_json_keeps_budget_proof_and_raw_schema_output():
    output = {
        "token": "schema-output-value",
        "max_tokens": 4096,
        "schema": {"properties": {"token": {"type": "string"}}},
    }
    result = CouncilResult(
        success=True,
        output=output,
        execution_plan={
            "required_phases": [],
            "phase_token_budgets": {"draft": {"max_tokens": 4096, "estimated_input_tokens": 1200}},
        },
    )
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=result)
        invoked = CliRunner().invoke(cli.app, ["run", "critic", "task", "--json"])
    assert invoked.exit_code == 0
    payload = json.loads(invoked.stdout)
    assert payload["output"] == output
    assert payload["execution_plan"]["phase_token_budgets"] == {
        "draft": {"max_tokens": 4096, "estimated_input_tokens": 1200},
    }
    assert "runtime_identity" not in result.execution_plan


@pytest.mark.parametrize(
    "model", ["codex:gpt-5", "vertex:gemini-pro", "openrouter:org/model:free", "claude-code:opus"]
)
def test_provider_prefixed_model_rejected_before_council(model):
    with patch("llm_council.Council", side_effect=AssertionError("must not construct")):
        invoked = CliRunner().invoke(
            cli.app, ["run", "critic", "task", "--models", model, "--json"]
        )
    assert invoked.exit_code == 1
    error = json.loads(invoked.stdout)["error"]
    assert "--models" in error and "positional" in error
    assert "must not construct" not in error


def test_openrouter_colon_suffix_is_a_literal_model_id():
    invoked = CliRunner().invoke(
        cli.app,
        [
            "run",
            "critic",
            "task",
            "--models",
            "org/model:free,org/second:extended",
            "--json",
            "--dry-run",
        ],
    )
    assert invoked.exit_code == 0
    assert json.loads(invoked.stdout)["models"] == ["org/model:free", "org/second:extended"]


def test_artifact_identical_content_preserves_phase_and_provider_occurrences(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "ledger.db")
    run = store.create_run("synthetic", "task")
    a = store.store_artifact(run.run_id, "same content", ArtifactType.DRAFT)
    b = store.store_artifact(run.run_id, "same content", ArtifactType.DRAFT, force_new=True)
    final = store.store_artifact(run.run_id, "same content", ArtifactType.SYNTHESIS)
    again = store.store_artifact(run.run_id, "same content", ArtifactType.SYNTHESIS)
    assert len({a.artifact_id, b.artifact_id, final.artifact_id}) == 3
    assert again.artifact_id == final.artifact_id
    assert a.content_hash == b.content_hash == final.content_hash
    assert a.file_path == b.file_path == final.file_path
    restored = ArtifactStore(store.artifact_dir, store.db_path).get_run_artifacts(run.run_id)
    assert sorted(item.artifact_type for item in restored) == ["draft", "draft", "synthesis"]
    assert all(store.get_artifact_content(item.artifact_id) == "same content" for item in restored)


def test_artifact_schema_unchanged_and_legacy_uuid_file_readable(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "ledger.db")
    run = store.create_run("synthetic", "task")
    legacy = store.store_artifact(run.run_id, "legacy", ArtifactType.DRAFT)
    legacy_path = store.artifact_dir / f"{legacy.artifact_id}.txt"
    Path(legacy.file_path).rename(legacy_path)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE artifacts SET file_path = ? WHERE artifact_id = ?",
            (str(legacy_path), legacy.artifact_id),
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(artifacts)")}
    assert columns == {
        "artifact_id",
        "run_id",
        "artifact_type",
        "content_hash",
        "byte_size",
        "token_estimate",
        "file_path",
        "processing_state",
        "created_at",
        "summary",
        "summary_tokens",
    }
    reopened = ArtifactStore(store.artifact_dir, store.db_path)
    old = reopened.get_run_artifacts(run.run_id)[0]
    assert old.artifact_id == legacy.artifact_id
    assert reopened.get_artifact_content(old.artifact_id) == "legacy"
    new = reopened.store_artifact(run.run_id, "legacy", ArtifactType.DRAFT, force_new=True)
    assert new.artifact_id != old.artifact_id


def test_force_new_keeps_content_dedup_but_creates_new_occurrence(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "ledger.db")
    run = store.create_run("synthetic", "task")
    first = store.store_artifact(run.run_id, "same", ArtifactType.DRAFT)
    second = store.store_artifact(run.run_id, "same", ArtifactType.DRAFT, force_new=True)
    assert first.artifact_id != second.artifact_id
    assert first.file_path == second.file_path


@pytest.mark.parametrize(
    "artifact_type,force_new",
    [(ArtifactType.DRAFT, True), (ArtifactType.SYNTHESIS, False)],
)
def test_existing_blob_occurrence_survives_windows_rename_semantics(
    tmp_path, monkeypatch, artifact_type, force_new
):
    original_rename = Path.rename

    def windows_rename(source, target):
        if Path(target).exists():
            raise FileExistsError("Windows rename cannot overwrite an existing destination")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", windows_rename)
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "ledger.db")
    run = store.create_run("synthetic", "task")
    first = store.store_artifact(run.run_id, "same content", ArtifactType.DRAFT)
    second = store.store_artifact(run.run_id, "same content", artifact_type, force_new=force_new)
    assert first.artifact_id != second.artifact_id
    assert first.file_path == second.file_path
    restored = store.get_run_artifacts(run.run_id)
    assert {item.artifact_id for item in restored} == {first.artifact_id, second.artifact_id}
    assert all(store.get_artifact_content(item.artifact_id) == "same content" for item in restored)
    assert list(store.artifact_dir.iterdir()) == [Path(first.file_path)]
