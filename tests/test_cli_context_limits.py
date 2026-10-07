"""Offline CLI regressions for character limits and request timeout boundaries."""

import json

import pytest
from typer.testing import CliRunner

from llm_council.cli.main import app
from llm_council.providers.registry import ProviderRegistry


@pytest.fixture
def invoke(tmp_path, monkeypatch, mock_provider, valid_json_response):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LLM_COUNCIL_LOCK_DIR", str(tmp_path / "locks"))
    config = tmp_path / "config.yaml"
    config.write_text("defaults:\n  providers: [mock]\n  max_retries: 1\n")
    response = json.loads(valid_json_response)
    response["implementation_title"] = "Synthetic implementation"
    mock_provider._response_text = json.dumps(response)
    monkeypatch.setattr(
        ProviderRegistry, "get_provider", lambda self, name, **kwargs: mock_provider
    )

    def run(*args):
        return CliRunner().invoke(
            app,
            [
                "--config",
                str(config),
                "run",
                "drafter",
                "Review synthetic evidence",
                "--mode",
                "impl",
                "--json",
                "--no-artifacts",
                *args,
            ],
        )

    return run


@pytest.mark.parametrize("length", [49_999, 50_000, 50_001])
@pytest.mark.parametrize("output_file", [False, True])
def test_file_character_boundary_and_json_diagnostics(invoke, tmp_path, length, output_file):
    reference = tmp_path / "reference.txt"
    # Multibyte UTF-8 distinguishes character limits from byte limits.
    reference.write_text("\u00e9" * length, encoding="utf-8")
    output = tmp_path / "result.json"
    args = ["--output", str(output)] if output_file else []
    result = invoke("--files", str(reference), *args)
    assert result.exit_code == 0, result.output
    if output_file:
        assert result.stdout == ""
    payload = json.loads(output.read_text() if output_file else result.stdout)
    truncated = length > 50_000
    assert payload["success"] is True
    assert payload["execution_status"] == ("degraded" if truncated else "completed")
    prepared = payload["execution_plan"]["context_preparation"]
    assert prepared["files"] == [
        {
            "path": str(reference),
            "status": "included",
            "original_chars": length,
            "retained_chars": min(length, 50_000),
            "truncated": truncated,
        }
    ]
    assert bool(prepared["warnings"]) is truncated
    assert ("Truncated" in result.stderr) is truncated


@pytest.mark.parametrize("enable_degradation", [True, False])
def test_truncation_warnings_are_in_degradation_report(invoke, tmp_path, enable_degradation):
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  providers: [mock]\n  max_retries: 1\n"
        f"  enable_degradation: {str(enable_degradation).lower()}\n"
    )
    reference = tmp_path / "reference.txt"
    reference.write_text("x" * 50_001)
    result = invoke("--files", str(reference))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    warnings = payload["execution_plan"]["context_preparation"]["warnings"]
    assert warnings
    assert payload["execution_status"] == "degraded"
    assert payload["provider_errors"] is None
    assert payload["degradation_report"]["context_warnings"] == [
        {
            "kind": "file_truncated",
            "path": str(reference),
            "original_chars": 50_001,
            "retained_chars": 50_000,
        }
    ]
    assert not payload["degradation_report"].get("failures")


@pytest.mark.parametrize("total", [199_999, 200_000, 200_001])
def test_total_character_cap_fails_before_provider_work(invoke, tmp_path, mock_provider, total):
    paths = []
    remaining = total
    while remaining:
        path = tmp_path / f"reference-{len(paths)}.txt"
        size = min(50_000, remaining)
        path.write_text("\u00e9" * size, encoding="utf-8")
        paths.append(str(path))
        remaining -= size
    result = invoke("--files", ",".join(paths))
    payload = json.loads(result.stdout)
    if total > 200_000:
        assert result.exit_code == 1
        assert payload["success"] is False
        assert payload["execution_status"] == "failed"
        assert "200000 character total limit" in payload["error"]
        assert paths[-1] in payload["error"]
        assert mock_provider.call_count == 0
    else:
        assert result.exit_code == 0, result.output
        assert payload["success"] is True
        warnings = (payload["degradation_report"] or {}).get("context_warnings", [])
        assert not any(item["kind"] == "file_truncated" for item in warnings)
        files = payload["execution_plan"]["context_preparation"]["files"]
        assert [item["path"] for item in files] == paths
        assert sum(item["retained_chars"] for item in files) == total
        assert all(not item["truncated"] for item in files)


def test_total_cap_counts_retained_not_original_characters(invoke, tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"reference-{index}.txt"
        path.write_text("x" * 50_001)
        paths.extend(["--files", str(path)])
    result = invoke(*paths, "--dry-run")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["execution_status"] == "not_executed"
    metadata = payload["context_metadata"]
    assert metadata["total_retained_chars"] == 200_000
    assert metadata["limits"] == {"max_total_chars": 200_000, "max_per_file_chars": 50_000}
    assert len(metadata["warnings"]) == 4
    assert all(item["truncated"] for item in metadata["files"])


@pytest.mark.parametrize("timeout", [10, 600, 601, 3600])
@pytest.mark.parametrize("dry_run", [False, True])
def test_timeout_accepts_finite_boundary_through_real_cli(invoke, timeout, dry_run):
    result = invoke("--timeout", str(timeout), *(["--dry-run"] if dry_run else []))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["execution_status"] == ("not_executed" if dry_run else "completed")
    if dry_run:
        assert payload["timeout"] == timeout


@pytest.mark.parametrize("timeout", [9, 3601])
def test_timeout_outside_range_is_failed_json_without_provider_work(invoke, mock_provider, timeout):
    result = invoke("--timeout", str(timeout))
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["execution_status"] == "failed"
    assert "10 and 3600" in payload["error"]
    assert mock_provider.call_count == 0


def test_default_timeout_remains_120(invoke):
    result = invoke("--dry-run")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["timeout"] == 120


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("default", {"draft": 3599.0, "critique": 3599.0, "synthesis": 3599.0}),
        ("bounded", {"draft": 45.0, "critique": 30.0, "synthesis": 45.0}),
    ],
)
def test_config_long_timeout_preserves_effective_runtime_caps(invoke, tmp_path, profile, expected):
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  providers: [openrouter]\n  timeout: 3600\n  max_retries: 1\n"
    )
    result = invoke("--runtime-profile", profile)
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)["execution_plan"]
    assert plan["provider_request_timeouts_by_provider"]["openrouter"] == expected
