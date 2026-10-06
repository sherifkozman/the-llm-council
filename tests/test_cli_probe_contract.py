"""Deep doctor must receive actual nonempty text before reporting reachability."""

import json

import pytest
from typer.testing import CliRunner

from llm_council.cli.main import app
from llm_council.providers.base import DoctorResult, GenerateResponse
from llm_council.providers.registry import ProviderRegistry


@pytest.fixture
def probe(tmp_path, monkeypatch, mock_provider):
    monkeypatch.setenv("LLM_COUNCIL_LOCK_DIR", str(tmp_path / "locks"))
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    monkeypatch.setattr(ProviderRegistry, "list_providers", lambda self: ["mock"])
    monkeypatch.setattr(
        ProviderRegistry, "get_provider", lambda self, name, **kwargs: mock_provider
    )

    def run(response, *, doctor_result=None, deep=True):
        async def generate(request):
            assert request.stream is False
            return response

        monkeypatch.setattr(mock_provider, "generate", generate)
        if doctor_result is not None:

            async def doctor():
                return doctor_result

            monkeypatch.setattr(mock_provider, "doctor", doctor)
        result = CliRunner().invoke(
            app,
            [
                "--config",
                str(config),
                "doctor",
                "--provider",
                "mock",
                "--json",
                *(["--deep"] if deep else []),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert len(payload["providers"]) == 1
        return payload["providers"][0]

    return run


@pytest.mark.parametrize("text", [None, "", " \t\n"])
def test_empty_deep_probe_has_typed_failure_not_reachability(probe, text):
    result = probe(GenerateResponse(text=text))
    assert result["name"] == "mock"
    assert result["ok"] is True
    assert result["probe_ok"] is False
    assert "empty_response" in result["probe_message"]
    assert "probe" in result["probe_message"]
    assert result["probe_latency_ms"] is None


def test_deep_probe_rejects_unexpected_stream_without_stringifying(probe):
    async def stream():
        yield GenerateResponse(text="OK")

    result = probe(stream())
    assert result["probe_ok"] is False
    assert "GenerateResponse" in result["probe_message"]
    assert "async_generator object" not in result["probe_message"]
    assert result["probe_latency_ms"] is None


def test_nonempty_deep_probe_reports_returned_text(probe):
    result = probe(GenerateResponse(text="  OK\n"))
    assert result["probe_ok"] is True
    assert result["probe_message"] == "OK"
    assert result["probe_latency_ms"] >= 0


@pytest.mark.parametrize(
    "deep,base_ok,text,probe_ok",
    [
        (False, True, "OK", None),
        (True, True, "OK", True),
        (True, True, "", False),
        (True, False, "OK", False),
    ],
)
def test_doctor_retains_native_details_without_changing_status(
    probe, tmp_path, deep, base_ok, text, probe_ok
):
    details = {"cli_path": str(tmp_path / "native" / "codex"), "cli_version": "codex-cli 0.153.3"}
    result = probe(
        GenerateResponse(text=text),
        deep=deep,
        doctor_result=DoctorResult(
            ok=base_ok, message="Original base diagnostic", latency_ms=12.5, details=details
        ),
    )
    assert result["details"] == details
    assert result["ok"] is base_ok
    assert result["message"] == "Original base diagnostic"
    assert result["latency_ms"] == 12.5
    if deep:
        assert result["probe_ok"] is probe_ok
        if not base_ok:
            assert result["probe_message"] == "Skipped because base doctor failed"
        elif probe_ok:
            assert result["probe_message"] == "OK"
        else:
            assert "empty_response" in result["probe_message"]
    else:
        assert "probe_ok" not in result


@pytest.mark.parametrize("details", [None, {}])
@pytest.mark.parametrize("deep", [False, True])
def test_doctor_omits_empty_optional_details(probe, details, deep):
    result = probe(
        GenerateResponse(text="OK"),
        deep=deep,
        doctor_result=DoctorResult(ok=True, details=details),
    )
    assert "details" not in result
    assert result["ok"] is True
    if deep:
        assert result["probe_ok"] is True
