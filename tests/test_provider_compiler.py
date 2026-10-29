"""Native compatibility metadata without changing reasoning compilation."""

import pytest

from llm_council.providers.base import GenerateRequest, ReasoningConfig
from llm_council.providers.compiler import compile_request_for_provider


@pytest.mark.parametrize("effort", ["high", "low"])
def test_codex_reasoning_metadata_reports_record_based_admission(effort):
    reasoning = ReasoningConfig(enabled=True, effort=effort)
    result = compile_request_for_provider(
        "codex", GenerateRequest(prompt="test", model="gpt-5.4", reasoning=reasoning)
    )
    metadata = result.to_dict()["reasoning_control"]
    assert result.request.reasoning == reasoning
    assert metadata == {
        "status": "requested",
        "effort": effort,
        "cli_version_requirement": "compatibility_record",
    }
    decision = next(item for item in result.decisions if item.option == "reasoning.effort")
    assert "matching compatibility record" in decision.detail
    assert "not verified until runtime" in decision.detail


@pytest.mark.parametrize("effort", ["high", "low"])
def test_claude_reasoning_metadata_uses_capabilities_not_builds(effort):
    reasoning = ReasoningConfig(enabled=True, effort=effort)
    result = compile_request_for_provider(
        "claude", GenerateRequest(prompt="test", model="claude-opus-5", reasoning=reasoning)
    )
    assert result.request.reasoning == reasoning
    assert result.to_dict()["reasoning_control"] == {
        "status": "requested",
        "effort": effort,
        "cli_version_requirement": "capability_based",
        "required_cli_flags": ["--effort"],
    }
    decision = next(item for item in result.decisions if item.option == "reasoning.effort")
    assert "checks required CLI flags" in decision.detail
    assert "not behavioral verification" in decision.detail
