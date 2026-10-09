"""Native compatibility metadata without changing reasoning compilation."""

import pytest

from llm_council.providers.base import GenerateRequest, ReasoningConfig
from llm_council.providers.compiler import compile_request_for_provider


@pytest.mark.parametrize(
    "provider,model,baseline,supported",
    [
        (
            "codex",
            "gpt-5.4",
            "codex-cli 0.149.1",
            ["codex-cli 0.149.1", "codex-cli 0.153.3", "codex-cli 0.160.1"],
        ),
    ],
)
@pytest.mark.parametrize("effort", ["high", "low"])
def test_native_metadata_distinguishes_baseline_from_supported_versions(
    provider, model, baseline, supported, effort
):
    reasoning = ReasoningConfig(enabled=True, effort=effort)
    result = compile_request_for_provider(
        provider, GenerateRequest(prompt="test", model=model, reasoning=reasoning)
    )
    metadata = result.to_dict()["reasoning_control"]
    assert result.request.reasoning == reasoning
    assert metadata["requires_cli_version"] == baseline
    assert metadata["cli_version_requirement"] == "verified_versions_only"
    assert metadata["verified_baseline_cli_version"] == baseline
    assert metadata["supported_cli_versions"] == supported
    decision = next(item for item in result.decisions if item.option == "reasoning.effort")
    assert "verified versions" in decision.detail
    assert supported[-1] in decision.detail


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
