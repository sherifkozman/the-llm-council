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
        (
            "claude",
            "claude-opus-5",
            "2.1.288 (Claude Code)",
            [
                "2.1.288 (Claude Code)",
                "2.1.289 (Claude Code)",
                "2.1.290 (Claude Code)",
                "2.1.291 (Claude Code)",
                "2.1.292 (Claude Code)",
            ],
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
