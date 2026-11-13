"""The critic may report no findings when none are supported."""

from llm_council.subagents import get_effective_system_prompt, load_subagent


def test_review_prompt_allows_no_evidence_based_findings():
    prompt = get_effective_system_prompt(load_subagent("critic"), "review")
    assert "MUST find at least one flaw or risk" not in prompt
    assert "no supported findings" in prompt
    assert "Do not fabricate" in prompt


def test_legacy_reviewer_entry_points_allow_no_supported_findings():
    for name in ("reviewer", "red-team"):
        prompt = get_effective_system_prompt(load_subagent(name))
        assert "MUST find at least one flaw or risk" not in prompt
        assert "no supported findings" in prompt
