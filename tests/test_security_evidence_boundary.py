"""Security analysis should respect executable control-flow evidence."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig
from llm_council.providers.base import GenerateResponse
from llm_council.subagents import get_effective_system_prompt, load_subagent


def test_security_draft_prompt_requires_control_flow_check():
    for name, mode in [("critic", "security"), ("red-team", None)]:
        prompt = get_effective_system_prompt(load_subagent(name), mode).lower()
        assert "short-circuit" in prompt
        assert "reachable" in prompt


@pytest.mark.asyncio
async def test_security_synthesis_requires_control_flow_check():
    orch = Orchestrator([], OrchestratorConfig(enable_artifacts=False))
    orch._task = "synthetic"
    orch._subagent_name = "critic"
    orch._resolved_mode = "security"
    orch._subagent_config = {}
    orch._schema = {"type": "object"}
    provider = MagicMock()
    provider.supports = AsyncMock(return_value=True)
    orch._candidate_providers_for_phase = AsyncMock(return_value=[("example", provider)])
    orch._call_provider = AsyncMock(return_value=GenerateResponse(text="{}"))
    await orch._run_synthesis({"example": "draft"}, "critique")
    system = orch._call_provider.await_args.args[2].messages[0].content.lower()
    assert "short-circuit" in system
    assert "reachable" in system
