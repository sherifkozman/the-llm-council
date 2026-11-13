"""Planner instructions must distinguish requirements from proposed assumptions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig
from llm_council.providers.base import GenerateResponse
from llm_council.subagents import get_effective_system_prompt, load_subagent


@pytest.mark.parametrize(
    "name,mode", [("planner", "plan"), ("planner", "assess"), ("assessor", None)]
)
def test_planner_requires_attributed_requirements_and_proposals(name, mode):
    prompt = get_effective_system_prompt(load_subagent(name), mode).lower()
    assert "explicitly supported" in prompt
    assert "proposed" in prompt
    assert "do not label" in prompt
    assert "drafts and critiques are not evidence" in prompt


def test_planner_uses_proportionate_scope_without_inventing_blockers():
    prompt = get_effective_system_prompt(load_subagent("planner"), "plan").lower()
    assert "proportionate" in prompt
    assert "empty repository" in prompt
    assert "brief phase names" in prompt
    assert "do not invent approval gates" in prompt
    assert "preserve safety and approval steps" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["planner", "assessor"])
async def test_planner_synthesis_retains_requirement_boundary(name):
    orch = Orchestrator([], OrchestratorConfig(enable_artifacts=False))
    orch._task = "synthetic"
    orch._subagent_name = name
    orch._subagent_config = {}
    orch._schema = {"type": "object"}
    provider = MagicMock()
    provider.supports = AsyncMock(return_value=True)
    orch._candidate_providers_for_phase = AsyncMock(return_value=[("example", provider)])
    orch._call_provider = AsyncMock(return_value=GenerateResponse(text="{}"))
    await orch._run_synthesis({"example": "synthetic draft"}, "synthetic critique")
    request = orch._call_provider.await_args.args[2]
    system = request.messages[0].content.lower()
    assert "explicitly supported" in system
    assert "do not label" in system
    assert "drafts and critiques are not evidence" in system
    assert "proportionate" in system
    assert "empty repository" in system
    assert "brief phase names" in system
    assert "do not invent approval gates" in system
    assert "safety and approval steps required by the user or evidence" in system
