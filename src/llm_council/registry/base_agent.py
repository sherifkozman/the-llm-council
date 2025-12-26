"""
Thin Base Agent Abstraction.

Provides a reusable wrapper for model + tools + loop that reduces
per-role boilerplate and enables consistent agent behavior.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from llm_council.registry.tool_registry import Tool, get_tool_registry

logger = logging.getLogger(__name__)


# Council Deliberation Protocol - injected into all agent prompts
COUNCIL_PROTOCOL = """
## Council Deliberation Protocol

### 1. Equal Standing
All council members have equal authority regardless of speaking order.
The synthesizer evaluates arguments on merit, not position.

### 2. Constructive Dissent (REQUIRED)
You MUST challenge assumptions and express unorthodox opinions
when grounded in logic, evidence, and facts.
- Do not simply agree with previous agents
- If you see a flaw, state it clearly with reasoning
- Groupthink is the enemy of good reasoning

### 3. Pass When Empty
If you have nothing substantive to add beyond what's been stated:
- Respond with: **PASS**
- Silence is better than redundancy

### 4. Collaborative Rivalry
Aim to produce the winning argument through merit:
- Accuracy, evidence, and clarity are rewarded
- Attack ideas, not agents

### 5. Evidence Required
All claims require supporting reasoning.
Cite sources, examples, or logical derivation.
"""


@dataclass
class AgentResult:
    """Result from an agent run."""

    content: str
    passed: bool = False  # True if agent responded with PASS
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)  # token counts
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        """Check if result has substantive content (not PASS)."""
        return not self.passed and bool(self.content.strip())


@dataclass
class AgentContext:
    """Context passed to agents during council deliberation."""

    task: str
    previous_drafts: dict[str, str] = field(default_factory=dict)
    critique: str | None = None
    round_number: int = 1
    mode: str | None = None  # Agent mode (e.g., 'impl', 'review')
    extra: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Thin wrapper for model + tools + loop.

    This base class encapsulates:
    - Model selection
    - Tool list management
    - System prompt with Council Protocol
    - PASS detection
    - Basic agent loop

    Subclasses implement role-specific behavior.
    """

    # Role-specific prompt additions (override in subclasses)
    ROLE_PROMPT: str = ""
    ROLE_NAME: str = "agent"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: list[Tool] | None = None,
        max_iterations: int = 10,
        include_protocol: bool = True,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'claude-opus-4-5')
            system_prompt: Role-specific system prompt
            tools: List of tools available to this agent
            max_iterations: Maximum tool-call loop iterations
            include_protocol: Whether to include Council Protocol in prompt
        """
        self.model = model
        self._base_prompt = system_prompt
        self._tools = tools or []
        self.max_iterations = max_iterations
        self.include_protocol = include_protocol

    @property
    def system_prompt(self) -> str:
        """Build full system prompt with protocol."""
        parts = [self._base_prompt]

        if self.include_protocol:
            parts.append(COUNCIL_PROTOCOL)

        if self.ROLE_PROMPT:
            parts.append(self.ROLE_PROMPT)

        return "\n\n".join(parts)

    @property
    def tools(self) -> list[Tool]:
        """Get tools available to this agent."""
        return self._tools

    def add_tools_from_registry(self, role: str | None = None) -> None:
        """Load tools from registry for this agent's role."""
        registry = get_tool_registry()
        role_tools = registry.get_tools_for_role(role or self.ROLE_NAME)
        self._tools.extend(role_tools)

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentResult:
        """
        Execute the agent on the given context.

        Args:
            context: Agent context with task and previous outputs

        Returns:
            AgentResult with content and metadata
        """
        pass

    def _detect_pass(self, content: str) -> bool:
        """Detect if agent response is a PASS."""
        stripped = content.strip().upper()
        return stripped == "PASS" or stripped.startswith("**PASS**")


class DrafterAgent(BaseAgent):
    """Agent role: Drafter - generates solutions."""

    ROLE_NAME = "drafter"
    ROLE_PROMPT = """
## Your Role: Drafter
Propose bold, well-reasoned solutions. Make clear recommendations.
Present your top choice with alternatives noted. Don't hedge excessively.
"""

    async def run(self, context: AgentContext) -> AgentResult:
        """Generate a draft solution."""
        # Implementation would call the model here
        # For now, return placeholder
        return AgentResult(
            content="Draft implementation placeholder",
            passed=False,
            metadata={"role": "drafter", "mode": context.mode},
        )


class CriticAgent(BaseAgent):
    """Agent role: Critic - evaluates and challenges."""

    ROLE_NAME = "critic"
    ROLE_PROMPT = """
## Your Role: Critic
You MUST find at least one flaw or risk. Challenge the strongest assumptions.
Propose edge cases. If genuinely excellent, still probe for hidden risks.
Do NOT simply validate or weakly agree.
"""

    async def run(self, context: AgentContext) -> AgentResult:
        """Critique the drafts."""
        return AgentResult(
            content="Critique placeholder",
            passed=False,
            metadata={"role": "critic", "mode": context.mode},
        )


class SynthesizerAgent(BaseAgent):
    """Agent role: Synthesizer - merges and finalizes."""

    ROLE_NAME = "synthesizer"
    ROLE_PROMPT = """
## Your Role: Synthesizer
Weigh arguments by evidence quality, not source.
Resolve disagreements with reasoning. Produce ONE coherent output.
Note consensus vs divergence.
"""

    async def run(self, context: AgentContext) -> AgentResult:
        """Synthesize drafts and critique into final output."""
        return AgentResult(
            content="Synthesis placeholder",
            passed=False,
            metadata={"role": "synthesizer"},
        )


class PlannerAgent(BaseAgent):
    """Agent role: Planner - creates actionable plans."""

    ROLE_NAME = "planner"
    ROLE_PROMPT = """
## Your Role: Planner
Create actionable plans with clear dependencies.
Identify risks and propose mitigations.
Be specific about what needs to happen, not when.
"""

    async def run(self, context: AgentContext) -> AgentResult:
        """Create a structured plan."""
        return AgentResult(
            content="Plan placeholder",
            passed=False,
            metadata={"role": "planner", "mode": context.mode},
        )


class ResearcherAgent(BaseAgent):
    """Agent role: Researcher - gathers and synthesizes information."""

    ROLE_NAME = "researcher"
    ROLE_PROMPT = """
## Your Role: Researcher
Gather comprehensive information with citations.
Present findings objectively, noting confidence levels.
Distinguish fact from inference.
"""

    async def run(self, context: AgentContext) -> AgentResult:
        """Conduct research and gather information."""
        return AgentResult(
            content="Research placeholder",
            passed=False,
            metadata={"role": "researcher"},
        )


# Agent factory for creating agents by role
AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "drafter": DrafterAgent,
    "critic": CriticAgent,
    "synthesizer": SynthesizerAgent,
    "planner": PlannerAgent,
    "researcher": ResearcherAgent,
}


def create_agent(
    role: str,
    model: str,
    system_prompt: str = "",
    tools: list[Tool] | None = None,
) -> BaseAgent:
    """
    Factory function to create an agent by role.

    Args:
        role: Agent role (drafter, critic, etc.)
        model: Model identifier
        system_prompt: Additional system prompt
        tools: Optional tool list

    Returns:
        Configured BaseAgent subclass
    """
    agent_class = AGENT_CLASSES.get(role, BaseAgent)
    return agent_class(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
    )
