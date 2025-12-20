"""
Council - Main facade class for LLM Council.

Provides a simple interface for running multi-LLM council tasks.
"""

from __future__ import annotations

from typing import Any

from llm_council.engine.orchestrator import CouncilResult, Orchestrator, OrchestratorConfig
from llm_council.protocol.types import CouncilConfig


class Council:
    """Multi-LLM Council Framework.

    Orchestrates multiple LLM backends to enable adversarial debate,
    cross-validation, and structured decision-making.

    Example:
        ```python
        council = Council(providers=["openrouter"])
        result = await council.run(
            task="Build a login page with OAuth",
            subagent="implementer"
        )
        print(result.output)
        ```
    """

    def __init__(
        self,
        providers: list[str] | None = None,
        config: CouncilConfig | None = None,
    ) -> None:
        """Initialize the Council.

        Args:
            providers: List of provider names to use. Defaults to ["openrouter"].
            config: Optional configuration override.
        """
        self._providers = providers or (config.providers if config else ["openrouter"])

        # Convert CouncilConfig to OrchestratorConfig
        orch_config = OrchestratorConfig(
            timeout=config.timeout if config else 120,
            max_retries=config.max_retries if config else 3,
            enable_artifacts=config.enable_artifact_store if config else True,
            enable_health_check=config.enable_health_check if config else False,
            enable_graceful_degradation=config.enable_graceful_degradation if config else True,
            models=config.models if config else None,
        )

        self.config = config or CouncilConfig(providers=self._providers)
        self._orchestrator = Orchestrator(providers=self._providers, config=orch_config)

    async def run(
        self,
        task: str,
        subagent: str = "router",
    ) -> CouncilResult:
        """Run a council task.

        Args:
            task: The task description.
            subagent: Subagent type (router, planner, implementer, etc.).

        Returns:
            CouncilResult with the result.
        """
        return await self._orchestrator.run(task=task, subagent=subagent)

    async def doctor(self) -> dict[str, Any]:
        """Check provider availability.

        Returns:
            Dict mapping provider names to their health status.
        """
        return await self._orchestrator.doctor()

    @property
    def providers(self) -> list[str]:
        """Get the list of configured providers."""
        return self.config.providers

    @classmethod
    def available_subagents(cls) -> list[str]:
        """Get list of available subagent types."""
        return [
            "router",
            "planner",
            "assessor",
            "researcher",
            "architect",
            "implementer",
            "reviewer",
            "test-designer",
            "shipper",
            "red-team",
        ]
