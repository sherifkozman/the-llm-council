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
        self.config = (
            config.model_copy(update={"providers": self._providers})
            if config
            else CouncilConfig(providers=self._providers)
        )
        self._orchestrator = self._build_orchestrator(self.config)

    def _build_orchestrator_config(self, config: CouncilConfig) -> OrchestratorConfig:
        """Convert CouncilConfig to OrchestratorConfig."""

        return OrchestratorConfig(
            timeout=config.timeout,
            max_retries=config.max_retries,
            enable_artifacts=config.enable_artifact_store,
            enable_health_check=config.enable_health_check,
            enable_graceful_degradation=config.enable_graceful_degradation,
            models=config.models,
            provider_configs=config.provider_configs,
            system_context=config.system_context,
            mode=config.mode,
            model_pack=config.model_pack,
            model_overrides=dict(config.model_overrides),
            reasoning_profile=config.reasoning_profile,
            execution_profile=config.execution_profile,
            budget_class=config.budget_class,
            required_capabilities=list(config.required_capabilities),
            disable_local_evidence=config.disable_local_evidence,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            runtime_profile=config.runtime_profile,
            output_schema=config.output_schema,
        )

    def _build_orchestrator(self, config: CouncilConfig) -> Orchestrator:
        """Create an orchestrator instance from CouncilConfig."""

        return Orchestrator(
            providers=config.providers,
            config=self._build_orchestrator_config(config),
        )

    async def run(
        self,
        task: str,
        subagent: str = "router",
        *,
        follow_router: bool | None = None,
    ) -> CouncilResult:
        """Run a council task.

        Args:
            task: The task description.
            subagent: Subagent type (router, planner, implementer, etc.).
            follow_router: When true, execute the router recommendation after a router run.

        Returns:
            CouncilResult with the result.
        """
        effective_follow_router = (
            follow_router if follow_router is not None else self.config.follow_router
        )
        if effective_follow_router and subagent != "router":
            raise ValueError("follow_router can only be used when subagent='router'")

        initial_result = await self._orchestrator.run(task=task, subagent=subagent)
        if not effective_follow_router:
            return initial_result

        return await self._run_router_follow_up(task, initial_result)

    async def _run_router_follow_up(self, task: str, router_result: CouncilResult) -> CouncilResult:
        """Run the subagent selected by the router, if the router result is usable."""

        if not router_result.success or not isinstance(router_result.output, dict):
            return router_result

        routed_subagent = router_result.output.get("subagent_to_run")
        routed_mode = router_result.output.get("mode")

        if routed_subagent in (None, "", "router"):
            return router_result

        routed_config = self.config.model_copy(
            update={
                "mode": routed_mode,
                "model_pack": router_result.output.get("model_pack"),
                "execution_profile": router_result.output.get("execution_profile"),
                "budget_class": router_result.output.get("budget_class"),
                "required_capabilities": list(router_result.output.get("required_capabilities") or []),
            }
        )
        routed_orchestrator = self._build_orchestrator(routed_config)
        routed_result = await routed_orchestrator.run(task=task, subagent=routed_subagent)

        if routed_result.execution_plan is None:
            routed_result.execution_plan = {}
        routed_result.execution_plan["routed_via_router"] = True
        routed_result.execution_plan["routing_subagent"] = routed_subagent
        routed_result.execution_plan["routing_mode"] = routed_mode
        routed_result.execution_plan["routing_model_pack"] = router_result.output.get("model_pack")
        routed_result.execution_plan["routing_execution_profile"] = router_result.output.get(
            "execution_profile"
        )
        routed_result.execution_plan["routing_budget_class"] = router_result.output.get(
            "budget_class"
        )
        routed_result.execution_plan["routing_required_capabilities"] = router_result.output.get(
            "required_capabilities"
        )

        routed_result.routed = True
        routed_result.routing_decision = dict(router_result.output)
        routed_result.routing_execution_plan = dict(router_result.execution_plan or {})
        return routed_result

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
