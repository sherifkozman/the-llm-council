"""
Council Engine - Orchestration logic for multi-LLM council.

The engine coordinates:
1. Parallel drafts from multiple providers
2. Adversarial critique
3. Synthesis with schema validation
4. Health checks and graceful degradation
"""

from llm_council.engine.degradation import (
    DegradationAction,
    DegradationDecision,
    DegradationPolicy,
    DegradationReport,
    FailureEvent,
    create_default_policy,
)
from llm_council.engine.health import (
    HealthChecker,
    HealthReport,
    HealthStatus,
    ProviderHealth,
    preflight_check,
)
from llm_council.engine.orchestrator import Orchestrator

__all__ = [
    # Orchestrator
    "Orchestrator",
    # Health checks
    "HealthChecker",
    "HealthStatus",
    "HealthReport",
    "ProviderHealth",
    "preflight_check",
    # Degradation
    "DegradationAction",
    "DegradationDecision",
    "DegradationPolicy",
    "DegradationReport",
    "FailureEvent",
    "create_default_policy",
]
