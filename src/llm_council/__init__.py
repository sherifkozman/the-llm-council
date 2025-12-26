"""llm-council package."""

from .council import Council
from .engine.orchestrator import CostEstimate, CouncilResult, OrchestratorConfig
from .protocol.types import CouncilConfig
from .providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    GenerateResult,
    Message,
    ProviderAdapter,
    ProviderCapabilities,
    ProviderCapabilityName,
)
from .providers.registry import ProviderRegistry, get_registry

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "Council",
    "CouncilConfig",
    "CouncilResult",
    "CostEstimate",
    "DoctorResult",
    "GenerateRequest",
    "GenerateResponse",
    "GenerateResult",
    "Message",
    "OrchestratorConfig",
    "ProviderAdapter",
    "ProviderCapabilityName",
    "ProviderCapabilities",
    "ProviderRegistry",
    "get_registry",
]
