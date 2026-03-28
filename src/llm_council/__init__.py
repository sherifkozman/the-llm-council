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

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("the-llm-council")
except Exception:
    __version__ = "0.7.1"  # fallback

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
