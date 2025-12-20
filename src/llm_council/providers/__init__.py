"""Provider adapters and registry utilities."""

from .base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    GenerateResult,
    Message,
    ProviderAdapter,
    ProviderCapabilities,
    ProviderCapabilityName,
)
from .registry import ProviderRegistry, get_registry

__all__ = [
    "DoctorResult",
    "GenerateRequest",
    "GenerateResponse",
    "GenerateResult",
    "Message",
    "ProviderAdapter",
    "ProviderCapabilityName",
    "ProviderCapabilities",
    "ProviderRegistry",
    "get_registry",
]
