"""
Provider Health Checks for LLM Council.

Implements preflight health checking for providers before council execution.
Uses the existing provider.doctor() method and integrates with error classification.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from llm_council.providers.base import DoctorResult, ErrorType, ProviderAdapter, classify_error

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Provider health status."""

    OK = "ok"
    DEGRADED = "degraded"  # Working but with issues
    DOWN = "down"  # Not usable
    UNKNOWN = "unknown"  # Check failed or not performed


@dataclass
class ProviderHealth:
    """Health status for a single provider."""

    provider: str
    status: HealthStatus
    message: str = ""
    latency_ms: float | None = None
    error_type: ErrorType | None = None
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def is_usable(self) -> bool:
        """Check if provider is usable for council execution."""
        return self.status in (HealthStatus.OK, HealthStatus.DEGRADED)


@dataclass
class HealthReport:
    """Aggregated health report for all providers."""

    providers: list[ProviderHealth]
    all_healthy: bool
    usable_count: int
    total_count: int
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    check_duration_ms: int = 0

    def get_usable_providers(self) -> list[str]:
        """Get list of provider names that are usable."""
        return [p.provider for p in self.providers if p.is_usable()]

    def get_down_providers(self) -> list[str]:
        """Get list of provider names that are down."""
        return [p.provider for p in self.providers if p.status == HealthStatus.DOWN]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "all_healthy": self.all_healthy,
            "usable_count": self.usable_count,
            "total_count": self.total_count,
            "checked_at": self.checked_at,
            "check_duration_ms": self.check_duration_ms,
            "providers": [
                {
                    "provider": p.provider,
                    "status": p.status.value,
                    "message": p.message,
                    "latency_ms": p.latency_ms,
                    "error_type": p.error_type.value if p.error_type else None,
                }
                for p in self.providers
            ],
        }


class HealthChecker:
    """Performs preflight health checks on providers."""

    # Default timeout for health checks (shorter than generation timeout)
    DEFAULT_TIMEOUT = 10.0

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        fail_on_any_down: bool = False,
    ) -> None:
        """Initialize the health checker.

        Args:
            timeout: Timeout for individual health checks in seconds
            fail_on_any_down: If True, treat any down provider as fatal
        """
        self._timeout = timeout
        self._fail_on_any_down = fail_on_any_down
        self._cache: dict[str, ProviderHealth] = {}
        self._cache_ttl = 60.0  # Cache results for 60 seconds

    async def check_provider(
        self,
        name: str,
        adapter: ProviderAdapter,
    ) -> ProviderHealth:
        """Check health of a single provider.

        Args:
            name: Provider name
            adapter: Provider adapter instance

        Returns:
            ProviderHealth with status and details
        """
        # Check cache
        cached = self._cache.get(name)
        if cached:
            # Parse ISO format timestamp (already uses +00:00, not Z)
            cache_age = time.time() - datetime.fromisoformat(cached.checked_at).timestamp()
            if cache_age < self._cache_ttl:
                return cached

        start = time.monotonic()
        try:
            result: DoctorResult = await asyncio.wait_for(
                adapter.doctor(),
                timeout=self._timeout,
            )
            latency_ms = (time.monotonic() - start) * 1000

            if result.ok:
                health = ProviderHealth(
                    provider=name,
                    status=HealthStatus.OK,
                    message=result.message or "Healthy",
                    latency_ms=result.latency_ms or latency_ms,
                    details=dict(result.details) if result.details else {},
                )
            else:
                # Provider returned not-ok
                error_type = classify_error(result.message or "", -1)
                health = ProviderHealth(
                    provider=name,
                    status=HealthStatus.DOWN,
                    message=result.message or "Health check failed",
                    latency_ms=result.latency_ms or latency_ms,
                    error_type=error_type,
                    details=dict(result.details) if result.details else {},
                )

        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - start) * 1000
            health = ProviderHealth(
                provider=name,
                status=HealthStatus.DEGRADED,
                message=f"Health check timed out after {self._timeout}s",
                latency_ms=latency_ms,
                error_type=ErrorType.TIMEOUT,
            )

        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            error_text = str(exc)
            error_type = classify_error(error_text, -1)

            # Determine severity based on error type
            if error_type in (ErrorType.AUTH, ErrorType.BILLING):
                status = HealthStatus.DOWN  # Non-retryable errors
            else:
                status = HealthStatus.DEGRADED  # Potentially retryable

            health = ProviderHealth(
                provider=name,
                status=status,
                message=f"Health check error: {error_text[:100]}",
                latency_ms=latency_ms,
                error_type=error_type,
            )
            logger.debug("Provider %s health check failed: %s", name, exc)

        # Cache result
        self._cache[name] = health
        return health

    async def check_all(
        self,
        providers: dict[str, ProviderAdapter],
    ) -> HealthReport:
        """Check health of all providers in parallel.

        Args:
            providers: Dict of provider_name -> adapter

        Returns:
            HealthReport with aggregated status
        """
        start = time.monotonic()

        tasks = [self.check_provider(name, adapter) for name, adapter in providers.items()]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        provider_healths: list[ProviderHealth] = []
        for name, result in zip(providers.keys(), results, strict=False):
            if isinstance(result, BaseException):
                provider_healths.append(
                    ProviderHealth(
                        provider=name,
                        status=HealthStatus.UNKNOWN,
                        message=f"Check failed: {result}",
                    )
                )
            else:
                # result is ProviderHealth here
                provider_healths.append(result)

        usable = [p for p in provider_healths if p.is_usable()]
        all_healthy = all(p.status == HealthStatus.OK for p in provider_healths)

        duration_ms = int((time.monotonic() - start) * 1000)

        return HealthReport(
            providers=provider_healths,
            all_healthy=all_healthy,
            usable_count=len(usable),
            total_count=len(provider_healths),
            check_duration_ms=duration_ms,
        )

    def clear_cache(self) -> None:
        """Clear the health check cache."""
        self._cache.clear()

    def should_skip_provider(self, health: ProviderHealth) -> bool:
        """Determine if a provider should be skipped based on health.

        Args:
            health: Provider health status

        Returns:
            True if provider should be skipped
        """
        if health.status == HealthStatus.DOWN:
            return True

        # Skip on non-retryable errors
        return health.error_type in (ErrorType.AUTH, ErrorType.BILLING, ErrorType.CLI_NOT_FOUND)


async def preflight_check(
    providers: dict[str, ProviderAdapter],
    timeout: float = 10.0,
    skip_on_failure: bool = True,
) -> tuple[dict[str, ProviderAdapter], HealthReport]:
    """Perform preflight health checks and return usable providers.

    Convenience function for quick preflight checks.

    Args:
        providers: Dict of provider_name -> adapter
        timeout: Timeout per check
        skip_on_failure: If True, exclude failed providers from result

    Returns:
        Tuple of (usable_providers, health_report)
    """
    checker = HealthChecker(timeout=timeout)
    report = await checker.check_all(providers)

    if skip_on_failure:
        usable_names = set(report.get_usable_providers())
        usable_providers = {
            name: adapter for name, adapter in providers.items() if name in usable_names
        }
    else:
        usable_providers = dict(providers)

    return usable_providers, report


__all__ = [
    "HealthChecker",
    "HealthStatus",
    "HealthReport",
    "ProviderHealth",
    "preflight_check",
]
