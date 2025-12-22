"""Tests for engine health checks and degradation."""

import asyncio

import pytest

from llm_council.engine import (
    DegradationAction,
    DegradationPolicy,
    HealthChecker,
    HealthReport,
    HealthStatus,
    ProviderHealth,
    create_default_policy,
    preflight_check,
)
from llm_council.providers.base import DoctorResult, ErrorType


class MockProvider:
    """Mock provider for testing."""

    def __init__(self, ok: bool = True, message: str = "OK", latency_ms: float = 10.0):
        self._ok = ok
        self._message = message
        self._latency_ms = latency_ms

    async def doctor(self) -> DoctorResult:
        return DoctorResult(ok=self._ok, message=self._message, latency_ms=self._latency_ms)


class TestHealthChecker:
    """Tests for HealthChecker."""

    @pytest.mark.asyncio
    async def test_check_healthy_provider(self):
        """Test checking a healthy provider."""
        checker = HealthChecker()
        provider = MockProvider(ok=True, message="Healthy")

        health = await checker.check_provider("test", provider)

        assert health.status == HealthStatus.OK
        assert health.provider == "test"
        assert health.is_usable()

    @pytest.mark.asyncio
    async def test_check_unhealthy_provider(self):
        """Test checking an unhealthy provider."""
        checker = HealthChecker()
        provider = MockProvider(ok=False, message="API key invalid")

        health = await checker.check_provider("test", provider)

        assert health.status == HealthStatus.DOWN
        assert not health.is_usable()

    @pytest.mark.asyncio
    async def test_check_timeout(self):
        """Test handling of timeout during health check."""

        class SlowProvider:
            async def doctor(self):
                await asyncio.sleep(10)
                return DoctorResult(ok=True, message="OK")

        checker = HealthChecker(timeout=0.1)
        provider = SlowProvider()

        health = await checker.check_provider("slow", provider)

        assert health.status == HealthStatus.DEGRADED
        assert health.error_type == ErrorType.TIMEOUT

    @pytest.mark.asyncio
    async def test_check_all_providers(self):
        """Test checking multiple providers."""
        checker = HealthChecker()
        providers = {
            "healthy1": MockProvider(ok=True),
            "healthy2": MockProvider(ok=True),
            "unhealthy": MockProvider(ok=False, message="Down"),
        }

        report = await checker.check_all(providers)

        assert report.total_count == 3
        assert report.usable_count == 2
        assert not report.all_healthy
        assert "unhealthy" in report.get_down_providers()
        assert "healthy1" in report.get_usable_providers()
        assert "healthy2" in report.get_usable_providers()

    @pytest.mark.asyncio
    async def test_health_caching(self):
        """Test that health results are cached."""
        checker = HealthChecker()
        provider = MockProvider(ok=True)

        # First check
        health1 = await checker.check_provider("test", provider)

        # Modify provider behavior
        provider._ok = False

        # Second check should return cached result
        health2 = await checker.check_provider("test", provider)

        assert health1.status == health2.status == HealthStatus.OK

        # Clear cache and check again
        checker.clear_cache()
        health3 = await checker.check_provider("test", provider)
        assert health3.status == HealthStatus.DOWN


class TestPreflightCheck:
    """Tests for preflight_check convenience function."""

    @pytest.mark.asyncio
    async def test_preflight_check_filters_down_providers(self):
        """Test that preflight check filters out down providers."""
        providers = {
            "healthy": MockProvider(ok=True),
            "unhealthy": MockProvider(ok=False),
        }

        usable, report = await preflight_check(providers, timeout=5.0)

        assert "healthy" in usable
        assert "unhealthy" not in usable
        assert report.usable_count == 1

    @pytest.mark.asyncio
    async def test_preflight_check_skip_on_failure_false(self):
        """Test preflight check with skip_on_failure=False."""
        providers = {
            "healthy": MockProvider(ok=True),
            "unhealthy": MockProvider(ok=False),
        }

        usable, report = await preflight_check(providers, timeout=5.0, skip_on_failure=False)

        # All providers should be returned
        assert len(usable) == 2


class TestDegradationPolicy:
    """Tests for DegradationPolicy."""

    def test_non_retryable_error_skips(self):
        """Test that non-retryable errors cause provider to be skipped."""
        policy = create_default_policy()

        decision = policy.decide(
            provider="test",
            error="insufficient_quota",
            phase="drafts",
            remaining_providers=2,
        )

        assert decision.action == DegradationAction.SKIP
        assert decision.billing_url is not None

    def test_retryable_error_retries(self):
        """Test that retryable errors trigger retry."""
        policy = create_default_policy(max_retries=2)

        decision = policy.decide(
            provider="test",
            error="rate_limit exceeded",
            phase="drafts",
            remaining_providers=2,
        )

        assert decision.action == DegradationAction.RETRY
        assert decision.retry_delay_ms > 0

    def test_max_retries_exceeded(self):
        """Test behavior when max retries exceeded."""
        policy = create_default_policy(max_retries=1)

        # First failure - should retry (use a proper rate limit error)
        decision1 = policy.decide("test", "rate_limit exceeded", "drafts", 2)
        assert decision1.action == DegradationAction.RETRY

        # Second failure - max retries exceeded
        decision2 = policy.decide("test", "rate_limit exceeded", "drafts", 2)
        assert decision2.action == DegradationAction.SKIP

    def test_fallback_provider(self):
        """Test using fallback provider."""
        policy = DegradationPolicy(
            fallback_providers={"primary": "backup"},
            max_retries=0,
        )

        decision = policy.decide(
            provider="primary",
            error="billing error",
            phase="synthesis",
            remaining_providers=0,
        )

        assert decision.action == DegradationAction.FALLBACK
        assert decision.fallback_provider == "backup"

    def test_abort_on_critical_failure(self):
        """Test abort when no providers remaining in critical phase."""
        policy = create_default_policy()

        decision = policy.decide(
            provider="last_provider",
            error="auth error",
            phase="synthesis",
            remaining_providers=0,
        )

        assert decision.action == DegradationAction.ABORT

    def test_degradation_report(self):
        """Test that failures are recorded in report."""
        policy = create_default_policy()

        policy.decide("provider1", "billing error", "drafts", 2)
        policy.decide("provider2", "rate_limit", "drafts", 1)

        report = policy.get_report()

        assert len(report.failures) == 2
        assert "provider1" in report.providers_skipped
        assert report.total_retries >= 0

    def test_policy_reset(self):
        """Test resetting policy state."""
        policy = create_default_policy()

        policy.decide("test", "error", "drafts", 2)
        assert len(policy.get_report().failures) == 1

        policy.reset()
        assert len(policy.get_report().failures) == 0


class TestHealthStatusMethods:
    """Tests for HealthStatus and related methods."""

    def test_provider_health_is_usable(self):
        """Test is_usable for different statuses."""
        ok = ProviderHealth(provider="test", status=HealthStatus.OK)
        degraded = ProviderHealth(provider="test", status=HealthStatus.DEGRADED)
        down = ProviderHealth(provider="test", status=HealthStatus.DOWN)
        unknown = ProviderHealth(provider="test", status=HealthStatus.UNKNOWN)

        assert ok.is_usable()
        assert degraded.is_usable()
        assert not down.is_usable()
        assert not unknown.is_usable()

    def test_health_report_to_dict(self):
        """Test HealthReport serialization."""
        report = HealthReport(
            providers=[
                ProviderHealth(provider="p1", status=HealthStatus.OK),
                ProviderHealth(provider="p2", status=HealthStatus.DOWN),
            ],
            all_healthy=False,
            usable_count=1,
            total_count=2,
            check_duration_ms=100,
        )

        data = report.to_dict()

        assert data["all_healthy"] is False
        assert data["usable_count"] == 1
        assert len(data["providers"]) == 2
