"""
Sakana AI (Fugu) provider adapter.

Fugu / Fugu Ultra is Sakana AI's OpenAI-compatible multi-agent orchestration
endpoint (https://api.sakana.ai/v1). The wire protocol is the standard OpenAI
``/chat/completions`` contract, so this adapter subclasses the OpenRouter
provider and reuses its request building, structured-output handling, and
response parsing. Only the base URL, auth env var, request headers, and default
model differ.

Fugu is NOT available on OpenRouter, so it must be a first-class *registered*
provider rather than a virtual ``vendor/model`` OpenRouter id (the orchestrator
only routes slash-containing names through OpenRouter).

Intended use: an optional, expensive "deep reviewer" critic
(``council run critic --providers sakana``), not a default council member. Fugu
is itself a multi-agent orchestrator, so it adds meta-review depth rather than
independent model-family diversity.

Environment variables:
    SAKANA_API_KEY:  Required. Your Sakana API key.
    SAKANA_BASE_URL: Optional. Override the API base URL.
    SAKANA_MODEL:    Optional. Override the default model (``fugu-ultra``).

NOTE: Live testing (2026-07-01) showed Fugu Ultra reliably handles prompt-based
JSON drafts (~15s-100s+ depending on task complexity) via council's fallback path,
so ``structured_output`` is reported as ``False`` here — Fugu is itself a slow
multi-agent orchestrator, and native ``response_format: json_schema`` decoding
over a large schema is still avoided in favor of that prompt-based fallback.
Fugu's latency varies widely (observed 8s-100s+ in the same session), which is
why callers should raise ``--timeout`` for runs that include it rather than
relying on the default.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from llm_council.providers.base import DoctorResult, ProviderCapabilities
from llm_council.providers.openrouter import OpenRouterProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llm_council.providers.base import GenerateRequest, GenerateResponse

# Truncation length for the response body appended to a re-raised HTTPStatusError.
# Bodies are typically short JSON error objects; 500 chars keeps the message readable
# in logs/degradation reports while still capturing the actual rejection reason.
_ERROR_BODY_MAX_CHARS = 500


class SakanaProvider(OpenRouterProvider):
    """Sakana AI (Fugu) OpenAI-compatible provider adapter."""

    name: ClassVar[str] = "sakana"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tool_use=False,
        structured_output=False,  # Fugu times out on native json_schema; use prompt-based JSON
        multimodal=False,
        max_tokens=None,  # Varies by model
    )

    BASE_URL = "https://api.sakana.ai/v1"
    DEFAULT_MODEL = "fugu-ultra"
    ENV_API_KEY = "SAKANA_API_KEY"

    # Fugu accepts reasoning_effort only in this set. The OpenRouter base builder
    # forwards OpenAI-style values ("low"/"medium"/"high"/"none"), and Fugu rejects
    # anything outside the set with HTTP 400 (surfaced as an opaque "unknown" failure
    # that silently degrades Fugu out of the council). Clamp sub-"high" values up to
    # "high" (Fugu is always a heavy multi-agent reasoner) and drop the disable sentinel.
    _FUGU_REASONING_EFFORTS: ClassVar[frozenset[str]] = frozenset({"high", "xhigh", "max"})

    def _build_request_body(self, request: GenerateRequest) -> dict[str, Any]:
        """Build the request body, sanitizing reasoning_effort for Fugu's API."""
        body = super()._build_request_body(request)
        effort = body.get("reasoning_effort")
        if effort is not None and effort not in self._FUGU_REASONING_EFFORTS:
            if effort == "none":
                body.pop("reasoning_effort", None)
            else:
                body["reasoning_effort"] = "high"
        return body

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate a response, surfacing Fugu's HTTP error body on failure.

        ``httpx.HTTPStatusError.__str__`` never includes the response body, so a
        rejected-parameter 400 (e.g. the reasoning_effort mismatch above) previously
        surfaced downstream only as an opaque "400 Bad Request" — indistinguishable
        from any other failure and hard to diagnose without re-running with a raw
        HTTP client. Only the non-streaming path is wrapped: a streaming request's
        errors surface lazily during iteration, past this method's own try/except.
        """
        try:
            return await super().generate(request)
        except httpx.HTTPStatusError as exc:
            body_text = exc.response.text[:_ERROR_BODY_MAX_CHARS]
            raise httpx.HTTPStatusError(
                f"{exc}\nResponse body: {body_text}",
                request=exc.request,
                response=exc.response,
            ) from exc

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get(self.ENV_API_KEY)
        self._base_url = base_url or os.environ.get("SAKANA_BASE_URL", self.BASE_URL)
        self._default_model = default_model or os.environ.get("SAKANA_MODEL") or self.DEFAULT_MODEL
        self._http_client = http_client
        self._owns_client = http_client is None

    def _get_headers(self) -> dict[str, str]:
        """Build request headers (plain OpenAI bearer auth; no OpenRouter extras)."""
        if not self._api_key:
            raise ValueError(
                "Sakana API key not configured. "
                "Set the SAKANA_API_KEY environment variable or pass api_key."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def doctor(self) -> DoctorResult:
        """Health check against the Sakana models endpoint."""
        start_time = time.time()

        if not self._api_key:
            return DoctorResult(
                ok=False,
                message="SAKANA_API_KEY environment variable not set",
                details={"error": "missing_api_key"},
            )

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._base_url}/models",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            latency_ms = (time.time() - start_time) * 1000
            return DoctorResult(
                ok=True,
                message="Sakana (Fugu) API is accessible",
                latency_ms=latency_ms,
                details={"models_available": True},
            )
        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            return DoctorResult(
                ok=False,
                message=f"API error: {e.response.status_code}",
                latency_ms=latency_ms,
                details={"status_code": e.response.status_code},
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return DoctorResult(
                ok=False,
                message=f"Connection error: {str(e)}",
                latency_ms=latency_ms,
                details={"error": str(e)},
            )


def _register() -> None:
    """Register the Sakana provider with the global registry."""
    from llm_council.providers.registry import get_registry

    with contextlib.suppress(ValueError):
        get_registry().register_provider("sakana", SakanaProvider)


_register()
