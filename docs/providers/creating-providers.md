# Creating Custom Providers

LLM Council uses a pluggable provider architecture that allows you to integrate any LLM backend via the `ProviderAdapter` interface.

## Why Create a Custom Provider?

- **Integrate proprietary models** - Add your company's internal LLM APIs
- **Support new LLM services** - Add providers like Cohere, AI21, Replicate
- **Local model inference** - Integrate Ollama, LM Studio, vLLM
- **Custom routing logic** - Implement specialized load balancing or failover
- **Proxy existing APIs** - Add middleware for logging, caching, or rate limiting

## Provider Architecture

```python
from llm_council.providers.base import (
    ProviderAdapter,
    ProviderCapabilities,
    GenerateRequest,
    GenerateResponse,
    DoctorResult,
    StructuredOutputConfig,
)

class MyProvider(ProviderAdapter):
    name = "myprovider"
    capabilities = ProviderCapabilities(
        streaming=True,
        tool_use=True,
        structured_output=True,
        multimodal=False,
        max_tokens=4096
    )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate a response."""
        pass

    async def supports(self, capability: str) -> bool:
        """Check if capability is supported."""
        pass

    async def doctor(self) -> DoctorResult:
        """Health check."""
        pass
```

## Structured Output Support

LLM Council provides a provider-agnostic `StructuredOutputConfig` class that each provider transforms into its API-specific format.

### StructuredOutputConfig

```python
from llm_council.providers.base import StructuredOutputConfig

config = StructuredOutputConfig(
    json_schema={"type": "object", "properties": {...}},
    name="my_output",     # Required by OpenAI/OpenRouter
    strict=True,          # Enforce strict schema adherence
)
```

### Provider-Specific Format Transformations

Each LLM provider has a different API format for structured outputs. Your provider must transform `StructuredOutputConfig` appropriately:

#### OpenAI / OpenRouter Format

```python
# request.structured_output is a StructuredOutputConfig
if request.structured_output:
    model = request.model or self._default_model
    if self._model_supports_structured_output(model):
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.structured_output.name,
                "strict": request.structured_output.strict,
                "schema": dict(request.structured_output.json_schema),
            },
        }
    elif model in JSON_MODE_ONLY_MODELS:
        # Fall back to simple JSON mode for older models
        kwargs["response_format"] = {"type": "json_object"}
```

**Supported Models:**
- GPT-5.x family (gpt-5.1, gpt-5.2, gpt-5.1-codex, gpt-5.2-codex)
- GPT-4o family (gpt-4o, gpt-4o-mini)
- GPT-4.1 family (gpt-4.1, gpt-4.1-mini, gpt-4.1-nano)
- o-series reasoning models (o1, o3-mini, o4-mini)

#### Anthropic Format

Anthropic requires a beta header and uses `output_format`:

```python
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"

if request.structured_output:
    if self._model_supports_structured_output(model):
        use_beta = True
        kwargs["output_format"] = {
            "type": "json_schema",
            "schema": dict(request.structured_output.json_schema),
        }

# When making the API call:
if use_beta:
    response = await client.beta.messages.create(
        betas=[STRUCTURED_OUTPUTS_BETA],
        **kwargs,
    )
else:
    response = await client.messages.create(**kwargs)
```

**Supported Models:**
- Claude Opus 4.x (claude-opus-4-5, claude-opus-4-1)
- Claude Sonnet 4.x (claude-sonnet-4-5)
- Claude Haiku 4.x (claude-haiku-4-5)

**Note:** Claude 3.x models do NOT support structured outputs.

#### Google/Gemini Format

Google uses `response_mime_type` and `response_schema` in generation config:

```python
if request.structured_output:
    if self._model_supports_structured_output(model):
        generation_config["response_mime_type"] = "application/json"
        generation_config["response_schema"] = dict(
            request.structured_output.json_schema
        )
    elif self._is_legacy_model(model):
        # Fall back to simple JSON mode for older models
        generation_config["response_mime_type"] = "application/json"
```

**Supported Models:**
- Gemini 3.x family (gemini-3-pro, gemini-3-flash, gemini-3-preview)
- Gemini 2.5 family (gemini-2.5-pro, gemini-2.5-flash)
- Gemini 2.0 family (gemini-2.0-flash, gemini-2.0-pro)
- Experimental models (gemini-exp-*)

**Legacy Models (JSON mode only, no schema enforcement):**
- Gemini 1.5 family (gemini-1.5-pro, gemini-1.5-flash)
- Gemini 1.0 family
- Original gemini-pro

### Model Capability Checking

Use prefix-based matching for robust model support:

```python
# Define supported model prefixes
STRUCTURED_OUTPUT_MODEL_PREFIXES = (
    "gpt-5",     # All GPT-5.x models
    "gpt-4o",    # All GPT-4o variants
    "gpt-4.1",   # All GPT-4.1 variants
)

# Explicit model names (with dated versions)
STRUCTURED_OUTPUT_MODELS = frozenset({
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    # ... etc
})

def _model_supports_structured_output(self, model: str) -> bool:
    """Check if model supports structured output."""
    # Direct match
    if model in STRUCTURED_OUTPUT_MODELS:
        return True

    # Prefix match (handles dated versions like gpt-4o-2024-08-06)
    if any(model.startswith(prefix) for prefix in STRUCTURED_OUTPUT_MODEL_PREFIXES):
        return True

    # Handle version-less model names (e.g., "gpt-4o-2024-08-06" -> "gpt-4o")
    for suffix in ("-2024", "-2025", "-2026"):
        if suffix in model:
            base_model = model.split(suffix)[0]
            if base_model in STRUCTURED_OUTPUT_MODELS:
                return True
            break

    return False
```

## Step-by-Step Guide

### 1. Create the Provider Class

Create a new file `my_provider.py`:

```python
from __future__ import annotations

import contextlib
import os
import time
from typing import Any, ClassVar
from collections.abc import AsyncIterator

import httpx

from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
)

# Define which models support structured output
STRUCTURED_OUTPUT_MODEL_PREFIXES = (
    "my-model-v2",
    "my-model-v3",
)

JSON_MODE_ONLY_MODELS = frozenset({
    "my-model-v1",
})


class MyProvider(ProviderAdapter):
    """Custom provider for My LLM Service."""

    name: ClassVar[str] = "myprovider"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tool_use=True,
        structured_output=True,
        multimodal=False,
        max_tokens=4096,
    )

    BASE_URL = "https://api.myllm.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("MY_API_KEY")
        self._base_url = base_url or self.BASE_URL
        self._default_model = default_model or "my-model-v2"
        self._http_client: httpx.AsyncClient | None = None

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate a response."""
        if not self._api_key:
            raise ValueError("API key required")

        client = await self._get_client()
        payload = self._build_payload(request)

        if request.stream:
            return self._generate_stream(client, payload)

        response = await client.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def _build_payload(self, request: GenerateRequest) -> dict[str, Any]:
        model = request.model or self._default_model
        payload: dict[str, Any] = {"model": model}

        if request.messages:
            payload["messages"] = [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ]
        elif request.prompt:
            payload["messages"] = [{"role": "user", "content": request.prompt}]

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        # Handle structured output
        if request.structured_output:
            if self._model_supports_structured_output(model):
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.structured_output.name,
                        "strict": request.structured_output.strict,
                        "schema": dict(request.structured_output.json_schema),
                    },
                }
            elif model in JSON_MODE_ONLY_MODELS:
                payload["response_format"] = {"type": "json_object"}
        elif request.response_format:
            # Legacy pass-through for backwards compatibility
            payload["response_format"] = dict(request.response_format)

        return payload

    def _model_supports_structured_output(self, model: str) -> bool:
        """Check if a specific model supports structured output."""
        return any(
            model.startswith(prefix) for prefix in STRUCTURED_OUTPUT_MODEL_PREFIXES
        )

    def _parse_response(self, data: dict[str, Any]) -> GenerateResponse:
        choice = data["choices"][0]
        message = choice["message"]

        usage = None
        if "usage" in data:
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }

        return GenerateResponse(
            text=message.get("content"),
            content=message.get("content"),
            tool_calls=message.get("tool_calls"),
            usage=usage,
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def _generate_stream(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> AsyncIterator[GenerateResponse]:
        """Stream responses."""
        payload["stream"] = True
        async with client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                import json
                data = json.loads(data_str)
                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                yield GenerateResponse(
                    text=delta.get("content"),
                    content=delta.get("content"),
                    finish_reason=choice.get("finish_reason"),
                )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0)
            )
        return self._http_client

    async def supports(self, capability: str) -> bool:
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

    async def doctor(self) -> DoctorResult:
        start_time = time.time()

        if not self._api_key:
            return DoctorResult(
                ok=False,
                message="MY_API_KEY not set",
                details={"error": "missing_api_key"},
            )

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            latency_ms = (time.time() - start_time) * 1000

            return DoctorResult(
                ok=True,
                message="API accessible",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return DoctorResult(
                ok=False,
                message=f"Error: {str(e)}",
                latency_ms=latency_ms,
                details={"error": str(e)},
            )


def _register() -> None:
    """Register the provider with the global registry."""
    from llm_council.providers.registry import get_registry

    registry = get_registry()
    with contextlib.suppress(ValueError):
        registry.register_provider("myprovider", MyProvider)


_register()
```

### 2. Register the Provider

Add to your `pyproject.toml`:

```toml
[project.entry-points."llm_council.providers"]
myprovider = "my_package.providers:MyProvider"
```

### 3. Use Your Provider

```python
from llm_council import Council

council = Council(providers=["myprovider"])
result = await council.run(
    task="Generate code",
    subagent="implementer"
)
```

## Provider Implementation Checklist

- [ ] Inherit from `ProviderAdapter`
- [ ] Set `name` class variable (unique identifier)
- [ ] Configure `capabilities` (streaming, tool_use, structured_output, etc.)
- [ ] Implement `generate()` with both streaming and non-streaming paths
- [ ] Implement `supports()` for capability checking
- [ ] Implement `doctor()` for health checks
- [ ] Handle `StructuredOutputConfig` → provider-specific format
- [ ] Define `STRUCTURED_OUTPUT_MODEL_PREFIXES` for model capability checking
- [ ] Handle legacy `response_format` for backward compatibility
- [ ] Register via `_register()` function and entry points
- [ ] Add proper error handling with descriptive messages
- [ ] Include usage tracking in `GenerateResponse`

## Provider Examples

See existing implementations:
- `src/llm_council/providers/openrouter.py` - HTTP API with streaming, OpenAI-compatible format
- `src/llm_council/providers/openai.py` - OpenAI native SDK with GPT-5.x support
- `src/llm_council/providers/anthropic.py` - Anthropic SDK with beta header for Claude 4.x
- `src/llm_council/providers/google.py` - Google Generative AI with Gemini 2.x/3.x support

## API Format Quick Reference

| Provider | Field | Format |
|----------|-------|--------|
| OpenAI | `response_format` | `{type: "json_schema", json_schema: {name, strict, schema}}` |
| OpenRouter | `response_format` | Same as OpenAI |
| Anthropic | `output_format` | `{type: "json_schema", schema}` + beta header |
| Google | `generation_config` | `{response_mime_type: "application/json", response_schema}` |

## Next Steps

- [OpenRouter Quickstart](../quickstart/openrouter.md)
- [Direct APIs Quickstart](../quickstart/direct-apis.md)
- [Main Documentation](../index.md)
