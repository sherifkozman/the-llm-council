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
from llm_council.providers.base import ProviderAdapter, ProviderCapabilities

class MyProvider(ProviderAdapter):
    name = "myprovider"
    capabilities = ProviderCapabilities(
        streaming=True,
        tool_use=True,
        structured_output=True,
        multimodal=False,
        max_tokens=4096
    )

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        """Generate a response."""
        pass

    async def supports(self, capability: str) -> bool:
        """Check if capability is supported."""
        pass

    async def doctor(self) -> DoctorResult:
        """Health check."""
        pass
```

## Step-by-Step Guide

### 1. Create the Provider Class

Create a new file `my_provider.py`:

```python
from __future__ import annotations

import os
import time
from typing import ClassVar, Optional, Union
from collections.abc import AsyncIterator

import httpx

from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
)


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

    BASE_URL = "https://api.mylllm.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("MY_API_KEY")
        self._base_url = base_url or self.BASE_URL
        self._default_model = default_model or "my-model-v1"
        self._http_client: Optional[httpx.AsyncClient] = None

    async def generate(
        self, request: GenerateRequest
    ) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """Generate a response."""
        if not self._api_key:
            raise ValueError("API key required")

        client = await self._get_client()
        payload = self._build_payload(request)

        response = await client.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def _build_payload(self, request: GenerateRequest) -> dict:
        payload = {"model": request.model or self._default_model}

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

        return payload

    def _parse_response(self, data: dict) -> GenerateResponse:
        choice = data["choices"][0]
        message = choice["message"]

        return GenerateResponse(
            text=message.get("content"),
            content=message.get("content"),
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
        )

    async def supports(self, capability: str) -> bool:
        return self.supports_capability(capability)

    async def doctor(self) -> DoctorResult:
        start_time = time.time()

        if not self._api_key:
            return DoctorResult(
                ok=False,
                message="MY_API_KEY not set",
            )

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()

            return DoctorResult(
                ok=True,
                message="API accessible",
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return DoctorResult(
                ok=False,
                message=f"Error: {str(e)}",
            )
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

## Provider Examples

See existing implementations:
- `src/llm_council/providers/openrouter.py` - HTTP API with streaming
- `src/llm_council/providers/anthropic.py` - Native SDK integration
- `src/llm_council/providers/cli/codex.py` - CLI-based provider

## Next Steps

- [OpenRouter Quickstart](../quickstart/openrouter.md)
- [Direct APIs Quickstart](../quickstart/direct-apis.md)
- [Main Documentation](../index.md)
