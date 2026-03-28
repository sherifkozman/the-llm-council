# Direct APIs Quickstart

This guide shows how to use LLM Council with direct API access to Anthropic, OpenAI, and Google, instead of going through OpenRouter.

## Why Use Direct APIs?

While OpenRouter is recommended for most users, direct APIs may be preferable when:
- You already have API keys for specific providers
- You need features only available in native SDKs
- You want guaranteed model versions without routing
- You have enterprise agreements with specific providers
- You're developing offline with local models

## Prerequisites

Install provider-specific dependencies:

```bash
# For Anthropic (Claude)
pip install the-llm-council[anthropic]

# For OpenAI (GPT)
pip install the-llm-council[openai]

# For Google (Gemini)
pip install the-llm-council[google]

# For Vertex AI (Enterprise GCP)
pip install the-llm-council[vertex]

# For all providers
pip install the-llm-council[all]
```

## Anthropic (Claude)

### Setup

1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. Set environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

3. Verify setup:

```bash
council doctor
```

### Usage

CLI:

```bash
council run drafter --mode impl "Build a user authentication system" --providers anthropic
```

Python:

```python
import asyncio
from llm_council import Council

async def main():
    council = Council(providers=["anthropic"])

    result = await council.run(
        task="Design a REST API for a blog platform",
        subagent="drafter"  # --mode arch
    )

    print(result.output)

asyncio.run(main())
```

### Model Selection

Default model: `claude-sonnet-4-6`

Override via config:

```yaml
# ~/.config/llm-council/config.yaml
providers:
  - name: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-opus-4-6
```

Available models:
- `claude-sonnet-4-6` - Latest Sonnet (recommended)
- `claude-opus-4-6` - Most capable, expensive
- `claude-haiku-4-5` - Fast and affordable

## OpenAI (GPT)

### Setup

1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. Set environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

3. Verify setup:

```bash
council doctor
```

### Usage

CLI:

```bash
council run researcher "Latest best practices for React 2026" --providers openai
```

Python:

```python
import asyncio
from llm_council import Council

async def main():
    council = Council(providers=["openai"])

    result = await council.run(
        task="Review this authentication code for security issues",
        subagent="critic"  # --mode review
    )

    print(result.output)

asyncio.run(main())
```

### Model Selection

Default model: `gpt-5.4`

Override via config:

```yaml
providers:
  - name: openai
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-5.4
```

Available models:
- `gpt-5.4` - Latest GPT (recommended)
- `gpt-5.4-codex` - Code-optimized
- `gpt-5.4-mini` - Fast and affordable

## Google (Gemini)

### Setup

1. Get an API key from [makersuite.google.com](https://makersuite.google.com)
2. Set environment variable:

```bash
export GOOGLE_API_KEY="..."
```

3. Verify setup:

```bash
council doctor
```

### Usage

CLI:

```bash
council run planner "Add payment processing to my app" --providers google
```

Python:

```python
import asyncio
from llm_council import Council

async def main():
    council = Council(providers=["google"])

    result = await council.run(
        task="Create a test plan for an e-commerce checkout flow",
        subagent="drafter"  # --mode test
    )

    print(result.output)

asyncio.run(main())
```

### Model Selection

Default model: `gemini-3.1-pro-preview`

Override via config:

```yaml
providers:
  - name: google
    api_key: ${GOOGLE_API_KEY}
    default_model: gemini-3.1-pro-preview
```

Available models:
- `gemini-3.1-pro-preview` - Most capable (recommended)
- `gemini-2.5-flash` - Fast and affordable
- `gemini-2.0-flash` - Previous generation

## Vertex AI (Enterprise GCP)

Vertex AI provides enterprise access to Gemini and Claude models through Google Cloud with unified billing and IAM.

The Vertex AI provider automatically routes to the appropriate SDK based on the model:
- **Gemini models**: Uses google-genai SDK (region: us-central1)
- **Claude models**: Uses anthropic[vertex] SDK (region: global)

### Setup

**Step 1: Authenticate with gcloud**

```bash
gcloud auth application-default login
```

Or use a service account:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

**Step 2: Configure for your model type**

For Gemini models:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"  # optional
export VERTEX_AI_MODEL="gemini-2.5-pro"     # optional, default: gemini-2.0-flash
```

For Claude models:

```bash
# pip install the-llm-council[vertex] includes both Gemini and Claude SDKs
export ANTHROPIC_VERTEX_PROJECT_ID="your-project-id"
export CLOUD_ML_REGION="global"              # Claude uses global region
export ANTHROPIC_MODEL="claude-opus-4-6@20260301"
```

**Step 3: Verify setup**

```bash
council doctor
```

### Usage

CLI:

```bash
council run drafter --mode arch "Design a distributed cache" --providers vertex-ai
```

Python:

```python
import asyncio
from llm_council import Council

async def main():
    council = Council(providers=["vertex-ai"])

    result = await council.run(
        task="Perform security analysis of JWT authentication",
        subagent="critic"  # --mode security
    )

    print(result.output)

asyncio.run(main())
```

### Model Selection

**Gemini Models** (via `VERTEX_AI_MODEL`):
- `gemini-2.0-flash` - Fast and capable (default)
- `gemini-2.5-pro` - Most capable Gemini
- `gemini-2.5-flash` - Balanced performance

**Claude Models** (via `ANTHROPIC_MODEL`):
- `claude-opus-4-6@20260301` - Most capable Claude
- `claude-sonnet-4-6@20260301` - Balanced Claude
- `claude-haiku-4-5@20250929` - Fast Claude

**Note:** Claude models require version suffix (e.g., `@20251101`).

### When to Use Vertex AI vs Direct APIs

| Use Case | Recommended Provider |
|----------|---------------------|
| Quick prototyping | Google API / Anthropic API |
| Enterprise/production | Vertex AI (unified GCP billing) |
| Claude with GCP billing | Vertex AI |
| Gemini with GCP billing | Vertex AI |
| Simple setup | Direct APIs |

## Using Multiple Providers

LLM Council's strength comes from orchestrating multiple providers for adversarial debate.

### All Three Providers

```python
import asyncio
from llm_council import Council

async def main():
    # Requires all three API keys to be set
    council = Council(providers=["anthropic", "openai", "google"])

    result = await council.run(
        task="Design a distributed caching system",
        subagent="drafter"  # --mode arch
    )

    # Result contains:
    # - drafts: Dict with keys "anthropic", "openai", "google"
    # - critique: Analysis of all three drafts
    # - output: Synthesized final result
    print(result.output)

asyncio.run(main())
```

### Mix Direct APIs with OpenRouter

```python
council = Council(providers=[
    "openrouter",  # For Llama, Mistral, etc.
    "anthropic",   # For guaranteed latest Claude
    "openai"       # For guaranteed latest GPT
])
```

### Provider-Specific Features

Each provider has unique capabilities:

```python
from llm_council.providers.anthropic import AnthropicProvider
from llm_council.providers.openai import OpenAIProvider
from llm_council.providers.google import GoogleProvider

# Check capabilities
anthropic = AnthropicProvider()
print(await anthropic.supports("streaming"))      # True
print(await anthropic.supports("tool_use"))       # True
print(await anthropic.supports("multimodal"))     # True

openai = OpenAIProvider()
print(await openai.supports("streaming"))         # True
print(await openai.supports("tool_use"))          # True
```

## Configuration

### Environment Variables

Priority order:
1. Explicitly passed to provider constructor
2. Environment variables
3. Config file
4. Built-in defaults

All providers support:

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_ORG_ID="org-..."  # Optional

# Google
export GOOGLE_API_KEY="..."
```

### Config File

Create `~/.config/llm-council/config.yaml`:

```yaml
providers:
  - name: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-sonnet-4-6

  - name: openai
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-5.4

  - name: google
    api_key: ${GOOGLE_API_KEY}
    default_model: gemini-3.1-pro-preview

defaults:
  timeout: 120
  max_retries: 3
  enable_schema_validation: true
```

### Programmatic Configuration

```python
from llm_council import Council
from llm_council.protocol.types import CouncilConfig

config = CouncilConfig(
    providers=["anthropic", "openai"],
    timeout=180,
    max_retries=5,
    model_overrides={
        "anthropic": "claude-opus-4-6",
        "openai": "gpt-5.4"
    }
)

council = Council(config=config)
```

## Advanced Usage

### Streaming Responses

```python
from llm_council.providers.anthropic import AnthropicProvider
from llm_council.providers.base import GenerateRequest, Message

async def stream_example():
    provider = AnthropicProvider()

    request = GenerateRequest(
        messages=[Message(role="user", content="Write a short poem")],
        stream=True,
        max_tokens=500
    )

    async for chunk in await provider.generate(request):
        if chunk.text:
            print(chunk.text, end="", flush=True)
```

### Custom Timeouts

```bash
# CLI
council run drafter --mode impl "Complex task" --timeout 300 --providers anthropic

# Python
config = CouncilConfig(
    providers=["anthropic"],
    timeout=300  # 5 minutes
)
council = Council(config=config)
```

### Retry Logic

```python
config = CouncilConfig(
    providers=["openai"],
    max_retries=5,  # Retry synthesis up to 5 times on validation failure
)
```

## Troubleshooting

### Issue: "API key not set"

**Solution**: Ensure environment variable is exported:

```bash
echo $ANTHROPIC_API_KEY  # Should print your key
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY
```

Add to shell profile for persistence:

```bash
# ~/.bashrc or ~/.zshrc
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

### Issue: "Package not installed"

**Solution**: Install provider-specific dependencies:

```bash
pip install the-llm-council[anthropic,openai,google]
```

### Issue: "Rate limit exceeded"

**Solution**: Each provider has rate limits. Check limits at:
- Anthropic: [console.anthropic.com](https://console.anthropic.com)
- OpenAI: [platform.openai.com/account/limits](https://platform.openai.com/account/limits)
- Google: [makersuite.google.com](https://makersuite.google.com)

Implement exponential backoff or reduce request rate.

### Issue: "Model not found"

**Solution**: Verify model name matches provider's current offerings:
- Anthropic models: [docs.anthropic.com/claude/docs/models-overview](https://docs.anthropic.com/claude/docs/models-overview)
- OpenAI models: [platform.openai.com/docs/models](https://platform.openai.com/docs/models)
- Google models: [ai.google.dev/models](https://ai.google.dev/models)

### Issue: Provider-specific errors

Run diagnostics:

```bash
council doctor
```

This checks:
- API key presence
- Network connectivity
- Model availability
- Latency

## Cost Comparison

Approximate costs per 1M tokens (as of 2026):

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| Anthropic | Claude Sonnet 4.6 | $3 | $15 |
| Anthropic | Claude Opus 4.6 | $15 | $75 |
| Anthropic | Claude Haiku 4.5 | $0.25 | $1.25 |
| OpenAI | GPT-5.4 | $5 | $15 |
| OpenAI | GPT-5.4 Mini | $0.50 | $1.50 |
| Google | Gemini 3.1 Pro | $3.50 | $10.50 |
| Google | Gemini 2.5 Flash | $0.35 | $1.05 |

Track your costs:

```python
result = await council.run(task="...", subagent="drafter")

if result.cost_estimate:
    print(f"Cost: ${result.cost_estimate.estimated_cost_usd:.4f}")
    print(f"Input tokens: {result.cost_estimate.total_input_tokens}")
    print(f"Output tokens: {result.cost_estimate.total_output_tokens}")
```

## CLI Providers (Offline/Local)

For local or offline use, LLM Council supports CLI-based providers:

### Codex CLI

```bash
# Requires codex CLI to be installed
council run drafter --mode impl "Build a parser" --providers codex
```

`codex` defaults to `gpt-5.4` so local ChatGPT-authenticated Codex CLI installs work
without extra config. If your local Codex environment supports `*-codex` variants,
you can still opt in via `providers[].default_model`.

### Gemini CLI

```bash
# Requires gemini CLI to be installed
council run researcher "AI safety research" --providers gemini
```

CLI providers execute local commands instead of making API calls, useful for:
- Air-gapped environments
- Local model inference
- Custom LLM wrappers

## Next Steps

- [Creating Custom Providers](../providers/creating-providers.md) - Build your own provider adapters
- [OpenRouter Quickstart](openrouter.md) - Simpler setup with unified API
- [Main Documentation](../index.md) - Full feature overview
