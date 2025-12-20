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
council run implementer "Build a user authentication system" --providers anthropic
```

Python:

```python
import asyncio
from llm_council import Council

async def main():
    council = Council(providers=["anthropic"])

    result = await council.run(
        task="Design a REST API for a blog platform",
        subagent="architect"
    )

    print(result.output)

asyncio.run(main())
```

### Model Selection

Default model: `claude-3-5-sonnet-20241022`

Override via config:

```yaml
# ~/.config/llm-council/config.yaml
providers:
  - name: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-3-opus-20240229
```

Available models:
- `claude-3-5-sonnet-20241022` - Latest Sonnet (recommended)
- `claude-3-opus-20240229` - Most capable, expensive
- `claude-3-haiku-20240307` - Fast and affordable

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
council run researcher "Latest best practices for React 2024" --providers openai
```

Python:

```python
import asyncio
from llm_council import Council

async def main():
    council = Council(providers=["openai"])

    result = await council.run(
        task="Review this authentication code for security issues",
        subagent="reviewer"
    )

    print(result.output)

asyncio.run(main())
```

### Model Selection

Default model: `gpt-4-turbo`

Override via config:

```yaml
providers:
  - name: openai
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4o
```

Available models:
- `gpt-4-turbo` - Latest GPT-4 (recommended)
- `gpt-4o` - Multimodal GPT-4
- `gpt-4` - Original GPT-4
- `gpt-3.5-turbo` - Fast and affordable

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
        subagent="test-designer"
    )

    print(result.output)

asyncio.run(main())
```

### Model Selection

Default model: `gemini-1.5-pro`

Override via config:

```yaml
providers:
  - name: google
    api_key: ${GOOGLE_API_KEY}
    default_model: gemini-1.5-flash
```

Available models:
- `gemini-1.5-pro` - Most capable (recommended)
- `gemini-1.5-flash` - Fast and affordable
- `gemini-pro` - Previous generation

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
        subagent="architect"
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
    default_model: claude-3-5-sonnet-20241022

  - name: openai
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4-turbo

  - name: google
    api_key: ${GOOGLE_API_KEY}
    default_model: gemini-1.5-pro

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
        "anthropic": "claude-3-opus-20240229",
        "openai": "gpt-4o"
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
council run implementer "Complex task" --timeout 300 --providers anthropic

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

Approximate costs per 1M tokens (as of 2024):

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| Anthropic | Claude 3.5 Sonnet | $3 | $15 |
| Anthropic | Claude 3 Opus | $15 | $75 |
| Anthropic | Claude 3 Haiku | $0.25 | $1.25 |
| OpenAI | GPT-4 Turbo | $10 | $30 |
| OpenAI | GPT-4o | $5 | $15 |
| OpenAI | GPT-3.5 Turbo | $0.50 | $1.50 |
| Google | Gemini 1.5 Pro | $3.50 | $10.50 |
| Google | Gemini 1.5 Flash | $0.35 | $1.05 |

Track your costs:

```python
result = await council.run(task="...", subagent="implementer")

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
council run implementer "Build a parser" --providers codex-cli
```

### Gemini CLI

```bash
# Requires gemini CLI to be installed
council run researcher "AI safety research" --providers gemini-cli
```

CLI providers execute local commands instead of making API calls, useful for:
- Air-gapped environments
- Local model inference
- Custom LLM wrappers

## Next Steps

- [Creating Custom Providers](../providers/creating-providers.md) - Build your own provider adapters
- [OpenRouter Quickstart](openrouter.md) - Simpler setup with unified API
- [Main Documentation](../index.md) - Full feature overview
