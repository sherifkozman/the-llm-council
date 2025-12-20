# OpenRouter Quickstart

OpenRouter is the recommended provider for LLM Council because it provides access to 100+ models through a single API key, including Claude, GPT, Gemini, and many others.

## Why OpenRouter?

- **Single API Key** - Access all models with one key instead of managing multiple API accounts
- **Automatic Failover** - Built-in redundancy if a model is unavailable
- **Competitive Pricing** - Often cheaper than direct APIs due to bulk purchasing
- **Model Diversity** - Easy to experiment with different providers without setup overhead

## Setup

### 1. Get an API Key

1. Visit [openrouter.ai](https://openrouter.ai)
2. Sign up for an account
3. Navigate to Keys section
4. Generate a new API key

### 2. Set Environment Variable

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

For persistent setup, add to your shell profile:

```bash
# ~/.bashrc or ~/.zshrc
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### 3. Verify Setup

```bash
council doctor
```

Expected output:

```
Provider Status
┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Provider  ┃ Status┃ Message                    ┃ Latency ┃
┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ openrouter│ OK    │ OpenRouter API is accessible│ 245ms  │
└───────────┴───────┴────────────────────────────┴─────────┘
```

## Basic Usage

### CLI Examples

```bash
# Run a planner task
council run planner "Add user authentication to my app"

# Generate code with the implementer
council run implementer "Create a Python function to validate email addresses"

# Get structured JSON output
council run architect "Design a REST API for a blog platform" --json

# Review code
council run reviewer "Check the auth.py file for security issues"

# Research a topic
council run researcher "What are the best practices for GraphQL API design in 2024?"
```

### Python API Examples

#### Basic Task

```python
import asyncio
from llm_council import Council

async def main():
    council = Council(providers=["openrouter"])

    result = await council.run(
        task="Build a REST API endpoint for user registration",
        subagent="implementer"
    )

    if result.success:
        print("Code generated successfully!")
        print(result.output)
    else:
        print("Failed:", result.validation_errors)

asyncio.run(main())
```

#### Multiple Subagents Workflow

```python
import asyncio
from llm_council import Council

async def build_feature():
    council = Council(providers=["openrouter"])

    # Step 1: Plan the feature
    plan_result = await council.run(
        task="Plan implementation of OAuth2 login flow",
        subagent="planner"
    )

    if not plan_result.success:
        print("Planning failed:", plan_result.validation_errors)
        return

    print("Plan:", plan_result.output)

    # Step 2: Design architecture
    arch_result = await council.run(
        task="Design OAuth2 architecture with token refresh",
        subagent="architect"
    )

    print("Architecture:", arch_result.output)

    # Step 3: Implement code
    impl_result = await council.run(
        task="Implement OAuth2 login based on the architecture",
        subagent="implementer"
    )

    print("Implementation:", impl_result.output)

    # Step 4: Review the code
    review_result = await council.run(
        task="Review OAuth2 implementation for security issues",
        subagent="reviewer"
    )

    print("Review findings:", review_result.output)

asyncio.run(build_feature())
```

## Model Selection

OpenRouter provides access to many models. You can specify which model to use in your requests:

### Default Models

LLM Council uses sensible defaults:
- **General tasks**: `anthropic/claude-3.5-sonnet`
- **Fast/cheap tasks**: `anthropic/claude-3-haiku`
- **Complex reasoning**: `anthropic/claude-3-opus`

### Overriding Models

Via config file (`~/.config/llm-council/config.yaml`):

```yaml
providers:
  - name: openrouter
    api_key: ${OPENROUTER_API_KEY}
    default_model: openai/gpt-4-turbo

defaults:
  timeout: 120
  max_retries: 3
```

Via environment variable:

```bash
export OPENROUTER_DEFAULT_MODEL="google/gemini-pro"
```

### Popular Models on OpenRouter

| Provider | Model ID | Use Case |
|----------|----------|----------|
| Anthropic | `anthropic/claude-3.5-sonnet` | Balanced performance/cost |
| Anthropic | `anthropic/claude-3-opus` | Complex reasoning |
| Anthropic | `anthropic/claude-3-haiku` | Fast, cheap tasks |
| OpenAI | `openai/gpt-4-turbo` | Latest GPT-4 |
| OpenAI | `openai/gpt-4o` | Multimodal GPT-4 |
| OpenAI | `openai/gpt-3.5-turbo` | Fast, affordable |
| Google | `google/gemini-pro` | Google's flagship |
| Google | `google/gemini-pro-vision` | Multimodal Gemini |
| Meta | `meta-llama/llama-3-70b` | Open source, powerful |

Full model list: [openrouter.ai/models](https://openrouter.ai/models)

## Advanced Configuration

### Timeout and Retries

```python
from llm_council import Council
from llm_council.protocol.types import CouncilConfig

config = CouncilConfig(
    providers=["openrouter"],
    timeout=180,        # 3 minutes per API call
    max_retries=5       # Up to 5 synthesis attempts
)

council = Council(config=config)
```

### Multiple Providers for Parallel Drafts

OpenRouter can be configured to use different models for drafting:

```python
# Use multiple models via OpenRouter
council = Council(providers=[
    "openrouter",  # Will use different models in parallel
])

# Or mix OpenRouter with direct APIs
council = Council(providers=[
    "openrouter",
    "anthropic",   # Direct Anthropic API
    "openai"       # Direct OpenAI API
])
```

### Verbose Output

```bash
council run implementer "Build a user service" --verbose
```

This shows:
- Duration per phase
- Number of synthesis attempts
- Token usage
- Cost estimates

### JSON Output for Scripting

```bash
council run planner "Add dark mode" --json | jq '.output.phases'
```

Pipe to `jq` or other tools for processing structured output.

## Troubleshooting

### Issue: "OPENROUTER_API_KEY environment variable not set"

**Solution**: Export the API key before running commands:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
council doctor
```

### Issue: "Rate limit exceeded"

**Solution**: OpenRouter enforces rate limits. Either:
1. Wait a few seconds and retry
2. Upgrade your OpenRouter plan for higher limits
3. Use a different provider temporarily

### Issue: "Model not available"

**Solution**: Some models may be temporarily unavailable. OpenRouter provides automatic failover, but you can also specify a different model:

```bash
council run implementer "task" --providers openrouter
```

Check model status at [openrouter.ai/models](https://openrouter.ai/models)

### Issue: Slow responses

**Solution**: For faster results:
1. Use a faster model like `claude-3-haiku` or `gpt-3.5-turbo`
2. Reduce timeout: `--timeout 60`
3. Use fewer providers for parallel drafts

## Cost Optimization

### Track Costs

```python
result = await council.run(task="...", subagent="implementer")

if result.cost_estimate:
    print(f"Estimated cost: ${result.cost_estimate.estimated_cost_usd:.4f}")
    print(f"Total tokens: {result.cost_estimate.tokens}")
```

### Reduce Costs

1. **Use cheaper models** - `claude-3-haiku` instead of `claude-3-opus`
2. **Reduce token limits** - Lower `max_tokens` in config
3. **Disable parallel drafts** - Use single provider mode
4. **Use caching** - OpenRouter supports prompt caching for repeated requests

### Example: Budget-Conscious Config

```yaml
providers:
  - name: openrouter
    api_key: ${OPENROUTER_API_KEY}
    default_model: anthropic/claude-3-haiku  # Cheapest Claude model

defaults:
  timeout: 60
  max_retries: 2
```

## Next Steps

- [Direct APIs Quickstart](direct-apis.md) - Use Anthropic, OpenAI, Google APIs directly
- [Creating Custom Providers](../providers/creating-providers.md) - Build your own provider adapters
- [Main Documentation](../index.md) - Full feature overview
