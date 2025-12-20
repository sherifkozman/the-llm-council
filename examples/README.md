# LLM Council Examples

This directory contains runnable Python examples demonstrating the llm-council package.

## Quick Start

1. **Install llm-council**:
   ```bash
   pip install the-llm-council
   ```

2. **Set up API key** (easiest with OpenRouter):
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   ```
   Get your key at: https://openrouter.ai/keys

3. **Run an example**:
   ```bash
   python examples/openrouter_only.py
   ```

## Examples

### 1. `openrouter_only.py` - Recommended Starting Point

**The simplest way to get started.** OpenRouter provides access to 100+ models with a single API key.

```bash
python examples/openrouter_only.py
```

**What it covers**:
- Basic setup and configuration
- Common tasks (implementation, review, planning, architecture)
- Health checks
- Cost tracking

**Best for**: First-time users, simple use cases, quick prototyping

---

### 2. `basic_council.py` - Core Features

Comprehensive overview of llm-council's main features.

```bash
python examples/basic_council.py
```

**What it covers**:
- Council creation and configuration
- Different subagent types (implementer, planner, reviewer, architect)
- Custom configuration with `OrchestratorConfig`
- Accessing structured output
- Phase timings and cost estimates
- Provider health checks

**Best for**: Learning all core features, understanding the API

---

### 3. `custom_subagent.py` - Advanced Customization

Learn how to create custom subagents with your own prompts and schemas.

```bash
python examples/custom_subagent.py
```

**What it covers**:
- Creating custom subagent configurations (YAML)
- Defining custom JSON schemas for output validation
- Custom prompts and system instructions
- Advanced orchestrator configuration
- Saving and reusing custom subagents

**Best for**: Advanced users, domain-specific use cases, custom workflows

---

### 4. `direct_apis.py` - Direct Provider APIs

Using direct API providers (Anthropic, OpenAI, Google) instead of OpenRouter.

```bash
# Set one or more provider API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."

python examples/direct_apis.py
```

**What it covers**:
- Single provider usage
- Multi-provider adversarial councils
- Model overrides per provider
- Custom cost tracking
- Provider comparison

**Best for**: Users with existing provider credits, specific model requirements, cost optimization

---

## Example Output Structure

All subagents return structured JSON validated against schemas:

### Implementer Output
```json
{
  "implementation_title": "Email Validator Function",
  "implementation_type": "feature",
  "summary": "Created email validation with regex...",
  "files": [
    {
      "path": "src/validators.py",
      "action": "create",
      "description": "Email validation function",
      "language": "python"
    }
  ],
  "testing_notes": {
    "manual_tests": [...],
    "automated_tests": [...]
  }
}
```

### Reviewer Output
```json
{
  "review_title": "Payment Code Security Review",
  "findings": [
    {
      "severity": "high",
      "issue_type": "SQL Injection",
      "cwe_id": "CWE-89",
      "description": "...",
      "recommendation": "..."
    }
  ]
}
```

### Planner Output
```json
{
  "plan_title": "OAuth2 Implementation Plan",
  "phases": [
    {
      "phase_name": "Foundation",
      "tasks": [...],
      "estimated_duration": "2 days"
    }
  ]
}
```

## Available Subagents

| Subagent | Purpose | Example Use |
|----------|---------|-------------|
| `implementer` | Code implementation | "Build a login API endpoint" |
| `reviewer` | Code review | "Review this payment code for security issues" |
| `planner` | Feature planning | "Plan the OAuth2 implementation" |
| `architect` | System design | "Design a microservices architecture" |
| `researcher` | Technical research | "Research caching strategies for Redis" |
| `assessor` | Build vs buy | "Should we build or buy a CMS?" |
| `test-designer` | Test suite design | "Design tests for the payment module" |
| `shipper` | Release notes | "Generate release notes from commits" |
| `red-team` | Security analysis | "Analyze authentication for vulnerabilities" |
| `router` | Task routing | "Route this task to the right subagent" |

## Configuration Options

### OrchestratorConfig Parameters

```python
from llm_council import Council, OrchestratorConfig

config = OrchestratorConfig(
    timeout=120,                    # Timeout per provider call (seconds)
    max_retries=3,                  # Max synthesis retry attempts
    max_draft_tokens=2000,          # Max tokens per draft
    max_critique_tokens=1200,       # Max tokens for critique
    max_synthesis_tokens=2000,      # Max tokens for synthesis
    draft_temperature=0.7,          # Draft creativity (0.0-2.0)
    critique_temperature=0.2,       # Critique strictness
    synthesis_temperature=0.2,      # Synthesis determinism
    enable_schema_validation=True,  # Validate against JSON schema
    strict_providers=True,          # Fail if providers missing
    model_overrides={               # Override default models
        "anthropic": "claude-3-opus-20240229",
        "openai": "gpt-4-turbo-preview"
    },
    cost_per_1k_input={             # Custom cost tracking
        "anthropic": 0.015,
        "openai": 0.01
    },
    cost_per_1k_output={
        "anthropic": 0.075,
        "openai": 0.03
    }
)

council = Council(providers=["openrouter"], config=config)
```

## Provider Setup

### OpenRouter (Recommended)
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```
- Get key: https://openrouter.ai/keys
- Access to 100+ models with one key
- Automatic failover and routing

### Anthropic Claude
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
- Get key: https://console.anthropic.com/
- Direct access to Claude models

### OpenAI GPT
```bash
export OPENAI_API_KEY="sk-..."
```
- Get key: https://platform.openai.com/api-keys
- Direct access to GPT models

### Google Gemini
```bash
export GOOGLE_API_KEY="..."
```
- Get key: https://makersuite.google.com/app/apikey
- Direct access to Gemini models

## Common Patterns

### Basic Usage
```python
from llm_council import Council

council = Council(providers=["openrouter"])
result = await council.run(
    task="Your task here",
    subagent="implementer"
)

if result.success:
    print(result.output)
```

### With Configuration
```python
from llm_council import Council, OrchestratorConfig

config = OrchestratorConfig(
    timeout=60,
    max_retries=5,
    draft_temperature=0.8
)

council = Council(providers=["openrouter"], config=config)
result = await council.run(task="...", subagent="implementer")
```

### Multi-Provider Council
```python
council = Council(providers=["anthropic", "openai", "google"])
result = await council.run(task="...", subagent="implementer")

# Each provider drafts, then adversarial critique, then synthesis
print(f"Drafts from: {list(result.drafts.keys())}")
```

### Health Check
```python
council = Council(providers=["openrouter"])
health = await council.doctor()

for provider, status in health.items():
    if status['ok']:
        print(f"✓ {provider}: {status['message']}")
```

## Troubleshooting

### "OPENROUTER_API_KEY not set"
```bash
export OPENROUTER_API_KEY="your-key-here"
```

### "Provider resolution failed"
Check that your API key is valid:
```python
council = Council(providers=["openrouter"])
health = await council.doctor()
print(health)
```

### "Schema validation failed"
The output didn't match the expected schema. Check `result.validation_errors`:
```python
if not result.success:
    print(result.validation_errors)
```

### Slow performance
Reduce token limits or timeout:
```python
config = OrchestratorConfig(
    timeout=60,
    max_draft_tokens=1000
)
```

## Next Steps

1. **Start simple**: Run `openrouter_only.py` to get familiar
2. **Explore features**: Try `basic_council.py` for all capabilities
3. **Customize**: Use `custom_subagent.py` to create domain-specific subagents
4. **Optimize**: Use `direct_apis.py` for cost optimization with multiple providers

## Resources

- **Documentation**: `docs/` directory
- **OpenRouter**: https://openrouter.ai/docs
- **Anthropic**: https://docs.anthropic.com/
- **OpenAI**: https://platform.openai.com/docs
- **Google AI**: https://ai.google.dev/docs

## Contributing

Have a useful example? Submit a PR! Examples should:
- Be self-contained and runnable
- Include clear comments and docstrings
- Handle missing API keys gracefully
- Show real-world use cases
