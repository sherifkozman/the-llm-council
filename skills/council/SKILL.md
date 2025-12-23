---
name: council
description: Run multi-LLM council for adversarial debate and cross-validation. Orchestrates Claude, GPT-4, and Gemini for production-grade implementation, code review, architecture design, research, and security analysis.
---

# LLM Council Skill

Multi-model council: parallel drafts → adversarial critique → validated synthesis.

> **Prerequisite:** This skill requires the `the-llm-council` Python package to be installed. The skill provides IDE integration but the actual council runs via the installed CLI. If you see `command not found: council`, run `pip install the-llm-council` first.

## Setup

### 1. Install
```bash
pip install the-llm-council>=0.2.0

# With specific provider SDKs
pip install the-llm-council[anthropic,openai,google]
```

### 2. Configure API Keys

| Provider | Environment Variable | Notes |
|----------|---------------------|-------|
| OpenRouter | `OPENROUTER_API_KEY` | **Recommended** - single key for all models |
| OpenAI | `OPENAI_API_KEY` | Direct GPT access |
| Anthropic | `ANTHROPIC_API_KEY` | Direct Claude access |
| Google | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Direct Gemini access |

```bash
# Minimum setup (OpenRouter)
export OPENROUTER_API_KEY="your-key"
```

### 3. Verify
```bash
council doctor
```

## Usage

```bash
council run <subagent> "<task>" [options]
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--json` | Output structured JSON |
| `--health-check` | Run preflight provider check |
| `--verbose, -v` | Verbose output |
| `--models, -m` | Comma-separated model IDs |
| `--providers, -p` | Comma-separated provider list |
| `--timeout, -t` | Timeout in seconds (default: 120) |
| `--max-retries` | Max retry attempts (default: 3) |
| `--no-artifacts` | Disable artifact storage (faster) |
| `--no-degradation` | Disable graceful degradation (strict mode) |

### Other Commands
```bash
council doctor    # Check provider health
council config    # Show current configuration
```

## Subagents

| Subagent | Use For | Details |
|----------|---------|---------|
| `implementer` | Feature code, bug fixes | See `subagents/implementer.md` |
| `reviewer` | Code review, security audit | See `subagents/reviewer.md` |
| `architect` | System design, APIs | See `subagents/architect.md` |
| `researcher` | Technical research | See `subagents/researcher.md` |
| `planner` | Roadmaps, execution plans | See `subagents/planner.md` |
| `assessor` | Build vs buy decisions | See `subagents/assessor.md` |
| `red-team` | Security threat analysis | See `subagents/red-team.md` |
| `test-designer` | Test suite design | See `subagents/test-designer.md` |
| `shipper` | Release notes | See `subagents/shipper.md` |
| `router` | Task classification | See `subagents/router.md` |

## Multi-Model Configuration

Run multiple models in parallel for adversarial debate:

```bash
# Via CLI flag
council run architect "Design caching layer" \
  --models "anthropic/claude-3.5-sonnet,openai/gpt-4o,google/gemini-pro"

# Via environment variable
export COUNCIL_MODELS="anthropic/claude-3.5-sonnet,openai/gpt-4o,google/gemini-pro"
```

### Model Pack Overrides

Fine-tune which models handle specific task types:

```bash
export COUNCIL_MODEL_FAST="anthropic/claude-3-haiku"      # Quick tasks
export COUNCIL_MODEL_REASONING="anthropic/claude-3-opus"  # Deep analysis
export COUNCIL_MODEL_CODE="openai/gpt-4o"                 # Code generation
export COUNCIL_MODEL_CRITIC="anthropic/claude-3.5-sonnet" # Adversarial critique
```

## Config File

Optional YAML configuration:

```yaml
# ~/.config/llm-council/config.yaml
providers:
  - name: openrouter
    api_key: ${OPENROUTER_API_KEY}
    default_model: anthropic/claude-3-opus

defaults:
  timeout: 120
  max_retries: 3
  summary_tier: actions
```

## Python API

```python
from llm_council import Council

council = Council(providers=["openrouter"])
result = await council.run(
    task="Build a login page with OAuth",
    subagent="implementer"
)
print(result.output)
```

## When to Use

**Use council for:**
- Feature implementation requiring production quality
- Code review with security analysis (CWE IDs)
- Architecture design decisions
- Technical research informing decisions
- Build vs buy assessments
- Security threat modeling

**Skip council for:**
- Quick file lookups
- Single-line fixes
- Simple questions

## Examples

```bash
# Feature implementation
council run implementer "Add pagination to users API" --json

# Code review
council run reviewer "Review the authentication changes" --json

# Multi-model architecture design
council run architect "Design caching layer" \
  --models "anthropic/claude-3.5-sonnet,openai/gpt-4o" --json

# Security threat model
council run red-team "Analyze auth system vulnerabilities" --json
```

## Security Notes

- **API Keys**: Never embed secrets in task descriptions or skill files. Use environment variables.
- **Data Sensitivity**: Avoid sending files containing secrets (`.env`, credentials) to the council. Context is sent to external LLM providers.
- **Skill Integrity**: Treat `SKILL.md` and `subagents/*.md` as configuration code. Keep under version control.

## Troubleshooting

```bash
# Check all providers
council doctor

# Verbose output for debugging
council run implementer "task" --verbose

# Faster runs (skip artifact storage)
council run implementer "task" --no-artifacts
```
