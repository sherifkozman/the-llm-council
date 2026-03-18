---
name: council
description: Run multi-LLM council for adversarial debate and cross-validation. Orchestrates Claude, GPT, and Gemini for production-grade implementation, code review, architecture design, research, and security analysis.
---

# LLM Council Skill

Multi-model council: parallel drafts -> adversarial critique -> validated synthesis.

> **Prerequisite:** This skill requires the `the-llm-council` Python package to be installed. The skill provides IDE integration but the actual council runs via the installed CLI. If you see `command not found: council`, run `pip install the-llm-council` first.

## Setup

### 1. Install
```bash
pip install the-llm-council>=0.6.0

# With specific provider SDKs
pip install the-llm-council[anthropic,openai,google]

# With Vertex AI (Enterprise GCP)
pip install the-llm-council[vertex]
```

### 2. Configure API Keys

| Provider | Environment Variable | Notes |
|----------|---------------------|-------|
| OpenRouter | `OPENROUTER_API_KEY` | **Recommended** - single key for all models |
| OpenAI | `OPENAI_API_KEY` | Direct GPT access |
| Anthropic | `ANTHROPIC_API_KEY` | Direct Claude access |
| Google | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Direct Gemini access |
| Vertex AI | `GOOGLE_CLOUD_PROJECT` or `ANTHROPIC_VERTEX_PROJECT_ID` + ADC | Enterprise GCP |

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

### CLI Options (v0.6.0)

| Option | Description |
|--------|-------------|
| `--mode` | Agent mode (e.g., `impl`/`arch`/`test` for drafter) |
| `--json` | Output structured JSON |
| `--verbose, -v` | Verbose output with metrics |
| `--models, -m` | Comma-separated OpenRouter model IDs |
| `--providers, -p` | Comma-separated provider list |
| `--timeout, -t` | Request timeout in seconds |
| `--temperature` | Model temperature (0.0-2.0) |
| `--max-tokens` | Max output tokens |
| `--input, -i` | Read task from file (use `-` for stdin) |
| `--output, -o` | Write output to file |
| `--context, --system` | Additional system context/instructions |
| `--schema` | Custom output schema JSON file |
| `--no-artifacts` | Disable artifact storage (faster) |
| `--dry-run` | Show what would run without executing |

### Other Commands
```bash
council doctor                     # Check provider health
council doctor --provider openai   # Check specific provider
council config --show              # Show current configuration
council config --init              # Create default config
council config --validate          # Validate config file
council --version                  # Show version
```

## Subagents (v0.5.0+)

### Core Agents

| Subagent | Modes | Use For | Details |
|----------|-------|---------|---------|
| `drafter` | `impl`, `arch`, `test` | Code, architecture, tests | See below |
| `critic` | `review`, `security` | Code review, security audit | See below |
| `synthesizer` | - | Merge and finalize outputs | See `subagents/synthesizer.md` |
| `researcher` | - | Technical research | See `subagents/researcher.md` |
| `planner` | `plan`, `assess` | Roadmaps, decisions | See `subagents/planner.md` |
| `router` | - | Task classification | See `subagents/router.md` |

### Agent Modes

**drafter modes:**
- `--mode impl` - Feature implementation, bug fixes (default)
- `--mode arch` - System design, API schemas
- `--mode test` - Test suite design

**critic modes:**
- `--mode review` - Code review with CWE IDs (default)
- `--mode security` - Security threat analysis

**planner modes:**
- `--mode plan` - Execution roadmaps (default)
- `--mode assess` - Build vs buy decisions

### Deprecated Aliases (Backwards Compatible)

The following legacy agent names still work but will be removed in v1.0:

| Old Name | Use Instead | Removed In |
|----------|-------------|------------|
| `implementer` | `drafter --mode impl` | v1.0 |
| `architect` | `drafter --mode arch` | v1.0 |
| `test-designer` | `drafter --mode test` | v1.0 |
| `reviewer` | `critic --mode review` | v1.0 |
| `red-team` | `critic --mode security` | v1.0 |
| `assessor` | `planner --mode assess` | v1.0 |
| `shipper` | `synthesizer` | v1.0 |

## Multi-Model Configuration

Run multiple models in parallel for adversarial debate:

```bash
# Via CLI flag
council run drafter --mode arch "Design caching layer" \
  --models "anthropic/claude-opus-4-6,openai/gpt-5.4,google/gemini-3.1-pro-preview"

# Via environment variable
export COUNCIL_MODELS="anthropic/claude-opus-4-6,openai/gpt-5.4,google/gemini-3.1-pro-preview"
```

### Model Pack Overrides

Fine-tune which models handle specific task types:

```bash
export COUNCIL_MODEL_FAST="anthropic/claude-haiku-4-5"        # Quick tasks (router, synthesizer)
export COUNCIL_MODEL_REASONING="anthropic/claude-opus-4-6"    # Deep analysis (planner, critic)
export COUNCIL_MODEL_CODE="openai/gpt-5.4"                   # Code generation (drafter)
export COUNCIL_MODEL_CRITIC="anthropic/claude-sonnet-4-6"     # Adversarial critique
export COUNCIL_MODEL_GROUNDED="google/gemini-3.1-pro-preview" # Research tasks
export COUNCIL_MODEL_CODE_COMPLEX="anthropic/claude-opus-4-6" # Complex refactoring
```

### Per-Provider Model Override

```bash
export OPENAI_MODEL="gpt-5.4"                # Override OpenAI default
export ANTHROPIC_MODEL="claude-opus-4-6"      # Override Anthropic default
export GOOGLE_MODEL="gemini-3.1-pro-preview"  # Override Google default
export OPENROUTER_MODEL="anthropic/claude-opus-4-6"  # Override OpenRouter default
```

## Config File

Optional YAML configuration:

```yaml
# ~/.config/llm-council/config.yaml
providers:
  - name: openrouter
    default_model: anthropic/claude-opus-4-6
  - name: openai
    default_model: gpt-5.4
  - name: google
    default_model: gemini-3.1-pro-preview

defaults:
  providers:
    - openrouter
  timeout: 120
  max_retries: 3
  summary_tier: actions
  output_format: json  # or "rich" (default)
```

## Python API

```python
from llm_council import Council
from llm_council.protocol.types import CouncilConfig

config = CouncilConfig(
    providers=["openrouter"],
    models=["anthropic/claude-opus-4-6", "openai/gpt-5.4"],
    mode="impl",
    timeout=120,
    temperature=0.7,
    max_tokens=4000,
    system_context="Additional instructions...",
)
council = Council(config=config)
result = await council.run(
    task="Build a login page with OAuth",
    subagent="drafter"
)
print(result.output)
print(result.success)
print(result.duration_ms)
print(result.cost_estimate.estimated_cost_usd)
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
council run drafter --mode impl "Add pagination to users API" --json

# Code review
council run critic --mode review "Review the authentication changes" --json

# Multi-model architecture design
council run drafter --mode arch "Design caching layer" \
  --models "anthropic/claude-opus-4-6,openai/gpt-5.4" --json

# Security threat model
council run critic --mode security "Analyze auth system vulnerabilities" --json

# Build vs buy decision
council run planner --mode assess "Should we build or buy a payment system?" --json

# Read task from file, write output to file
council run drafter --mode impl -i task.md -o result.json --json

# Dry run (see what would execute)
council run drafter --mode impl "Build login page" --dry-run

# With additional context
council run critic --mode review "Review auth changes" \
  --context "Focus on OWASP Top 10 vulnerabilities"

# Legacy syntax (still works, shows deprecation warning)
council run implementer "Add pagination" --json
council run reviewer "Review changes" --json
```

## Security Notes

- **API Keys**: Never embed secrets in task descriptions or skill files. Use environment variables.
- **Data Sensitivity**: Avoid sending files containing secrets (`.env`, credentials) to the council. Context is sent to external LLM providers.
- **Context Injection**: User-provided `--context` content is wrapped in XML delimiters and framed as reference data to mitigate prompt injection.
- **Skill Integrity**: Treat `SKILL.md` and `subagents/*.md` as configuration code. Keep under version control.

## Troubleshooting

```bash
# Check all providers
council doctor

# Check specific provider
council doctor --provider openrouter --json

# Verbose output for debugging
council run drafter --mode impl "task" --verbose

# Debug logging
council --debug run drafter "task"

# Faster runs (skip artifact storage)
council run drafter "task" --no-artifacts

# Dry run to see configuration
council run drafter --mode impl "task" --dry-run
```
