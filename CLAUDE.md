# The LLM Council - Claude Code Integration

## Quick Reference

```bash
# Run a council task
council run <subagent> "<task>"

# Check provider setup
council doctor

# Configure providers
council config
```

## Subagents

### Core Agents (v2)

| Subagent | Use For | Modes |
|----------|---------|-------|
| `drafter` | Generate solutions | `impl`, `arch`, `test` |
| `critic` | Evaluate and challenge | `review`, `security` |
| `synthesizer` | Merge and finalize | - |
| `researcher` | Technical research | - |
| `planner` | Plans and decisions | `plan`, `assess` |
| `router` | Classify and route | - |

### Deprecated Aliases (Backwards Compatible)

| Old Name | Maps To | Removed In |
|----------|---------|------------|
| `implementer` | `drafter --mode impl` | v1.0 |
| `architect` | `drafter --mode arch` | v1.0 |
| `test-designer` | `drafter --mode test` | v1.0 |
| `reviewer` | `critic --mode review` | v1.0 |
| `red-team` | `critic --mode security` | v1.0 |
| `assessor` | `planner --mode assess` | v1.0 |
| `shipper` | `synthesizer` | v1.0 |

## When to Use Council

**Use council for:**
- Feature implementation (`council run drafter --mode impl "..."`)
- Code review (`council run critic --mode review "..."`)
- Architecture design (`council run drafter --mode arch "..."`)
- Security review (`council run critic --mode security "..."`)
- Build vs buy decisions (`council run planner --mode assess "..."`)

**Skip council for:**
- Quick file lookups
- Single-line fixes
- Simple questions

## CLI Options

```bash
# Use a specific mode
council run drafter --mode impl "Add login feature"

# Disable artifact storage (faster)
council run drafter "task" --no-artifacts

# Get structured JSON output
council run planner "Add dark mode" --json

# Verbose output for debugging
council run drafter "task" --verbose

# Specify providers
council run drafter --providers openrouter,anthropic "task"
```

## Provider Setup

### OpenRouter (Recommended)

```bash
export OPENROUTER_API_KEY="your-key"
council doctor
```

### Direct APIs

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
council doctor
```

### Vertex AI (Enterprise GCP)

Access Gemini and Claude models through Google Cloud with enterprise billing and IAM.

**For Gemini models:**
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"  # optional
export VERTEX_AI_MODEL="gemini-2.5-pro"     # optional, default: gemini-2.0-flash

council doctor
council run architect "Design a cache" --providers vertex-ai
```

**For Claude models:**
```bash
gcloud auth application-default login
export ANTHROPIC_VERTEX_PROJECT_ID="your-project-id"
export CLOUD_ML_REGION="global"  # Claude uses global region
export ANTHROPIC_MODEL="claude-opus-4-5@20251101"

council doctor
council run architect "Design a cache" --providers vertex-ai
```

**Note:** `pip install the-llm-council[vertex]` includes both Gemini and Claude SDKs.

## Troubleshooting

```bash
# Check all providers
council doctor

# Run with verbose output
council run drafter "task" --verbose

# Disable artifact store (faster, less context)
council run drafter "task" --no-artifacts
```
