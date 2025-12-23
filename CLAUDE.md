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

| Subagent | Use For |
|----------|---------|
| `router` | Classify and route tasks |
| `planner` | Create roadmaps and plans |
| `assessor` | Build vs buy decisions |
| `researcher` | Technical research |
| `architect` | System design |
| `implementer` | Feature implementation |
| `reviewer` | Code review |
| `test-designer` | Test suite design |
| `shipper` | Release notes |
| `red-team` | Security analysis |

## When to Use Council

**Use council for:**
- Feature implementation (`council run implementer "..."`)
- Code review (`council run reviewer "..."`)
- Architecture design (`council run architect "..."`)
- Security review (`council run red-team "..."`)
- Build vs buy decisions (`council run assessor "..."`)

**Skip council for:**
- Quick file lookups
- Single-line fixes
- Simple questions

## CLI Options

```bash
# Health check before running
council run implementer "task" --health-check

# Disable artifact storage (faster)
council run implementer "task" --no-artifacts

# Disable graceful degradation (strict mode)
council run implementer "task" --no-degradation

# Get structured JSON output
council run planner "Add dark mode" --json

# Verbose output for debugging
council run implementer "task" --verbose
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
council run implementer "task" --verbose

# Disable artifact store (faster, less context)
council run implementer "task" --no-artifacts
```
