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

Access 200+ models (Gemini, Claude, Llama, Mistral) through Google Cloud.

```bash
# Option 1: Application Default Credentials
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"  # optional

# Option 2: Service Account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Verify setup
council doctor

# Use with council
council run architect "Design a cache" --providers vertex-ai
```

## Troubleshooting

```bash
# Check all providers
council doctor

# Run with verbose output
council run implementer "task" --verbose

# Disable artifact store (faster, less context)
council run implementer "task" --no-artifacts
```
