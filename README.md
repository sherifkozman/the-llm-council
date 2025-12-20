# The LLM Council

```
                              ┌─────────────────────────────────────────┐
                              │           THE LLM COUNCIL               │
                              └─────────────────────────────────────────┘

         ┌───────────┐         ┌───────────┐         ┌───────────┐
         │  ╭─────╮  │         │  ╭─────╮  │         │  ╭─────╮  │
         │  │ GPT │  │         │  │CLAUD│  │         │  │GEMIN│  │
         │  ╰─────╯  │         │  ╰─────╯  │         │  ╰─────╯  │
         │  ┌─────┐  │         │  ┌─────┐  │         │  ┌─────┐  │
         │  │ ⚡  │  │         │  │ 🧠  │  │         │  │ ✨  │  │
         │  └─────┘  │         │  └─────┘  │         │  └─────┘  │
         └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
               │                     │                     │
               │    PARALLEL DRAFTS  │                     │
               └──────────┬──────────┴──────────┬──────────┘
                          │                     │
                          ▼                     ▼
                    ┌───────────────────────────────┐
                    │      ADVERSARIAL CRITIQUE     │
                    │  "I disagree because..."      │
                    │  "Have you considered..."     │
                    │  "This contradicts..."        │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         SYNTHESIS             │
                    │   Best ideas merged into      │
                    │   schema-validated output     │
                    └───────────────────────────────┘
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

A Multi-LLM Council Framework that orchestrates multiple LLM backends to enable **adversarial debate**, **cross-validation**, and **structured decision-making**.

## Why Use a Council?

Single-model outputs have blind spots. By running multiple models in parallel and having them critique each other, the council:

- **Catches errors** that any single model might miss
- **Reduces hallucination** through cross-validation
- **Produces higher-quality outputs** via adversarial refinement
- **Validates structure** with JSON schema enforcement and retry logic

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Orchestration** | Run parallel drafts from Claude, OpenAI, Gemini, or any provider |
| **Adversarial Critique** | Built-in critique phase identifies weaknesses and blind spots |
| **Schema Validation** | JSON schema validation with automatic retry for structured outputs |
| **Provider Agnostic** | Swap between OpenRouter, direct APIs, or CLI-based providers |
| **Health Checks** | Preflight provider health checks with latency tracking |
| **Graceful Degradation** | Automatic retry, fallback, and skip strategies for failures |
| **Artifact Store** | Persistent storage of drafts with tiered summarization |
| **Secret-Safe Logging** | Redaction pipeline prevents credential leakage |

## Requirements

- **Python**: 3.10 or higher
- **OS**: macOS, Linux, Windows (WSL recommended)
- **API Keys**: At least one of:
  - `OPENROUTER_API_KEY` (recommended - single key for all models)
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY` or `GEMINI_API_KEY`

## Installation

```bash
pip install llm-council
```

With specific providers:

```bash
# OpenRouter (recommended - single API key for all models)
pip install llm-council

# Direct APIs
pip install llm-council[anthropic,openai,google]

# All providers
pip install llm-council[all]

# Development
pip install llm-council[dev]
```

## Quick Start

### CLI Usage

```bash
# Set your API key
export OPENROUTER_API_KEY="your-key"

# Run a council task
council run implementer "Build a login page with OAuth"

# With health check and verbose output
council run implementer "Build a login page" --health-check --verbose

# Disable artifact storage for faster runs
council run implementer "Quick fix" --no-artifacts

# Get structured JSON output
council run planner "Add user authentication" --json
```

### Python API

```python
from llm_council import Council

council = Council(providers=["openrouter"])
result = await council.run(
    task="Build a login page with OAuth",
    subagent="implementer"
)
print(result.output)
```

### Check Provider Health

```bash
council doctor
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           LLM Council                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │    CLI      │───▶│  Council    │───▶│     Orchestrator        │ │
│  │  (typer)    │    │   (API)     │    │                         │ │
│  └─────────────┘    └─────────────┘    │  ┌───────────────────┐  │ │
│                                        │  │  Health Checker   │  │ │
│  ┌─────────────────────────────────┐   │  ├───────────────────┤  │ │
│  │        Provider Registry        │◀──│  │ Degradation Policy│  │ │
│  │  ┌─────────┐ ┌─────────┐       │   │  ├───────────────────┤  │ │
│  │  │OpenRouter│ │Anthropic│ ...  │   │  │  Artifact Store   │  │ │
│  │  └─────────┘ └─────────┘       │   │  └───────────────────┘  │ │
│  └─────────────────────────────────┘   └─────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Subagent Configs                          │   │
│  │  router | planner | architect | implementer | reviewer | ... │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     JSON Schemas                             │   │
│  │  Validation & retry logic for structured outputs             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
0. HEALTH CHECK (optional)
   └── Preflight check of all providers, skip unhealthy ones

1. PARALLEL DRAFTS
   ├── Provider A generates draft
   ├── Provider B generates draft
   └── Provider C generates draft
   └── (Graceful degradation on failures)

2. ADVERSARIAL CRITIQUE
   └── Critic identifies weaknesses, contradictions, blind spots

3. SYNTHESIS
   └── Merge best elements, address critique, validate schema

4. VALIDATION
   └── JSON schema check with retry on failure

5. ARTIFACT STORAGE (optional)
   └── Store drafts and outputs for context management
```

## Subagents

| Subagent | Purpose | Example |
|----------|---------|---------|
| `router` | Classify and route tasks | "Is this a bug or feature?" |
| `planner` | Create execution roadmaps | "Plan the auth implementation" |
| `assessor` | Make build/no-build decisions | "Should we use Redis or Memcached?" |
| `researcher` | Deep market/technical research | "Research OAuth providers" |
| `architect` | Design systems and APIs | "Design the caching layer" |
| `implementer` | Generate production code | "Build the login page" |
| `reviewer` | Review code for issues | "Review this PR for security" |
| `test-designer` | Design test suites | "Design tests for auth module" |
| `shipper` | Create release notes | "Generate changelog for v1.2" |
| `red-team` | Security threat analysis | "Analyze attack vectors" |

## Writing a Provider

Providers are pluggable via Python entry points:

```python
from llm_council.providers.base import ProviderAdapter, GenerateRequest, GenerateResponse

class MyProvider(ProviderAdapter):
    name = "myprovider"

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        # Your implementation
        return GenerateResponse(text="...", content="...")

    async def doctor(self) -> DoctorResult:
        return DoctorResult(ok=True, message="Healthy")
```

Register via `pyproject.toml`:

```toml
[project.entry-points."llm_council.providers"]
myprovider = "my_package.providers:MyProvider"
```

## Configuration

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

## CLI Reference

```bash
council run <subagent> "<task>"    # Run a council task
council doctor                      # Check provider health
council config                      # Show configuration

# Options
--providers, -p    Comma-separated provider list
--timeout, -t      Timeout in seconds (default: 120)
--max-retries      Max retry attempts (default: 3)
--health-check     Run preflight health check
--no-artifacts     Disable artifact storage
--no-degradation   Disable graceful degradation
--json             Output structured JSON
--verbose, -v      Verbose output
```

## Development

```bash
# Clone the repository
git clone https://github.com/sherifkozman/the-llm-council.git
cd the-llm-council

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
mypy src/llm_council
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/the-llm-council.git
cd the-llm-council

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/ && mypy src/llm_council
```

### Contribution Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Test** your changes (`pytest`)
5. **Lint** your code (`ruff check src/ && mypy src/llm_council`)
6. **Commit** with a clear message (`git commit -m 'Add amazing feature'`)
7. **Push** to your branch (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request

### What We're Looking For

- **New Providers**: Add support for more LLM backends
- **New Subagents**: Create specialized agents for specific tasks
- **Bug Fixes**: Found a bug? We'd love a fix!
- **Documentation**: Improvements to docs are always welcome
- **Tests**: More test coverage is great

## Security

For security concerns, please see our [Security Policy](SECURITY.md) or email security@example.com.

**Key security features:**
- CLI adapters use exec-style subprocess (no shell injection)
- Environment variable allowlisting prevents secret leakage
- Path traversal protection in artifact storage
- Configurable secret redaction in logs

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting
- [httpx](https://www.python-httpx.org/) - Async HTTP client

---

<p align="center">
  <i>When one model isn't enough, convene a council.</i>
</p>

<p align="center">
  <sub>~ vibe coded by <a href="https://twitter.com/sherifkozman">Sherif Kozman</a> & The LLM Council ~</sub>
</p>
