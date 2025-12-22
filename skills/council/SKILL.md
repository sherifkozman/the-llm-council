---
name: council
description: Run multi-LLM council for adversarial debate and cross-validation. Orchestrates Claude, GPT-4, and Gemini for production-grade implementation, code review, architecture design, research, and security analysis.
---

# LLM Council

Multi-model council that runs parallel drafts, adversarial critique, and validated synthesis.

## Prerequisites
- Install: `pip install the-llm-council>=0.2.0`
- Configure: `export OPENROUTER_API_KEY="your-key"` (or direct API keys)
- Verify: `council doctor`

## Security Notes
- **API Keys**: Never embed secrets in task descriptions or skill files. Use environment variables.
- **Data Sensitivity**: Avoid sending files containing secrets (`.env`, credentials) to the council. The council sends context to external LLM providers.
- **Skill Integrity**: Treat `SKILL.md` and `subagents/*.md` as configuration code. Keep under version control and review changes.

## Usage
```bash
council run <subagent> "<task>" [--json] [--health-check] [--verbose]
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

## When to Use Council
- Feature implementation requiring production quality
- Code review with security analysis (CWE IDs)
- Architecture design decisions
- Technical research informing decisions
- Build vs buy assessments
- Security threat modeling

## When NOT to Use
- Quick file lookups
- Single-line fixes
- Simple questions

## Examples
```bash
# Feature implementation
council run implementer "Add pagination to users API" --json

# Code review
council run reviewer "Review the authentication changes" --json

# Research
council run researcher "Compare Redis vs Memcached for our caching needs" --json

# Architecture
council run architect "Design caching layer for the API" --json

# Security
council run red-team "Analyze authentication system for vulnerabilities" --json
```

For detailed guidance on each subagent, read the corresponding file in `subagents/`.
