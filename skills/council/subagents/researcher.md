# Researcher Subagent

## Purpose
Technical research with formal citations and recommendations to inform decisions.

## When to Use
- "What should we use for X?" decisions
- Comparing technologies/libraries/frameworks
- Evaluating build vs buy decisions
- Any research informing a technical decision

## Output Schema
Returns JSON with:
- `executive_summary`: 2-3 sentence overview
- `findings`: Array with topics, options, evaluation
- `tradeoff_matrix`: Comparison table
- `recommendations`: Suggested choices with rationale
- `citations`: Formal citations with URLs
- `confidence`: Confidence score (0.0-1.0)

```json
{
  "executive_summary": "SQLAlchemy is recommended for FastAPI due to mature ecosystem...",
  "findings": [
    {"topic": "ORM Options", "options": ["SQLAlchemy", "Tortoise", "Piccolo"]}
  ],
  "tradeoff_matrix": {"headers": ["ORM", "Async", "Maturity"], "rows": [...]},
  "recommendations": [{"choice": "SQLAlchemy", "rationale": "Best ecosystem support"}],
  "citations": [{"title": "SQLAlchemy Docs", "url": "https://..."}],
  "confidence": 0.87
}
```

## CLI Options
```bash
council run researcher "task" --health-check --json --verbose
```

## Example
```bash
council run researcher "Research and recommend an ORM for Python FastAPI" --json
```

## Cost & Time
- **Cost**: ~$0.42 per research task
- **Time**: ~4 minutes
