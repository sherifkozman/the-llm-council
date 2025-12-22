# Architect Subagent

## Purpose
System design, API schemas, and data model design with multi-perspective validation.

## When to Use
- Designing new systems or major features
- API schema design (REST, GraphQL, gRPC)
- Database schema and data model design
- Technology stack selection
- Microservice boundary definition

## Output Schema
Returns JSON with:
- `architecture`: High-level system design with components
- `api_schema`: Formal API specification
- `data_models`: Database schemas, entity definitions
- `technology_choices`: Justified selections
- `tradeoffs`: Pros/cons of key decisions
- `implementation_phases`: Suggested build sequence
- `confidence`: Confidence score (0.0-1.0)

```json
{
  "architecture": {
    "components": ["API Gateway", "Auth Service", "Data Layer"],
    "interactions": [...]
  },
  "api_schema": {"endpoints": [...]},
  "data_models": {"entities": [...]},
  "technology_choices": {"database": "PostgreSQL", "rationale": "..."},
  "tradeoffs": [{"decision": "...", "pros": [...], "cons": [...]}],
  "implementation_phases": ["Phase 1: Core API", "Phase 2: Auth"],
  "confidence": 0.88
}
```

## CLI Options
```bash
council run architect "task" --health-check --json --verbose
```

## Example
```bash
council run architect "Design a multi-tenant SaaS API with row-level security" --json
```

## Cost & Time
- **Cost**: ~$0.57 per design
- **Time**: ~5 minutes

## Security Note
Architecture designs should be reviewed by security-aware engineers before implementation. For systems handling sensitive data, run `council run red-team` on the design.
