# Implementer Subagent

## Purpose
Production-grade feature implementation with adversarial multi-model review.

## When to Use
- Feature implementation (new endpoints, components, modules)
- Bug fixes requiring non-trivial logic changes
- Refactoring complex code
- Any production code that will be committed

## Output Schema
Returns JSON with:
- `implementation`: Complete code with file paths
- `rationale`: Design decisions explanation
- `test_plan`: How to verify the implementation
- `edge_cases`: Identified edge cases and handling
- `dependencies`: Required packages or configuration
- `confidence`: Confidence score (0.0-1.0)

```json
{
  "implementation": {
    "files": [
      {"path": "src/api/users.py", "changes": "...code..."}
    ]
  },
  "rationale": "Used cursor-based pagination for scalability...",
  "test_plan": ["Test empty results", "Test page boundaries"],
  "edge_cases": ["Empty dataset", "Invalid cursor"],
  "dependencies": [],
  "confidence": 0.92
}
```

## CLI Options
```bash
council run implementer "task" --health-check --json --verbose
```

## Example
```bash
council run implementer "Add pagination to the users API endpoint" --json
```

## Cost & Time
- **Cost**: ~$0.24 per task
- **Time**: ~3 minutes

## Security Note
Generated code should be reviewed before merging. For changes involving authentication, payments, PII, or infrastructure, require additional human security review.
