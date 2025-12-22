# Router Subagent

## Purpose
Task classification and routing to the appropriate subagent.

## When to Use
- When unsure which subagent to use
- Classifying incoming tasks automatically
- Building meta-agents that dispatch work
- Analyzing task complexity

## Output Schema
Returns JSON with:
- `task_type`: Classified task type
- `recommended_subagent`: Best subagent for this task
- `complexity`: simple/moderate/complex
- `reasoning`: Why this subagent was chosen
- `alternative_subagents`: Other valid choices
- `confidence`: Confidence score (0.0-1.0)

## CLI Options
```bash
council run router "task" --health-check --json --verbose
```

## Example
```bash
council run router "Should we use Redis or Memcached for caching?" --json
```

## Cost & Time
- **Cost**: ~$0.10 per classification
- **Time**: ~30 seconds
