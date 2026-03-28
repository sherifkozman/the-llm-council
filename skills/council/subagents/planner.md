# Planner

Primary planning and assessment subagent.

## Modes

| Mode | Use for |
|------|---------|
| `plan` | execution roadmaps, migration plans, phased delivery |
| `assess` | build-vs-buy, go/no-go, tradeoff analysis |

## Examples

```bash
council run planner --mode plan "Plan MongoDB to PostgreSQL migration" --json
council run planner --mode assess "Redis vs Memcached for sessions" --json

# With file context
council run planner --mode plan --files PRD.md "Plan implementation" --json
```
