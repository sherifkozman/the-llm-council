# Router

Primary task-classification subagent.

Use it when you want the package to choose the next subagent and mode.

Typical output fields:

- `subagent_to_run`
- `mode`
- `model_pack`
- `execution_profile`
- `budget_class`
- `required_capabilities`

## Examples

```bash
council run router "Add pagination to users API" --json
# -> drafter --mode impl

council run router "Is our auth system secure?" --json
# -> critic --mode security

council run router "Redis vs Memcached for caching" --json
# -> planner --mode assess

council run router "Assess whether we should adopt a hosted vector store" --route
```
