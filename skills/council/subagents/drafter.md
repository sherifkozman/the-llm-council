# Drafter

Primary implementation/design subagent.

## Modes

| Mode | Use for |
|------|---------|
| `impl` | implementation, bug fixes, refactors |
| `arch` | system design, API design, component boundaries |
| `test` | test strategy, coverage gaps, fixture planning |

## Examples

```bash
council run drafter --mode impl "Add pagination to users API" --json
council run drafter --mode arch "Design a multi-tenant SaaS API" --json
council run drafter --mode test "Design tests for cursor-based pagination" --json

# With file context
council run drafter --mode impl "Refactor auth" --files src/auth.py --json

# Lower latency/cost
council run drafter --mode arch "Design caching layer" \
  --runtime-profile bounded \
  --reasoning-profile light
```
