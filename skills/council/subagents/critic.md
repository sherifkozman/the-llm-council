# Critic

Primary review and security-analysis subagent.

## Modes

| Mode | Use for |
|------|---------|
| `review` | code review, correctness, maintainability, test gaps |
| `security` | threat modeling, attack paths, security findings |

## Examples

```bash
council run critic --mode review "Review the authentication middleware" --json
council run critic --mode security "Analyze OAuth2 implementation" --json

# With file context
council run critic --mode review --files src/auth.py,src/middleware.py "Review these files" --json

# Lower latency/cost
council run critic --mode review "Review auth changes" \
  --runtime-profile bounded \
  --reasoning-profile off
```
