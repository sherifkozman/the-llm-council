# Assessor (Legacy Alias)

`assessor` is a backwards-compatible alias.

Prefer:

```bash
council run planner --mode assess "<task>"
```

Use it for:

- build-vs-buy decisions
- tradeoff analysis
- recommendation scoring

Example:

```bash
council run planner --mode assess "Should we build custom auth or use Auth0?" --json
```
