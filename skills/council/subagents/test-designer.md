# Test Designer (Legacy Alias)

`test-designer` is a backwards-compatible alias.

Prefer:

```bash
council run drafter --mode test "<task>"
```

Use it for:

- test plan generation
- coverage gap analysis
- test strategy for new features

Example:

```bash
council run drafter --mode test "Design tests for cursor-based pagination" --json
```
