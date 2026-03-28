# Implementer (Legacy Alias)

`implementer` is a backwards-compatible alias.

Prefer:

```bash
council run drafter --mode impl "<task>"
```

Use it for:

- feature implementation
- non-trivial bug fixes
- refactors that produce code changes

Example:

```bash
council run drafter --mode impl "Add pagination to the users API endpoint" --json
```
