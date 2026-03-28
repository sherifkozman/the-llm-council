# Reviewer (Legacy Alias)

`reviewer` is a backwards-compatible alias.

Prefer:

```bash
council run critic --mode review "<task>"
```

Use it for:

- code review
- maintainability and correctness checks
- API and test-impact review

Example:

```bash
council run critic --mode review "Review the authentication middleware" --json
```
