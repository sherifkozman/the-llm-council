# Red Team (Legacy Alias)

`red-team` is a backwards-compatible alias.

Prefer:

```bash
council run critic --mode security "<task>"
```

Use it for:

- threat modeling
- attack-path analysis
- security review of designs or code

Example:

```bash
council run critic --mode security "Threat model the OAuth2 implementation" --json
```
