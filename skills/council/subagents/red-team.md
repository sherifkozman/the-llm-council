# Red Team Subagent

## Purpose
Adversarial security analysis with threat modeling and attack vector identification.

## When to Use
- Security-critical feature review (auth, payments, PII)
- Pre-deployment security audit
- Threat modeling for new architectures
- Penetration test planning
- Compliance preparation (SOC2, HIPAA, PCI-DSS)

## Output Schema
Returns JSON with:
- `threat_model`: Identified threats with attack vectors
- `exploitation_scenarios`: Step-by-step attack walkthroughs
- `security_controls`: Existing controls and effectiveness
- `gaps`: Missing security controls
- `recommendations`: Prioritized security improvements
- `cwe_ids`: Related CWE identifiers
- `confidence`: Confidence score (0.0-1.0)

```json
{
  "threat_model": {
    "threat_actors": ["External attacker", "Malicious insider"],
    "attack_surface": ["API endpoints", "Session management"]
  },
  "exploitation_scenarios": [
    {"vector": "Session fixation", "steps": [...], "impact": "high"}
  ],
  "security_controls": [{"control": "CSRF tokens", "status": "implemented"}],
  "gaps": ["No rate limiting on auth endpoints"],
  "recommendations": [{"priority": "high", "action": "Add rate limiting"}],
  "cwe_ids": ["CWE-384", "CWE-307"],
  "confidence": 0.85
}
```

## CLI Options
```bash
council run red-team "task" --health-check --json --verbose
```

## Example
```bash
council run red-team "Threat model the OAuth2 implementation" --json
```

## Cost & Time
- **Cost**: ~$1.30 per analysis
- **Time**: ~11 minutes

## Security Note
Red-team outputs may contain sensitive attack details. Handle findings with appropriate confidentiality. Do not share raw threat models publicly without redaction.
