# Assessor Subagent

## Purpose
Build-vs-buy, go/no-go, and technical tradeoff analysis with weighted criteria.

## When to Use
- Build vs buy decisions
- Go/no-go decisions
- Technology A vs B comparisons
- Cost-benefit analysis
- Vendor selection
- Prioritizing technical debt vs features

## Output Schema
Returns JSON with:
- `decision_question`: What is being evaluated
- `options`: Array with scores, pros, cons, costs, risks
- `evaluation_criteria`: Weighted scoring criteria
- `recommendation`: Suggested choice with rationale
- `sensitivity_analysis`: Robustness of the decision
- `next_steps`: Concrete actions
- `confidence`: Confidence score (0.0-1.0)

## CLI Options
```bash
council run assessor "task" --health-check --json --verbose
```

## Example
```bash
council run assessor "Should we build custom auth or use Auth0?" --json
```

## Cost & Time
- **Cost**: ~$0.35 per assessment
- **Time**: ~2.5 minutes
