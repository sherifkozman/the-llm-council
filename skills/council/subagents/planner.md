# Planner Subagent

## Purpose
Execution roadmaps with phased plans, dependency analysis, and risk mitigation.

## When to Use
- Planning multi-phase implementations
- Creating sprint/milestone roadmaps
- Breaking down large epics into stories
- Sequencing work with complex dependencies
- Migration planning

## Output Schema
Returns JSON with:
- `phases`: Array with objectives, tasks, dependencies
- `critical_path`: Tasks that gate the project
- `parallelizable_work`: Tasks that can be concurrent
- `risk_mitigation`: How to handle identified risks
- `rollback_plan`: How to undo changes if needed
- `success_criteria`: How to know each phase is complete
- `confidence`: Confidence score (0.0-1.0)

## CLI Options
```bash
council run planner "task" --health-check --json --verbose
```

## Example
```bash
council run planner "Plan the migration from MongoDB to PostgreSQL" --json
```

## Cost & Time
- **Cost**: ~$0.40 per plan
- **Time**: ~5 minutes
