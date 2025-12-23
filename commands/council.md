---
name: council
description: Run a multi-LLM council task with adversarial debate
arguments:
  - name: subagent
    description: "Subagent type: implementer, reviewer, architect, researcher, planner, assessor, red-team, test-designer, shipper, router"
    required: true
  - name: task
    description: Task description in quotes
    required: true
---

Run the LLM Council with the specified subagent and task.

> **Requires:** `pip install the-llm-council` and at least one API key configured.

## Subagents
- `implementer` - Feature code, bug fixes
- `reviewer` - Code review, security audit
- `architect` - System design, APIs
- `researcher` - Technical research
- `planner` - Roadmaps, execution plans
- `assessor` - Build vs buy decisions
- `red-team` - Security threat analysis
- `test-designer` - Test suite design
- `shipper` - Release notes
- `router` - Task classification

## Execution

```bash
council run $subagent "$task" --json
```
