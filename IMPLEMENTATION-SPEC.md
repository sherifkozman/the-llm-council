# Council V2 Implementation Spec

**Branch:** `feature/council-v2`
**Status:** APPROVED
**Decisions:**
- Consolidation: Aliases (backwards compatible)
- CLI Flags: Accept 45% reduction (11→6)
- Phase 4: Include (full refactoring)
**Created:** 2025-12-25

---

## Baseline Metrics (Captured)

| Metric | Current Value | Target | Threshold |
|--------|---------------|--------|-----------|
| Subagent count | 10 | 5-6 | ≥40% reduction |
| Provider count | 7 (5 API + 2 CLI) | 5 | Deprecate CLI adapters |
| CLI flags | 11 | 5-6 | ≥50% reduction |
| Python LOC | 7,137 | ~6,500 | <10% increase |

---

## Phase 1: Prompt Enhancements (No Code Changes)

**Goal:** Add Council Deliberation Protocol to all agent system prompts.

### 1.1 Files to Modify

| File | Change |
|------|--------|
| `src/llm_council/subagents/implementer.yaml` | Add protocol + drafter role |
| `src/llm_council/subagents/architect.yaml` | Add protocol + drafter role |
| `src/llm_council/subagents/researcher.yaml` | Add protocol + researcher role |
| `src/llm_council/subagents/reviewer.yaml` | Add protocol + critic role |
| `src/llm_council/subagents/red-team.yaml` | Add protocol + critic role |
| `src/llm_council/subagents/planner.yaml` | Add protocol + planner role |
| `src/llm_council/subagents/assessor.yaml` | Add protocol + planner role |
| `src/llm_council/subagents/router.yaml` | Add protocol (minimal) |
| `src/llm_council/subagents/shipper.yaml` | Add protocol + synthesizer role |
| `src/llm_council/subagents/test-designer.yaml` | Add protocol + drafter role |

### 1.2 Protocol Template (Add to ALL agents)

```yaml
# Add to system_prompt in each YAML file:

## Council Deliberation Protocol

### 1. Equal Standing
All council members have equal authority regardless of speaking order.
The synthesizer evaluates arguments on merit, not position.

### 2. Constructive Dissent (REQUIRED)
You MUST challenge assumptions and express unorthodox opinions
when grounded in logic, evidence, and facts.
- Do not simply agree with previous agents
- If you see a flaw, state it clearly with reasoning
- Groupthink is the enemy of good reasoning

### 3. Pass When Empty
If you have nothing substantive to add beyond what's been stated:
- Respond with: **PASS**
- Silence is better than redundancy

### 4. Collaborative Rivalry
Aim to produce the winning argument through merit:
- Accuracy, evidence, and clarity are rewarded
- Attack ideas, not agents

### 5. Evidence Required
All claims require supporting reasoning.
Cite sources, examples, or logical derivation.
```

### 1.3 Role-Specific Additions

**Drafter roles (implementer, architect, test-designer):**
```yaml
## Your Role: Drafter
Propose bold, well-reasoned solutions. Make clear recommendations.
Present your top choice with alternatives noted. Don't hedge excessively.
```

**Critic roles (reviewer, red-team):**
```yaml
## Your Role: Critic
You MUST find at least one flaw or risk. Challenge the strongest assumptions.
Propose edge cases. If genuinely excellent, still probe for hidden risks.
Do NOT simply validate or weakly agree.
```

**Synthesizer roles (shipper):**
```yaml
## Your Role: Synthesizer
Weigh arguments by evidence quality, not source.
Resolve disagreements with reasoning. Produce ONE coherent output.
Note consensus vs divergence.
```

**Planner roles (planner, assessor):**
```yaml
## Your Role: Planner
Create actionable plans with clear dependencies.
Identify risks and propose mitigations.
Be specific about what needs to happen, not when.
```

**Researcher roles (researcher):**
```yaml
## Your Role: Researcher
Gather comprehensive information with citations.
Present findings objectively, noting confidence levels.
Distinguish fact from inference.
```

### 1.4 Exit Criteria

- [ ] All 10 subagent YAML files updated with protocol
- [ ] Role-specific additions applied to each agent
- [ ] Manual test: Run `council run reviewer "test"` and verify dissent language appears
- [ ] No schema validation errors

---

## Phase 2: Agent Consolidation

**Goal:** Reduce 10 agents to 5-6 core agents using modes.

### 2.1 Consolidation Map

| Core Agent | Absorbs | Mode Flag | Primary Function |
|------------|---------|-----------|------------------|
| **drafter** | implementer, architect, test-designer | `--mode impl\|arch\|test` | Generate solutions |
| **critic** | reviewer, red-team | `--mode review\|security` | Evaluate and challenge |
| **synthesizer** | shipper | (none) | Merge and finalize |
| **researcher** | (standalone) | (none) | Gather information |
| **planner** | assessor | `--mode plan\|assess` | Strategic planning |
| **router** | (standalone) | (none) | Task classification |

**Result:** 6 agents (from 10) = 40% reduction ✓

### 2.2 Implementation Approach

**Option A: Aliases (Recommended)**
- Keep original YAML files but mark as deprecated
- Create new consolidated YAML files with mode switching
- Route old names to new agents via CLI alias handling
- Backwards compatible

**Option B: Full Replacement**
- Delete old YAML files
- Create new consolidated files only
- Breaking change for existing scripts

**Decision needed from user.**

### 2.3 Files to Create/Modify

| Action | File | Notes |
|--------|------|-------|
| CREATE | `src/llm_council/subagents/drafter.yaml` | Merge impl/arch/test prompts with mode switching |
| CREATE | `src/llm_council/subagents/critic.yaml` | Merge reviewer/red-team with mode switching |
| MODIFY | `src/llm_council/subagents/planner.yaml` | Add assessor mode |
| DEPRECATE | `src/llm_council/subagents/implementer.yaml` | Alias to drafter --mode impl |
| DEPRECATE | `src/llm_council/subagents/architect.yaml` | Alias to drafter --mode arch |
| DEPRECATE | `src/llm_council/subagents/test-designer.yaml` | Alias to drafter --mode test |
| DEPRECATE | `src/llm_council/subagents/reviewer.yaml` | Alias to critic --mode review |
| DEPRECATE | `src/llm_council/subagents/red-team.yaml` | Alias to critic --mode security |
| DEPRECATE | `src/llm_council/subagents/assessor.yaml` | Alias to planner --mode assess |
| DEPRECATE | `src/llm_council/subagents/shipper.yaml` | Alias to synthesizer |
| MODIFY | `src/llm_council/cli/main.py` | Add alias resolution + deprecation warnings |

### 2.4 Schema Consolidation

| Core Schema | Absorbs |
|-------------|---------|
| `drafter.json` | implementer.json, architect.json, test-designer.json |
| `critic.json` | reviewer.json, red-team.json |
| `planner.json` | assessor.json (already similar) |

**Approach:** Use JSON Schema `oneOf` or mode-specific required fields.

### 2.5 Exit Criteria

- [ ] 6 core agent files created/modified
- [ ] Alias resolution working (`council run implementer` → `council run drafter --mode impl`)
- [ ] Deprecation warnings printed for old names
- [ ] All existing functionality preserved
- [ ] Test suite passes

---

## Phase 3: CLI/Provider Simplification

**Goal:** Reduce CLI surface area and deprecate CLI providers.

### 3.1 CLI Flag Reduction

| Current Flag | Action | Rationale |
|--------------|--------|-----------|
| `--health-check` | REMOVE | Use lazy failure instead |
| `--init` | KEEP | Useful for setup |
| `--json` | KEEP | Essential for scripting |
| `--max-retries` | MOVE TO CONFIG | Rarely changed |
| `--models` | KEEP | Core functionality |
| `--no-artifacts` | KEEP | Performance option |
| `--no-degradation` | MOVE TO CONFIG | Rarely changed |
| `--providers` | KEEP | Core functionality |
| `--show` | KEEP | Debugging |
| `--timeout` | MOVE TO CONFIG | Rarely changed |
| `--verbose` | KEEP | Debugging |

**Result:** 11 → 7 flags = 36% reduction (below 50% threshold)

**Additional removal candidates:**
- Combine `--show` and `--verbose` into single `--debug`
- Result: 11 → 6 flags = 45% reduction (still below threshold)

**Alternative:** Accept 36-45% reduction as sufficient given low effort.

### 3.2 Provider Deprecation

| Provider | Action | Replacement |
|----------|--------|-------------|
| `providers/cli/codex.py` | DEPRECATE | Use `openai` provider directly |
| `providers/cli/gemini.py` | DEPRECATE | Use `google` provider directly |

**Implementation:**
1. Add deprecation warning when CLI providers are used
2. Remove from default provider list
3. Keep files for one version, then delete

### 3.3 Files to Modify

| File | Change |
|------|--------|
| `src/llm_council/cli/main.py` | Remove/consolidate flags |
| `src/llm_council/providers/cli/codex.py` | Add deprecation warning |
| `src/llm_council/providers/cli/gemini.py` | Add deprecation warning |
| `src/llm_council/providers/registry.py` | Remove CLI providers from defaults |
| `src/llm_council/config/models.py` | Add config-based defaults for moved flags |

### 3.4 Exit Criteria

- [ ] CLI flags reduced (target: ≥45%)
- [ ] CLI providers show deprecation warning
- [ ] CLI providers removed from default list
- [ ] Config file supports moved flags
- [ ] No functionality loss

---

## Phase 4: Internal Refactoring (CONDITIONAL)

**Gate:** Only proceed if Phases 1-3 succeed AND prototype shows ≥20% gains.

### 4.1 Declarative Tool Registry

**Proceed only if:**
- Baseline tool wiring LOC measured
- Prototype on single agent shows ≥20% reduction

### 4.2 Thin Base Agent

**Proceed only if:**
- Baseline per-role LOC measured
- Prototype on single agent shows ≥20% reduction

**Decision:** Skip for initial implementation. Revisit after Phases 1-3 complete.

---

## Implementation Order

```
Phase 1: Prompt Enhancements
    ├── Update all 10 YAML files with protocol
    ├── Add role-specific prompts
    ├── Test manually
    └── Council review

Phase 2: Agent Consolidation
    ├── Create drafter.yaml (merge impl/arch/test)
    ├── Create critic.yaml (merge reviewer/red-team)
    ├── Update planner.yaml (add assessor mode)
    ├── Add CLI alias resolution
    ├── Add deprecation warnings
    ├── Update schemas
    └── Council review + red-team

Phase 3: CLI/Provider Simplification
    ├── Remove/consolidate CLI flags
    ├── Deprecate CLI providers
    ├── Update config for moved flags
    └── Council review

Final: Commit
    ├── Run full test suite
    ├── Council red-team final
    ├── Fix any issues
    └── Commit with gains report
```

---

## Success Metrics

| Metric | Before | Target | Threshold | Pass? |
|--------|--------|--------|-----------|-------|
| Subagent count | 10 | 6 | ≥40% reduction | TBD |
| CLI flags | 11 | 6 | ≥45% reduction | TBD |
| Provider count | 7 | 5 | Deprecate 2 CLI | TBD |
| Quality regression | 0 | 0 | 0 failures | TBD |
| Dissent in critic | - | ≥70% | Sample review | TBD |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing scripts using old agent names | Aliases + deprecation warnings |
| Quality regression from consolidation | Test each agent individually |
| CLI flag removal breaks workflows | Document in CHANGELOG |
| Schema consolidation causes validation failures | Use `oneOf` for mode-specific fields |

---

## Open Questions (Need User Decision)

1. **Aliases vs Full Replacement?**
   - Aliases: Backwards compatible, more files, deprecation path
   - Full Replacement: Cleaner, breaking change

2. **Accept 45% CLI reduction (vs 50% target)?**
   - Removing more flags would hurt usability

3. **Skip Phase 4 (Internal Refactoring)?**
   - Defer to separate branch after v2 ships

---

## Approval Checklist

Before proceeding, confirm:

- [ ] Phase 1 scope approved (prompt changes only)
- [ ] Phase 2 approach: Aliases or Full Replacement?
- [ ] Phase 3 flag reduction acceptable (45% vs 50%)
- [ ] Phase 4 deferred to future branch
- [ ] Implementation order approved

---

*Awaiting user confirmation to proceed.*
