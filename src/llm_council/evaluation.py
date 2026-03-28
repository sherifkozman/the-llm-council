"""Dataset-driven evaluation harness for council runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from llm_council.council import Council
from llm_council.engine.orchestrator import CouncilResult
from llm_council.protocol.types import CouncilConfig, ReasoningProfile, RuntimeProfile


class EvalExpectations(BaseModel):
    """Deterministic checks for a single evaluation case."""

    model_config = ConfigDict(extra="forbid")

    success: bool | None = True
    routed: bool | None = None
    execution_plan_values: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)
    executed_capabilities: list[str] = Field(default_factory=list)
    pending_capabilities: list[str] | None = None
    minimum_evidence_items: int | None = None
    output_keys: list[str] = Field(default_factory=list)
    output_contains: list[str] = Field(default_factory=list)
    output_contains_any: list[str] = Field(default_factory=list)
    minimum_output_contains_any_matches: int | None = None
    output_not_contains: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    """One evaluation case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    task: str
    subagent: str
    mode: str | None = None
    follow_router: bool = False
    model_pack: str | None = None
    execution_profile: str | None = None
    budget_class: str | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    disable_local_evidence: bool = False
    context_file: str | None = None
    system_context: str | None = None
    expectations: EvalExpectations = Field(default_factory=EvalExpectations)

    @property
    def mode_key(self) -> str:
        """Return a stable mode bucket for scorecards."""

        return f"{self.subagent}:{self.mode or 'default'}"


class EvalDataset(BaseModel):
    """Evaluation dataset definition."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str
    description: str | None = None
    cases: list[EvalCase]
    _source_path: Path | None = PrivateAttr(default=None)


class EvalCriterionResult(BaseModel):
    """Outcome of one deterministic criterion."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    expected: Any | None = None
    actual: Any | None = None
    message: str | None = None


class EvalCaseResult(BaseModel):
    """Evaluation result for a single case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    mode_key: str
    passed: bool
    success: bool
    error: str | None = None
    validation_errors: list[str] | None = None
    provider_errors: dict[str, str] | None = None
    duration_ms: int
    criteria: list[EvalCriterionResult]
    execution_plan: dict[str, Any] = Field(default_factory=dict)


class EvalModeScorecard(BaseModel):
    """Aggregated scorecard for one subagent/mode bucket."""

    model_config = ConfigDict(extra="forbid")

    mode_key: str
    total_cases: int
    passed_cases: int
    case_pass_rate: float
    total_criteria: int
    passed_criteria: int
    criteria_pass_rate: float
    average_duration_ms: int


class EvalReport(BaseModel):
    """Full evaluation report."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_description: str | None = None
    total_cases: int
    passed_cases: int
    failed_cases: int
    case_pass_rate: float
    total_criteria: int
    passed_criteria: int
    criteria_pass_rate: float
    duration_ms: int
    mode_scorecards: list[EvalModeScorecard]
    case_results: list[EvalCaseResult]


class EvalVariant(BaseModel):
    """One named runtime variant for eval comparison."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    providers: list[str] = Field(default_factory=list)
    models: list[str] | None = None
    model_overrides: dict[str, str] = Field(default_factory=dict)
    provider_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    enable_health_check: bool | None = None
    runtime_profile: RuntimeProfile | None = None
    reasoning_profile: ReasoningProfile | None = None


class EvalVariantsFile(BaseModel):
    """Collection of named eval variants loaded from disk."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str
    description: str | None = None
    variants: list[EvalVariant]


class EvalVariantResult(BaseModel):
    """Evaluation outcome for one named variant."""

    model_config = ConfigDict(extra="forbid")

    variant_name: str
    description: str | None = None
    providers: list[str]
    models: list[str] | None = None
    model_overrides: dict[str, str] = Field(default_factory=dict)
    report: EvalReport


class EvalComparisonReport(BaseModel):
    """Comparison report across multiple runtime variants."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_description: str | None = None
    variants_name: str | None = None
    variants_description: str | None = None
    best_variant: str | None = None
    ranking: list[str] = Field(default_factory=list)
    variant_results: list[EvalVariantResult]


def load_eval_dataset(path: str | Path) -> EvalDataset:
    """Load an evaluation dataset from YAML or JSON."""

    dataset_path = Path(path)
    raw = dataset_path.read_text(encoding="utf-8")
    data = json.loads(raw) if dataset_path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Evaluation dataset must be an object: {dataset_path}")
    dataset = EvalDataset.model_validate(data)
    dataset._source_path = dataset_path
    return dataset


def load_eval_variants(path: str | Path) -> EvalVariantsFile:
    """Load a named variant set from YAML or JSON."""

    variants_path = Path(path)
    raw = variants_path.read_text(encoding="utf-8")
    data = (
        json.loads(raw) if variants_path.suffix.lower() == ".json" else yaml.safe_load(raw)
    )
    if not isinstance(data, dict):
        raise ValueError(f"Evaluation variants file must be an object: {variants_path}")
    return EvalVariantsFile.model_validate(data)


async def run_eval_dataset(
    dataset: EvalDataset,
    *,
    base_config: CouncilConfig | None = None,
    case_ids: list[str] | None = None,
    max_cases: int | None = None,
    fail_fast: bool = False,
) -> EvalReport:
    """Run a dataset and compute deterministic scorecards."""

    started = time.perf_counter()
    selected_cases = _select_cases(dataset.cases, case_ids=case_ids, max_cases=max_cases)
    results: list[EvalCaseResult] = []
    config = base_config or CouncilConfig()

    for case in selected_cases:
        case_config = config.model_copy(
            update={
                "mode": case.mode,
                "model_pack": case.model_pack,
                "execution_profile": case.execution_profile,
                "budget_class": case.budget_class,
                "required_capabilities": list(case.required_capabilities),
                "disable_local_evidence": case.disable_local_evidence,
                "system_context": _load_case_context(case, dataset),
                "follow_router": case.follow_router,
            }
        )
        council = Council(config=case_config)
        result = await council.run(
            task=case.task,
            subagent=case.subagent,
            follow_router=case.follow_router,
        )
        case_result = evaluate_case_result(case, result)
        results.append(case_result)
        if fail_fast and not case_result.passed:
            break

    duration_ms = int((time.perf_counter() - started) * 1000)
    return build_eval_report(dataset, results, duration_ms=duration_ms)


async def run_eval_comparison(
    dataset: EvalDataset,
    variants: EvalVariantsFile,
    *,
    base_config: CouncilConfig | None = None,
    variant_names: list[str] | None = None,
    case_ids: list[str] | None = None,
    max_cases: int | None = None,
    fail_fast: bool = False,
) -> EvalComparisonReport:
    """Run the same dataset against multiple named runtime variants."""

    base = base_config or CouncilConfig()
    results: list[EvalVariantResult] = []
    selected_variants = _select_variants(variants.variants, variant_names=variant_names)

    for variant in selected_variants:
        variant_config = _merge_variant_config(base, variant)
        report = await run_eval_dataset(
            dataset,
            base_config=variant_config,
            case_ids=case_ids,
            max_cases=max_cases,
            fail_fast=fail_fast,
        )
        results.append(
            EvalVariantResult(
                variant_name=variant.name,
                description=variant.description,
                providers=list(variant_config.providers),
                models=list(variant_config.models) if variant_config.models else None,
                model_overrides=dict(variant_config.model_overrides),
                report=report,
            )
        )

    ranked = sorted(
        results,
        key=lambda item: (
            item.report.case_pass_rate,
            item.report.criteria_pass_rate,
            -item.report.duration_ms,
        ),
        reverse=True,
    )
    ranking = [item.variant_name for item in ranked]
    return EvalComparisonReport(
        dataset_name=dataset.name,
        dataset_description=dataset.description,
        variants_name=variants.name,
        variants_description=variants.description,
        best_variant=ranking[0] if ranking else None,
        ranking=ranking,
        variant_results=results,
    )


def evaluate_case_result(case: EvalCase, result: CouncilResult) -> EvalCaseResult:
    """Evaluate one council result against deterministic expectations."""

    criteria: list[EvalCriterionResult] = []
    expectations = case.expectations
    execution_plan = result.execution_plan or {}
    output = result.output or {}
    output_text = json.dumps(output, sort_keys=True).lower()

    if expectations.success is not None:
        criteria.append(
            _criterion(
                "success",
                result.success == expectations.success,
                expected=expectations.success,
                actual=result.success,
            )
        )

    if expectations.routed is not None:
        criteria.append(
            _criterion(
                "routed",
                result.routed == expectations.routed,
                expected=expectations.routed,
                actual=result.routed,
            )
        )

    for key, expected_value in expectations.execution_plan_values.items():
        actual_value = execution_plan.get(key)
        criteria.append(
            _criterion(
                f"execution_plan.{key}",
                actual_value == expected_value,
                expected=expected_value,
                actual=actual_value,
            )
        )

    if expectations.required_capabilities:
        actual_capabilities = list(execution_plan.get("required_capabilities") or [])
        missing = [item for item in expectations.required_capabilities if item not in actual_capabilities]
        criteria.append(
            _criterion(
                "required_capabilities",
                not missing,
                expected=expectations.required_capabilities,
                actual=actual_capabilities,
                message=f"Missing required capabilities: {', '.join(missing)}" if missing else None,
            )
        )

    if expectations.executed_capabilities:
        actual_executed = list(execution_plan.get("executed_capabilities") or [])
        missing = [item for item in expectations.executed_capabilities if item not in actual_executed]
        criteria.append(
            _criterion(
                "executed_capabilities",
                not missing,
                expected=expectations.executed_capabilities,
                actual=actual_executed,
                message=f"Missing executed capabilities: {', '.join(missing)}" if missing else None,
            )
        )

    if expectations.pending_capabilities is not None:
        actual_pending = sorted(execution_plan.get("pending_capabilities") or [])
        expected_pending = sorted(expectations.pending_capabilities)
        criteria.append(
            _criterion(
                "pending_capabilities",
                actual_pending == expected_pending,
                expected=expected_pending,
                actual=actual_pending,
            )
        )

    if expectations.minimum_evidence_items is not None:
        actual_evidence_items = int(execution_plan.get("evidence_items") or 0)
        criteria.append(
            _criterion(
                "minimum_evidence_items",
                actual_evidence_items >= expectations.minimum_evidence_items,
                expected=expectations.minimum_evidence_items,
                actual=actual_evidence_items,
            )
        )

    if expectations.output_keys:
        actual_keys = sorted(output.keys())
        missing = [item for item in expectations.output_keys if item not in output]
        criteria.append(
            _criterion(
                "output_keys",
                not missing,
                expected=expectations.output_keys,
                actual=actual_keys,
                message=f"Missing output keys: {', '.join(missing)}" if missing else None,
            )
        )

    for item in expectations.output_contains:
        lowered = item.lower()
        criteria.append(
            _criterion(
                f"output_contains:{item}",
                lowered in output_text,
                expected=item,
                actual=output_text,
            )
        )

    if expectations.output_contains_any:
        matched_items = [item for item in expectations.output_contains_any if item.lower() in output_text]
        minimum_matches = expectations.minimum_output_contains_any_matches or 1
        criteria.append(
            _criterion(
                "output_contains_any",
                len(matched_items) >= minimum_matches,
                expected={
                    "minimum_matches": minimum_matches,
                    "phrases": expectations.output_contains_any,
                },
                actual={
                    "matched_count": len(matched_items),
                    "matched_phrases": matched_items,
                },
                message=f"Matched {len(matched_items)} expected clues",
            )
        )

    for item in expectations.output_not_contains:
        lowered = item.lower()
        criteria.append(
            _criterion(
                f"output_not_contains:{item}",
                lowered not in output_text,
                expected=item,
                actual=output_text,
            )
        )

    passed = all(item.passed for item in criteria)
    return EvalCaseResult(
        case_id=case.id,
        mode_key=case.mode_key,
        passed=passed,
        success=result.success,
        error=result.error,
        validation_errors=list(result.validation_errors) if result.validation_errors else None,
        provider_errors=dict(result.provider_errors) if result.provider_errors else None,
        duration_ms=result.duration_ms,
        criteria=criteria,
        execution_plan=execution_plan,
    )


def build_eval_report(
    dataset: EvalDataset,
    case_results: list[EvalCaseResult],
    *,
    duration_ms: int,
) -> EvalReport:
    """Aggregate case results into a report and per-mode scorecards."""

    total_cases = len(case_results)
    passed_cases = sum(1 for item in case_results if item.passed)
    failed_cases = total_cases - passed_cases
    total_criteria = sum(len(item.criteria) for item in case_results)
    passed_criteria = sum(1 for item in case_results for criterion in item.criteria if criterion.passed)

    return EvalReport(
        dataset_name=dataset.name,
        dataset_description=dataset.description,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        case_pass_rate=_ratio(passed_cases, total_cases),
        total_criteria=total_criteria,
        passed_criteria=passed_criteria,
        criteria_pass_rate=_ratio(passed_criteria, total_criteria),
        duration_ms=duration_ms,
        mode_scorecards=_build_mode_scorecards(case_results),
        case_results=case_results,
    )


def _select_cases(
    cases: list[EvalCase],
    *,
    case_ids: list[str] | None,
    max_cases: int | None,
) -> list[EvalCase]:
    selected = cases
    if case_ids:
        case_id_set = set(case_ids)
        selected = [case for case in selected if case.id in case_id_set]
    if max_cases is not None:
        selected = selected[:max_cases]
    return selected


def _select_variants(
    variants: list[EvalVariant],
    *,
    variant_names: list[str] | None,
) -> list[EvalVariant]:
    """Select requested variants while preserving file order."""

    if not variant_names:
        return variants

    requested = {name.strip() for name in variant_names if name.strip()}
    selected = [variant for variant in variants if variant.name in requested]
    missing = sorted(requested - {variant.name for variant in selected})
    if missing:
        raise ValueError(f"Unknown eval variant(s): {', '.join(missing)}")
    return selected


def _merge_variant_config(base: CouncilConfig, variant: EvalVariant) -> CouncilConfig:
    """Merge a named variant onto a base council config."""

    provider_configs = {
        key: dict(value) for key, value in base.provider_configs.items()
    }
    for provider_name, config in variant.provider_configs.items():
        provider_configs[provider_name] = dict(config)

    model_overrides = dict(base.model_overrides)
    model_overrides.update(variant.model_overrides)

    return base.model_copy(
        update={
            "providers": list(variant.providers) if variant.providers else list(base.providers),
            "models": list(variant.models) if variant.models is not None else base.models,
            "provider_configs": provider_configs,
            "model_overrides": model_overrides,
            "runtime_profile": (
                variant.runtime_profile
                if variant.runtime_profile is not None
                else base.runtime_profile
            ),
            "reasoning_profile": (
                variant.reasoning_profile
                if variant.reasoning_profile is not None
                else base.reasoning_profile
            ),
            "enable_health_check": (
                variant.enable_health_check
                if variant.enable_health_check is not None
                else base.enable_health_check
            ),
        }
    )


def _load_case_context(case: EvalCase, dataset: EvalDataset) -> str | None:
    parts: list[str] = []
    if case.system_context:
        parts.append(case.system_context)
    if case.context_file:
        context_path = Path(case.context_file)
        if not context_path.is_absolute() and dataset._source_path is not None:
            context_path = dataset._source_path.parent / context_path
        parts.append(context_path.read_text(encoding="utf-8"))
    if not parts:
        return None
    return "\n\n".join(part for part in parts if part)


def _build_mode_scorecards(case_results: list[EvalCaseResult]) -> list[EvalModeScorecard]:
    buckets: dict[str, list[EvalCaseResult]] = {}
    for result in case_results:
        buckets.setdefault(result.mode_key, []).append(result)

    scorecards: list[EvalModeScorecard] = []
    for mode_key in sorted(buckets):
        items = buckets[mode_key]
        total_cases = len(items)
        passed_cases = sum(1 for item in items if item.passed)
        total_criteria = sum(len(item.criteria) for item in items)
        passed_criteria = sum(1 for item in items for criterion in item.criteria if criterion.passed)
        average_duration_ms = int(sum(item.duration_ms for item in items) / total_cases) if total_cases else 0
        scorecards.append(
            EvalModeScorecard(
                mode_key=mode_key,
                total_cases=total_cases,
                passed_cases=passed_cases,
                case_pass_rate=_ratio(passed_cases, total_cases),
                total_criteria=total_criteria,
                passed_criteria=passed_criteria,
                criteria_pass_rate=_ratio(passed_criteria, total_criteria),
                average_duration_ms=average_duration_ms,
            )
        )
    return scorecards


def _criterion(
    name: str,
    passed: bool,
    *,
    expected: Any | None = None,
    actual: Any | None = None,
    message: str | None = None,
) -> EvalCriterionResult:
    return EvalCriterionResult(
        name=name,
        passed=passed,
        expected=expected,
        actual=actual,
        message=message,
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


__all__ = [
    "EvalCase",
    "EvalCaseResult",
    "EvalComparisonReport",
    "EvalCriterionResult",
    "EvalDataset",
    "EvalExpectations",
    "EvalModeScorecard",
    "EvalReport",
    "EvalVariant",
    "EvalVariantResult",
    "EvalVariantsFile",
    "build_eval_report",
    "evaluate_case_result",
    "load_eval_dataset",
    "load_eval_variants",
    "run_eval_comparison",
    "run_eval_dataset",
]
