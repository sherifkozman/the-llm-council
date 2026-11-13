"""Evaluation must distinguish usable output from a complete run."""

from llm_council.engine.orchestrator import CouncilResult
from llm_council.evaluation import EvalCase, evaluate_case_result


def test_successful_but_degraded_run_does_not_pass_default_success_case():
    case = EvalCase(id="review", task="Review", subagent="critic")
    result = CouncilResult(
        success=True,
        output={"review_summary": "partial"},
        execution_plan={"required_phases": [], "degraded_output": True},
    )
    evaluated = evaluate_case_result(case, result)
    assert evaluated.execution_status == "degraded"
    assert evaluated.passed is False
    assert any(item.name == "execution_status" and not item.passed for item in evaluated.criteria)


def test_expected_degraded_run_can_pass_a_degradation_test():
    case = EvalCase.model_validate(
        {
            "id": "fallback",
            "task": "Review",
            "subagent": "critic",
            "expectations": {"success": True, "execution_status": "degraded"},
        }
    )
    result = CouncilResult(
        success=True,
        output={"review_summary": "partial"},
        execution_plan={"required_phases": [], "degraded_output": True},
    )
    evaluated = evaluate_case_result(case, result)
    assert evaluated.execution_status == "degraded"
    assert evaluated.passed is True


def test_expected_failure_keeps_failure_status():
    case = EvalCase.model_validate(
        {
            "id": "failure",
            "task": "Review",
            "subagent": "critic",
            "expectations": {"success": False, "execution_status": "failed"},
        }
    )
    evaluated = evaluate_case_result(case, CouncilResult(success=False))
    assert evaluated.execution_status == "failed"
    assert evaluated.passed is True


async def test_comparison_excludes_fail_fast_partial_coverage(monkeypatch):
    from unittest.mock import AsyncMock

    from llm_council.evaluation import (
        EvalDataset,
        EvalVariantsFile,
        build_eval_report,
        run_eval_comparison,
    )

    dataset = EvalDataset.model_validate(
        {
            "name": "coverage",
            "cases": [
                {"id": "a", "task": "A", "subagent": "critic"},
                {"id": "b", "task": "B", "subagent": "critic"},
            ],
        }
    )
    variants = EvalVariantsFile.model_validate(
        {
            "name": "variants",
            "variants": [
                {"name": "partial"},
                {"name": "complete"},
            ],
        }
    )
    failure = CouncilResult(success=False)
    partial = build_eval_report(
        dataset, [evaluate_case_result(dataset.cases[0], failure)], duration_ms=1
    )
    complete = build_eval_report(
        dataset, [evaluate_case_result(case, failure) for case in dataset.cases], duration_ms=10
    )
    monkeypatch.setattr(
        "llm_council.evaluation.run_eval_dataset", AsyncMock(side_effect=[partial, complete])
    )
    result = await run_eval_comparison(dataset, variants, fail_fast=True)
    assert result.ranking == ["complete"]
    assert result.ineligible_variants["partial"]
    assert result.variant_results[0].report.execution_complete is False


async def test_empty_comparison_has_no_winner(monkeypatch):
    from unittest.mock import AsyncMock

    from llm_council.evaluation import (
        EvalDataset,
        EvalVariantsFile,
        build_eval_report,
        run_eval_comparison,
    )

    dataset = EvalDataset.model_validate({"name": "empty", "cases": []})
    variants = EvalVariantsFile.model_validate({"name": "variants", "variants": [{"name": "one"}]})
    empty = build_eval_report(dataset, [], duration_ms=1)
    monkeypatch.setattr("llm_council.evaluation.run_eval_dataset", AsyncMock(return_value=empty))
    result = await run_eval_comparison(dataset, variants)
    assert result.best_variant is None
    assert result.ranking == []
    assert result.ineligible_variants["one"]


async def test_comparison_does_not_rank_unexpected_degradation(monkeypatch):
    from unittest.mock import AsyncMock

    from llm_council.evaluation import (
        EvalDataset,
        EvalVariantsFile,
        build_eval_report,
        run_eval_comparison,
    )

    dataset = EvalDataset.model_validate(
        {
            "name": "ambiguous",
            "cases": [
                {"id": "a", "task": "A", "subagent": "critic", "expectations": {"success": None}},
            ],
        }
    )
    variants = EvalVariantsFile.model_validate({"name": "variants", "variants": [{"name": "one"}]})
    degraded = CouncilResult(success=True, execution_plan={"degraded_output": True})
    case_result = evaluate_case_result(dataset.cases[0], degraded)
    assert case_result.passed is True  # Explicitly unconstrained assertion stays compatible.
    report = build_eval_report(dataset, [case_result], duration_ms=1)
    monkeypatch.setattr("llm_council.evaluation.run_eval_dataset", AsyncMock(return_value=report))
    comparison = await run_eval_comparison(dataset, variants)
    assert comparison.ranking == []
    assert comparison.ineligible_variants["one"]
