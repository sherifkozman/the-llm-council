"""Tests for dataset-driven evaluation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_council.engine.orchestrator import CouncilResult
from llm_council.evaluation import (
    EvalDataset,
    EvalReport,
    EvalVariantsFile,
    load_eval_dataset,
    load_eval_variants,
    run_eval_comparison,
    run_eval_dataset,
)
from llm_council.protocol.types import ReasoningProfile, RuntimeProfile


class TestEvalDatasetLoading:
    """Tests for loading eval datasets from disk."""

    def test_load_eval_dataset_yaml(self, tmp_path):
        """YAML datasets should load into structured models."""
        dataset_path = tmp_path / "dataset.yaml"
        dataset_path.write_text(
            """
version: 1
name: smoke
cases:
  - id: planner-plan
    task: Plan this
    subagent: planner
    mode: plan
    expectations:
      execution_plan_values:
        mode: plan
      output_keys: [objective]
"""
        )

        dataset = load_eval_dataset(dataset_path)

        assert isinstance(dataset, EvalDataset)
        assert dataset.name == "smoke"
        assert dataset.cases[0].mode_key == "planner:plan"
        assert dataset._source_path == dataset_path

    def test_load_eval_variants_yaml(self, tmp_path):
        """Variant files should load into structured models."""
        variants_path = tmp_path / "variants.yaml"
        variants_path.write_text(
            """
version: 1
name: compare
variants:
  - name: openai-default
    providers: [openai]
    provider_configs:
      openai:
        default_model: gpt-5.4
"""
        )

        variants = load_eval_variants(variants_path)

        assert isinstance(variants, EvalVariantsFile)
        assert variants.name == "compare"
        assert variants.variants[0].provider_configs["openai"]["default_model"] == "gpt-5.4"


class TestRunEvalDataset:
    """Tests for eval execution and scorecards."""

    @pytest.mark.asyncio
    async def test_run_eval_dataset_scores_case_and_mode(self):
        """Eval should score deterministic expectations and aggregate per mode."""
        dataset = EvalDataset.model_validate(
            {
                "version": 1,
                "name": "smoke",
                "cases": [
                    {
                        "id": "planner-plan",
                        "task": "Plan rollout",
                        "subagent": "planner",
                        "mode": "plan",
                        "expectations": {
                            "execution_plan_values": {
                                "mode": "plan",
                                "execution_profile": "light_tools",
                            },
                            "required_capabilities": ["planning-assess", "repo-analysis"],
                            "executed_capabilities": ["planning-assess", "repo-analysis"],
                            "pending_capabilities": [],
                            "minimum_evidence_items": 2,
                            "output_keys": ["objective", "phases", "risks", "success_criteria"],
                        },
                    }
                ],
            }
        )
        council_result = CouncilResult(
            success=True,
            output={
                "objective": "Ship the first scoped milestone safely.",
                "phases": [
                    {"phase_number": 1, "name": "Phase 1", "tasks": [{}], "deliverables": ["spec"]}
                ],
                "risks": [{"risk": "scope creep", "severity": "medium", "mitigation": "timebox"}],
                "success_criteria": ["Milestone shipped"],
            },
            duration_ms=42,
            execution_plan={
                "mode": "plan",
                "execution_profile": "light_tools",
                "required_capabilities": ["planning-assess", "repo-analysis"],
                "executed_capabilities": ["planning-assess", "repo-analysis"],
                "pending_capabilities": [],
                "evidence_items": 2,
            },
        )

        with patch("llm_council.evaluation.Council") as mock_council_class:
            council = MagicMock()
            council.run = AsyncMock(return_value=council_result)
            mock_council_class.return_value = council

            report = await run_eval_dataset(dataset)

        assert report.total_cases == 1
        assert report.passed_cases == 1
        assert report.case_pass_rate == 1.0
        assert report.mode_scorecards[0].mode_key == "planner:plan"
        assert report.mode_scorecards[0].criteria_pass_rate == 1.0
        orch_config = mock_council_class.call_args.kwargs["config"]
        assert orch_config.mode == "plan"

    @pytest.mark.asyncio
    async def test_run_eval_dataset_can_fail_fast(self):
        """Fail-fast should stop after the first failing case."""
        dataset = EvalDataset.model_validate(
            {
                "version": 1,
                "name": "smoke",
                "cases": [
                    {
                        "id": "case-one",
                        "task": "Review this",
                        "subagent": "critic",
                        "mode": "review",
                        "expectations": {"success": True},
                    },
                    {
                        "id": "case-two",
                        "task": "Research this",
                        "subagent": "researcher",
                        "expectations": {"success": True},
                    },
                ],
            }
        )
        failing_result = CouncilResult(success=False, output=None, duration_ms=5, execution_plan={})

        with patch("llm_council.evaluation.Council") as mock_council_class:
            council = MagicMock()
            council.run = AsyncMock(return_value=failing_result)
            mock_council_class.return_value = council

            report = await run_eval_dataset(dataset, fail_fast=True)

        assert report.total_cases == 1
        assert report.failed_cases == 1
        council.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_eval_dataset_preserves_failure_diagnostics(self):
        """Eval case results should keep top-level provider failure details."""
        dataset = EvalDataset.model_validate(
            {
                "version": 1,
                "name": "smoke",
                "cases": [
                    {
                        "id": "critic-review",
                        "task": "Review this PR",
                        "subagent": "critic",
                        "mode": "review",
                        "expectations": {"success": True},
                    }
                ],
            }
        )
        failed_result = CouncilResult(
            success=False,
            error="Council run failed.",
            provider_errors={
                "openai": "Provider call aborted: Below minimum required providers (1)"
            },
            validation_errors=["Empty synthesis response."],
            duration_ms=20,
            execution_plan={},
        )

        with patch("llm_council.evaluation.Council") as mock_council_class:
            council = MagicMock()
            council.run = AsyncMock(return_value=failed_result)
            mock_council_class.return_value = council

            report = await run_eval_dataset(dataset)

        case_result = report.case_results[0]
        assert case_result.error == "Council run failed."
        assert case_result.provider_errors == {
            "openai": "Provider call aborted: Below minimum required providers (1)"
        }
        assert case_result.validation_errors == ["Empty synthesis response."]

    @pytest.mark.asyncio
    async def test_run_eval_dataset_loads_context_file(self, tmp_path):
        """Case context files should be loaded relative to the dataset path."""
        context_path = tmp_path / "context.md"
        context_path.write_text("## Imported PR Context\npatched diff here\n")
        dataset_path = tmp_path / "dataset.yaml"
        dataset_path.write_text(
            """
version: 1
name: smoke
cases:
  - id: imported-review
    task: Review the supplied PR
    subagent: critic
    mode: review
    disable_local_evidence: true
    context_file: context.md
    expectations:
      success: true
"""
        )
        dataset = load_eval_dataset(dataset_path)
        council_result = CouncilResult(
            success=True, output={"review_summary": "ok"}, duration_ms=10
        )

        with patch("llm_council.evaluation.Council") as mock_council_class:
            council = MagicMock()
            council.run = AsyncMock(return_value=council_result)
            mock_council_class.return_value = council

            await run_eval_dataset(dataset)

        orch_config = mock_council_class.call_args.kwargs["config"]
        assert orch_config.disable_local_evidence is True
        assert "Imported PR Context" in orch_config.system_context

    @pytest.mark.asyncio
    async def test_run_eval_dataset_supports_output_contains_any(self):
        """Weak-label expectations can require a minimum number of clue matches."""
        dataset = EvalDataset.model_validate(
            {
                "version": 1,
                "name": "weak-labels",
                "cases": [
                    {
                        "id": "critic-review",
                        "task": "Review this PR",
                        "subagent": "critic",
                        "mode": "review",
                        "expectations": {
                            "output_contains_any": [
                                "cloudbuild",
                                "provisioning failed",
                                "cache-control",
                            ],
                            "minimum_output_contains_any_matches": 2,
                        },
                    }
                ],
            }
        )
        council_result = CouncilResult(
            success=True,
            output={
                "review_summary": "cloudbuild is broken and cache-control is too aggressive",
                "reasoning": "The cloudbuild deploy flags conflict, and cache-control on icons is too sticky.",
            },
            duration_ms=12,
            execution_plan={},
        )

        with patch("llm_council.evaluation.Council") as mock_council_class:
            council = MagicMock()
            council.run = AsyncMock(return_value=council_result)
            mock_council_class.return_value = council

            report = await run_eval_dataset(dataset)

        assert report.total_cases == 1
        assert report.passed_cases == 1
        criterion = report.case_results[0].criteria[1]
        assert criterion.name == "output_contains_any"
        assert criterion.passed is True

    @pytest.mark.asyncio
    async def test_run_eval_comparison_ranks_variants(self):
        """Comparison should run all variants and rank them by score."""
        dataset = EvalDataset.model_validate({"version": 1, "name": "smoke", "cases": []})
        variants = EvalVariantsFile.model_validate(
            {
                "version": 1,
                "name": "providers",
                "variants": [
                    {"name": "openai", "providers": ["openai"]},
                    {"name": "dual", "providers": ["openai", "google"]},
                ],
            }
        )
        weak_report = EvalReport(
            dataset_name="smoke",
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            case_pass_rate=0.0,
            total_criteria=2,
            passed_criteria=1,
            criteria_pass_rate=0.5,
            duration_ms=50,
            mode_scorecards=[],
            case_results=[],
        )
        strong_report = EvalReport(
            dataset_name="smoke",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            case_pass_rate=1.0,
            total_criteria=2,
            passed_criteria=2,
            criteria_pass_rate=1.0,
            duration_ms=80,
            mode_scorecards=[],
            case_results=[],
        )

        with patch(
            "llm_council.evaluation.run_eval_dataset", new_callable=AsyncMock
        ) as mock_run_eval:
            mock_run_eval.side_effect = [weak_report, strong_report]

            report = await run_eval_comparison(dataset, variants)

        assert report.best_variant == "dual"
        assert report.ranking == ["dual", "openai"]
        assert len(report.variant_results) == 2
        assert report.variant_results[0].variant_name == "openai"

    @pytest.mark.asyncio
    async def test_run_eval_comparison_can_filter_variants(self):
        """Comparison should run only the requested named variants."""
        dataset = EvalDataset.model_validate({"version": 1, "name": "smoke", "cases": []})
        variants = EvalVariantsFile.model_validate(
            {
                "version": 1,
                "name": "providers",
                "variants": [
                    {"name": "openai", "providers": ["openai"]},
                    {"name": "vertex", "providers": ["vertex-ai"]},
                    {"name": "dual", "providers": ["openai", "vertex-ai"]},
                ],
            }
        )
        passing_report = EvalReport(
            dataset_name="smoke",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            case_pass_rate=1.0,
            total_criteria=1,
            passed_criteria=1,
            criteria_pass_rate=1.0,
            duration_ms=10,
            mode_scorecards=[],
            case_results=[],
        )

        with patch(
            "llm_council.evaluation.run_eval_dataset", new_callable=AsyncMock
        ) as mock_run_eval:
            mock_run_eval.side_effect = [passing_report, passing_report]

            report = await run_eval_comparison(
                dataset,
                variants,
                variant_names=["vertex", "dual"],
            )

        assert report.ranking == ["vertex", "dual"]
        assert [item.variant_name for item in report.variant_results] == ["vertex", "dual"]
        assert mock_run_eval.await_count == 2

    @pytest.mark.asyncio
    async def test_run_eval_comparison_variant_can_override_reasoning_profile(self):
        """Variants can force lighter reasoning for bounded benchmark runs."""
        dataset = EvalDataset.model_validate({"version": 1, "name": "smoke", "cases": []})
        variants = EvalVariantsFile.model_validate(
            {
                "version": 1,
                "name": "providers",
                "variants": [
                    {"name": "light-openai", "providers": ["openai"], "reasoning_profile": "light"},
                ],
            }
        )
        passing_report = EvalReport(
            dataset_name="smoke",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            case_pass_rate=1.0,
            total_criteria=1,
            passed_criteria=1,
            criteria_pass_rate=1.0,
            duration_ms=10,
            mode_scorecards=[],
            case_results=[],
        )

        with patch(
            "llm_council.evaluation.run_eval_dataset", new_callable=AsyncMock
        ) as mock_run_eval:
            mock_run_eval.return_value = passing_report

            await run_eval_comparison(dataset, variants)

        base_config = mock_run_eval.await_args.kwargs["base_config"]
        assert base_config.reasoning_profile == ReasoningProfile.LIGHT

    @pytest.mark.asyncio
    async def test_run_eval_comparison_variant_can_override_runtime_profile(self):
        """Variants can force bounded runtime budgets for benchmark runs."""
        dataset = EvalDataset.model_validate({"version": 1, "name": "smoke", "cases": []})
        variants = EvalVariantsFile.model_validate(
            {
                "version": 1,
                "name": "providers",
                "variants": [
                    {
                        "name": "bounded-openai",
                        "providers": ["openai"],
                        "runtime_profile": "bounded",
                    },
                ],
            }
        )
        passing_report = EvalReport(
            dataset_name="smoke",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            case_pass_rate=1.0,
            total_criteria=1,
            passed_criteria=1,
            criteria_pass_rate=1.0,
            duration_ms=10,
            mode_scorecards=[],
            case_results=[],
        )

        with patch(
            "llm_council.evaluation.run_eval_dataset", new_callable=AsyncMock
        ) as mock_run_eval:
            mock_run_eval.return_value = passing_report

            await run_eval_comparison(dataset, variants)

        base_config = mock_run_eval.await_args.kwargs["base_config"]
        assert base_config.runtime_profile == RuntimeProfile.BOUNDED
