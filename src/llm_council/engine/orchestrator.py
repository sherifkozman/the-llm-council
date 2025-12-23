"""Council orchestration engine for multi-LLM workflows.

This module implements a provider-agnostic, async-first orchestrator that runs a
three-phase council flow:

1) Parallel drafts from multiple providers (async :func:`asyncio.gather`)
2) Adversarial critique over those drafts
3) Synthesis into a final JSON response, with optional JSON Schema validation
   and automatic retry on validation failures

The orchestrator is intentionally thin: it delegates all provider interaction to
the :class:`llm_council.providers.base.ProviderAdapter` interface and all output
shape enforcement to JSON Schema + Pydantic models.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from typing import (
    Any,
    TypeVar,
)

from jsonschema import Draft7Validator
from pydantic import BaseModel, ConfigDict, Field

from llm_council.config.models import get_council_models, is_multi_model_enabled
from llm_council.engine.degradation import DegradationAction, DegradationPolicy
from llm_council.engine.health import HealthReport, preflight_check
from llm_council.protocol.types import PhaseTiming, SummaryTier
from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    Message,
    ProviderAdapter,
    StructuredOutputConfig,
)
from llm_council.providers.registry import get_registry
from llm_council.schemas import load_schema
from llm_council.storage.artifacts import ArtifactStore, ArtifactType, get_store
from llm_council.subagents import load_subagent

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OrchestratorConfig(BaseModel):
    """Configuration for the council orchestrator."""

    model_config = ConfigDict(extra="allow")

    timeout: int = Field(default=120, ge=10, le=600, description="Timeout per provider call.")
    max_retries: int = Field(default=3, ge=1, le=10, description="Max synthesis retries.")
    summary_tier: SummaryTier = Field(
        default=SummaryTier.ACTIONS, description="Desired summarization depth."
    )
    max_draft_tokens: int = Field(default=4000, description="Max tokens per draft call.")
    max_critique_tokens: int = Field(default=2000, description="Max tokens for critique call.")
    max_synthesis_tokens: int = Field(default=8000, description="Max tokens for synthesis call.")
    draft_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    critique_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    synthesis_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    enable_schema_validation: bool = Field(default=True)
    strict_providers: bool = Field(
        default=True,
        description=(
            "If True, fail the run when any configured provider cannot be resolved from the "
            "registry. If False, missing providers are skipped."
        ),
    )
    model_overrides: dict[str, str] = Field(default_factory=dict)
    cost_per_1k_input: dict[str, float] = Field(default_factory=dict)
    cost_per_1k_output: dict[str, float] = Field(default_factory=dict)
    enable_health_check: bool = Field(
        default=False, description="Enable preflight health checks before running drafts."
    )
    enable_artifacts: bool = Field(
        default=True, description="Store drafts and synthesis results as artifacts."
    )
    enable_graceful_degradation: bool = Field(
        default=True, description="Use degradation policy for provider failure handling."
    )
    models: list[str] | None = Field(
        default=None,
        description=(
            "List of OpenRouter model IDs for multi-model council. "
            "If set, creates virtual providers for each model. "
            "Example: ['anthropic/claude-3.5-sonnet', 'openai/gpt-4o', 'google/gemini-pro']"
        ),
    )


class CostEstimate(BaseModel):
    """Estimated cost for a council run."""

    model_config = ConfigDict(frozen=True)

    provider_calls: dict[str, int] = Field(default_factory=dict)
    tokens: int = Field(default=0, description="Total tokens (input + output).")
    total_input_tokens: int = Field(default=0)
    total_output_tokens: int = Field(default=0)
    estimated_cost_usd: float = Field(default=0.0)


class CouncilResult(BaseModel):
    """Result payload for an orchestrator run."""

    model_config = ConfigDict(extra="allow")

    success: bool = Field(...)
    error: str | None = Field(default=None, description="Top-level error message, if any.")
    output: dict[str, Any] | None = Field(default=None)
    drafts: dict[str, str] | None = Field(default=None)
    critique: str | None = Field(default=None)
    synthesis_attempts: int = Field(default=1)
    duration_ms: int = Field(default=0)
    phase_timings: list[PhaseTiming] | None = Field(default=None)
    validation_errors: list[str] | None = Field(default=None)
    provider_errors: dict[str, str] | None = Field(
        default=None,
        description="Provider resolution/call errors keyed by provider name.",
    )
    cost_estimate: CostEstimate | None = Field(default=None)
    run_id: str | None = Field(default=None)
    health_report: dict[str, Any] | None = Field(
        default=None, description="Preflight health check report."
    )
    degradation_report: dict[str, Any] | None = Field(
        default=None, description="Degradation events during execution."
    )


class ValidationResult(BaseModel):
    """Validation result for a synthesis attempt."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    data: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    raw: str | None = None


class Orchestrator:
    """Coordinates multi-LLM council runs using provider adapters."""

    def __init__(self, providers: list[str], config: OrchestratorConfig) -> None:
        self._provider_names = providers
        self._config = config
        self._registry = get_registry()
        self._providers: dict[str, ProviderAdapter] = {}
        self._provider_init_errors: dict[str, str] = {}
        self._initialize_providers()
        self._cost_calls: dict[str, int] = {}
        self._input_tokens: dict[str, int] = {}
        self._output_tokens: dict[str, int] = {}
        self._task: str | None = None
        self._subagent_name: str | None = None
        self._subagent_config: dict[str, Any] | None = None
        self._schema: dict[str, Any] | None = None
        self._artifact_store: ArtifactStore | None = None
        self._degradation_policy: DegradationPolicy | None = None
        self._health_report: HealthReport | None = None
        self._run_id: str | None = None

        if self._config.enable_artifacts:
            self._artifact_store = get_store(enabled=True)

        if self._config.enable_graceful_degradation:
            self._degradation_policy = DegradationPolicy(
                max_retries=2,
                min_providers_required=1,
                abort_on_all_failures=True,
            )

    async def run(self, task: str, subagent: str) -> CouncilResult:
        """Run a full council workflow for the given task and subagent."""

        self._cost_calls = {}
        self._input_tokens = {}
        self._output_tokens = {}
        self._task = task
        self._subagent_name = subagent
        phase_timings: list[PhaseTiming] = []
        start_time = time.monotonic()

        try:
            self._subagent_config = load_subagent(subagent)
            schema_name = self._subagent_config.get("schema")
            self._schema = load_schema(schema_name) if schema_name else None
        except Exception as exc:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return CouncilResult(
                success=False,
                error=f"Failed to load subagent/schema: {exc}",
                duration_ms=duration_ms,
                phase_timings=phase_timings or None,
                provider_errors=dict(self._provider_init_errors) or None,
                cost_estimate=self._build_cost_estimate(),
            )

        provider_errors = self._validate_providers_for_run()
        if provider_errors:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return CouncilResult(
                success=False,
                error="Provider resolution failed.",
                duration_ms=duration_ms,
                phase_timings=phase_timings or None,
                provider_errors=provider_errors,
                cost_estimate=self._build_cost_estimate(),
            )

        # Refresh provider adapters for this run in case registry changed.
        self._initialize_providers()

        # Initialize degradation policy for this run
        if self._degradation_policy:
            self._degradation_policy.reset()

        # Create artifact run if enabled
        if self._artifact_store:
            run_record = self._artifact_store.create_run(
                subagent=subagent,
                task=task,
                budget_tokens=4000,
            )
            self._run_id = run_record.run_id

        # Run preflight health checks if enabled
        if self._config.enable_health_check and self._providers:
            try:
                usable_providers, health_report = await preflight_check(
                    self._providers,
                    timeout=10.0,
                    skip_on_failure=True,
                )
                self._health_report = health_report
                self._providers = usable_providers
                logger.info(
                    "Health check: %d/%d providers usable",
                    health_report.usable_count,
                    health_report.total_count,
                )
            except Exception as exc:
                logger.warning("Health check failed: %s", exc)

        if not self._providers:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return CouncilResult(
                success=False,
                error="No usable providers configured.",
                duration_ms=duration_ms,
                phase_timings=phase_timings or None,
                provider_errors=dict(self._provider_init_errors) or None,
                cost_estimate=self._build_cost_estimate(),
            )

        try:
            drafts, draft_timing = await self._timed(self._run_parallel_drafts, "drafts")
            phase_timings.append(draft_timing)

            # Store drafts as artifacts if enabled
            if self._artifact_store and self._run_id:
                for _provider_name, draft_text in drafts.items():
                    if draft_text:
                        try:
                            self._artifact_store.store_artifact(
                                run_id=self._run_id,
                                content=draft_text,
                                artifact_type=ArtifactType.DRAFT,
                            )
                        except Exception as exc:
                            logger.debug("Failed to store draft artifact: %s", exc)

            critique, critique_timing = await self._timed(
                lambda: self._run_critique(drafts), "critique"
            )
            phase_timings.append(critique_timing)

            # Store critique as artifact if enabled
            if self._artifact_store and self._run_id and critique:
                try:
                    self._artifact_store.store_artifact(
                        run_id=self._run_id,
                        content=critique,
                        artifact_type=ArtifactType.CRITIQUE,
                    )
                except Exception as exc:
                    logger.debug("Failed to store critique artifact: %s", exc)

            synth_result, synth_timing = await self._timed(
                lambda: self._run_synthesis(drafts, critique), "synthesis"
            )
            synthesis_result, synth_attempts = synth_result
            phase_timings.append(synth_timing)

            # Store synthesis as artifact if enabled
            if self._artifact_store and self._run_id and synthesis_result.raw:
                try:
                    self._artifact_store.store_artifact(
                        run_id=self._run_id,
                        content=synthesis_result.raw,
                        artifact_type=ArtifactType.SYNTHESIS,
                    )
                except Exception as exc:
                    logger.debug("Failed to store synthesis artifact: %s", exc)
        except Exception as exc:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.exception("Council run failed.")
            return CouncilResult(
                success=False,
                error=str(exc),
                drafts=None,
                critique=None,
                synthesis_attempts=0,
                duration_ms=duration_ms,
                phase_timings=phase_timings or None,
                provider_errors=dict(self._provider_init_errors) or None,
                cost_estimate=self._build_cost_estimate(),
            )

        duration_ms = int((time.monotonic() - start_time) * 1000)

        cost_estimate = self._build_cost_estimate()

        # Complete the run in artifact store if enabled
        if self._artifact_store and self._run_id:
            try:
                status = "completed" if synthesis_result.ok else "failed"
                self._artifact_store.complete_run(self._run_id, status=status)
            except Exception as exc:
                logger.debug("Failed to complete run in artifact store: %s", exc)

        # Get degradation report if policy is enabled
        degradation_report_dict = None
        if self._degradation_policy:
            degradation_report_dict = self._degradation_policy.get_report().to_dict()

        # Get health report if available
        health_report_dict = None
        if self._health_report:
            health_report_dict = self._health_report.to_dict()

        return CouncilResult(
            success=synthesis_result.ok,
            output=synthesis_result.data,
            drafts=drafts,
            critique=critique,
            synthesis_attempts=synth_attempts,
            duration_ms=duration_ms,
            phase_timings=phase_timings,
            validation_errors=synthesis_result.errors or None,
            provider_errors=dict(self._provider_init_errors) or None,
            cost_estimate=cost_estimate,
            run_id=self._run_id,
            health_report=health_report_dict,
            degradation_report=degradation_report_dict,
        )

    async def _run_parallel_drafts(self) -> dict[str, str]:
        """Generate draft responses from all configured providers in parallel."""

        if self._task is None or self._subagent_config is None:
            raise RuntimeError("Orchestrator.run must be called before drafting.")

        draft_tasks = [
            self._generate_draft(provider_name, adapter)
            for provider_name, adapter in self._providers.items()
        ]

        results = await asyncio.gather(*draft_tasks, return_exceptions=True)

        drafts: dict[str, str] = {}
        for provider_name, result in zip(self._providers.keys(), results, strict=True):
            if isinstance(result, BaseException):
                error_msg = str(result)
                self._provider_init_errors[provider_name] = error_msg

                # Use degradation policy to decide action
                if self._degradation_policy:
                    remaining = len(self._providers) - len(self._provider_init_errors)
                    decision = self._degradation_policy.decide(
                        provider=provider_name,
                        error=result if isinstance(result, Exception) else Exception(str(result)),
                        phase="drafts",
                        remaining_providers=remaining,
                    )
                    if decision.action == DegradationAction.ABORT:
                        raise RuntimeError(f"Aborting due to provider failure: {decision.reason}")

                drafts[provider_name] = ""
                continue
            # result is tuple[str, str] here
            name, text = result
            drafts[name] = text

        return drafts

    async def _run_critique(self, drafts: dict[str, str]) -> str:
        """Run an adversarial critique over the drafts."""

        if self._task is None or self._subagent_config is None:
            raise RuntimeError("Orchestrator.run must be called before critique.")

        provider_name, adapter = await self._select_provider_for_phase("critique")
        system_prompt = (
            "You are an adversarial reviewer. Identify errors, gaps, contradictions, "
            "and schema violations. Provide concrete fixes."
        )
        user_prompt = self._format_critique_prompt(self._task, drafts)

        request = GenerateRequest(
            model=self._model_override(provider_name),
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            max_tokens=self._config.max_critique_tokens,
            temperature=self._config.critique_temperature,
        )
        response = await self._call_provider(provider_name, adapter, request)
        return response.text or ""

    async def _run_synthesis(
        self, drafts: dict[str, str], critique: str
    ) -> tuple[ValidationResult, int]:
        """Synthesize final response with schema validation and retry."""

        if self._task is None or self._subagent_config is None:
            raise RuntimeError("Orchestrator.run must be called before synthesis.")

        provider_name, adapter = await self._select_provider_for_phase("synthesis")
        schema = self._schema
        errors: list[str] = []
        last_raw: str | None = None

        for attempt in range(1, self._config.max_retries + 1):
            system_prompt = (
                "You are the synthesizer. Combine drafts and critique into a single response. "
                "Return ONLY valid JSON that matches the provided schema."
            )
            user_prompt = self._format_synthesis_prompt(
                task=self._task,
                drafts=drafts,
                critique=critique,
                schema=schema,
                errors=errors,
            )

            request = GenerateRequest(
                model=self._model_override(provider_name),
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ],
                max_tokens=self._config.max_synthesis_tokens,
                temperature=self._config.synthesis_temperature,
            )

            if schema and await adapter.supports("structured_output"):
                # Use StructuredOutputConfig for provider-agnostic structured output
                # Each provider transforms this to their native API format
                request.structured_output = StructuredOutputConfig(
                    json_schema=schema,
                    name=self._subagent_name or "council_output",
                    strict=True,
                )

            response = await self._call_provider(provider_name, adapter, request)
            last_raw = response.text or ""
            result = self._validate_response(last_raw)
            if result.ok:
                return result, attempt
            errors = result.errors

        return ValidationResult(ok=False, errors=errors, raw=last_raw), self._config.max_retries

    async def _generate_draft(
        self, provider_name: str, adapter: ProviderAdapter
    ) -> tuple[str, str]:
        """Generate a draft response from a single provider."""

        if self._task is None or self._subagent_config is None:
            raise RuntimeError("Orchestrator.run must be called before drafting.")

        system_prompt = self._subagent_config.get("prompts", {}).get("system", "")
        user_prompt = self._format_draft_prompt(self._task)

        request = GenerateRequest(
            model=self._model_override(provider_name),
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            max_tokens=self._config.max_draft_tokens,
            temperature=self._config.draft_temperature,
        )
        response = await self._call_provider(provider_name, adapter, request)
        return provider_name, response.text or ""

    async def _call_provider(
        self, provider_name: str, adapter: ProviderAdapter, request: GenerateRequest
    ) -> GenerateResponse:
        """Execute a provider request with timeout and usage tracking."""

        async def _consume_stream(stream: AsyncIterator[GenerateResponse]) -> GenerateResponse:
            text_parts: list[str] = []
            usage: dict[str, int] | None = None
            async for chunk in stream:
                if chunk.text:
                    text_parts.append(chunk.text)
                if chunk.usage:
                    usage = dict(chunk.usage)
            return GenerateResponse(text="".join(text_parts), usage=usage)

        try:
            result = await asyncio.wait_for(adapter.generate(request), timeout=self._config.timeout)
            if isinstance(result, GenerateResponse):
                response = result
            else:
                response = await asyncio.wait_for(
                    _consume_stream(result), timeout=self._config.timeout
                )

            self._record_usage(provider_name, response.usage)
            return response

        except Exception as exc:
            # Let degradation policy decide if this should be handled
            if self._degradation_policy:
                remaining = len(self._providers) - len(self._provider_init_errors) - 1
                decision = self._degradation_policy.decide(
                    provider=provider_name,
                    error=exc,
                    phase="call",
                    remaining_providers=remaining,
                )

                if decision.action == DegradationAction.ABORT:
                    raise RuntimeError(f"Provider call aborted: {decision.reason}") from exc
                elif decision.action == DegradationAction.RETRY and decision.retry_delay_ms > 0:
                    await asyncio.sleep(decision.retry_delay_ms / 1000.0)

            # Re-raise the exception to let caller handle it
            raise

    def _record_usage(self, provider: str, usage: Mapping[str, int] | None) -> None:
        """Accumulate token usage and call counts for cost estimation."""

        self._cost_calls[provider] = self._cost_calls.get(provider, 0) + 1
        if not usage:
            return
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        self._input_tokens[provider] = self._input_tokens.get(provider, 0) + prompt_tokens
        self._output_tokens[provider] = self._output_tokens.get(provider, 0) + completion_tokens

    def _build_cost_estimate(self) -> CostEstimate:
        """Compute a cost estimate from recorded token usage."""

        total_input = sum(self._input_tokens.values())
        total_output = sum(self._output_tokens.values())
        total_tokens = total_input + total_output
        estimated_cost = 0.0

        for provider in self._provider_names:
            in_tokens = self._input_tokens.get(provider, 0)
            out_tokens = self._output_tokens.get(provider, 0)
            in_rate = self._config.cost_per_1k_input.get(provider, 0.0)
            out_rate = self._config.cost_per_1k_output.get(provider, 0.0)
            estimated_cost += (in_tokens / 1000.0) * in_rate
            estimated_cost += (out_tokens / 1000.0) * out_rate

        return CostEstimate(
            provider_calls=dict(self._cost_calls),
            tokens=total_tokens,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            estimated_cost_usd=round(estimated_cost, 6),
        )

    def _validate_response(self, raw_text: str) -> ValidationResult:
        """Parse and validate the synthesis response against the schema."""

        if not raw_text:
            return ValidationResult(ok=False, errors=["Empty synthesis response."])

        parsed = self._extract_json(raw_text)
        if parsed is None:
            return ValidationResult(ok=False, errors=["Failed to parse JSON."])

        if self._config.enable_schema_validation and self._schema:
            validator = Draft7Validator(self._schema)
            errors = [err.message for err in validator.iter_errors(parsed)]
            if errors:
                return ValidationResult(ok=False, errors=errors, raw=raw_text)

        return ValidationResult(ok=True, data=parsed, raw=raw_text)

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract the first JSON object from a response string.

        Uses balanced brace matching to find complete JSON objects,
        which is more robust than simple find/rfind.
        """
        cleaned = text.strip()

        # Handle markdown code blocks
        if cleaned.startswith("```"):
            # Find closing triple backticks
            end_fence = cleaned.rfind("```")
            cleaned = cleaned[3:end_fence].strip() if end_fence > 3 else cleaned.strip("`")
            # Remove optional language identifier
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        # Try direct parsing first
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Use balanced brace matching to extract JSON object
        extracted = self._extract_balanced_json(cleaned)
        if extracted:
            try:
                parsed = json.loads(extracted)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    def _extract_balanced_json(self, text: str) -> str | None:
        """Extract the first balanced JSON object using brace counting.

        This handles cases where LLMs include commentary after the JSON
        or when there are multiple JSON-like structures in the response.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None

    def _format_draft_prompt(self, task: str) -> str:
        """Format the draft prompt with task details."""

        schema_hint = ""
        if self._schema:
            schema_hint = "\nReturn a draft that aligns with the JSON schema."
        tier_hint = f"\nSummary tier: {self._config.summary_tier.value}"
        return f"Task:\n{task}\n{schema_hint}{tier_hint}\n"

    def _format_critique_prompt(self, task: str, drafts: dict[str, str]) -> str:
        """Format critique prompt with all drafts."""

        draft_blocks = "\n\n".join(
            f"Provider: {name}\nDraft:\n{content}" for name, content in drafts.items()
        )
        schema_hint = ""
        if self._schema:
            schema_hint = "\nSchema (JSON):\n" + json.dumps(self._schema, indent=2)
        tier_hint = f"\nSummary tier: {self._config.summary_tier.value}"
        return f"Task:\n{task}\n{schema_hint}{tier_hint}\n\nDrafts:\n{draft_blocks}"

    def _format_synthesis_prompt(
        self,
        task: str,
        drafts: dict[str, str],
        critique: str,
        schema: dict[str, Any] | None,
        errors: Iterable[str],
    ) -> str:
        """Format synthesis prompt with drafts, critique, schema, and errors."""

        draft_blocks = "\n\n".join(
            f"Provider: {name}\nDraft:\n{content}" for name, content in drafts.items()
        )
        schema_block = json.dumps(schema, indent=2) if schema else "{}"
        error_block = "\n".join(f"- {err}" for err in errors) if errors else "None"
        return (
            f"Task:\n{task}\n\n"
            f"Schema (JSON):\n{schema_block}\n\n"
            f"Summary tier: {self._config.summary_tier.value}\n\n"
            f"Critique:\n{critique}\n\n"
            f"Drafts:\n{draft_blocks}\n\n"
            f"Validation errors to fix (if any):\n{error_block}\n\n"
            "Return ONLY JSON that matches the schema."
        )

    def _model_override(self, provider_name: str) -> str | None:
        """Return the model override for a provider if configured."""

        return self._config.model_overrides.get(provider_name)

    async def _select_provider_for_phase(self, phase: str) -> tuple[str, ProviderAdapter]:
        """Select a provider adapter for a given phase.

        Selection rules:
        - For synthesis: prefer a provider that supports structured output (JSON schema).
        - Otherwise: use the first configured provider that is available.
        """

        if not self._providers:
            raise RuntimeError("No providers available for selection.")

        if phase == "synthesis":
            for name, provider in self._providers.items():
                if await provider.supports("structured_output"):
                    return name, provider

        # Fallback: first available provider.
        name = next(iter(self._providers.keys()))
        return name, self._providers[name]

    async def _timed(
        self, coro_factory: Callable[[], Awaitable[T]], phase: str
    ) -> tuple[T, PhaseTiming]:
        """Measure duration for an async phase."""

        start = time.monotonic()
        result = await coro_factory()
        duration_ms = int((time.monotonic() - start) * 1000)
        return result, PhaseTiming(phase=phase, duration_ms=duration_ms)

    async def doctor(self) -> dict[str, Any]:
        """Check provider availability.

        Returns:
            A dict mapping provider name to a serialized DoctorResult-like payload.
        """

        results: dict[str, Any] = {}

        async def _check(name: str) -> tuple[str, DoctorResult]:
            try:
                adapter = self._registry.get_provider(name)
            except Exception as exc:
                return name, DoctorResult(ok=False, message=str(exc))
            try:
                return name, await adapter.doctor()
            except Exception as exc:
                return name, DoctorResult(ok=False, message=str(exc))

        pairs = await asyncio.gather(*[_check(name) for name in self._provider_names])
        for name, result in pairs:
            results[name] = result.model_dump()
        return results

    def _initialize_providers(self) -> None:
        """Instantiate provider adapters from the registry.

        If multi-model council is enabled (via config.models or COUNCIL_MODELS env var),
        and only 'openrouter' is in the provider list, this method creates virtual
        providers for each model to enable parallel drafts from different LLMs.
        """
        from llm_council.providers.openrouter import create_openrouter_for_model

        self._providers = {}
        self._provider_init_errors = {}

        # Check for multi-model configuration
        models = self._config.models
        if models is None and is_multi_model_enabled():
            models = get_council_models()

        # If we have multiple models and only openrouter is configured,
        # create virtual providers for each model
        if models and len(models) > 1 and self._provider_names == ["openrouter"]:
            logger.info(
                "Multi-model council enabled with %d models: %s",
                len(models),
                ", ".join(models),
            )
            for model in models:
                # Use model name as provider name (e.g., "anthropic/claude-3.5-sonnet")
                try:
                    self._providers[model] = create_openrouter_for_model(model)
                except Exception as exc:
                    self._provider_init_errors[model] = str(exc)
        else:
            # Standard provider initialization
            for name in self._provider_names:
                try:
                    self._providers[name] = self._registry.get_provider(name)
                except Exception as exc:
                    self._provider_init_errors[name] = str(exc)

        if self._provider_init_errors:
            logger.debug("Provider initialization errors: %s", self._provider_init_errors)

        if self._provider_init_errors and self._config.strict_providers:
            # Keep _providers populated only with those that are actually usable.
            for name in list(self._providers.keys()):
                if name in self._provider_init_errors:
                    self._providers.pop(name, None)

    def _validate_providers_for_run(self) -> dict[str, str]:
        """Validate configured providers and return any resolution errors."""

        if not self._provider_names:
            return {"__all__": "No providers configured."}

        if self._provider_init_errors and self._config.strict_providers:
            return dict(self._provider_init_errors)

        return {}


__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "CouncilResult",
    "CostEstimate",
    "ValidationResult",
]
