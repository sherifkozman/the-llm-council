"""Provider-aware request compilation for generation requests.

This module centralizes request option handling before adapters execute calls.
Adapters still own wire-format serialization, but compatibility decisions such
as dropping unsupported options or coercing provider-specific defaults are made
here so they can be tested and surfaced in execution metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from llm_council.providers.anthropic import (
    STRUCTURED_OUTPUT_MODEL_PREFIXES as ANTHROPIC_STRUCTURED_OUTPUT_MODEL_PREFIXES,
)
from llm_council.providers.anthropic import (
    STRUCTURED_OUTPUT_MODELS as ANTHROPIC_STRUCTURED_OUTPUT_MODELS,
)
from llm_council.providers.base import GenerateRequest, ReasoningConfig, StructuredOutputConfig
from llm_council.providers.gemini import LEGACY_MODEL_PREFIXES, STRUCTURED_OUTPUT_MODEL_PREFIXES
from llm_council.providers.openai import (
    JSON_MODE_ONLY_MODELS,
    REASONING_MODEL_PREFIXES,
    REASONING_MODELS,
)
from llm_council.providers.openai import (
    STRUCTURED_OUTPUT_MODEL_PREFIXES as OPENAI_STRUCTURED_OUTPUT_MODEL_PREFIXES,
)
from llm_council.providers.openai import (
    STRUCTURED_OUTPUT_MODELS as OPENAI_STRUCTURED_OUTPUT_MODELS,
)
from llm_council.providers.openrouter import _prepare_structured_output_schema
from llm_council.providers.registry import provider_identity
from llm_council.providers.vertex import CLAUDE_MODEL_PREFIXES

CompilationAction = Literal[
    "supported",
    "transformed",
    "dropped",
    "downgraded",
    "ignored",
]


@dataclass(frozen=True)
class CompilationDecision:
    """One provider-specific request-compilation decision."""

    option: str
    action: CompilationAction
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "option": self.option,
            "action": self.action,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CompiledProviderRequest:
    """Compiled request plus decision log."""

    request: GenerateRequest
    decisions: tuple[CompilationDecision, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.request.model,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def compile_request_for_provider(
    provider_name: str,
    request: GenerateRequest,
) -> CompiledProviderRequest:
    """Compile a generic request to the effective request for one provider."""

    compiled = request.model_copy(deep=True)
    decisions: list[CompilationDecision] = []
    identity = provider_identity(provider_name)
    model = compiled.model or ""

    def update(**changes: Any) -> None:
        nonlocal compiled, model
        compiled = compiled.model_copy(update=changes)
        if "model" in changes and isinstance(changes["model"], str):
            model = changes["model"]

    def decide(option: str, action: CompilationAction, detail: str) -> None:
        decisions.append(CompilationDecision(option=option, action=action, detail=detail))

    if identity == "openai":
        if compiled.temperature is not None and not _openai_supports_temperature(model):
            update(temperature=None)
            decide("temperature", "dropped", f"{model} rejects temperature")

        if compiled.reasoning and compiled.reasoning.enabled:
            if _openai_supports_reasoning(model):
                if compiled.reasoning.effort == "none" and model.startswith(REASONING_MODEL_PREFIXES):
                    update(
                        reasoning=compiled.reasoning.model_copy(update={"effort": "medium"})
                    )
                    decide(
                        "reasoning.effort",
                        "transformed",
                        f"{model} does not support effort=none; using medium",
                    )
            else:
                update(reasoning=None)
                decide("reasoning", "dropped", f"{model} does not support reasoning controls")

        if compiled.structured_output:
            if _openai_supports_structured_output(model):
                decide("structured_output", "supported", f"{model} supports json_schema")
            elif model in JSON_MODE_ONLY_MODELS:
                update_kwargs: dict[str, Any] = {"structured_output": None}
                detail = f"{model} only supports json_object mode"
                if compiled.response_format != {"type": "json_object"}:
                    update_kwargs["response_format"] = {"type": "json_object"}
                    if compiled.response_format is not None:
                        detail += "; coerced incompatible response_format to json_object"
                update(**update_kwargs)
                decide(
                    "structured_output",
                    "downgraded",
                    detail,
                )
            else:
                update(structured_output=None)
                decide(
                    "structured_output",
                    "dropped",
                    f"{model} does not support structured output",
                )

    elif identity == "anthropic":
        if compiled.top_p is not None:
            update(top_p=None)
            decide("top_p", "dropped", "Anthropic adapter does not support top_p")
        if compiled.tool_choice is not None:
            update(tool_choice=None)
            decide("tool_choice", "dropped", "Anthropic adapter does not support tool_choice")
        if compiled.reasoning and compiled.reasoning.enabled and compiled.temperature not in (
            None,
            1,
        ):
            update(temperature=1.0)
            decide(
                "temperature",
                "transformed",
                "Anthropic thinking requires temperature=1",
            )
        if compiled.structured_output and not _anthropic_supports_structured_output(model):
            update(structured_output=None)
            decide(
                "structured_output",
                "dropped",
                f"{model} does not support Anthropic structured outputs",
            )

    elif identity == "gemini":
        if compiled.tools:
            update(tools=None)
            decide("tools", "dropped", "Gemini API adapter does not yet forward tool definitions")
        if compiled.tool_choice is not None:
            update(tool_choice=None)
            decide("tool_choice", "dropped", "Gemini API adapter does not support tool_choice")
        if compiled.reasoning and compiled.reasoning.enabled and compiled.reasoning.effort is not None:
            update(reasoning=compiled.reasoning.model_copy(update={"effort": None}))
            decide("reasoning.effort", "ignored", "Gemini uses thinking_level/budget, not effort")
        if compiled.structured_output:
            if _gemini_supports_structured_output(model):
                decide("structured_output", "supported", f"{model} supports response_schema")
            elif _gemini_is_legacy_model(model):
                decide(
                    "structured_output",
                    "supported",
                    f"{model} uses adapter-managed JSON MIME fallback without schema enforcement",
                )
            else:
                update(structured_output=None)
                decide(
                    "structured_output",
                    "dropped",
                    f"{model} does not support Gemini structured output",
                )

    elif identity == "vertex-ai":
        if _vertex_is_claude_model(model):
            if compiled.top_p is not None:
                update(top_p=None)
                decide("top_p", "dropped", "Vertex Claude path does not support top_p")
            if compiled.tool_choice is not None:
                update(tool_choice=None)
                decide(
                    "tool_choice",
                    "dropped",
                    "Vertex Claude path does not support tool_choice",
                )
            if compiled.structured_output and not _anthropic_supports_structured_output(
                _strip_vertex_revision(model)
            ):
                update(structured_output=None)
                decide(
                    "structured_output",
                    "dropped",
                    f"{model} does not support Claude structured outputs on Vertex",
                )
        else:
            if compiled.tools:
                update(tools=None)
                decide("tools", "dropped", "Vertex Gemini path does not yet forward tool definitions")
            if compiled.tool_choice is not None:
                update(tool_choice=None)
                decide("tool_choice", "dropped", "Vertex Gemini path does not support tool_choice")
            if compiled.reasoning and compiled.reasoning.enabled and compiled.reasoning.effort is not None:
                update(reasoning=compiled.reasoning.model_copy(update={"effort": None}))
                decide(
                    "reasoning.effort",
                    "ignored",
                    "Vertex Gemini uses thinking_level/budget, not effort",
                )
            if compiled.structured_output:
                gemini_model = _strip_vertex_revision(model)
                if _gemini_supports_structured_output(gemini_model):
                    decide("structured_output", "supported", f"{model} supports response_schema")
                elif _gemini_is_legacy_model(gemini_model):
                    decide(
                        "structured_output",
                        "supported",
                        f"{model} uses adapter-managed JSON MIME fallback without schema enforcement",
                    )
                else:
                    update(structured_output=None)
                    decide(
                        "structured_output",
                        "dropped",
                        f"{model} does not support Gemini structured output on Vertex",
                    )

    elif identity == "openrouter":
        if compiled.structured_output and compiled.reasoning and compiled.reasoning.enabled:
            update(reasoning=None)
            decide(
                "reasoning",
                "dropped",
                "OpenRouter structured-output requests do not forward reasoning controls",
            )
        if compiled.structured_output:
            schema, strict_mode = _prepare_structured_output_schema(
                dict(compiled.structured_output.json_schema),
                strict=compiled.structured_output.strict,
            )
            structured_output_updates: dict[str, Any] = {}
            if schema != compiled.structured_output.json_schema:
                structured_output_updates["json_schema"] = schema
                decide(
                    "structured_output.json_schema",
                    "transformed",
                    "OpenRouter sanitized schema metadata before request dispatch",
                )
            if compiled.structured_output.strict and not strict_mode:
                structured_output_updates["strict"] = False
                decide(
                    "structured_output.strict",
                    "downgraded",
                    "OpenRouter strict mode disabled because schema is not strict-compatible",
                )
            if structured_output_updates:
                update(
                    structured_output=compiled.structured_output.model_copy(
                        update=structured_output_updates
                    )
                )

    elif identity == "codex":
        _drop_cli_only_options(
            compiled,
            update,
            decide,
            allow_structured_output=True,
            provider_label="Codex CLI",
        )

    elif identity == "claude":
        _drop_cli_only_options(
            compiled,
            update,
            decide,
            allow_structured_output=False,
            provider_label="Claude Code CLI",
        )

    elif identity == "gemini-cli":
        _drop_cli_only_options(
            compiled,
            update,
            decide,
            allow_structured_output=False,
            provider_label="Gemini CLI",
        )

    return CompiledProviderRequest(request=compiled, decisions=tuple(decisions))


def _drop_cli_only_options(
    compiled: GenerateRequest,
    update: Any,
    decide: Any,
    *,
    allow_structured_output: bool,
    provider_label: str,
) -> None:
    if compiled.temperature is not None:
        update(temperature=None)
        decide("temperature", "dropped", f"{provider_label} does not support temperature")
    if compiled.top_p is not None:
        update(top_p=None)
        decide("top_p", "dropped", f"{provider_label} does not support top_p")
    if compiled.stop:
        update(stop=None)
        decide("stop", "dropped", f"{provider_label} does not support stop sequences")
    if compiled.tools:
        update(tools=None)
        decide("tools", "dropped", f"{provider_label} does not support tool definitions")
    if compiled.tool_choice is not None:
        update(tool_choice=None)
        decide("tool_choice", "dropped", f"{provider_label} does not support tool_choice")
    if compiled.response_format is not None:
        update(response_format=None)
        decide("response_format", "dropped", f"{provider_label} does not support response_format")
    if compiled.reasoning and compiled.reasoning.enabled:
        update(reasoning=None)
        decide("reasoning", "dropped", f"{provider_label} does not support reasoning controls")
    if compiled.structured_output and not allow_structured_output:
        update(structured_output=None)
        decide(
            "structured_output",
            "dropped",
            f"{provider_label} does not support structured output",
        )
def _openai_supports_structured_output(model: str) -> bool:
    return model in OPENAI_STRUCTURED_OUTPUT_MODELS or any(
        model.startswith(prefix) for prefix in OPENAI_STRUCTURED_OUTPUT_MODEL_PREFIXES
    )


def _openai_supports_reasoning(model: str) -> bool:
    return model in REASONING_MODELS or model.startswith(REASONING_MODEL_PREFIXES)


def _openai_supports_temperature(model: str) -> bool:
    return not _openai_supports_reasoning(model)


def _anthropic_supports_structured_output(model: str) -> bool:
    normalized = _normalize_model_name(model)
    return normalized in ANTHROPIC_STRUCTURED_OUTPUT_MODELS or any(
        normalized.startswith(prefix) for prefix in ANTHROPIC_STRUCTURED_OUTPUT_MODEL_PREFIXES
    )


def _gemini_supports_structured_output(model: str) -> bool:
    normalized = _normalize_model_name(model)
    return any(normalized.startswith(prefix) for prefix in STRUCTURED_OUTPUT_MODEL_PREFIXES)


def _gemini_is_legacy_model(model: str) -> bool:
    normalized = _normalize_model_name(model)
    return any(normalized.startswith(prefix) for prefix in LEGACY_MODEL_PREFIXES)


def _vertex_is_claude_model(model: str) -> bool:
    return _normalize_model_name(model).startswith(CLAUDE_MODEL_PREFIXES)


def _strip_vertex_revision(model: str) -> str:
    return model.split("@", 1)[0]


def _normalize_model_name(model: str) -> str:
    """Normalize revisioned and publisher-qualified model identifiers."""

    normalized = _strip_vertex_revision(model)
    if "/models/" in normalized:
        normalized = normalized.rsplit("/models/", 1)[1]
    return normalized
