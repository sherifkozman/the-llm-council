"""Tests for provider-aware request compilation."""

from __future__ import annotations

import pytest

from llm_council.providers.base import (
    GenerateRequest,
    PromptCacheConfig,
    ReasoningConfig,
    StructuredOutputConfig,
)
from llm_council.providers.compiler import compile_request_for_provider


def _schema(*, required: list[str] | None = None, additional_properties: bool | None = False):
    payload: dict[str, object] = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "summary": {"type": "string"},
        },
    }
    if required is not None:
        payload["required"] = required
    if additional_properties is not None:
        payload["additionalProperties"] = additional_properties
    return payload


class TestRequestCompiler:
    """Tests for request compilation across all current providers."""

    @pytest.mark.parametrize("provider,model", [("codex", "gpt-5.4"), ("claude", "claude-opus-5")])
    @pytest.mark.parametrize("max_tokens", [None, 4000])
    def test_cli_output_budget_is_explicitly_dropped_only_when_requested(
        self, provider: str, model: str, max_tokens: int | None
    ) -> None:
        original = GenerateRequest(
            prompt="review",
            model=model,
            max_tokens=max_tokens,
            timeout_seconds=12.0,
            temperature=0.2,
            reasoning=ReasoningConfig(enabled=True, effort="high"),
        )
        original_payload = original.model_dump()
        compiled = compile_request_for_provider(provider, original)

        assert original.model_dump() == original_payload
        assert compiled.request.max_tokens is None
        assert compiled.request.timeout_seconds == 12.0
        max_token_decisions = [
            item for item in compiled.to_dict()["decisions"] if item["option"] == "max_tokens"
        ]
        if max_tokens is None:
            assert max_token_decisions == []
        else:
            assert max_token_decisions == [
                {
                    "option": "max_tokens",
                    "action": "dropped",
                    "detail": f"requested max_tokens={max_tokens} is not enforced; "
                    "no verified whole-generation output token limit; "
                    "native output may exceed requested limit; timeout still applies",
                }
            ]
        assert compiled.request.temperature is None
        assert any(
            item.option == "temperature" and item.action == "dropped" for item in compiled.decisions
        )
        assert compiled.request.reasoning == ReasoningConfig(enabled=True, effort="high")
        control = compiled.to_dict()["reasoning_control"]
        assert control["status"] == "requested"
        assert control["effort"] == "high"

    @pytest.mark.parametrize(
        "provider", ["openai", "anthropic", "openrouter", "gemini", "vertex-ai", "gemini-cli"]
    )
    @pytest.mark.parametrize("max_tokens", [None, 4000])
    def test_output_budget_compilation_is_unchanged_for_other_routes(
        self, provider: str, max_tokens: int | None
    ) -> None:
        original = GenerateRequest(prompt="test", max_tokens=max_tokens, timeout_seconds=12.0)
        compiled = compile_request_for_provider(provider, original)

        assert compiled.request.max_tokens == max_tokens
        assert original.max_tokens == max_tokens
        assert compiled.request.timeout_seconds == 12.0
        assert not any(item.option == "max_tokens" for item in compiled.decisions)

    @pytest.mark.parametrize("provider,model", [("codex", "gpt-5.4"), ("claude", "claude-opus-5")])
    @pytest.mark.parametrize("effort", ["high", "low"])
    def test_cli_forwards_verified_effort_without_other_provider_knobs(
        self, provider, model, effort
    ):
        original = GenerateRequest(
            prompt="review",
            model=model,
            reasoning=ReasoningConfig(
                enabled=True, effort=effort, budget_tokens=16384, thinking_level="high"
            ),
        )
        compiled = compile_request_for_provider(provider, original)
        assert compiled.request.reasoning == ReasoningConfig(enabled=True, effort=effort)
        assert original.reasoning.budget_tokens == 16384
        actions = {(item.option, item.action) for item in compiled.decisions}
        assert ("reasoning.effort", "supported") in actions
        assert ("reasoning.budget_tokens", "ignored") in actions
        assert ("reasoning.thinking_level", "ignored") in actions
        metadata = compiled.to_dict()["reasoning_control"]
        assert metadata["status"] == "requested"
        assert metadata["effort"] == effort
        assert metadata["requires_cli_version"]

    def test_codex_explicit_off_is_not_conflated_with_uncontrolled_default(self):
        off = compile_request_for_provider(
            "codex", GenerateRequest(prompt="test", reasoning=ReasoningConfig(enabled=False))
        )
        absent = compile_request_for_provider("codex", GenerateRequest(prompt="test"))
        assert off.request.reasoning == ReasoningConfig(enabled=False)
        assert off.to_dict()["reasoning_control"]["effort"] == "none"
        assert absent.request.reasoning is None
        assert absent.to_dict()["reasoning_control"] == {"status": "uncontrolled"}

    @pytest.mark.parametrize(
        "provider,model,reasoning",
        [
            ("claude", "claude-opus-5", ReasoningConfig(enabled=False)),
            ("claude", "sonnet", ReasoningConfig(enabled=True, effort="high")),
            ("claude", None, ReasoningConfig(enabled=True, effort="high")),
            ("claude", "claude-opus-5", ReasoningConfig(enabled=True, effort="medium")),
            ("codex", "gpt-5.4", ReasoningConfig(enabled=True, effort="medium")),
            ("codex", "gpt-5.4", ReasoningConfig(enabled=True, budget_tokens=4096)),
            ("codex", "gpt-5.4", ReasoningConfig(enabled=False, effort="high")),
            ("gemini-cli", "gemini-3-pro-preview", ReasoningConfig(enabled=False)),
        ],
    )
    def test_unsupported_cli_reasoning_is_explicitly_uncontrolled(self, provider, model, reasoning):
        compiled = compile_request_for_provider(
            provider, GenerateRequest(prompt="test", model=model, reasoning=reasoning)
        )
        assert compiled.request.reasoning is None
        assert compiled.to_dict()["reasoning_control"]["status"] == "uncontrolled"
        assert any(
            item.option == "reasoning" and item.action == "dropped" for item in compiled.decisions
        )

    @pytest.mark.parametrize(
        "provider", ["openai", "openrouter", "anthropic", "gemini", "vertex-ai"]
    )
    def test_explicit_disabled_reasoning_keeps_api_payload_behavior(self, provider):
        compiled = compile_request_for_provider(
            provider, GenerateRequest(prompt="test", reasoning=ReasoningConfig(enabled=False))
        )
        assert compiled.request.reasoning == ReasoningConfig(enabled=False)
        assert "reasoning_control" not in compiled.to_dict()

    def test_openai_drops_temperature_for_reasoning_models(self):
        compiled = compile_request_for_provider(
            "openai",
            GenerateRequest(
                prompt="test",
                model="o3-mini",
                temperature=0.2,
            ),
        )

        assert compiled.request.temperature is None
        assert compiled.decisions[0].option == "temperature"
        assert compiled.decisions[0].action == "dropped"

    def test_openai_downgrades_json_schema_to_json_mode_for_legacy_models(self):
        compiled = compile_request_for_provider(
            "openai",
            GenerateRequest(
                prompt="test",
                model="gpt-3.5-turbo",
                structured_output=StructuredOutputConfig(
                    json_schema=_schema(required=["result"], additional_properties=False),
                    name="legacy",
                    strict=True,
                ),
            ),
        )

        assert compiled.request.structured_output is None
        assert compiled.request.response_format == {"type": "json_object"}
        assert any(decision.action == "downgraded" for decision in compiled.decisions)

    def test_openai_legacy_json_mode_coerces_incompatible_response_format(self):
        compiled = compile_request_for_provider(
            "openai",
            GenerateRequest(
                prompt="test",
                model="gpt-3.5-turbo",
                response_format={"type": "text"},
                structured_output=StructuredOutputConfig(
                    json_schema=_schema(required=["result"], additional_properties=False),
                    name="legacy",
                    strict=True,
                ),
            ),
        )

        assert compiled.request.structured_output is None
        assert compiled.request.response_format == {"type": "json_object"}
        assert compiled.decisions[0].detail.endswith(
            "coerced incompatible response_format to json_object"
        )

    def test_anthropic_forces_temperature_for_reasoning_and_drops_unsupported_fields(self):
        compiled = compile_request_for_provider(
            "anthropic",
            GenerateRequest(
                prompt="test",
                model="claude-opus-4-6",
                temperature=0.3,
                top_p=0.8,
                tool_choice={"type": "auto"},
                reasoning=ReasoningConfig(enabled=True, budget_tokens=2048),
            ),
        )

        assert compiled.request.temperature == 1.0
        assert compiled.request.top_p is None
        assert compiled.request.tool_choice is None
        actions = {(decision.option, decision.action) for decision in compiled.decisions}
        assert ("temperature", "transformed") in actions
        assert ("top_p", "dropped") in actions
        assert ("tool_choice", "dropped") in actions

    def test_anthropic_keeps_automatic_prompt_cache_config(self):
        compiled = compile_request_for_provider(
            "anthropic",
            GenerateRequest(
                prompt="test",
                model="claude-opus-4-6",
                prompt_cache=PromptCacheConfig(ttl="1h"),
            ),
        )

        assert compiled.request.prompt_cache is not None
        assert compiled.request.prompt_cache.ttl == "1h"
        actions = {(decision.option, decision.action) for decision in compiled.decisions}
        assert ("prompt_cache", "supported") in actions

    def test_non_cache_control_providers_drop_prompt_cache_config(self):
        for provider, model in (
            ("openai", "gpt-5.4"),
            ("gemini", "gemini-3.1-pro-preview"),
            ("vertex-ai", "gemini-3.1-pro-preview"),
        ):
            compiled = compile_request_for_provider(
                provider,
                GenerateRequest(
                    prompt="test",
                    model=model,
                    prompt_cache=PromptCacheConfig(),
                ),
            )

            assert compiled.request.prompt_cache is None
            assert any(
                decision.option == "prompt_cache" and decision.action == "dropped"
                for decision in compiled.decisions
            )

    def test_openrouter_keeps_prompt_cache_for_anthropic_routes(self):
        compiled = compile_request_for_provider(
            "openrouter",
            GenerateRequest(
                prompt="test",
                model="anthropic/claude-opus-4-6",
                prompt_cache=PromptCacheConfig(ttl="1h"),
            ),
        )

        assert compiled.request.prompt_cache is not None
        assert compiled.request.prompt_cache.ttl == "1h"
        assert any(
            decision.option == "prompt_cache" and decision.action == "supported"
            for decision in compiled.decisions
        )

    def test_openrouter_drops_prompt_cache_for_unsupported_routes(self):
        compiled = compile_request_for_provider(
            "openrouter",
            GenerateRequest(
                prompt="test",
                model="openai/gpt-5.4",
                prompt_cache=PromptCacheConfig(),
            ),
        )

        assert compiled.request.prompt_cache is None
        assert any(
            decision.option == "prompt_cache"
            and decision.action == "dropped"
            and "route-dependent" in decision.detail
            for decision in compiled.decisions
        )

    def test_openrouter_virtual_provider_names_keep_prompt_cache_for_anthropic_routes(self):
        compiled = compile_request_for_provider(
            "anthropic/claude-opus-4-6",
            GenerateRequest(
                prompt="test",
                model="anthropic/claude-opus-4-6",
                prompt_cache=PromptCacheConfig(),
            ),
        )

        assert compiled.request.prompt_cache is not None
        assert any(
            decision.option == "prompt_cache" and decision.action == "supported"
            for decision in compiled.decisions
        )

    def test_gemini_keeps_cached_content_prompt_cache(self):
        compiled = compile_request_for_provider(
            "gemini",
            GenerateRequest(
                prompt="test",
                model="gemini-3.1-pro-preview",
                prompt_cache=PromptCacheConfig(
                    mode="cached_content",
                    cached_content_name="cachedContents/example",
                ),
            ),
        )

        assert compiled.request.prompt_cache is not None
        assert compiled.request.prompt_cache.mode == "cached_content"
        assert any(
            decision.option == "prompt_cache" and decision.action == "supported"
            for decision in compiled.decisions
        )

    def test_gemini_drops_automatic_prompt_cache_request_controls(self):
        compiled = compile_request_for_provider(
            "gemini",
            GenerateRequest(
                prompt="test",
                model="gemini-3.1-pro-preview",
                prompt_cache=PromptCacheConfig(),
            ),
        )

        assert compiled.request.prompt_cache is None
        assert any(
            decision.option == "prompt_cache"
            and decision.action == "dropped"
            and "cached-content" in decision.detail
            for decision in compiled.decisions
        )

    def test_vertex_keeps_prompt_cache_for_split_paths(self):
        claude = compile_request_for_provider(
            "vertex-ai",
            GenerateRequest(
                prompt="test",
                model="claude-opus-4-6@20260301",
                prompt_cache=PromptCacheConfig(),
            ),
        )
        gemini = compile_request_for_provider(
            "vertex-ai",
            GenerateRequest(
                prompt="test",
                model="gemini-3.1-pro-preview",
                prompt_cache=PromptCacheConfig(
                    mode="cached_content",
                    cached_content_name="cachedContents/example",
                ),
            ),
        )

        assert claude.request.prompt_cache is not None
        assert gemini.request.prompt_cache is not None
        actions = {(decision.option, decision.action) for decision in claude.decisions}
        assert ("prompt_cache", "supported") in actions
        actions = {(decision.option, decision.action) for decision in gemini.decisions}
        assert ("prompt_cache", "supported") in actions

    def test_vertex_drops_incompatible_prompt_cache_modes_for_split_paths(self):
        claude = compile_request_for_provider(
            "vertex-ai",
            GenerateRequest(
                prompt="test",
                model="claude-opus-4-6@20260301",
                prompt_cache=PromptCacheConfig(
                    mode="cached_content",
                    cached_content_name="cachedContents/example",
                ),
            ),
        )
        gemini = compile_request_for_provider(
            "vertex-ai",
            GenerateRequest(
                prompt="test",
                model="gemini-3.1-pro-preview",
                prompt_cache=PromptCacheConfig(),
            ),
        )

        assert claude.request.prompt_cache is None
        assert gemini.request.prompt_cache is None
        assert any(
            decision.option == "prompt_cache" and decision.action == "dropped"
            for decision in claude.decisions
        )
        assert any(
            decision.option == "prompt_cache" and decision.action == "dropped"
            for decision in gemini.decisions
        )

    def test_gemini_drops_tooling_and_ignores_reasoning_effort(self):
        compiled = compile_request_for_provider(
            "gemini",
            GenerateRequest(
                prompt="test",
                model="gemini-3.1-pro-preview",
                tools=[{"type": "function"}],
                tool_choice={"type": "auto"},
                reasoning=ReasoningConfig(enabled=True, effort="high", thinking_level="low"),
            ),
        )

        assert compiled.request.tools is None
        assert compiled.request.tool_choice is None
        assert compiled.request.reasoning is not None
        assert compiled.request.reasoning.effort is None
        actions = {(decision.option, decision.action) for decision in compiled.decisions}
        assert ("tools", "dropped") in actions
        assert ("tool_choice", "dropped") in actions
        assert ("reasoning.effort", "ignored") in actions

    def test_vertex_claude_and_gemini_paths_compile_differently(self):
        claude = compile_request_for_provider(
            "vertex-ai",
            GenerateRequest(
                prompt="test",
                model="claude-opus-4-6@20260301",
                top_p=0.5,
                tool_choice={"type": "auto"},
            ),
        )
        gemini = compile_request_for_provider(
            "vertex-ai",
            GenerateRequest(
                prompt="test",
                model="gemini-3.1-pro-preview",
                tools=[{"type": "function"}],
                tool_choice={"type": "auto"},
            ),
        )

        assert claude.request.top_p is None
        assert claude.request.tool_choice is None
        assert gemini.request.tools is None
        assert gemini.request.tool_choice is None

    def test_openrouter_drops_reasoning_for_structured_output_and_downgrades_strict(self):
        compiled = compile_request_for_provider(
            "openrouter",
            GenerateRequest(
                prompt="test",
                model="qwen/qwen3-max-thinking",
                structured_output=StructuredOutputConfig(
                    json_schema=_schema(required=["result"], additional_properties=False),
                    name="or",
                    strict=True,
                ),
                reasoning=ReasoningConfig(enabled=True, effort="high"),
            ),
        )

        assert compiled.request.reasoning is None
        assert compiled.request.structured_output is not None
        assert compiled.request.structured_output.strict is False
        actions = {(decision.option, decision.action) for decision in compiled.decisions}
        assert ("reasoning", "dropped") in actions
        assert ("structured_output.strict", "downgraded") in actions

    def test_openrouter_persists_schema_sanitization(self):
        compiled = compile_request_for_provider(
            "openrouter",
            GenerateRequest(
                prompt="test",
                model="qwen/qwen3-max-thinking",
                structured_output=StructuredOutputConfig(
                    json_schema={
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                        "required": ["result"],
                        "additionalProperties": False,
                    },
                    name="or",
                    strict=True,
                ),
            ),
        )

        assert compiled.request.structured_output is not None
        assert "$schema" not in compiled.request.structured_output.json_schema
        actions = {(decision.option, decision.action) for decision in compiled.decisions}
        assert ("structured_output.json_schema", "transformed") in actions

    def test_openrouter_virtual_provider_names_keep_openrouter_compilation(self):
        compiled = compile_request_for_provider(
            "qwen/qwen3-max-thinking",
            GenerateRequest(
                prompt="test",
                model="qwen/qwen3-max-thinking",
                structured_output=StructuredOutputConfig(
                    json_schema=_schema(required=["result"], additional_properties=False),
                    name="or",
                    strict=True,
                ),
                reasoning=ReasoningConfig(enabled=True, effort="high"),
            ),
        )

        assert compiled.request.reasoning is None
        assert compiled.request.structured_output is not None
        assert compiled.request.structured_output.strict is False
        actions = {(decision.option, decision.action) for decision in compiled.decisions}
        assert ("reasoning", "dropped") in actions
        assert ("structured_output.strict", "downgraded") in actions

    def test_codex_keeps_structured_output_but_drops_other_unsupported_controls(self):
        compiled = compile_request_for_provider(
            "codex",
            GenerateRequest(
                prompt="test",
                temperature=0.1,
                reasoning=ReasoningConfig(enabled=True, effort="medium"),
                structured_output=StructuredOutputConfig(
                    json_schema=_schema(required=["result"], additional_properties=False),
                    name="codex",
                    strict=True,
                ),
            ),
        )

        assert compiled.request.temperature is None
        assert compiled.request.reasoning is None
        assert compiled.request.structured_output is not None

    def test_claude_code_and_gemini_cli_drop_structured_output(self):
        claude = compile_request_for_provider(
            "claude",
            GenerateRequest(
                prompt="test",
                structured_output=StructuredOutputConfig(
                    json_schema=_schema(required=["result"], additional_properties=False),
                    name="claude",
                    strict=True,
                ),
                reasoning=ReasoningConfig(enabled=True, effort="medium"),
            ),
        )
        gemini_cli = compile_request_for_provider(
            "gemini-cli",
            GenerateRequest(
                prompt="test",
                structured_output=StructuredOutputConfig(
                    json_schema=_schema(required=["result"], additional_properties=False),
                    name="gemini-cli",
                    strict=True,
                ),
                tools=[{"type": "function"}],
            ),
        )

        assert claude.request.structured_output is None
        assert claude.request.reasoning is None
        assert gemini_cli.request.structured_output is None
        assert gemini_cli.request.tools is None

    def test_vertex_uses_claude_path_for_publisher_qualified_models(self):
        compiled = compile_request_for_provider(
            "vertex-ai",
            GenerateRequest(
                prompt="test",
                model="publishers/anthropic/models/claude-opus-4-6@20260301",
                top_p=0.5,
                tool_choice={"type": "auto"},
            ),
        )

        assert compiled.request.top_p is None
        assert compiled.request.tool_choice is None
        actions = {(decision.option, decision.action) for decision in compiled.decisions}
        assert ("top_p", "dropped") in actions
        assert ("tool_choice", "dropped") in actions
