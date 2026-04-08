"""Tests for provider-aware request compilation."""

from __future__ import annotations

from llm_council.providers.base import GenerateRequest, ReasoningConfig, StructuredOutputConfig
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
