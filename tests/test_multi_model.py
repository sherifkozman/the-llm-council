"""Tests for multi-model council functionality."""

from __future__ import annotations

import os

from llm_council.config.models import (
    DEFAULT_COUNCIL_MODELS,
    DEFAULT_MODEL_PACKS,
    ModelConfig,
    ModelPack,
    get_council_models,
    get_model_for_pack,
    is_multi_model_enabled,
    parse_models_string,
)
from llm_council.providers.openrouter import OpenRouterProvider, create_openrouter_for_model


class TestModelConfig:
    """Tests for ModelConfig class."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        ModelConfig.reset()

    def teardown_method(self) -> None:
        """Clean up env vars after each test."""
        ModelConfig.reset()
        for key in [
            "COUNCIL_MODELS",
            "COUNCIL_MODEL_FAST",
            "COUNCIL_MODEL_REASONING",
            "COUNCIL_MODEL_CODE",
            "COUNCIL_MODEL_CRITIC",
        ]:
            os.environ.pop(key, None)

    def test_default_models_single(self) -> None:
        """Without COUNCIL_MODELS env, returns single default model."""
        models = get_council_models()
        assert len(models) == 1
        assert models[0] == DEFAULT_MODEL_PACKS[ModelPack.DEFAULT]

    def test_multi_model_from_env(self) -> None:
        """COUNCIL_MODELS env var enables multi-model."""
        os.environ["COUNCIL_MODELS"] = "anthropic/claude-3.5-sonnet,openai/gpt-4o,google/gemini-pro"
        ModelConfig.reset()

        models = get_council_models()
        assert len(models) == 3
        assert "anthropic/claude-3.5-sonnet" in models
        assert "openai/gpt-4o" in models
        assert "google/gemini-pro" in models

    def test_is_multi_model_enabled(self) -> None:
        """is_multi_model_enabled returns correct value."""
        assert not is_multi_model_enabled()

        os.environ["COUNCIL_MODELS"] = "model1,model2"
        ModelConfig.reset()
        assert is_multi_model_enabled()

    def test_single_model_not_multi(self) -> None:
        """Single model in COUNCIL_MODELS is not multi-model."""
        os.environ["COUNCIL_MODELS"] = "anthropic/claude-3.5-sonnet"
        ModelConfig.reset()

        assert not is_multi_model_enabled()
        models = get_council_models()
        assert len(models) == 1

    def test_model_pack_defaults(self) -> None:
        """Model packs have correct defaults (December 2025)."""
        assert get_model_for_pack(ModelPack.FAST) == "anthropic/claude-3-5-haiku"
        assert get_model_for_pack(ModelPack.REASONING) == "anthropic/claude-opus-4-5"
        assert get_model_for_pack(ModelPack.CODE) == "openai/gpt-5.1"
        assert get_model_for_pack(ModelPack.CRITIC) == "anthropic/claude-sonnet-4-5"

    def test_model_pack_overrides(self) -> None:
        """Model pack env vars override defaults."""
        os.environ["COUNCIL_MODEL_FAST"] = "custom/fast-model"
        os.environ["COUNCIL_MODEL_CODE"] = "custom/code-model"
        ModelConfig.reset()

        assert get_model_for_pack(ModelPack.FAST) == "custom/fast-model"
        assert get_model_for_pack(ModelPack.CODE) == "custom/code-model"
        # Non-overridden packs still use defaults (December 2025)
        assert get_model_for_pack(ModelPack.REASONING) == "anthropic/claude-opus-4-5"

    def test_parse_models_string(self) -> None:
        """parse_models_string correctly parses comma-separated models."""
        result = parse_models_string("model1, model2, model3")
        assert result == ["model1", "model2", "model3"]

        result = parse_models_string("model1,model2")
        assert result == ["model1", "model2"]

        result = parse_models_string("  model1  ")
        assert result == ["model1"]

        result = parse_models_string("")
        assert result == []

    def test_models_with_whitespace(self) -> None:
        """Models are trimmed of whitespace."""
        os.environ["COUNCIL_MODELS"] = "  model1  ,  model2  "
        ModelConfig.reset()

        models = get_council_models()
        assert models == ["model1", "model2"]


class TestOpenRouterFactory:
    """Tests for OpenRouter provider factory function."""

    def test_create_openrouter_for_model(self) -> None:
        """Factory creates provider with specified model."""
        provider = create_openrouter_for_model("openai/gpt-4o")
        assert isinstance(provider, OpenRouterProvider)
        assert provider._default_model == "openai/gpt-4o"

    def test_different_models_different_providers(self) -> None:
        """Different models create distinct provider instances."""
        provider1 = create_openrouter_for_model("anthropic/claude-3.5-sonnet")
        provider2 = create_openrouter_for_model("openai/gpt-4o")

        assert provider1 is not provider2
        assert provider1._default_model == "anthropic/claude-3.5-sonnet"
        assert provider2._default_model == "openai/gpt-4o"


class TestOrchestratorMultiModel:
    """Tests for orchestrator multi-model functionality."""

    def setup_method(self) -> None:
        """Reset model config before each test."""
        ModelConfig.reset()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelConfig.reset()
        os.environ.pop("COUNCIL_MODELS", None)

    def test_orchestrator_creates_virtual_providers(self) -> None:
        """Orchestrator creates virtual providers when models configured."""
        from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(
            models=["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "google/gemini-pro"]
        )
        orchestrator = Orchestrator(providers=["openrouter"], config=config)

        # Should have 3 providers (one per model)
        assert len(orchestrator._providers) == 3
        assert "anthropic/claude-3.5-sonnet" in orchestrator._providers
        assert "openai/gpt-4o" in orchestrator._providers
        assert "google/gemini-pro" in orchestrator._providers

    def test_orchestrator_uses_env_models(self) -> None:
        """Orchestrator uses COUNCIL_MODELS env var."""
        os.environ["COUNCIL_MODELS"] = "model1,model2"
        ModelConfig.reset()

        from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig()
        orchestrator = Orchestrator(providers=["openrouter"], config=config)

        assert len(orchestrator._providers) == 2
        assert "model1" in orchestrator._providers
        assert "model2" in orchestrator._providers

    def test_orchestrator_config_models_override_env(self) -> None:
        """Config models take precedence over env var."""
        os.environ["COUNCIL_MODELS"] = "env-model1,env-model2"
        ModelConfig.reset()

        from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(models=["config-model1", "config-model2", "config-model3"])
        orchestrator = Orchestrator(providers=["openrouter"], config=config)

        # Should use config models, not env
        assert len(orchestrator._providers) == 3
        assert "config-model1" in orchestrator._providers
        assert "env-model1" not in orchestrator._providers

    def test_single_provider_no_multi_model(self) -> None:
        """Single model does not trigger multi-model mode."""
        from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(models=["single-model"])
        orchestrator = Orchestrator(providers=["openrouter"], config=config)

        # Should fall back to standard provider initialization
        # (single model doesn't trigger multi-model)
        assert "openrouter" in orchestrator._providers

    def test_non_openrouter_provider_no_multi_model(self) -> None:
        """Multi-model only works with openrouter provider."""
        from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(models=["model1", "model2"])
        # Using a different provider list
        orchestrator = Orchestrator(providers=["anthropic"], config=config)

        # Should not create virtual providers for non-openrouter
        assert (
            "anthropic" in orchestrator._providers
            or "anthropic" in orchestrator._provider_init_errors
        )


class TestCouncilMultiModel:
    """Tests for Council class with multi-model configuration."""

    def setup_method(self) -> None:
        """Reset model config before each test."""
        ModelConfig.reset()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelConfig.reset()
        os.environ.pop("COUNCIL_MODELS", None)

    def test_council_with_models_config(self) -> None:
        """Council accepts models in config."""
        from llm_council import Council
        from llm_council.protocol.types import CouncilConfig

        config = CouncilConfig(models=["anthropic/claude-3.5-sonnet", "openai/gpt-4o"])
        council = Council(config=config)

        # Verify the orchestrator was configured with models
        assert council._orchestrator._config.models == [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
        ]


class TestDefaultCouncilModels:
    """Tests for default council model configuration."""

    def test_default_models_defined(self) -> None:
        """DEFAULT_COUNCIL_MODELS contains expected models (December 2025)."""
        assert len(DEFAULT_COUNCIL_MODELS) == 3
        assert "anthropic/claude-opus-4-5" in DEFAULT_COUNCIL_MODELS
        assert "openai/gpt-5.1" in DEFAULT_COUNCIL_MODELS
        assert "google/gemini-3-flash-preview" in DEFAULT_COUNCIL_MODELS

    def test_all_model_packs_defined(self) -> None:
        """All ModelPack variants have default models."""
        for pack in ModelPack:
            assert pack in DEFAULT_MODEL_PACKS
            assert DEFAULT_MODEL_PACKS[pack] is not None
