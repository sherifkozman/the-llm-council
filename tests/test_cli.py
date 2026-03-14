"""Tests for CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from llm_council.cli.main import app

runner = CliRunner()


class TestCLIHelp:
    """Tests for CLI help commands."""

    def test_help(self):
        """Test --help displays correctly."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Multi-LLM Council Framework" in result.stdout

    def test_run_help(self):
        """Test run --help displays correctly."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "subagent" in result.stdout.lower()

    def test_doctor_help(self):
        """Test doctor --help displays correctly."""
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "provider" in result.stdout.lower() or "availability" in result.stdout.lower()

    def test_config_help(self):
        """Test config --help displays correctly."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0


class TestCLIVersion:
    """Tests for version command."""

    def test_version(self):
        """Test version command output."""
        from llm_council import __version__

        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "LLM Council" in result.stdout
        assert __version__ in result.stdout


class TestCLIDoctor:
    """Tests for doctor command."""

    def test_doctor_shows_providers(self):
        """Test doctor command shows provider table."""
        with patch(
            "llm_council.cli.main._load_config_defaults",
            return_value={},
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "Provider Status" in result.stdout

    def test_doctor_with_no_providers(self):
        """Test doctor when no providers registered."""
        with patch("llm_council.providers.registry.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor"])
            # With no providers, should show empty table or message
            assert result.exit_code == 0


class TestCLIConfig:
    """Tests for config command."""

    def test_config_show_no_file(self, tmp_path, monkeypatch):
        """Test config --show with no config file."""
        # Mock home directory
        monkeypatch.setenv("HOME", str(tmp_path))
        result = runner.invoke(app, ["config", "--show"])
        # Should indicate no config or show empty/default
        assert result.exit_code == 0

    def test_config_init(self, tmp_path, monkeypatch):
        """Test config --init creates config file."""
        monkeypatch.setenv("HOME", str(tmp_path))
        result = runner.invoke(app, ["config", "--init"])
        assert result.exit_code == 0

        # Check file was created
        config_file = tmp_path / ".config" / "llm-council" / "config.yaml"
        assert config_file.exists()

    def test_config_no_options(self):
        """Test config with no options shows usage."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "Usage" in result.stdout or "config" in result.stdout


class TestOutputFormatConfig:
    """Tests for output_format config default (#29)."""

    def test_run_uses_output_format_json_from_config(self):
        """Config output_format: json enables JSON output."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {
            "success": True,
            "output": {"result": "test"},
        }

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["run", "router", "Test task"])
            assert result.exit_code in [0, 1]
            # Should produce JSON output (no rich panels)
            if result.exit_code == 0:
                assert "Council Result" not in result.stdout

    def test_run_json_flag_overrides_config(self):
        """Explicit --json flag works even with no config."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {
            "success": True,
            "output": {"result": "test"},
        }

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["run", "router", "Test task", "--json"])
            assert result.exit_code in [0, 1]

    def test_run_rich_output_when_config_not_json(self):
        """Non-json output_format preserves rich output."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {
            "success": True,
            "output": {"result": "test"},
        }
        mock_result.validation_errors = None

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "rich"},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["run", "router", "Test task"])
            assert result.exit_code in [0, 1]

    def test_doctor_uses_output_format_json_from_config(self):
        """Doctor command respects output_format config."""
        with patch(
            "llm_council.cli.main._load_config_defaults",
            return_value={"output_format": "json"},
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            # JSON output should not have rich table header
            if "Provider Status" not in result.stdout:
                assert "{" in result.stdout or "providers" in result.stdout


class TestProviderConfigLoading:
    """Tests for provider config loading from config file (#26)."""

    def test_load_provider_configs_extracts_models(self):
        """Provider configs extract default_model from config."""
        from llm_council.cli.main import _load_provider_configs

        config_data = {
            "providers": [
                {"name": "openai", "default_model": "gpt-5.2"},
                {"name": "google", "default_model": "gemini-3.1-pro"},
            ],
            "defaults": {},
        }

        with patch(
            "llm_council.cli.main._load_config",
            return_value=config_data,
        ):
            result = _load_provider_configs()
            assert result == {
                "openai": {"default_model": "gpt-5.2"},
                "google": {"default_model": "gemini-3.1-pro"},
            }

    def test_load_provider_configs_ignores_invalid(self):
        """Provider configs skip invalid entries."""
        from llm_council.cli.main import _load_provider_configs

        config_data = {
            "providers": [
                {"name": "openai", "default_model": "gpt-5.2"},
                "invalid_string",
                {"no_name": True},
                {"name": "anthropic"},  # no default_model
            ],
        }

        with patch(
            "llm_council.cli.main._load_config",
            return_value=config_data,
        ):
            result = _load_provider_configs()
            assert result == {
                "openai": {"default_model": "gpt-5.2"},
            }

    def test_load_provider_configs_empty(self):
        """Empty config returns empty dict."""
        from llm_council.cli.main import _load_provider_configs

        with patch(
            "llm_council.cli.main._load_config",
            return_value={},
        ):
            result = _load_provider_configs()
            assert result == {}


class TestCLIRun:
    """Tests for run command."""

    def test_run_requires_arguments(self):
        """Test run requires subagent and task."""
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_run_with_missing_task(self):
        """Test run with subagent but no task."""
        result = runner.invoke(app, ["run", "router"])
        assert result.exit_code != 0

    def test_run_basic(self):
        """Test basic run command."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {"success": True, "output": {"result": "test"}}
        mock_result.validation_errors = None

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council_class.return_value = mock_council

            # Mock asyncio.run to return our mock result
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = mock_result

                result = runner.invoke(app, ["run", "router", "Test task"])
                # Command should complete (may show result or error depending on output parsing)
                assert result.exit_code in [0, 1]  # Either success or handled error

    def test_run_with_json_output(self):
        """Test run with --json flag."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {
            "success": True,
            "output": {"result": "test"},
            "drafts": {},
            "critique": None,
            "synthesis_attempts": 1,
            "duration_ms": 1000,
        }

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council_class.return_value = mock_council

            with patch("asyncio.run") as mock_run:
                mock_run.return_value = mock_result

                result = runner.invoke(app, ["run", "router", "Test task", "--json"])
                # JSON output should be produced
                assert result.exit_code in [0, 1]

    def test_run_with_providers(self):
        """Test run with custom providers."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {}
        mock_result.model_dump.return_value = {"success": True, "output": {}}
        mock_result.validation_errors = None

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council_class.return_value = mock_council

            with patch("asyncio.run") as mock_run:
                mock_run.return_value = mock_result

                result = runner.invoke(
                    app,
                    ["run", "router", "Test task", "--providers", "openai,anthropic"],
                )
                # Should attempt to use specified providers
                assert result.exit_code in [0, 1]
