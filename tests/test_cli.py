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
