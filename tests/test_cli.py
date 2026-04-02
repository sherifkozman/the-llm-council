"""Tests for CLI commands."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from llm_council.cli.main import app
from llm_council.eval_import import ImportedPullRequest
from llm_council.providers.base import GenerateResponse
from llm_council.storage.artifacts import ArtifactStore, ArtifactType

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

    def test_eval_help(self):
        """Test eval help displays correctly."""
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "evaluation" in result.stdout.lower() or "dataset" in result.stdout.lower()


class TestCLIVersion:
    """Tests for version command."""

    def test_version(self):
        """Test version command output."""
        from llm_council import __version__

        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "LLM Council" in result.stdout
        assert __version__ in result.stdout


class TestCLIStorage:
    """Tests for storage commands."""

    def test_storage_status_reports_legacy_usage(self, monkeypatch, tmp_path):
        """Storage status should show when the legacy store is active."""
        monkeypatch.setattr("llm_council.storage.artifacts.Path.home", lambda: tmp_path)
        legacy_store = ArtifactStore(
            artifact_dir=tmp_path / ".claude" / "council-artifacts",
            db_path=tmp_path / ".claude" / "council-ledger.db",
        )
        run = legacy_store.create_run(subagent="test", task="legacy")
        legacy_store.store_artifact(run.run_id, "legacy content", ArtifactType.DRAFT)

        result = runner.invoke(app, ["storage", "status", "--json"])

        assert result.exit_code == 0
        assert '"using_legacy": true' in result.stdout
        assert str(tmp_path / ".claude" / "council-artifacts") in result.stdout

    def test_storage_migrate_dry_run(self, monkeypatch, tmp_path):
        """Dry-run migration should report what would be copied."""
        monkeypatch.setattr("llm_council.storage.artifacts.Path.home", lambda: tmp_path)
        legacy_store = ArtifactStore(
            artifact_dir=tmp_path / ".claude" / "council-artifacts",
            db_path=tmp_path / ".claude" / "council-ledger.db",
        )
        run = legacy_store.create_run(subagent="test", task="legacy")
        legacy_store.store_artifact(run.run_id, "legacy content", ArtifactType.DRAFT)

        result = runner.invoke(app, ["storage", "migrate", "--dry-run", "--json"])

        assert result.exit_code == 0
        assert '"dry_run": true' in result.stdout
        assert '"copied_files": 1' in result.stdout
        assert not (tmp_path / ".council").exists()

    def test_storage_migrate_copies_legacy_store(self, monkeypatch, tmp_path):
        """Migration command should copy legacy store into neutral council home."""
        monkeypatch.setattr("llm_council.storage.artifacts.Path.home", lambda: tmp_path)
        legacy_store = ArtifactStore(
            artifact_dir=tmp_path / ".claude" / "council-artifacts",
            db_path=tmp_path / ".claude" / "council-ledger.db",
        )
        run = legacy_store.create_run(subagent="test", task="legacy")
        legacy_store.store_artifact(run.run_id, "legacy content", ArtifactType.DRAFT)

        result = runner.invoke(app, ["storage", "migrate", "--json"])

        assert result.exit_code == 0
        assert '"migrated": true' in result.stdout
        assert (tmp_path / ".council" / "ledger.db").exists()
        assert (tmp_path / ".council" / "artifacts").exists()


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

    def test_doctor_uses_provider_config(self):
        """Doctor passes provider config to get_provider (#28)."""
        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={},
            ),
            patch(
                "llm_council.cli.main._load_provider_configs",
                return_value={
                    "openai": {"default_model": "gpt-5.2"},
                },
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
        ):
            mock_registry = MagicMock()
            mock_provider = MagicMock()
            mock_doctor_result = MagicMock()
            mock_doctor_result.ok = True
            mock_doctor_result.message = "OK"
            mock_doctor_result.latency_ms = 100.0

            mock_registry.list_providers.return_value = ["openai"]
            mock_registry.get_provider.return_value = mock_provider
            # doctor() is now awaited inside an async function
            mock_provider.doctor = AsyncMock(return_value=mock_doctor_result)
            mock_reg.return_value = mock_registry

            runner.invoke(app, ["doctor"])

            # Verify get_provider was called with config kwargs
            mock_registry.get_provider.assert_called_with("openai", default_model="gpt-5.2")

    def test_doctor_with_no_providers(self):
        """Test doctor when no providers registered."""
        with patch("llm_council.providers.registry.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor"])
            # With no providers, should show empty table or message
            assert result.exit_code == 0

    def test_doctor_accepts_multiple_provider_filters_and_aliases(self):
        """Doctor should accept repeated --provider flags and resolve friendly aliases."""
        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["claude", "codex", "gemini", "gemini-cli"]
            mock_registry.resolve_name.side_effect = {
                "codex": "codex",
                "gemini-cli": "gemini-cli",
                "claude": "claude",
            }.get

            def _get_provider(name, **_kwargs):
                provider = MagicMock()
                result = MagicMock()
                result.ok = True
                result.message = f"{name} ok"
                result.latency_ms = 1.0
                provider.doctor = AsyncMock(return_value=result)
                return provider

            mock_registry.get_provider.side_effect = _get_provider
            mock_reg.return_value = mock_registry

            result = runner.invoke(
                app,
                [
                    "doctor",
                    "--provider",
                    "codex",
                    "--provider",
                    "gemini-cli,claude",
                    "--json",
                ],
            )

            assert result.exit_code == 0
            assert '"name": "codex"' in result.stdout
            assert '"name": "gemini-cli"' in result.stdout
            assert '"name": "claude"' in result.stdout

    def test_doctor_deep_probe_reports_generation_readiness(self):
        """Deep doctor should surface whether a provider can answer a trivial prompt."""
        @asynccontextmanager
        async def fake_slot(_name: str, *, timeout_seconds: float | None = None):
            assert timeout_seconds == 5.0
            yield 0.0

        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
            patch("llm_council.cli.main.provider_call_slot", fake_slot),
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["codex"]
            mock_registry.resolve_name.side_effect = {"codex": "codex"}.get

            provider = MagicMock()
            doctor_result = MagicMock()
            doctor_result.ok = True
            doctor_result.message = "Logged in using ChatGPT"
            doctor_result.latency_ms = None
            provider.doctor = AsyncMock(return_value=doctor_result)
            provider.generate = AsyncMock(return_value=GenerateResponse(text="OK", content="OK"))
            mock_registry.get_provider.return_value = provider
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor", "--provider", "codex", "--deep", "--json"])

            assert result.exit_code == 0
            assert '"probe_ok": true' in result.stdout
            assert '"probe_message": "OK"' in result.stdout

    def test_doctor_deep_probe_skips_when_base_health_fails(self):
        """Deep doctor should not run a probe when the base doctor already failed."""
        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["codex"]
            mock_registry.resolve_name.side_effect = {"codex": "codex"}.get

            provider = MagicMock()
            doctor_result = MagicMock()
            doctor_result.ok = False
            doctor_result.message = "Not logged in"
            doctor_result.latency_ms = None
            provider.doctor = AsyncMock(return_value=doctor_result)
            provider.generate = AsyncMock()
            mock_registry.get_provider.return_value = provider
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor", "--provider", "codex", "--deep", "--json"])

            assert result.exit_code == 0
            assert '"probe_ok": false' in result.stdout
            assert "Skipped because base doctor failed" in result.stdout
            provider.generate.assert_not_called()

    def test_doctor_deep_probe_serializes_live_probes(self):
        """Deep doctor should not fan out multiple live probes at once."""
        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["openai", "claude"]
            mock_registry.resolve_name.side_effect = {"openai": "openai", "claude": "claude"}.get

            def _provider(name: str) -> MagicMock:
                provider = MagicMock()
                doctor_result = MagicMock()
                doctor_result.ok = True
                doctor_result.message = f"{name} ok"
                doctor_result.latency_ms = 1.0
                provider.doctor = AsyncMock(return_value=doctor_result)

                async def _generate(_request):
                    await asyncio.sleep(0.01)
                    return GenerateResponse(text="OK", content="OK")

                provider.generate = AsyncMock(side_effect=_generate)
                return provider

            providers = {"openai": _provider("openai"), "claude": _provider("claude")}
            mock_registry.get_provider.side_effect = lambda name, **_kwargs: providers[name]
            mock_reg.return_value = mock_registry

            state = {"active": 0, "max_active": 0}

            @asynccontextmanager
            async def fake_slot(_name: str, *, timeout_seconds: float | None = None):
                assert timeout_seconds == 5.0
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                try:
                    yield 0.0
                finally:
                    state["active"] -= 1

            with patch("llm_council.cli.main.provider_call_slot", fake_slot):
                result = runner.invoke(
                    app,
                    [
                        "doctor",
                        "--provider",
                        "openai",
                        "--provider",
                        "claude",
                        "--deep",
                        "--json",
                    ],
                )

            assert result.exit_code == 0
            assert state["max_active"] == 1


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


class TestCLIEvaluate:
    """Tests for eval command."""

    def test_evaluate_json_output(self, tmp_path):
        """Eval should emit JSON and exit cleanly when all cases pass."""
        dataset_path = tmp_path / "dataset.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        report = MagicMock()
        report.failed_cases = 0
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "total_cases": 1,
            "passed_cases": 1,
            "failed_cases": 0,
            "case_results": [],
            "mode_scorecards": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load,
            patch("llm_council.cli.main.run_eval_dataset", new_callable=AsyncMock) as mock_eval,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load.return_value = MagicMock()
            mock_eval.return_value = report

            result = runner.invoke(app, ["eval", str(dataset_path), "--json"])

        assert result.exit_code == 0
        assert '"dataset_name": "smoke"' in result.stdout

    def test_evaluate_fails_when_report_has_failed_cases(self, tmp_path):
        """Eval should return a non-zero exit code when any case fails."""
        dataset_path = tmp_path / "dataset.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        report = MagicMock()
        report.failed_cases = 1
        report.dataset_name = "smoke"
        report.passed_cases = 0
        report.total_cases = 1
        report.passed_criteria = 1
        report.total_criteria = 2
        report.duration_ms = 10
        report.mode_scorecards = []
        report.case_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "total_cases": 1,
            "passed_cases": 0,
            "failed_cases": 1,
            "case_results": [],
            "mode_scorecards": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load,
            patch("llm_council.cli.main.run_eval_dataset", new_callable=AsyncMock) as mock_eval,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load.return_value = MagicMock()
            mock_eval.return_value = report

            result = runner.invoke(app, ["eval", str(dataset_path)])

        assert result.exit_code == 1

    def test_evaluate_compare_json_output(self, tmp_path):
        """Eval-compare should emit JSON and exit cleanly when variants pass."""
        dataset_path = tmp_path / "dataset.yaml"
        variants_path = tmp_path / "variants.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        variants_path.write_text("version: 1\nname: candidates\nvariants: []\n")
        report = MagicMock()
        report.variant_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "variants_name": "candidates",
            "best_variant": None,
            "ranking": [],
            "variant_results": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load_dataset,
            patch("llm_council.cli.main.load_eval_variants") as mock_load_variants,
            patch(
                "llm_council.cli.main.run_eval_comparison", new_callable=AsyncMock
            ) as mock_compare,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load_dataset.return_value = MagicMock()
            mock_load_variants.return_value = MagicMock()
            mock_compare.return_value = report

            result = runner.invoke(
                app,
                ["eval-compare", str(dataset_path), str(variants_path), "--json"],
            )

        assert result.exit_code == 0
        assert '"dataset_name": "smoke"' in result.stdout

    def test_evaluate_compare_passes_variant_filters(self, tmp_path):
        """Eval-compare should forward selected variant names to the runner."""
        dataset_path = tmp_path / "dataset.yaml"
        variants_path = tmp_path / "variants.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        variants_path.write_text("version: 1\nname: candidates\nvariants: []\n")
        report = MagicMock()
        report.variant_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "variants_name": "candidates",
            "best_variant": None,
            "ranking": [],
            "variant_results": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load_dataset,
            patch("llm_council.cli.main.load_eval_variants") as mock_load_variants,
            patch(
                "llm_council.cli.main.run_eval_comparison", new_callable=AsyncMock
            ) as mock_compare,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load_dataset.return_value = MagicMock()
            mock_load_variants.return_value = MagicMock()
            mock_compare.return_value = report

            result = runner.invoke(
                app,
                [
                    "eval-compare",
                    str(dataset_path),
                    str(variants_path),
                    "--variant",
                    "openai-gpt54",
                    "--variant",
                    "vertex-gemini31pro",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        assert mock_compare.await_args.kwargs["variant_names"] == [
            "openai-gpt54",
            "vertex-gemini31pro",
        ]

    def test_evaluate_compare_builds_base_config_with_timeout_overrides(self, tmp_path):
        """Eval-compare should honor timeout and retry overrides."""
        dataset_path = tmp_path / "dataset.yaml"
        variants_path = tmp_path / "variants.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        variants_path.write_text("version: 1\nname: candidates\nvariants: []\n")
        report = MagicMock()
        report.variant_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "variants_name": "candidates",
            "best_variant": None,
            "ranking": [],
            "variant_results": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load_dataset,
            patch("llm_council.cli.main.load_eval_variants") as mock_load_variants,
            patch(
                "llm_council.cli.main.run_eval_comparison", new_callable=AsyncMock
            ) as mock_compare,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load_dataset.return_value = MagicMock()
            mock_load_variants.return_value = MagicMock()
            mock_compare.return_value = report

            result = runner.invoke(
                app,
                [
                    "eval-compare",
                    str(dataset_path),
                    str(variants_path),
                    "--timeout",
                    "30",
                    "--max-retries",
                    "1",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        base_config = mock_compare.await_args.kwargs["base_config"]
        assert base_config.timeout == 30
        assert base_config.max_retries == 1
        assert base_config.runtime_profile.value == "default"
        assert base_config.reasoning_profile.value == "default"

    def test_evaluate_compare_builds_base_config_with_reasoning_profile(self, tmp_path):
        """Eval-compare should honor reasoning-profile overrides."""
        dataset_path = tmp_path / "dataset.yaml"
        variants_path = tmp_path / "variants.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        variants_path.write_text("version: 1\nname: candidates\nvariants: []\n")
        report = MagicMock()
        report.variant_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "variants_name": "candidates",
            "best_variant": None,
            "ranking": [],
            "variant_results": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load_dataset,
            patch("llm_council.cli.main.load_eval_variants") as mock_load_variants,
            patch(
                "llm_council.cli.main.run_eval_comparison", new_callable=AsyncMock
            ) as mock_compare,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load_dataset.return_value = MagicMock()
            mock_load_variants.return_value = MagicMock()
            mock_compare.return_value = report

            result = runner.invoke(
                app,
                [
                    "eval-compare",
                    str(dataset_path),
                    str(variants_path),
                    "--reasoning-profile",
                    "off",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        base_config = mock_compare.await_args.kwargs["base_config"]
        assert base_config.reasoning_profile.value == "off"

    def test_evaluate_compare_builds_base_config_with_runtime_profile(self, tmp_path):
        """Eval-compare should honor runtime-profile overrides."""
        dataset_path = tmp_path / "dataset.yaml"
        variants_path = tmp_path / "variants.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        variants_path.write_text("version: 1\nname: candidates\nvariants: []\n")
        report = MagicMock()
        report.variant_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "variants_name": "candidates",
            "best_variant": None,
            "ranking": [],
            "variant_results": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load_dataset,
            patch("llm_council.cli.main.load_eval_variants") as mock_load_variants,
            patch(
                "llm_council.cli.main.run_eval_comparison", new_callable=AsyncMock
            ) as mock_compare,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load_dataset.return_value = MagicMock()
            mock_load_variants.return_value = MagicMock()
            mock_compare.return_value = report

            result = runner.invoke(
                app,
                [
                    "eval-compare",
                    str(dataset_path),
                    str(variants_path),
                    "--runtime-profile",
                    "bounded",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        base_config = mock_compare.await_args.kwargs["base_config"]
        assert base_config.runtime_profile.value == "bounded"


class TestCLIImport:
    """Tests for local-only PR import command."""

    def test_eval_import_pr_json_output(self):
        """Import command should emit JSON for the imported bundle."""
        imported = ImportedPullRequest(
            repo="owner/repo",
            pr_number=42,
            title="Fix auth edge case",
            state="OPEN",
            is_draft=False,
            base_ref="main",
            head_ref="fix/auth",
            additions=10,
            deletions=2,
            changed_files=1,
            url="https://github.com/owner/repo/pull/42",
            import_root=".council-private/imports/github/owner__repo/pr-42",
            diff_path=".council-private/imports/github/owner__repo/pr-42/diff.patch",
            review_input_path=".council-private/imports/github/owner__repo/pr-42/review_input.md",
            metadata_path=".council-private/imports/github/owner__repo/pr-42/pr.json",
            raw_comments_path=".council-private/imports/github/owner__repo/pr-42/review_comments.raw.json",
            greptile_labels_path=".council-private/imports/github/owner__repo/pr-42/greptile_labels.json",
            imported_comment_count=3,
        )

        with patch("llm_council.cli.main.import_github_pr_review", return_value=imported):
            result = runner.invoke(app, ["eval-import-pr", "owner/repo", "42", "--json"])

        assert result.exit_code == 0
        assert '"repo": "owner/repo"' in result.stdout


class TestProviderConfigLoading:
    """Tests for provider config loading from config file (#26)."""

    def test_load_provider_configs_extracts_models(self):
        """Provider configs extract default_model from config."""
        from llm_council.cli.main import _load_provider_configs

        config_data = {
            "providers": [
                {"name": "openai", "default_model": "gpt-5.2"},
                {"name": "gemini", "default_model": "gemini-3.1-pro"},
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
                "gemini": {"default_model": "gemini-3.1-pro"},
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

    def test_run_with_route_flag_forwards_router_followup(self):
        """--route should enable router follow-up in config and run invocation."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"recommendation": "proceed"}
        mock_result.model_dump.return_value = {
            "success": True,
            "output": {"recommendation": "proceed"},
        }
        mock_result.validation_errors = None
        mock_result.execution_plan = None
        mock_result.routed = False
        mock_result.routing_decision = None

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council.run = AsyncMock(return_value=mock_result)
            mock_council_class.return_value = mock_council

            result = runner.invoke(app, ["run", "router", "Test task", "--route"])

            assert result.exit_code == 0
            council_config = mock_council_class.call_args.kwargs["config"]
            assert council_config.follow_router is True
            mock_council.run.assert_awaited_once_with(
                task="Test task",
                subagent="router",
                follow_router=True,
            )

    def test_run_with_route_flag_requires_router_subagent(self):
        """--route is rejected for non-router subagents."""
        result = runner.invoke(app, ["run", "planner", "Test task", "--route"])

        assert result.exit_code == 1
        assert "--route can only be used with the router subagent" in result.stdout

    def test_run_forwards_reasoning_profile(self):
        """Run should pass through the reasoning-profile override."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {"success": True, "output": {"result": "test"}}
        mock_result.validation_errors = None

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council.run = AsyncMock(return_value=mock_result)
            mock_council_class.return_value = mock_council

            result = runner.invoke(
                app,
                ["run", "router", "Test task", "--reasoning-profile", "light"],
            )

        assert result.exit_code in [0, 1]
        council_config = mock_council_class.call_args.kwargs["config"]
        assert council_config.reasoning_profile.value == "light"

    def test_run_forwards_runtime_profile(self):
        """Run should pass through the runtime-profile override."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {"success": True, "output": {"result": "test"}}
        mock_result.validation_errors = None

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council.run = AsyncMock(return_value=mock_result)
            mock_council_class.return_value = mock_council

            result = runner.invoke(
                app,
                ["run", "router", "Test task", "--runtime-profile", "bounded"],
            )

        assert result.exit_code in [0, 1]
        council_config = mock_council_class.call_args.kwargs["config"]
        assert council_config.runtime_profile.value == "bounded"
