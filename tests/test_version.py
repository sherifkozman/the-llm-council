"""Tests for package version detection."""

from __future__ import annotations

from pathlib import Path

from llm_council import _detect_version, _read_local_project_version


class TestVersionDetection:
    """Tests for local-source version resolution."""

    def test_read_local_project_version_from_nearest_pyproject(self, tmp_path: Path):
        """The package should read the version from the closest project root."""
        package_dir = tmp_path / "repo" / "src" / "llm_council"
        package_dir.mkdir(parents=True)
        module_file = package_dir / "__init__.py"
        module_file.write_text('"""stub"""', encoding="utf-8")
        (tmp_path / "repo" / "pyproject.toml").write_text(
            """
[project]
name = "the-llm-council"
version = "9.9.9"
""".strip(),
            encoding="utf-8",
        )

        assert _read_local_project_version(module_file) == "9.9.9"

    def test_detect_version_prefers_local_project_over_installed_metadata(
        self, monkeypatch, tmp_path: Path
    ):
        """Editable/source runs should not report whatever wheel is installed globally."""
        package_dir = tmp_path / "repo" / "src" / "llm_council"
        package_dir.mkdir(parents=True)
        module_file = package_dir / "__init__.py"
        module_file.write_text('"""stub"""', encoding="utf-8")
        (tmp_path / "repo" / "pyproject.toml").write_text(
            """
[project]
name = "the-llm-council"
version = "1.2.3"
""".strip(),
            encoding="utf-8",
        )

        def _stale_version(_dist_name: str) -> str:
            return "0.7.1"

        monkeypatch.setattr("importlib.metadata.version", _stale_version)

        assert _detect_version(module_file) == "1.2.3"

    def test_detect_version_falls_back_to_installed_metadata(self, monkeypatch, tmp_path: Path):
        """Installed distributions should still use package metadata when no repo is present."""
        module_file = tmp_path / "site-packages" / "llm_council" / "__init__.py"
        module_file.parent.mkdir(parents=True)
        module_file.write_text('"""stub"""', encoding="utf-8")

        monkeypatch.setattr("importlib.metadata.version", lambda _dist_name: "2.0.0")

        assert _detect_version(module_file) == "2.0.0"

    def test_detect_version_ignores_malformed_local_pyproject(self, monkeypatch, tmp_path: Path):
        """Malformed local project metadata should fall back to installed package metadata."""
        package_dir = tmp_path / "repo" / "src" / "llm_council"
        package_dir.mkdir(parents=True)
        module_file = package_dir / "__init__.py"
        module_file.write_text('"""stub"""', encoding="utf-8")
        (tmp_path / "repo" / "pyproject.toml").write_text(
            """
[project]
name = "the-llm-council"
version = "1.2.3
""".strip(),
            encoding="utf-8",
        )

        monkeypatch.setattr("importlib.metadata.version", lambda _dist_name: "2.0.0")

        assert _detect_version(module_file) == "2.0.0"

    def test_detect_version_skips_unrelated_nearest_pyproject(self, monkeypatch, tmp_path: Path):
        """Nearest workspace pyproject should be ignored when it belongs to another package."""
        package_dir = tmp_path / "workspace" / "repo" / "src" / "llm_council"
        package_dir.mkdir(parents=True)
        module_file = package_dir / "__init__.py"
        module_file.write_text('"""stub"""', encoding="utf-8")
        (tmp_path / "workspace" / "repo" / "pyproject.toml").write_text(
            """
[project]
name = "other-package"
version = "9.9.9"
""".strip(),
            encoding="utf-8",
        )
        (tmp_path / "workspace" / "pyproject.toml").write_text(
            """
[project]
name = "the-llm-council"
version = "3.4.5"
""".strip(),
            encoding="utf-8",
        )

        monkeypatch.setattr("importlib.metadata.version", lambda _dist_name: "0.7.1")

        assert _detect_version(module_file) == "3.4.5"

    def test_detect_version_returns_fallback_when_all_sources_fail(
        self, monkeypatch, tmp_path: Path
    ):
        """The package should return a stable fallback when local and installed metadata fail."""
        module_file = tmp_path / "site-packages" / "llm_council" / "__init__.py"
        module_file.parent.mkdir(parents=True)
        module_file.write_text('"""stub"""', encoding="utf-8")

        def _raise(_dist_name: str) -> str:
            raise RuntimeError("no metadata")

        monkeypatch.setattr("importlib.metadata.version", _raise)

        assert _detect_version(module_file) == "0.0.0+unknown"
