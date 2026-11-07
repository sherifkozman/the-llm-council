"""Run bounded SDK wire and CLI parser checks in disposable environments."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

EXTRAS = ("base", "anthropic", "openai", "gemini", "google", "vertex", "all")
EXPECTED = {
    "base": {"test_openrouter_serializes_basic_request"},
    "anthropic": {
        "test_openrouter_serializes_basic_request",
        "test_sdk_serializes_basic_request_and_parses_response[anthropic]",
        "test_vertex_anthropic_sdk_serializes_basic_request",
        "test_anthropic_serializes_council_temperature_and_reasoning_together",
    },
    "openai": {
        "test_openrouter_serializes_basic_request",
        "test_sdk_serializes_basic_request_and_parses_response[openai]",
    },
    "gemini": {
        "test_openrouter_serializes_basic_request",
        "test_google_sdk_constructs_gemini_and_vertex_clients_with_retry_options",
        "test_google_sdk_preserves_requested_high_thinking_level",
        "test_google_sdk_serializes_requested_thinking_without_provider_traffic[gemini]",
        "test_google_sdk_serializes_requested_thinking_without_provider_traffic[vertex]",
    },
    "google": {
        "test_openrouter_serializes_basic_request",
        "test_google_sdk_constructs_gemini_and_vertex_clients_with_retry_options",
        "test_google_sdk_preserves_requested_high_thinking_level",
        "test_google_sdk_serializes_requested_thinking_without_provider_traffic[gemini]",
        "test_google_sdk_serializes_requested_thinking_without_provider_traffic[vertex]",
    },
    "vertex": {
        "test_openrouter_serializes_basic_request",
        "test_sdk_serializes_basic_request_and_parses_response[anthropic]",
        "test_anthropic_serializes_council_temperature_and_reasoning_together",
        "test_google_sdk_constructs_gemini_and_vertex_clients_with_retry_options",
        "test_google_sdk_preserves_requested_high_thinking_level",
        "test_google_sdk_serializes_requested_thinking_without_provider_traffic[gemini]",
        "test_google_sdk_serializes_requested_thinking_without_provider_traffic[vertex]",
        "test_vertex_anthropic_sdk_serializes_basic_request",
    },
    "all": {
        "test_openrouter_serializes_basic_request",
        "test_sdk_serializes_basic_request_and_parses_response[openai]",
        "test_sdk_serializes_basic_request_and_parses_response[anthropic]",
        "test_anthropic_serializes_council_temperature_and_reasoning_together",
        "test_google_sdk_constructs_gemini_and_vertex_clients_with_retry_options",
        "test_google_sdk_preserves_requested_high_thinking_level",
        "test_google_sdk_serializes_requested_thinking_without_provider_traffic[gemini]",
        "test_google_sdk_serializes_requested_thinking_without_provider_traffic[vertex]",
        "test_vertex_anthropic_sdk_serializes_basic_request",
    },
}


def run(command: list[str], env: dict[str, str]) -> int:
    return subprocess.run(command, env=env, check=False).returncode


def main() -> int:
    resolution = os.environ.get("COUNCIL_MATRIX_RESOLUTION", "latest")
    if resolution not in {"lowest-direct", "latest"}:
        raise SystemExit("Invalid resolution")

    with tempfile.TemporaryDirectory(prefix="council-upgrade-matrix-") as temporary:
        root = Path(temporary)
        home = root / "home"
        home.mkdir()
        env = {
            key: os.environ[key]
            for key in ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TMP", "TEMP")
            if key in os.environ
        }
        env.update(
            HOME=str(home),
            USERPROFILE=str(home),
            UV_CACHE_DIR=str(root / "cache"),
            PYTHONDONTWRITEBYTECODE="1",
        )
        failures: list[tuple[str, str]] = []
        for extra in EXTRAS:
            print(f"::group::Python {sys.version.split()[0]} {resolution} {extra}", flush=True)
            venv = root / extra
            py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            dependency = ".[dev]" if extra == "base" else f".[{extra},dev]"
            report = root / f"{extra}-results.xml"
            commands = [
                ["uv", "venv", str(venv), "--python", sys.executable],
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(py),
                    "--default-index",
                    "https://pypi.org/simple",
                    *(["--resolution", "lowest-direct"] if resolution == "lowest-direct" else []),
                    "-e",
                    dependency,
                ],
                ["uv", "pip", "check", "--python", str(py)],
                ["uv", "pip", "list", "--python", str(py), "--format", "freeze"],
                [
                    str(py),
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "-r",
                    "s",
                    f"--junitxml={report}",
                    "tests/test_provider_sdk_wire.py",
                    "tests/test_cli_contract.py::test_parser_error_remains_stderr_exit_two",
                ],
            ]
            for phase, command in zip(
                ("venv", "install", "dependency-check", "resolved-versions", "tests"),
                commands,
                strict=True,
            ):
                if run(command, env):
                    failures.append((extra, phase))
                    print(f"RESULT extra={extra} phase={phase} failed", flush=True)
                    break
                print(f"RESULT extra={extra} phase={phase} passed", flush=True)

            if report.exists():
                cases = list(ET.parse(report).iter("testcase"))
                passed = {
                    case.get("name")
                    for case in cases
                    if case.get("classname", "").endswith("test_provider_sdk_wire")
                    and not any(
                        case.find(tag) is not None for tag in ("skipped", "failure", "error")
                    )
                }
                missing = EXPECTED[extra] - passed
                print(
                    f"RESULT extra={extra} wire-passed={len(passed)} "
                    f"expected={len(EXPECTED[extra])} missing={sorted(missing)}",
                    flush=True,
                )
                if missing:
                    failures.append((extra, "missing-provider-wire-coverage"))
                parser_passed = any(
                    case.get("classname", "").endswith("test_cli_contract")
                    and case.get("name") == "test_parser_error_remains_stderr_exit_two"
                    and not any(
                        case.find(tag) is not None for tag in ("skipped", "failure", "error")
                    )
                    for case in cases
                )
                print(f"RESULT extra={extra} parser-test-passed={parser_passed}", flush=True)
                if not parser_passed:
                    failures.append((extra, "missing-passing-cli-parser-test"))
            else:
                failures.append((extra, "missing-test-report"))
            print("::endgroup::", flush=True)

        if failures:
            raise SystemExit(f"Failed matrix cells: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
