"""Portable native admission; packaging differences must not bypass verification."""

import hashlib
import json
from pathlib import Path

import pytest

from llm_council.providers.cli import _codex_snapshot_worker as worker
from llm_council.providers.cli import codex


def pe_image(machine=0x8664, *, dll=False):
    data = bytearray(128)
    data[:2] = b"MZ"
    data[60:64] = (64).to_bytes(4, "little")
    data[64:68] = b"PE\0\0"
    data[68:70] = machine.to_bytes(2, "little")
    data[86:88] = (0x2002 if dll else 2).to_bytes(2, "little")
    data[88:90] = (0x20B).to_bytes(2, "little")
    return bytes(data)


@pytest.mark.parametrize(
    "machine,arch,expected",
    [(0x8664, "x86_64", True), (0xAA64, "arm64", True), (0x8664, "arm64", False)],
)
def test_windows_native_header(tmp_path, machine, arch, expected):
    path = tmp_path / "codex.exe"
    path.write_bytes(pe_image(machine))
    assert codex._native_executable_matches(path, "win32", arch) is expected


def test_windows_header_rejects_dll(tmp_path):
    path = tmp_path / "codex.exe"
    path.write_bytes(pe_image(dll=True))
    assert not codex._native_executable_matches(path, "win32", "x86_64")


def npm_package(tmp_path):
    modules = tmp_path / "node_modules"
    package = modules / "@openai/codex"
    (package / "bin").mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps({"name": "@openai/codex", "bin": {"codex": "bin/codex.js"}})
    )
    (package / "bin/codex.js").write_text("// standard package entrypoint")
    platform_package = modules / "@openai/codex-linux-x64"
    binary = platform_package / "vendor/x86_64-unknown-linux-musl/bin/codex"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native executable")
    (platform_package / "package.json").write_text(json.dumps({"name": "@openai/codex-linux-x64"}))
    return package, binary


def test_resolve_npm_javascript_without_executing_it(tmp_path):
    package, binary = npm_package(tmp_path)
    assert codex._resolve_codex_executable(package / "bin/codex.js", "linux", "x86_64") == binary


def test_resolve_npm_platform_alias(tmp_path):
    package, binary = npm_package(tmp_path)
    metadata = json.loads((package / "package.json").read_text())
    metadata["optionalDependencies"] = {
        "@openai/codex-linux-x64": "npm:@openai/codex@0.162.1-linux-x64"
    }
    (package / "package.json").write_text(json.dumps(metadata))
    platform_package = package.parent / "codex-linux-x64"
    (platform_package / "package.json").write_text(
        json.dumps({"name": "@openai/codex", "version": "0.162.1-linux-x64"})
    )
    assert codex._resolve_codex_executable(package / "bin/codex.js", "linux", "x86_64") == binary


def test_resolve_pnpm_shim_without_executing_it(tmp_path):
    package, binary = npm_package(tmp_path)
    shim = package.parent.parent / ".bin/codex"
    shim.parent.mkdir()
    shim.write_text('#!/bin/sh\nexec node "$basedir/../@openai/codex/bin/codex.js" "$@"\n')
    assert codex._resolve_codex_executable(shim, "linux", "x86_64") == binary


def test_unrelated_wrapper_does_not_select_adjacent_package(tmp_path):
    package, _ = npm_package(tmp_path)
    shim = package.parent.parent / ".bin/codex"
    shim.parent.mkdir()
    shim.write_text("#!/bin/sh\nexec another-program\n")
    assert codex._resolve_codex_executable(shim, "linux", "x86_64") == shim


@pytest.mark.parametrize("extension", ["cmd", "ps1"])
def test_resolve_windows_npm_launcher(tmp_path, extension):
    package, _ = npm_package(tmp_path)
    native = package.parent / "codex-win32-x64"
    binary = native / "vendor/x86_64-pc-windows-msvc/bin/codex.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(pe_image())
    (native / "package.json").write_text(json.dumps({"name": "@openai/codex-win32-x64"}))
    launcher = tmp_path / f"codex.{extension}"
    launcher.write_text(r'node "%dp0%\node_modules\@openai\codex\bin\codex.js" %*')
    assert codex._resolve_codex_executable(launcher, "win32", "x86_64") == binary


@pytest.mark.parametrize("metadata", [{"name": "@openai/codex", "bin": "unexpected"}, "x" * 65537])
def test_resolve_rejects_invalid_or_oversized_package_metadata(tmp_path, metadata):
    package, _ = npm_package(tmp_path)
    (package / "package.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError):
        codex._resolve_codex_executable(package / "bin/codex.js", "linux", "x86_64")


def test_windows_snapshot_rejects_junction(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    from types import SimpleNamespace

    monkeypatch.setattr(
        Path, "lstat", lambda self: SimpleNamespace(st_mode=0o40755, st_file_attributes=0x400)
    )
    with pytest.raises(ValueError, match="reparse"):
        worker._windows_source(source, worker.PurePosixPath("codex.exe"))


def test_windows_readonly_snapshot_cleanup(tmp_path, monkeypatch):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "codex.exe").write_bytes(pe_image())
    (root / "codex.exe").chmod(0o400)
    monkeypatch.setattr(codex.sys, "platform", "win32")
    codex._cleanup_home(root)
    assert not root.exists()


def test_windows_snapshot_uses_exe_and_rejects_changed_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    binary = source / "codex.exe"
    binary.write_bytes(pe_image())
    expected = {"bin/codex": hashlib.sha256(binary.read_bytes()).hexdigest()}
    dest = tmp_path / "snapshot"
    dest.mkdir()
    worker.copy_windows_runtime(
        source, dest, expected, ["bin/codex"], source_entrypoint="codex.exe"
    )
    assert (dest / "bin/codex.exe").read_bytes() == binary.read_bytes()
    binary.write_bytes(b"changed")
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(ValueError, match="digest"):
        worker.copy_windows_runtime(
            source, second, expected, ["bin/codex"], source_entrypoint="codex.exe"
        )


def test_windows_snapshot_rejects_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    real = source / "real.exe"
    real.write_bytes(pe_image())
    (source / "codex.exe").symlink_to(real)
    dest = tmp_path / "snapshot"
    dest.mkdir()
    with pytest.raises((ValueError, OSError)):
        worker.copy_windows_runtime(
            source,
            dest,
            {"bin/codex": hashlib.sha256(real.read_bytes()).hexdigest()},
            ["bin/codex"],
            source_entrypoint="codex.exe",
        )


def test_windows_environment_keeps_only_os_bootstrap_and_isolated_user_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(codex.sys, "platform", "win32")
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    monkeypatch.setenv("USERPROFILE", "C:\\original-user")
    monkeypatch.setenv("SECRET_TOKEN", "never-copy")
    env = codex.CodexCLIProvider._record_environment(str(tmp_path))
    assert env["SystemRoot"] == "C:\\Windows"
    assert env["USERPROFILE"] == str(tmp_path)
    assert env["APPDATA"].startswith(str(tmp_path))
    assert "SECRET_TOKEN" not in env
