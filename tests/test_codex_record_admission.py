"""Record admission must inspect the owned snapshot without copying credentials."""

import asyncio
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llm_council.providers.base import GenerateRequest
from llm_council.providers.cli import codex

CATALOG = b'{"models":[]}'


def synthetic_package(tmp_path, catalog=CATALOG):
    root = tmp_path / "package"
    executable = root / "bin/codex"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    sys.stdout.write('codex-cli 0.165.0\\n')\n"
        "elif sys.argv[1:] == ['debug', 'models', '--bundled']:\n"
        f"    sys.stdout.buffer.write({catalog!r})\n"
        "elif sys.argv[-2:] == ['login', 'status']:\n"
        "    print('Logged in using an API key')\n"
        "elif sys.argv[1] == 'exec':\n"
        '    print(\'{"type":"item.completed","item":{"type":"agent_message","text":"FINAL"}}\')\n'
        '    print(\'{"type":"turn.completed","usage":{"input_tokens":2,"output_tokens":1}}\')\n'
        "else:\n"
        "    sys.exit(4)\n"
    )
    executable.chmod(0o700)
    return root, executable


def payload_for(executable, *, catalog=CATALOG, version="codex-cli 0.165.0"):
    now = datetime.now(timezone.utc)
    system = {"darwin": "darwin", "linux": "linux"}[sys.platform]
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x86_64"}[platform.machine().lower()]
    record = {
        "profile": "generation-v1",
        "version": version,
        "os": system,
        "arch": arch,
        "model": "synthetic-model",
        "auth": "apikey",
        "endpoint": "openai",
        "reasoning": "default",
        "output": "text",
        "catalog_sha256": hashlib.sha256(catalog).hexdigest(),
        "files": {"bin/codex": hashlib.sha256(executable.read_bytes()).hexdigest()},
        "executables": ["bin/codex"],
    }
    value = {
        "schema": 1,
        "issued_at": (now - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": (now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "records": [record],
    }
    return json.dumps(value).encode()


def runtime():
    return codex._CodexRuntime(
        {"PATH": os.defpath, "SYNTHETIC_SECRET_TOKEN": "do-not-copy"},
        b'{"auth_mode":"apikey","OPENAI_API_KEY":"do-not-copy"}',
        "apikey",
        {"model_provider": "openai", "cli_auth_credentials_store": "file"},
    )


@pytest.fixture
def allow_synthetic_script(monkeypatch):
    # Exercise the real snapshot and child ownership with a harmless script;
    # native header checks have separate, non-mocked tests below.
    monkeypatch.setattr(codex, "_native_executable_matches", lambda *args: True)


@pytest.mark.asyncio
async def test_admission_uses_the_same_owned_snapshot_for_credential_free_probes(
    tmp_path, monkeypatch, allow_synthetic_script
):
    root, executable = synthetic_package(tmp_path)
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(
        provider, "_fetch_compatibility_record", AsyncMock(return_value=payload_for(executable))
    )
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    real_run = provider._run_cli
    calls = []

    async def capture(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[-1] == "--version" or cmd[-3:] == ["debug", "models", "--bundled"]:
            assert "SYNTHETIC_SECRET_TOKEN" not in kwargs["env"]
            assert not (Path(kwargs["env"]["CODEX_HOME"]) / "auth.json").exists()
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", capture)
    identity = {"cli_version": None}
    request = GenerateRequest(prompt="synthetic", model="synthetic-model")
    async with provider._admit_recorded_codex(
        runtime(), request, deadline=asyncio.get_running_loop().time() + 5, identity=identity
    ) as admission:
        assert admission.version == "codex-cli 0.165.0"
        assert admission.catalog == CATALOG
        assert admission.is_current()
        assert identity["cli_version"] == admission.version
        assert admission.snapshot.executable != executable
        probes = [
            entry
            for entry in calls
            if entry[0][-1] == "--version" or entry[0][-3:] == ["debug", "models", "--bundled"]
        ]
        assert len(probes) == 2
        assert all(cmd[0] == str(admission.snapshot.executable) for cmd, _ in probes)
        assert all(
            kw["cleanup_confirmed"] is admission.snapshot.cleanup_confirmed for _, kw in probes
        )
        copied = admission.snapshot.executable.parent.parent
    assert not copied.exists()
    assert not admission.is_current()


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["binary", "catalog", "version", "context"])
async def test_admission_rejects_unmatched_observations(
    tmp_path, monkeypatch, mismatch, allow_synthetic_script
):
    root, executable = synthetic_package(tmp_path)
    payload = payload_for(
        executable,
        catalog=b"different" if mismatch == "catalog" else CATALOG,
        version="codex-cli 0.165.1" if mismatch == "version" else "codex-cli 0.165.0",
    )
    if mismatch == "binary":
        executable.write_text(executable.read_text() + "# changed\n")
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(provider, "_fetch_compatibility_record", AsyncMock(return_value=payload))
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    request = GenerateRequest(
        prompt="synthetic", model="other" if mismatch == "context" else "synthetic-model"
    )
    with pytest.raises(RuntimeError):
        async with provider._admit_recorded_codex(
            runtime(),
            request,
            deadline=asyncio.get_running_loop().time() + 5,
            identity={"cli_version": None},
        ):
            pytest.fail("mismatched candidate was admitted")
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
async def test_admission_rejects_unrecorded_custom_provider_route(tmp_path, monkeypatch):
    _, executable = synthetic_package(tmp_path)
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    fetch = AsyncMock(return_value=payload_for(executable))
    monkeypatch.setattr(provider, "_fetch_compatibility_record", fetch)
    configured = runtime()
    configured.config["openai_base_url"] = "http://127.0.0.1:1234"
    with pytest.raises(RuntimeError, match="compatibility record"):
        async with provider._admit_recorded_codex(
            configured,
            GenerateRequest(prompt="synthetic", model="synthetic-model"),
            deadline=asyncio.get_running_loop().time() + 5,
            identity={"cli_version": None},
        ):
            pytest.fail("unrecorded route was admitted")
    fetch.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("expire_at", ["snapshot", "version"])
async def test_admission_never_probes_after_record_expiry(
    tmp_path, monkeypatch, expire_at, allow_synthetic_script
):
    from contextlib import asynccontextmanager

    _, executable = synthetic_package(tmp_path)
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(
        provider, "_fetch_compatibility_record", AsyncMock(return_value=payload_for(executable))
    )
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    start_time = datetime.now(timezone.utc)

    class ControlledDatetime(datetime):
        current = start_time

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(codex, "datetime", ControlledDatetime)
    real_snapshot = provider._snapshot_compatible_codex
    real_run = provider._run_cli
    probes = []

    @asynccontextmanager
    async def slow_snapshot(*args, **kwargs):
        async with real_snapshot(*args, **kwargs) as snapshot:
            if expire_at == "snapshot":
                ControlledDatetime.current = start_time + timedelta(days=2)
            yield snapshot

    async def capture(cmd, **kwargs):
        is_probe = cmd[-1] == "--version" or cmd[-3:] == ["debug", "models", "--bundled"]
        if is_probe:
            probes.append(cmd)
        result = await real_run(cmd, **kwargs)
        if cmd[-1] == "--version" and expire_at == "version":
            ControlledDatetime.current = start_time + timedelta(days=2)
        return result

    monkeypatch.setattr(provider, "_snapshot_compatible_codex", slow_snapshot)
    monkeypatch.setattr(provider, "_run_cli", capture)
    with pytest.raises(RuntimeError, match="record"):
        async with provider._admit_recorded_codex(
            runtime(),
            GenerateRequest(prompt="synthetic", model="synthetic-model"),
            deadline=asyncio.get_running_loop().time() + 5,
            identity={"cli_version": None},
        ):
            pytest.fail("expired record was admitted")
    assert len(probes) == (0 if expire_at == "snapshot" else 1)


def test_record_environment_has_no_inherited_proxy_ca_or_loader_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted.invalid")
    monkeypatch.setenv("SSL_CERT_FILE", "/synthetic/untrusted.pem")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/synthetic/untrusted.dylib")
    provider = codex.CodexCLIProvider(cli_path="synthetic")
    env = provider._record_environment(str(tmp_path))
    assert env == {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "CODEX_HOME": str(tmp_path / ".codex"),
        "XDG_CONFIG_HOME": str(tmp_path / ".config"),
        "TMPDIR": str(tmp_path),
        "TMP": str(tmp_path),
        "TEMP": str(tmp_path),
        "LANG": "C",
        "LC_ALL": "C",
    }


@pytest.mark.parametrize(
    "system,arch,content,expected",
    [
        (
            "darwin",
            "arm64",
            b"\xcf\xfa\xed\xfe\x0c\x00\x00\x01" + bytes(4) + b"\x02\x00\x00\x00" + bytes(4),
            True,
        ),
        (
            "darwin",
            "x86_64",
            b"\xcf\xfa\xed\xfe\x07\x00\x00\x01" + bytes(4) + b"\x02\x00\x00\x00" + bytes(4),
            True,
        ),
        (
            "darwin",
            "arm64",
            b"\xcf\xfa\xed\xfe\x07\x00\x00\x01" + bytes(4) + b"\x02\x00\x00\x00" + bytes(4),
            False,
        ),
        ("linux", "x86_64", b"\x7fELF\x02\x01\x01" + bytes(9) + b"\x02\x00\x3e\x00", True),
        ("linux", "arm64", b"\x7fELF\x02\x01\x01" + bytes(9) + b"\x02\x00\xb7\x00", True),
        ("linux", "arm64", b"\x7fELF\x02\x01\x01" + bytes(9) + b"\x02\x00\x3e\x00", False),
        ("darwin", "arm64", b"#!/usr/bin/env python3\n", False),
    ],
)
def test_native_executable_header_matches_recorded_platform(
    tmp_path, system, arch, content, expected
):
    path = tmp_path / "codex"
    path.write_bytes(content)
    assert codex._native_executable_matches(path, system, arch) is expected


def test_record_profile_sanitizes_verified_catalog(tmp_path):
    catalog = {
        "models": [
            {
                "slug": "synthetic-model",
                "apply_patch_tool_type": "freeform",
                "experimental_supported_tools": ["send_user_message_async", "clock"],
                "tool_mode": "code_mode_only",
                "multi_agent_version": "v2",
                "supported_reasoning_levels": [{"effort": "high"}],
            }
        ]
    }
    provider = codex.CodexCLIProvider(cli_path="synthetic")
    path = provider._write_generation_catalog(
        json.dumps(catalog),
        "synthetic-model",
        "high",
        root=tmp_path,
        profile="generation-v1",
    )
    selected = json.loads(Path(path).read_text())["models"][0]
    assert selected["apply_patch_tool_type"] is None
    assert selected["experimental_supported_tools"] == []
    assert selected["tool_mode"] == "direct"
    assert selected["multi_agent_version"] == "disabled"
    assert selected["supported_reasoning_levels"] == [{"effort": "high"}]


@pytest.mark.parametrize(
    "field,value",
    [
        ("experimental_supported_tools", ["unknown-tool"]),
        ("tool_mode", "unknown-mode"),
        ("multi_agent_version", "unknown-version"),
    ],
)
def test_record_profile_rejects_unknown_catalog_controls(tmp_path, field, value):
    model = {
        "slug": "synthetic-model",
        "apply_patch_tool_type": "freeform",
        "experimental_supported_tools": [],
        "tool_mode": "direct",
        "multi_agent_version": "disabled",
        "supported_reasoning_levels": [{"effort": "high"}],
    }
    model[field] = value
    provider = codex.CodexCLIProvider(cli_path="synthetic")
    with pytest.raises(ValueError, match="tool shape"):
        provider._write_generation_catalog(
            json.dumps({"models": [model]}),
            "synthetic-model",
            "high",
            root=tmp_path,
            profile="generation-v1",
        )


def generation_catalog():
    return json.dumps(
        {
            "models": [
                {
                    "slug": "synthetic-model",
                    "apply_patch_tool_type": "freeform",
                    "experimental_supported_tools": ["send_user_message_async", "clock"],
                    "tool_mode": "code_mode_only",
                    "multi_agent_version": "v2",
                    "supported_reasoning_levels": [{"effort": "high"}],
                }
            ]
        }
    ).encode()


@pytest.mark.asyncio
async def test_recorded_generation_uses_owned_copy_and_profile(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    source, executable = synthetic_package(tmp_path, catalog)
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(
        provider,
        "_fetch_compatibility_record",
        AsyncMock(return_value=payload_for(executable, catalog=catalog)),
    )
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    real_run = provider._run_cli
    observed = []

    async def capture(cmd, **kwargs):
        if len(cmd) > 1 and cmd[1] == "exec":
            observed.append((cmd, kwargs))
            assert cmd[0] != str(executable)
            assert kwargs["cleanup_confirmed"] is not None
            home = Path(kwargs["env"]["HOME"])
            assert kwargs["env"] == provider._record_environment(str(home))
            assert (home / ".codex/auth.json").read_bytes() == runtime().auth
            arguments = list(cmd)
            assert "features.goals=false" in arguments
            assert "features.shell_tool=false" in arguments
            catalog_arg = next(arg for arg in arguments if arg.startswith("model_catalog_json="))
            rewritten = json.loads(Path(json.loads(catalog_arg.split("=", 1)[1])).read_text())
            selected = rewritten["models"][0]
            assert selected["apply_patch_tool_type"] is None
            assert selected["experimental_supported_tools"] == []
            assert selected["tool_mode"] == "direct"
            assert selected["multi_agent_version"] == "disabled"
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", capture)
    response = await provider.generate(GenerateRequest(prompt="synthetic", model="synthetic-model"))
    assert response.text == "FINAL"
    assert len(observed) == 1
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))
    assert not list(tmp_path.glob("llm-council-codex-home-*"))


@pytest.mark.asyncio
async def test_recorded_generation_rechecks_freshness_before_spawn(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(
        provider,
        "_fetch_compatibility_record",
        AsyncMock(return_value=payload_for(executable, catalog=catalog)),
    )
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr(codex._CodexAdmission, "is_current", lambda self: False)
    real_run = provider._run_cli
    observed = []

    async def capture(cmd, **kwargs):
        if len(cmd) > 1 and cmd[1] == "exec":
            observed.append(cmd)
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", capture)
    with pytest.raises(RuntimeError, match="record"):
        await provider._generate(
            GenerateRequest(prompt="synthetic", model="synthetic-model"), {"cli_version": None}
        )
    assert observed == []
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
async def test_public_generation_does_not_run_changed_binary(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    payload = payload_for(executable, catalog=catalog)
    executable.write_text(executable.read_text() + "# changed\n")
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(provider, "_fetch_compatibility_record", AsyncMock(return_value=payload))
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    real_run = provider._run_cli
    probes = []

    async def capture(cmd, **kwargs):
        if cmd[-1] == "--version" or cmd[-3:] == ["debug", "models", "--bundled"]:
            probes.append(cmd)
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", capture)
    with pytest.raises(RuntimeError):
        await provider.generate(GenerateRequest(prompt="synthetic", model="synthetic-model"))
    assert probes == []


def test_legacy_record_profile_preserves_existing_catalog_controls(tmp_path):
    model = {
        "slug": "synthetic-model",
        "apply_patch_tool_type": "freeform",
        "experimental_supported_tools": [],
        "tool_mode": "code_mode",
        "multi_agent_version": "v1",
        "supported_reasoning_levels": [{"effort": "high"}],
    }
    path = codex.CodexCLIProvider(cli_path="synthetic")._write_generation_catalog(
        json.dumps({"models": [model]}),
        "synthetic-model",
        "high",
        root=tmp_path,
        profile="generation-legacy-v1",
    )
    selected = json.loads(Path(path).read_text())["models"][0]
    assert selected["apply_patch_tool_type"] is None
    assert selected["experimental_supported_tools"] == []
    assert selected["tool_mode"] == "direct"
    assert selected["multi_agent_version"] == "disabled"


@pytest.mark.asyncio
async def test_legacy_record_profile_does_not_add_newer_goal_flag(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = json.dumps(
        {
            "models": [
                {
                    "slug": "synthetic-model",
                    "apply_patch_tool_type": "freeform",
                    "experimental_supported_tools": [],
                    "tool_mode": "code_mode",
                    "multi_agent_version": "v1",
                    "supported_reasoning_levels": [{"effort": "high"}],
                }
            ]
        }
    ).encode()
    _, executable = synthetic_package(tmp_path, catalog)
    value = json.loads(payload_for(executable, catalog=catalog))
    value["records"][0]["profile"] = "generation-legacy-v1"
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(
        provider, "_fetch_compatibility_record", AsyncMock(return_value=json.dumps(value).encode())
    )
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    real_run = provider._run_cli
    observed = []

    async def capture(cmd, **kwargs):
        if len(cmd) > 1 and cmd[1] == "exec":
            observed.append(cmd)
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", capture)
    response = await provider.generate(GenerateRequest(prompt="synthetic", model="synthetic-model"))
    assert response.text == "FINAL"
    assert len(observed) == 1
    assert "features.goals=false" not in observed[0]
    assert "features.shell_tool=false" in observed[0]


@pytest.mark.asyncio
async def test_doctor_uses_the_same_recorded_snapshot_as_generation(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    provider = codex.CodexCLIProvider(cli_path=str(executable), default_model="synthetic-model")
    monkeypatch.setattr(
        provider,
        "_fetch_compatibility_record",
        AsyncMock(return_value=payload_for(executable, catalog=catalog)),
    )
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    real_run = provider._run_cli
    logins = []

    async def capture(cmd, **kwargs):
        if cmd[-2:] == ["login", "status"]:
            logins.append((cmd, kwargs))
            assert cmd[0] != str(executable)
            assert kwargs["cleanup_confirmed"] is not None
            assert kwargs["env"] == provider._record_environment(kwargs["env"]["HOME"])
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", capture)
    result = await provider.doctor()
    assert result.ok, result.message
    assert len(logins) == 1
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
async def test_doctor_rejects_missing_record_before_running_codex(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    provider = codex.CodexCLIProvider(cli_path=str(executable), default_model="synthetic-model")
    monkeypatch.setattr(provider, "_fetch_compatibility_record", AsyncMock(return_value=None))
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    real_run = provider._run_cli
    calls = []

    async def capture(cmd, **kwargs):
        calls.append(cmd)
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", capture)
    result = await provider.doctor()
    assert not result.ok
    assert "record" in result.message
    assert calls == []


@pytest.mark.asyncio
async def test_doctor_never_starts_login_after_record_expiry(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    provider = codex.CodexCLIProvider(cli_path=str(executable), default_model="synthetic-model")
    monkeypatch.setattr(
        provider,
        "_fetch_compatibility_record",
        AsyncMock(return_value=payload_for(executable, catalog=catalog)),
    )
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr(codex._CodexAdmission, "is_current", lambda self: False)
    real_run = provider._run_cli
    logins = []

    async def capture(cmd, **kwargs):
        if cmd[-2:] == ["login", "status"]:
            logins.append(cmd)
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", capture)
    result = await provider.doctor()
    assert not result.ok
    assert "record" in result.message
    assert logins == []
    assert not list(tmp_path.glob("llm-council-codex-home-*"))
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code,expected_ok", [(0, True), (1, False)])
async def test_doctor_recognizes_native_api_key_status_only_on_success(
    tmp_path, monkeypatch, allow_synthetic_script, exit_code, expected_ok
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    content = executable.read_text().replace(
        "print('Logged in using an API key')",
        f"print('API key configured - synthetic***masked'); sys.exit({exit_code})",
    )
    executable.write_text(content)
    provider = codex.CodexCLIProvider(cli_path=str(executable), default_model="synthetic-model")
    monkeypatch.setattr(
        provider,
        "_fetch_compatibility_record",
        AsyncMock(return_value=payload_for(executable, catalog=catalog)),
    )
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    result = await provider.doctor()
    assert result.ok is expected_ok
    assert "synthetic***masked" not in result.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout,stderr,category",
    [
        ("API key configured - visible***fragment", "not logged in", "apikey"),
        ("API key configured - visible***fragment\nnot logged in", "", "apikey"),
        ("API key configured - visible***fragment", "", "chatgpt"),
        ("Logged in using ChatGPT", "", "apikey"),
    ],
)
async def test_doctor_rejects_conflicting_auth_status_without_disclosure(
    tmp_path, monkeypatch, allow_synthetic_script, stdout, stderr, category
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    document = json.loads(payload_for(executable, catalog=catalog))
    document["records"][0]["auth"] = category
    document["records"][0]["endpoint"] = "chatgpt" if category == "chatgpt" else "openai"
    selected = runtime()
    selected.category = category
    provider = codex.CodexCLIProvider(cli_path=str(executable), default_model="synthetic-model")
    monkeypatch.setattr(
        provider,
        "_fetch_compatibility_record",
        AsyncMock(return_value=json.dumps(document).encode()),
    )
    monkeypatch.setattr(provider, "_resolve_runtime", lambda: selected)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    real_run = provider._run_cli

    async def status(cmd, **kwargs):
        if cmd[-2:] == ["login", "status"]:
            return 0, codex._LiveCodexState(stdout_parts=[stdout], stderr_parts=[stderr])
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", status)
    result = await provider.doctor()
    assert not result.ok
    assert "visible***fragment" not in result.message


@pytest.mark.asyncio
async def test_doctor_accepts_chatgpt_status_only_for_selected_chatgpt_auth(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    document = json.loads(payload_for(executable, catalog=catalog))
    document["records"][0].update(auth="chatgpt", endpoint="chatgpt")
    selected = codex._CodexRuntime(
        {},
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {
                    "id_token": "synthetic",
                    "access_token": "synthetic",
                    "refresh_token": "synthetic",
                    "account_id": "synthetic",
                },
            }
        ).encode(),
        "chatgpt",
        {"model_provider": "openai", "cli_auth_credentials_store": "file"},
    )
    provider = codex.CodexCLIProvider(cli_path=str(executable), default_model="synthetic-model")
    monkeypatch.setattr(
        provider,
        "_fetch_compatibility_record",
        AsyncMock(return_value=json.dumps(document).encode()),
    )
    monkeypatch.setattr(provider, "_resolve_runtime", lambda: selected)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    real_run = provider._run_cli

    async def status(cmd, **kwargs):
        if cmd[-2:] == ["login", "status"]:
            return 0, codex._LiveCodexState(stdout_parts=["Logged in using ChatGPT"])
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", status)
    result = await provider.doctor()
    assert result.ok
    assert "Logged in using ChatGPT" in result.message


@pytest.mark.asyncio
async def test_public_generation_accepts_exact_context_from_compact_record(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    document = json.loads(payload_for(executable, catalog=catalog))
    flat = document["records"][0]
    fields = {"model", "auth", "endpoint", "reasoning", "output"}
    build = {key: value for key, value in flat.items() if key not in fields}
    build["contexts"] = [{key: flat[key] for key in fields}]
    document.update(schema=2, records=[build])
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(
        provider,
        "_fetch_compatibility_record",
        AsyncMock(return_value=json.dumps(document).encode()),
    )
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    result = await provider.generate(GenerateRequest(prompt="synthetic", model="synthetic-model"))
    assert result.text == "FINAL"


@pytest.mark.asyncio
async def test_expired_record_at_spawn_task_prevents_generation(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(
        provider,
        "_fetch_compatibility_record",
        AsyncMock(return_value=payload_for(executable, catalog=catalog)),
    )
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    checks = iter([True, False])
    monkeypatch.setattr(codex._CodexAdmission, "is_current", lambda self, **kwargs: next(checks))
    real_run = provider._run_cli
    launched = []

    async def capture(cmd, **kwargs):
        if len(cmd) > 1 and cmd[1] == "exec":
            launched.append(cmd)
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", capture)
    with pytest.raises(RuntimeError, match="record"):
        await provider.generate(GenerateRequest(prompt="synthetic", model="synthetic-model"))
    assert len(launched) == 1  # Attempted wrapper call; guard must prevent OS spawn.
    assert not list(tmp_path.glob("llm-council-codex-home-*"))
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
async def test_custom_record_uses_only_the_user_selected_api_route(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    value = json.loads(payload_for(executable, catalog=catalog))
    value["records"][0]["endpoint"] = "custom"
    selected = runtime()
    selected.config["openai_base_url"] = "https://synthetic-route.invalid/v1"
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(
        provider, "_fetch_compatibility_record", AsyncMock(return_value=json.dumps(value).encode())
    )
    monkeypatch.setattr(provider, "_resolve_runtime", lambda: selected)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    real_run = provider._run_cli
    generation = []

    async def capture(cmd, **kwargs):
        if len(cmd) > 1 and cmd[1] == "exec":
            generation.append(cmd)
        return await real_run(cmd, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", capture)
    result = await provider.generate(GenerateRequest(prompt="synthetic", model="synthetic-model"))
    assert result.text == "FINAL"
    assert len(generation) == 1
    assert 'openai_base_url="https://synthetic-route.invalid/v1"' in generation[0]


@pytest.mark.asyncio
async def test_custom_record_does_not_authorize_default_route(
    tmp_path, monkeypatch, allow_synthetic_script
):
    catalog = generation_catalog()
    _, executable = synthetic_package(tmp_path, catalog)
    value = json.loads(payload_for(executable, catalog=catalog))
    value["records"][0]["endpoint"] = "custom"
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    monkeypatch.setattr(
        provider, "_fetch_compatibility_record", AsyncMock(return_value=json.dumps(value).encode())
    )
    monkeypatch.setattr(provider, "_resolve_runtime", runtime)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with pytest.raises(RuntimeError, match="compatibility record"):
        await provider.generate(GenerateRequest(prompt="synthetic", model="synthetic-model"))


@pytest.mark.asyncio
async def test_chatgpt_custom_route_is_not_assumed_qualified(tmp_path, monkeypatch):
    _, executable = synthetic_package(tmp_path)
    provider = codex.CodexCLIProvider(cli_path=str(executable))
    selected = runtime()
    selected.category = "chatgpt"
    selected.config["openai_base_url"] = "https://synthetic-route.invalid/v1"
    fetch = AsyncMock(return_value=payload_for(executable))
    monkeypatch.setattr(provider, "_fetch_compatibility_record", fetch)
    with pytest.raises(RuntimeError, match="route"):
        async with provider._admit_recorded_codex(
            selected,
            GenerateRequest(prompt="synthetic", model="synthetic-model"),
            deadline=asyncio.get_running_loop().time() + 5,
            identity={"cli_version": None},
        ):
            pytest.fail("unqualified ChatGPT custom route was admitted")
    fetch.assert_not_awaited()


@pytest.mark.parametrize(
    "item_type",
    [
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "collab_tool_call",
        "web_search",
        "todo_list",
        "unknown_future_tool",
    ],
)
@pytest.mark.parametrize("event_type", ["item.started", "item.updated", "item.completed"])
def test_tool_or_unknown_item_is_not_accepted_as_generation_output(item_type, event_type):
    state = codex._LiveCodexState()
    codex._ingest_codex_stdout_line(
        json.dumps({"type": event_type, "item": {"type": item_type}}), state
    )
    assert state.protocol_error


@pytest.mark.asyncio
async def test_unexpected_tool_event_terminates_child_without_waiting_for_deadline(tmp_path):
    marker = tmp_path / "after-event"
    script = (
        "import json, sys, time\n"
        "print(json.dumps({'type': 'item.started', 'item': {'type': 'mcp_tool_call'}}), flush=True)\n"
        "time.sleep(10)\n"
        "open(sys.argv[1], 'w').write('unexpected')\n"
    )
    provider = codex.CodexCLIProvider(cli_path=sys.executable)
    start = asyncio.get_running_loop().time()
    with pytest.raises(RuntimeError, match="unexpected.*item"):
        await provider._run_cli(
            [sys.executable, "-c", script, str(marker)],
            env={"PATH": os.defpath},
            cwd=str(tmp_path),
            deadline=start + 3,
            events=True,
        )
    assert asyncio.get_running_loop().time() - start < 3
    assert not marker.exists()


@pytest.mark.parametrize("event_type", ["tool.started", "mcp_tool_call", "future.event"])
def test_unknown_top_level_event_is_not_accepted_as_generation_output(event_type):
    state = codex._LiveCodexState()
    codex._ingest_codex_stdout_line(json.dumps({"type": event_type}), state)
    assert state.protocol_error


@pytest.mark.asyncio
async def test_unknown_top_level_event_terminates_child(tmp_path):
    script = (
        "import json, time; print(json.dumps({'type': 'tool.started'}), flush=True); time.sleep(10)"
    )
    provider = codex.CodexCLIProvider(cli_path=sys.executable)
    start = asyncio.get_running_loop().time()
    with pytest.raises(RuntimeError, match="unexpected.*event"):
        await provider._run_cli(
            [sys.executable, "-c", script],
            env={"PATH": os.defpath},
            cwd=str(tmp_path),
            deadline=start + 3,
            events=True,
        )
    assert asyncio.get_running_loop().time() - start < 3


def test_catalog_rejects_missing_record_profile(tmp_path):
    """A version string alone cannot select a generation contract."""
    provider = codex.CodexCLIProvider(cli_path="synthetic")
    with pytest.raises(ValueError, match="compatibility profile"):
        provider._write_generation_catalog(
            generation_catalog(),
            "synthetic-model",
            "high",
            root=tmp_path,
        )
