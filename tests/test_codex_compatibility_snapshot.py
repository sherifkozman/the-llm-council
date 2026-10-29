"""A verified runtime copy must survive normal package updates and fail closed."""

import asyncio
import hashlib
import importlib
import os
from pathlib import Path

import pytest


def snapshotter():
    module = importlib.import_module("llm_council.providers.cli.codex")
    function = getattr(
        module.CodexCLIProvider(cli_path="synthetic"), "_snapshot_verified_codex", None
    )
    assert callable(function), "verified Codex snapshot is not implemented"

    def snapshot(source, files, executables, deadline, *, candidates=None):
        return function(
            source, candidates or [{"files": files, "executables": executables}], deadline
        )

    return snapshot


def package(root):
    (root / "bin").mkdir(parents=True)
    (root / "codex-resources").mkdir()
    (root / "bin" / "codex").write_bytes(b"synthetic executable")
    (root / "bin" / "codex").chmod(0o700)
    (root / "codex-resources" / "data").write_bytes(b"synthetic resource")
    return {
        "bin/codex": hashlib.sha256(b"synthetic executable").hexdigest(),
        "codex-resources/data": hashlib.sha256(b"synthetic resource").hexdigest(),
    }


@pytest.mark.asyncio
async def test_snapshot_retains_verified_copy_during_normal_upgrade(tmp_path, monkeypatch):
    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    deadline = asyncio.get_running_loop().time() + 5
    async with snapshotter()(source, files, ["bin/codex"], deadline) as snapshot:
        executable = snapshot.executable
        assert executable.read_bytes() == b"synthetic executable"
        assert os.access(executable, os.X_OK)
        replacement = source / "bin" / "replacement"
        replacement.write_bytes(b"new upstream executable")
        replacement.replace(source / "bin" / "codex")
        assert executable.read_bytes() == b"synthetic executable"
        assert (
            executable.parent.parent / "codex-resources" / "data"
        ).read_bytes() == b"synthetic resource"
        copied_root = executable.parent.parent
    assert not copied_root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["changed", "missing", "file_symlink", "directory_symlink", "traversal", "mode_changed"],
)
async def test_snapshot_rejects_unverified_files_and_cleans_up(tmp_path, monkeypatch, failure):
    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    if failure == "changed":
        (source / "bin" / "codex").write_bytes(b"changed")
    elif failure == "missing":
        (source / "codex-resources" / "data").unlink()
    elif failure == "file_symlink":
        (source / "codex-resources" / "data").unlink()
        (source / "codex-resources" / "data").symlink_to(source / "bin" / "codex")
    elif failure == "directory_symlink":
        (source / "codex-resources").rename(source / "real-resources")
        (source / "codex-resources").symlink_to(source / "real-resources", target_is_directory=True)
    elif failure == "mode_changed":
        (source / "bin" / "codex").chmod(0o600)
    else:
        files["../outside"] = hashlib.sha256(b"outside").hexdigest()
    with pytest.raises((OSError, ValueError, RuntimeError)):
        async with snapshotter()(
            source,
            {},
            [],
            asyncio.get_running_loop().time() + 5,
            candidates=[{"files": files, "executables": ["bin/codex"]}],
        ):
            pytest.fail("unverified file was accepted")
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
async def test_snapshot_rejects_an_expired_deadline_without_leaving_files(tmp_path, monkeypatch):
    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with pytest.raises((TimeoutError, RuntimeError)):
        async with snapshotter()(
            source, files, ["bin/codex"], asyncio.get_running_loop().time() - 1
        ):
            pytest.fail("expired request was accepted")
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
async def test_snapshot_retains_directory_if_worker_cleanup_cannot_be_confirmed(
    tmp_path, monkeypatch
):
    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")

    async def cleanup_failed(*args, **kwargs):
        kwargs["cleanup_confirmed"].settled = False
        raise RuntimeError("Codex cleanup incomplete: process did not settle")

    monkeypatch.setattr(provider, "_run_cli", cleanup_failed)
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        async with provider._snapshot_verified_codex(
            source,
            [{"files": files, "executables": ["bin/codex"]}],
            asyncio.get_running_loop().time() + 5,
        ):
            pytest.fail("snapshot was accepted")
    remaining = list(tmp_path.glob("llm-council-codex-snapshot-*"))
    assert len(remaining) == 1
    remaining[0].rmdir()


@pytest.mark.asyncio
async def test_snapshot_timeout_kills_worker_before_deleting_copy(tmp_path, monkeypatch):
    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    real_spawn = asyncio.create_subprocess_exec
    children = []

    async def start_stalled_worker(*args, **kwargs):
        child = await real_spawn(args[0], "-I", "-c", "import time; time.sleep(30)", **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start_stalled_worker)
    with pytest.raises(RuntimeError, match="timed out"):
        async with snapshotter()(
            source, files, ["bin/codex"], asyncio.get_running_loop().time() + 0.3
        ):
            pytest.fail("stalled snapshot was accepted")
    assert children and children[0].returncode is not None
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
async def test_snapshot_cleans_up_after_definite_spawn_failure(tmp_path, monkeypatch):
    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    async def spawn_fails(*args, **kwargs):
        raise FileNotFoundError("synthetic worker missing")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_fails)
    with pytest.raises(FileNotFoundError):
        async with snapshotter()(
            source,
            {},
            [],
            asyncio.get_running_loop().time() + 5,
            candidates=[{"files": files, "executables": ["bin/codex"]}],
        ):
            pytest.fail("worker failure was accepted")
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
async def test_snapshot_retained_if_later_child_cleanup_is_unconfirmed(tmp_path, monkeypatch):
    import shutil
    import sys

    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")
    with pytest.raises(RuntimeError, match="snapshot cleanup incomplete"):
        async with provider._snapshot_verified_codex(
            source,
            [{"files": files, "executables": ["bin/codex"]}],
            asyncio.get_running_loop().time() + 5,
        ) as snapshot:
            copied_root = snapshot.executable.parent.parent
            real_cleanup = module.terminate_process_tree

            async def uncertain_cleanup(proc, grace):
                await real_cleanup(proc, grace)
                raise asyncio.TimeoutError("synthetic cleanup uncertainty")

            monkeypatch.setattr(module, "terminate_process_tree", uncertain_cleanup)
            with pytest.raises(RuntimeError, match="cleanup incomplete"):
                await provider._run_cli(
                    [sys.executable, "-I", "-c", "print('synthetic')"],
                    env={"PATH": os.defpath},
                    cwd=str(copied_root),
                    deadline=asyncio.get_running_loop().time() + 5,
                    cleanup_confirmed=snapshot.cleanup_confirmed,
                )
    assert copied_root.exists()
    shutil.rmtree(copied_root)


@pytest.mark.asyncio
async def test_snapshot_removed_after_later_child_cancel_is_reaped(tmp_path, monkeypatch):
    import sys

    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")
    real_spawn = asyncio.create_subprocess_exec
    started = asyncio.Event()
    children = []
    async with provider._snapshot_verified_codex(
        source,
        [{"files": files, "executables": ["bin/codex"]}],
        asyncio.get_running_loop().time() + 5,
    ) as snapshot:
        copied_root = snapshot.executable.parent.parent

        async def observe_spawn(*args, **kwargs):
            child = await real_spawn(*args, **kwargs)
            children.append(child)
            started.set()
            return child

        monkeypatch.setattr(asyncio, "create_subprocess_exec", observe_spawn)
        task = asyncio.create_task(
            provider._run_cli(
                [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                env={"PATH": os.defpath},
                cwd=str(copied_root),
                deadline=asyncio.get_running_loop().time() + 5,
                cleanup_confirmed=snapshot.cleanup_confirmed,
            )
        )
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert children and children[0].returncode is not None
        assert snapshot.cleanup_confirmed.settled is True
    assert not copied_root.exists()


@pytest.mark.asyncio
async def test_snapshot_refuses_overlapping_owned_children(tmp_path, monkeypatch):
    import sys

    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")
    real_spawn = asyncio.create_subprocess_exec
    started = asyncio.Event()
    async with provider._snapshot_verified_codex(
        source,
        [{"files": files, "executables": ["bin/codex"]}],
        asyncio.get_running_loop().time() + 5,
    ) as snapshot:

        async def observe_spawn(*args, **kwargs):
            child = await real_spawn(*args, **kwargs)
            started.set()
            return child

        monkeypatch.setattr(asyncio, "create_subprocess_exec", observe_spawn)
        task = asyncio.create_task(
            provider._run_cli(
                [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                env={"PATH": os.defpath},
                cwd=str(snapshot.executable.parent),
                deadline=asyncio.get_running_loop().time() + 5,
                cleanup_confirmed=snapshot.cleanup_confirmed,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 2)
            with pytest.raises(RuntimeError, match="snapshot.*active|snapshot.*cleanup"):
                await provider._run_cli(
                    [sys.executable, "-I", "-c", "print('synthetic')"],
                    env={"PATH": os.defpath},
                    cwd=str(snapshot.executable.parent),
                    deadline=asyncio.get_running_loop().time() + 5,
                    cleanup_confirmed=snapshot.cleanup_confirmed,
                )
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_snapshot_rejects_group_execute_without_user_permission(tmp_path, monkeypatch):
    source = tmp_path / "package"
    files = package(source)
    executable = source / "bin" / "codex"
    executable.chmod(0o610)
    if os.access(executable, os.X_OK):
        pytest.skip("current identity can execute the synthetic source")
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with pytest.raises(RuntimeError, match="snapshot"):
        async with snapshotter()(
            source,
            {},
            [],
            asyncio.get_running_loop().time() + 5,
            candidates=[{"files": files, "executables": ["bin/codex"]}],
        ):
            pytest.fail("snapshot widened executable permissions")
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
async def test_snapshot_rejects_subprocess_from_another_event_loop(tmp_path, monkeypatch):
    import sys

    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")
    async with provider._snapshot_verified_codex(
        source,
        [{"files": files, "executables": ["bin/codex"]}],
        asyncio.get_running_loop().time() + 5,
    ) as snapshot:

        async def other_loop_run():
            return await provider._run_cli(
                [sys.executable, "-I", "-c", "print('synthetic')"],
                env={"PATH": os.defpath},
                cwd=str(snapshot.executable.parent),
                deadline=asyncio.get_running_loop().time() + 5,
                cleanup_confirmed=snapshot.cleanup_confirmed,
            )

        with pytest.raises(RuntimeError, match="event loop"):
            await asyncio.to_thread(lambda: asyncio.run(other_loop_run()))


@pytest.mark.asyncio
async def test_snapshot_rejects_children_after_context_exit(tmp_path, monkeypatch):
    import sys

    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")
    async with provider._snapshot_verified_codex(
        source,
        [{"files": files, "executables": ["bin/codex"]}],
        asyncio.get_running_loop().time() + 5,
    ) as snapshot:
        marker = snapshot.cleanup_confirmed
    with pytest.raises(RuntimeError, match="snapshot.*closed"):
        await provider._run_cli(
            [sys.executable, "-I", "-c", "print('synthetic')"],
            env={"PATH": os.defpath},
            cwd=str(tmp_path),
            deadline=asyncio.get_running_loop().time() + 5,
            cleanup_confirmed=marker,
        )


@pytest.mark.asyncio
async def test_snapshot_scope_reports_unsettled_child(tmp_path, monkeypatch):
    import shutil
    import sys

    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")
    real_spawn = asyncio.create_subprocess_exec
    started = asyncio.Event()
    child_task = None
    copied_root = None
    try:
        with pytest.raises(RuntimeError, match="snapshot.*cleanup incomplete"):
            async with provider._snapshot_verified_codex(
                source,
                [{"files": files, "executables": ["bin/codex"]}],
                asyncio.get_running_loop().time() + 5,
            ) as snapshot:
                copied_root = snapshot.executable.parent.parent

                async def observe_spawn(*args, **kwargs):
                    child = await real_spawn(*args, **kwargs)
                    started.set()
                    return child

                monkeypatch.setattr(asyncio, "create_subprocess_exec", observe_spawn)
                child_task = asyncio.create_task(
                    provider._run_cli(
                        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                        env={"PATH": os.defpath},
                        cwd=str(copied_root),
                        deadline=asyncio.get_running_loop().time() + 5,
                        cleanup_confirmed=snapshot.cleanup_confirmed,
                    )
                )
                await asyncio.wait_for(started.wait(), 2)
        assert copied_root.exists()
    finally:
        if child_task is not None:
            child_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await child_task
        if copied_root is not None and copied_root.exists():
            shutil.rmtree(copied_root)


def test_candidate_copy_selects_only_the_exact_executable_and_inventory(tmp_path):
    module = importlib.import_module("llm_council.providers.cli._codex_snapshot_worker")
    source = tmp_path / "source"
    expected = package(source)
    destination = tmp_path / "copy"
    destination.mkdir()
    candidates = [
        {"files": {**expected, "bin/codex": "0" * 64}, "executables": ["bin/codex"]},
        {"files": expected, "executables": ["bin/codex"]},
    ]
    digest = module.copy_candidate_runtime(source, destination, candidates)
    assert digest == expected["bin/codex"]
    assert (destination / "bin/codex").read_bytes() == b"synthetic executable"
    assert (destination / "codex-resources/data").read_bytes() == b"synthetic resource"


def test_candidate_copy_rejects_unknown_executable(tmp_path):
    module = importlib.import_module("llm_council.providers.cli._codex_snapshot_worker")
    source = tmp_path / "source"
    expected = package(source)
    destination = tmp_path / "copy"
    destination.mkdir()
    candidates = [{"files": {**expected, "bin/codex": "0" * 64}, "executables": ["bin/codex"]}]
    with pytest.raises(ValueError):
        module.copy_candidate_runtime(source, destination, candidates)
    assert list(destination.iterdir()) == []


def test_candidate_copy_rejects_ambiguous_inventories(tmp_path):
    module = importlib.import_module("llm_council.providers.cli._codex_snapshot_worker")
    source = tmp_path / "source"
    expected = package(source)
    destination = tmp_path / "copy"
    destination.mkdir()
    candidates = [
        {"files": expected, "executables": ["bin/codex"]},
        {"files": {**expected, "codex-resources/data": "0" * 64}, "executables": ["bin/codex"]},
    ]
    with pytest.raises(ValueError):
        module.copy_candidate_runtime(source, destination, candidates)
    assert list(destination.iterdir()) == []


def test_candidate_copy_does_not_follow_executable_symlinks(tmp_path):
    module = importlib.import_module("llm_council.providers.cli._codex_snapshot_worker")
    source = tmp_path / "source"
    expected = package(source)
    (source / "bin/codex").rename(source / "bin/original")
    (source / "bin/codex").symlink_to("original")
    destination = tmp_path / "copy"
    destination.mkdir()
    with pytest.raises((OSError, ValueError)):
        module.copy_candidate_runtime(
            source, destination, [{"files": expected, "executables": ["bin/codex"]}]
        )


@pytest.mark.asyncio
async def test_snapshot_selects_a_recorded_inventory_in_owned_worker(tmp_path, monkeypatch):
    module = importlib.import_module("llm_council.providers.cli.codex")
    source = tmp_path / "source"
    expected = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    provider = module.CodexCLIProvider(cli_path=str(source / "bin/codex"))
    candidates = [
        {"files": {**expected, "bin/codex": "0" * 64}, "executables": ["bin/codex"]},
        {"files": expected, "executables": ["bin/codex"]},
    ]
    async with provider._snapshot_compatible_codex(
        candidates, asyncio.get_running_loop().time() + 5
    ) as snapshot:
        assert snapshot.executable_sha256 == expected["bin/codex"]
        assert snapshot.inventory_sha256 == module._inventory_sha256(expected, ["bin/codex"])
        assert snapshot.executable.read_bytes() == b"synthetic executable"
        root = snapshot.executable.parent.parent
    assert not root.exists()


@pytest.mark.asyncio
async def test_snapshot_rejects_unrecorded_candidate_in_owned_worker(tmp_path, monkeypatch):
    module = importlib.import_module("llm_council.providers.cli.codex")
    source = tmp_path / "source"
    expected = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    provider = module.CodexCLIProvider(cli_path=str(source / "bin/codex"))
    candidates = [{"files": {**expected, "bin/codex": "0" * 64}, "executables": ["bin/codex"]}]
    with pytest.raises(RuntimeError):
        async with provider._snapshot_compatible_codex(
            candidates, asyncio.get_running_loop().time() + 5
        ):
            pytest.fail("unrecorded executable was accepted")
    assert not list(tmp_path.glob("llm-council-codex-snapshot-*"))


@pytest.mark.asyncio
async def test_validated_record_candidates_feed_the_snapshot_worker(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone

    module = importlib.import_module("llm_council.providers.cli.codex")
    source = tmp_path / "source"
    expected = package(source)
    now = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
    criteria = {
        "os": "darwin",
        "arch": "arm64",
        "model": "synthetic-model",
        "auth": "apikey",
        "endpoint": "openai",
        "reasoning": "high",
        "output": "text",
    }
    record = {
        **criteria,
        "version": "codex-cli 0.165.0",
        "catalog_sha256": "b" * 64,
        "profile": "generation-v1",
        "files": expected,
        "executables": ["bin/codex"],
    }
    payload = json.dumps(
        {
            "schema": 1,
            "issued_at": "2026-10-29T00:00:00Z",
            "expires_at": "2026-11-29T00:00:00Z",
            "records": [record],
        }
    ).encode()
    candidates = module._compatibility_candidates(payload, criteria, now)
    assert candidates == [record]
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    provider = module.CodexCLIProvider(cli_path=str(source / "bin/codex"))
    async with provider._snapshot_compatible_codex(
        candidates, asyncio.get_running_loop().time() + 5
    ) as snapshot:
        assert snapshot.executable_sha256 == expected["bin/codex"]
        assert snapshot.executable.read_bytes() == b"synthetic executable"


@pytest.mark.asyncio
async def test_candidate_snapshot_uses_selected_executable_symlink(tmp_path, monkeypatch):
    module = importlib.import_module("llm_council.providers.cli.codex")
    source = tmp_path / "source"
    expected = package(source)
    selected = tmp_path / "selected-codex"
    selected.symlink_to(source / "bin/codex")
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    provider = module.CodexCLIProvider(cli_path=str(selected))
    candidates = [{"files": expected, "executables": ["bin/codex"]}]
    async with provider._snapshot_compatible_codex(
        candidates, asyncio.get_running_loop().time() + 5
    ) as snapshot:
        assert snapshot.executable_sha256 == expected["bin/codex"]


@pytest.mark.asyncio
async def test_candidate_snapshot_rejects_empty_candidate_inventory(tmp_path):
    module = importlib.import_module("llm_council.providers.cli.codex")
    selected = tmp_path / "codex"
    selected.write_bytes(b"synthetic")
    provider = module.CodexCLIProvider(cli_path=str(selected))
    with pytest.raises(RuntimeError, match="did not match"):
        async with provider._snapshot_compatible_codex([], asyncio.get_running_loop().time() + 5):
            pytest.fail("empty candidate inventory was accepted")


@pytest.mark.asyncio
async def test_snapshot_inventory_binding_survives_caller_mutation(tmp_path, monkeypatch):
    module = importlib.import_module("llm_council.providers.cli.codex")
    source = tmp_path / "source"
    expected = package(source)
    initial_inventory = module._inventory_sha256(expected, ["bin/codex"])
    candidate = {"files": expected, "executables": ["bin/codex"]}
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    provider = module.CodexCLIProvider(cli_path=str(source / "bin/codex"))
    real_run = provider._run_cli

    async def mutate_after_serialization(*args, **kwargs):
        candidate["files"]["codex-resources/data"] = "0" * 64
        return await real_run(*args, **kwargs)

    monkeypatch.setattr(provider, "_run_cli", mutate_after_serialization)
    async with provider._snapshot_compatible_codex(
        [candidate], asyncio.get_running_loop().time() + 5
    ) as snapshot:
        assert snapshot.inventory_sha256 == initial_inventory
        assert (
            snapshot.executable.parent.parent / "codex-resources/data"
        ).read_bytes() == b"synthetic resource"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["codex", "codex-custom"])
async def test_candidate_snapshot_binds_explicit_nonstandard_native_path(
    tmp_path, monkeypatch, name
):
    module = importlib.import_module("llm_council.providers.cli.codex")
    source = tmp_path / "vendor"
    expected = package(source)
    selected = source / name
    (source / "bin/codex").rename(selected)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    provider = module.CodexCLIProvider(cli_path=str(selected))
    candidates = [{"files": expected, "executables": ["bin/codex"]}]
    async with provider._snapshot_compatible_codex(
        candidates, asyncio.get_running_loop().time() + 5
    ) as snapshot:
        assert snapshot.executable_sha256 == expected["bin/codex"]
        assert snapshot.executable.read_bytes() == b"synthetic executable"
        assert (
            snapshot.executable.parent.parent / "codex-resources/data"
        ).read_bytes() == b"synthetic resource"


@pytest.mark.asyncio
async def test_nonstandard_selected_executable_cannot_use_other_recorded_binary(
    tmp_path, monkeypatch
):
    module = importlib.import_module("llm_council.providers.cli.codex")
    source = tmp_path / "vendor"
    expected = package(source)
    selected = source / "custom-codex"
    selected.write_bytes(b"different executable")
    selected.chmod(0o700)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    provider = module.CodexCLIProvider(cli_path=str(selected))
    with pytest.raises(RuntimeError):
        async with provider._snapshot_compatible_codex(
            [{"files": expected, "executables": ["bin/codex"]}],
            asyncio.get_running_loop().time() + 5,
        ):
            pytest.fail("a different binary was substituted for the selected executable")


@pytest.mark.parametrize(
    "entrypoint", ["../codex", "/tmp/codex", "nested/codex", "a\\\\b", "bad:codex", ""]
)
def test_candidate_copy_rejects_unsafe_alternate_entrypoint(tmp_path, entrypoint):
    module = importlib.import_module("llm_council.providers.cli._codex_snapshot_worker")
    source = tmp_path / "source"
    expected = package(source)
    destination = tmp_path / "copy"
    destination.mkdir()
    with pytest.raises(ValueError):
        module.copy_candidate_runtime(
            source,
            destination,
            [{"files": expected, "executables": ["bin/codex"]}],
            source_entrypoint=entrypoint,
        )
    assert list(destination.iterdir()) == []


def test_candidate_copy_rejects_symlinked_alternate_entrypoint(tmp_path):
    module = importlib.import_module("llm_council.providers.cli._codex_snapshot_worker")
    source = tmp_path / "source"
    expected = package(source)
    (source / "alternate").symlink_to(source / "bin/codex")
    destination = tmp_path / "copy"
    destination.mkdir()
    with pytest.raises((OSError, ValueError)):
        module.copy_candidate_runtime(
            source,
            destination,
            [{"files": expected, "executables": ["bin/codex"]}],
            source_entrypoint="alternate",
        )
    assert list(destination.iterdir()) == []


@pytest.mark.asyncio
async def test_candidate_snapshot_supports_renamed_executable_in_bin(tmp_path, monkeypatch):
    module = importlib.import_module("llm_council.providers.cli.codex")
    source = tmp_path / "package"
    expected = package(source)
    selected = source / "bin/custom-codex"
    (source / "bin/codex").rename(selected)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    provider = module.CodexCLIProvider(cli_path=str(selected))
    async with provider._snapshot_compatible_codex(
        [{"files": expected, "executables": ["bin/codex"]}],
        asyncio.get_running_loop().time() + 5,
    ) as snapshot:
        assert snapshot.executable_sha256 == expected["bin/codex"]
        assert (
            snapshot.executable.parent.parent / "codex-resources/data"
        ).read_bytes() == b"synthetic resource"


@pytest.mark.asyncio
async def test_pre_spawn_guard_cannot_consume_remaining_deadline(tmp_path, monkeypatch):
    import sys
    import time
    from unittest.mock import AsyncMock

    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")
    spawn = AsyncMock(side_effect=AssertionError("subprocess spawned after deadline"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    def slow_guard():
        time.sleep(0.1)
        return True

    with pytest.raises(RuntimeError, match="timed out"):
        await provider._run_cli(
            [sys.executable, "-c", "print('synthetic')"],
            env={"PATH": os.defpath},
            cwd=str(tmp_path),
            deadline=asyncio.get_running_loop().time() + 0.05,
            pre_spawn_guard=slow_guard,
        )
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconfirmed_cleanup_fails_inside_an_outer_exception_handler(tmp_path, monkeypatch):
    source = tmp_path / "package"
    files = package(source)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    try:
        raise LookupError("unrelated handled error")
    except LookupError:
        with pytest.raises(RuntimeError, match="snapshot cleanup incomplete"):
            async with snapshotter()(
                source, files, ["bin/codex"], asyncio.get_running_loop().time() + 5
            ) as snapshot:
                snapshot.cleanup_confirmed.settled = False


def test_cleanup_failure_does_not_rethrow_an_unrelated_outer_exception(tmp_path):
    module = importlib.import_module("llm_council.providers.cli.codex")
    try:
        raise LookupError("unrelated handled error")
    except LookupError:
        with pytest.raises(RuntimeError, match="cleanup incomplete"):
            module._cleanup_home(tmp_path / "missing")
