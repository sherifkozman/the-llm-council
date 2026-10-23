"""Automatic evidence must not disclose content outside the selected repository."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llm_council.engine.evidence import _candidate_files, _search_files
from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("link_kind", ["file", "search_root", "nested_directory"])
async def test_outside_marker_never_reaches_provider(
    tmp_path, monkeypatch, mock_registry, mock_provider, link_kind
):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("token HARMLESS_OUTSIDE_MARKER\n")
    if link_kind == "search_root":
        (repo / "src").symlink_to(outside, target_is_directory=True)
    else:
        (repo / "src").mkdir()
        if link_kind == "file":
            (repo / "src" / "auth.py").symlink_to(outside / "private.txt")
        else:
            (repo / "src" / "nested").symlink_to(outside, target_is_directory=True)
    (repo / "tests").mkdir()
    (repo / "tests" / "auth.py").write_text("token INTENDED_REPOSITORY_MARKER\n")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("LLM_COUNCIL_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr("llm_council.engine.orchestrator.get_registry", lambda: mock_registry)
    capture = AsyncMock(wraps=mock_provider.generate)
    monkeypatch.setattr(mock_provider, "generate", capture)
    orch = Orchestrator(
        providers=["mock"], config=OrchestratorConfig(mode="review", enable_artifacts=False)
    )
    orch._task = "Review token handling"
    orch._subagent_name = "critic"
    orch._prepare_run("critic")
    orch._evidence_bundle = await orch._collect_evidence_for_run()
    await orch._generate_draft("mock", mock_provider)
    request = capture.call_args.args[0]
    delivered = "\n".join(message.content for message in request.messages)
    assert "INTENDED_REPOSITORY_MARKER" in delivered
    assert "HARMLESS_OUTSIDE_MARKER" not in delivered


@pytest.mark.parametrize("replacement", ["file", "parent"])
def test_reader_rechecks_paths_after_discovery(tmp_path, replacement):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    source = repo / "src" / "auth.py"
    source.write_text("token original\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "auth.py").write_text("token HARMLESS_OUTSIDE_MARKER\n")
    candidates = _candidate_files(repo)
    assert candidates == [source]
    if replacement == "file":
        source.unlink()
        source.symlink_to(outside / "auth.py")
    else:
        (repo / "src").rename(repo / "original")
        (repo / "src").symlink_to(outside, target_is_directory=True)
    assert _search_files(candidates, ["token"], repo, limit=10) == []


def test_direct_outside_candidate_is_not_read(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "auth.py"
    outside.write_text("token HARMLESS_OUTSIDE_MARKER\n")
    assert _search_files([outside], ["token"], repo, limit=10) == []


def test_outside_alias_to_inside_file_does_not_abort_search(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "auth.py"
    source.write_text("token safe\n")
    alias = tmp_path / "alias.py"
    alias.symlink_to(source)
    assert _search_files([alias], ["token"], repo, limit=10) == [f"{alias}:1: token safe"]


def test_normal_nested_files_keep_source_anchors(tmp_path):
    source = tmp_path / "src" / "nested" / "auth.py"
    source.parent.mkdir(parents=True)
    source.write_text("# heading\ntoken INTENDED_REPOSITORY_MARKER\n")
    assert _search_files(_candidate_files(tmp_path), ["token"], tmp_path, limit=10) == [
        "src/nested/auth.py:2: token INTENDED_REPOSITORY_MARKER"
    ]


def test_broken_and_looping_links_are_skipped(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "missing.py").symlink_to(tmp_path / "absent")
    loop = tmp_path / "src" / "loop.py"
    loop.symlink_to(loop)
    assert _search_files(_candidate_files(tmp_path), ["token"], tmp_path, limit=10) == []


def test_in_repo_file_link_and_selected_root_link_keep_anchors(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    target = repo / "local.txt"
    target.write_text("token INTENDED_REPOSITORY_MARKER\n")
    (repo / "src" / "auth.py").symlink_to(target)
    alias = tmp_path / "selected-repo"
    alias.symlink_to(repo, target_is_directory=True)
    assert _search_files(_candidate_files(alias), ["token"], alias, limit=10) == [
        "src/auth.py:1: token INTENDED_REPOSITORY_MARKER"
    ]


@pytest.mark.parametrize("replacement", ["file", "parent"])
def test_symlink_swap_between_resolution_and_open_is_rejected(tmp_path, monkeypatch, replacement):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    source = repo / "src" / "auth.py"
    source.write_text("token original\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "auth.py").write_text("token HARMLESS_OUTSIDE_MARKER\n")
    original_resolve = Path.resolve
    swapped = False

    def resolve_and_swap(path, *args, **kwargs):
        nonlocal swapped
        resolved = original_resolve(path, *args, **kwargs)
        if path == source and not swapped:
            swapped = True
            if replacement == "file":
                source.unlink()
                source.symlink_to(outside / "auth.py")
            else:
                (repo / "src").rename(repo / "original")
                (repo / "src").symlink_to(outside, target_is_directory=True)
        return resolved

    monkeypatch.setattr(Path, "resolve", resolve_and_swap)
    assert _search_files([source], ["token"], repo, limit=10) == []
    assert swapped


def test_unsupported_safe_open_is_visible_and_does_not_read(tmp_path, monkeypatch, caplog):
    source = tmp_path / "auth.py"
    source.write_text("token MUST_NOT_BE_READ\n")
    monkeypatch.setattr(os, "supports_dir_fd", set())
    assert _search_files([source], ["token"], tmp_path, limit=10) == []
    assert "automatic evidence" in caplog.text.lower()
    assert "unavailable" in caplog.text.lower()


def test_non_regular_files_are_not_opened_for_content(tmp_path):
    source = tmp_path / "auth.py"
    source.mkdir()
    assert _search_files([source], ["token"], tmp_path, limit=10) == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO fixture requires POSIX")
def test_fifo_does_not_block_evidence_read(tmp_path):
    source = tmp_path / "auth.py"
    os.mkfifo(source)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "from llm_council.engine.evidence import _search_files; "
            "assert _search_files([Path(sys.argv[1])], ['token'], Path(sys.argv[2]), 10) == []",
            str(source),
            str(tmp_path),
        ],
        timeout=3,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("read_fails", [False, True])
def test_safe_reader_closes_every_open_descriptor(tmp_path, monkeypatch, read_fails):
    source = tmp_path / "src" / "nested" / "auth.py"
    source.parent.mkdir(parents=True)
    source.write_text("token safe\n")
    original_open = os.open
    opened = []

    def tracked_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, tracked_open})
    if read_fails:

        def fail_read(*args, **kwargs):
            raise OSError("synthetic read error")

        monkeypatch.setattr(os, "fdopen", fail_read)
    hits = _search_files([source], ["token"], tmp_path, limit=1)
    assert hits == ([] if read_fails else ["src/nested/auth.py:1: token safe"])
    assert opened
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)
