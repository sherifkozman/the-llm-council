"""The suite-wide home isolation itself (conftest._hermetic_home), pinned so it cannot regress.

These tests deliberately request NO home fixture: a test that asks for ``isolated_home`` would
pass even if the autouse flag were dropped, because asking instantiates the fixture anyway.
What must hold is that a test which never thinks about its home still gets its own.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_council.providers import concurrency
from llm_council.storage import artifacts
from llm_council.storage.artifacts import ArtifactStore, get_store, reset_store


def test_a_test_that_never_asks_still_gets_its_own_home(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    home = Path.home()

    assert home.is_relative_to(tmp_path_factory.getbasetemp())
    assert list(home.iterdir()) == [], "each test starts in an empty home"


def test_the_default_store_lives_in_the_tests_own_home(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The store a test gets without passing paths must never be the real ledger."""
    store = get_store()

    assert store.db_path.is_relative_to(Path.home())
    assert store.artifact_dir.is_relative_to(Path.home())
    assert Path.home().is_relative_to(tmp_path_factory.getbasetemp())


@pytest.fixture(scope="module")
def _a_store_left_behind(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A cached store planted BEFORE any function-scoped fixture of the test runs."""
    root = tmp_path_factory.mktemp("left-behind")
    artifacts._default_store = ArtifactStore(
        artifact_dir=root / "artifacts", db_path=root / "ledger.db"
    )
    yield root
    reset_store()


def test_a_store_cached_by_an_earlier_test_is_not_handed_on(
    _a_store_left_behind: Path,
) -> None:
    """get_store() caches a singleton; each test must start without the previous one's.

    Deterministic: the stale store exists before this test's fixtures run, whatever
    the test order or worker.
    """
    assert not get_store().db_path.is_relative_to(_a_store_left_behind)


def test_provider_locks_are_the_tests_own(tmp_path_factory: pytest.TempPathFactory) -> None:
    """A developer's shell cannot move a test's locks or switch them off."""
    assert "LLM_COUNCIL_DISABLE_PROVIDER_LOCKS" not in os.environ
    assert concurrency._locks_enabled() == (concurrency.fcntl is not None)
    assert concurrency._lock_root().is_relative_to(Path.home())
    assert Path.home().is_relative_to(tmp_path_factory.getbasetemp())
