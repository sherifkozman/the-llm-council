"""Fail-closed checks for independently maintained Codex compatibility records."""

import hashlib
import importlib
import json
from datetime import datetime, timezone

import pytest


def validator():
    module = importlib.import_module("llm_council.providers.cli.codex")

    def match(payload, identity, now):
        fields = ("os", "arch", "model", "auth", "endpoint", "reasoning", "output")
        criteria = {key: identity[key] for key in fields}
        catalog = (
            CATALOG
            if identity["catalog_sha256"] == hashlib.sha256(CATALOG).hexdigest()
            else b"different"
        )
        record = module._match_snapshot_identity(
            payload,
            criteria,
            identity["files"]["bin/codex"],
            identity["version"],
            catalog,
            module._inventory_sha256(identity["files"], identity["executables"]),
            now,
        )
        return record["profile"] if record is not None else None

    return match


CATALOG = b"synthetic-catalog"

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
IDENTITY = {
    "version": "codex-cli 0.165.0",
    "os": "darwin",
    "arch": "arm64",
    "model": "synthetic-model",
    "auth": "apikey",
    "endpoint": "openai",
    "reasoning": "high",
    "output": "text",
    "catalog_sha256": hashlib.sha256(CATALOG).hexdigest(),
    "executables": ["bin/codex", "codex-path/rg"],
    "files": {"bin/codex": "a" * 64, "codex-path/rg": "c" * 64},
}


def document():
    return {
        "schema": 1,
        "issued_at": "2026-10-29T00:00:00Z",
        "expires_at": "2026-11-29T00:00:00Z",
        "records": [{**IDENTITY, "profile": "generation-v1"}],
    }


def test_record_accepts_only_matching_build_and_runtime():
    assert validator()(json.dumps(document()).encode(), IDENTITY, NOW) == "generation-v1"


@pytest.mark.parametrize(
    "field,changed",
    [
        ("version", "codex-cli 0.166.0"),
        ("os", "linux"),
        ("arch", "x86_64"),
        ("model", "another-model"),
        ("auth", "chatgpt"),
        ("endpoint", "other"),
        ("reasoning", "medium"),
        ("output", "json_schema"),
        ("catalog_sha256", "d" * 64),
        ("executables", ["bin/codex"]),
        ("files", {"bin/codex": "e" * 64, "codex-path/rg": "c" * 64}),
    ],
)
def test_record_rejects_changed_build_or_context(field, changed):
    assert validator()(json.dumps(document()).encode(), {**IDENTITY, field: changed}, NOW) is None


@pytest.mark.parametrize(
    "mutation", ["expired", "future", "long_lived", "unknown_profile", "duplicate"]
)
def test_record_rejects_untrustworthy_record_shape(mutation):
    value = document()
    if mutation == "expired":
        value["expires_at"] = "2026-10-29T12:00:00Z"
    if mutation == "future":
        value["issued_at"] = "2026-10-30T00:00:00Z"
    if mutation == "long_lived":
        value["expires_at"] = "2027-10-29T00:00:00Z"
    if mutation == "unknown_profile":
        value["records"][0]["profile"] = "future-profile"
    if mutation == "duplicate":
        value["records"].append(value["records"][0].copy())
    assert validator()(json.dumps(value).encode(), IDENTITY, NOW) is None


def test_record_rejects_duplicate_json_keys_and_oversized_content():
    assert validator()(b'{"schema":1,"schema":1}', IDENTITY, NOW) is None
    assert validator()(b" " * (256 * 1024 + 1), IDENTITY, NOW) is None


def test_record_rejects_nesting_beyond_python_recursion_limit():
    import sys

    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(500)
        assert validator()(b"[" * 1000 + b"0" + b"]" * 1000, IDENTITY, NOW) is None
    finally:
        sys.setrecursionlimit(original_limit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("os", "unsupported"),
        ("arch", "unsupported"),
        ("auth", "unsupported"),
        ("endpoint", "unsupported"),
        ("reasoning", "unsupported"),
        ("output", "unsupported"),
    ],
)
def test_record_rejects_unsupported_context_even_when_identity_matches(field, value):
    identity = {**IDENTITY, field: value}
    record = document()
    record["records"][0][field] = value
    assert validator()(json.dumps(record).encode(), identity, NOW) is None


def test_record_rejects_unproven_auth_endpoint_pair():
    identity = {**IDENTITY, "endpoint": "chatgpt"}
    record = document()
    record["records"][0]["endpoint"] = "chatgpt"
    assert validator()(json.dumps(record).encode(), identity, NOW) is None


def test_record_rejects_null_byte_in_file_path():
    files = {"bin/codex": "a" * 64, "bad\x00file": "c" * 64}
    identity = {**IDENTITY, "files": files}
    record = document()
    record["records"][0]["files"] = files
    assert validator()(json.dumps(record).encode(), identity, NOW) is None


@pytest.mark.parametrize(
    "executables", [[], ["bin/codex", "bin/codex"], ["bin/codex", "not-in-files"]]
)
def test_record_rejects_invalid_executable_inventory(executables):
    identity = {**IDENTITY, "executables": executables}
    record = document()
    record["records"][0]["executables"] = executables
    assert validator()(json.dumps(record).encode(), identity, NOW) is None


def test_record_candidates_require_full_document_validation_before_selection():
    module = importlib.import_module("llm_council.providers.cli.codex")
    select = getattr(module, "_compatibility_candidates", None)
    assert callable(select), "compatibility candidate selection is not implemented"
    value = document()
    criteria = {
        key: IDENTITY[key]
        for key in ("os", "arch", "model", "auth", "endpoint", "reasoning", "output")
    }
    assert select(json.dumps(value).encode(), criteria, NOW) == value["records"]
    assert select(json.dumps(value).encode(), {**criteria, "model": "other"}, NOW) == []
    value["records"].append({**value["records"][0], "profile": "unknown"})
    assert select(json.dumps(value).encode(), criteria, NOW) == []


def test_record_candidates_reject_expired_and_unrecognized_selection_context():
    module = importlib.import_module("llm_council.providers.cli.codex")
    select = getattr(module, "_compatibility_candidates", None)
    assert callable(select), "compatibility candidate selection is not implemented"
    criteria = {
        key: IDENTITY[key]
        for key in ("os", "arch", "model", "auth", "endpoint", "reasoning", "output")
    }
    expired = document()
    expired["expires_at"] = "2026-10-29T12:00:00Z"
    assert select(json.dumps(expired).encode(), criteria, NOW) == []
    assert select(json.dumps(document()).encode(), {**criteria, "unexpected": "x"}, NOW) == []


def test_snapshot_identity_requires_observed_version_catalog_and_fresh_record():
    import hashlib

    module = importlib.import_module("llm_council.providers.cli.codex")
    match = getattr(module, "_match_snapshot_identity", None)
    assert callable(match), "snapshot identity matching is not implemented"
    catalog = b'{"models":[]}'
    criteria = {
        key: IDENTITY[key]
        for key in ("os", "arch", "model", "auth", "endpoint", "reasoning", "output")
    }
    value = document()
    value["records"][0]["catalog_sha256"] = hashlib.sha256(catalog).hexdigest()
    payload = json.dumps(value).encode()
    inventory = module._inventory_sha256(IDENTITY["files"], IDENTITY["executables"])
    assert (
        match(payload, criteria, "a" * 64, "codex-cli 0.165.0", catalog, inventory, NOW)
        == value["records"][0]
    )
    assert match(payload, criteria, "0" * 64, "codex-cli 0.165.0", catalog, inventory, NOW) is None
    assert match(payload, criteria, "a" * 64, "codex-cli 0.165.1", catalog, inventory, NOW) is None
    assert match(payload, criteria, "a" * 64, "codex-cli 0.165.0", b"{}", inventory, NOW) is None
    assert (
        match(
            payload,
            {**criteria, "model": "other"},
            "a" * 64,
            "codex-cli 0.165.0",
            catalog,
            inventory,
            NOW,
        )
        is None
    )
    value["expires_at"] = "2026-10-29T12:00:00Z"
    assert (
        match(
            json.dumps(value).encode(),
            criteria,
            "a" * 64,
            "codex-cli 0.165.0",
            catalog,
            inventory,
            NOW,
        )
        is None
    )


def test_snapshot_identity_rejects_ambiguous_records():
    import hashlib

    module = importlib.import_module("llm_council.providers.cli.codex")
    match = getattr(module, "_match_snapshot_identity", None)
    assert callable(match), "snapshot identity matching is not implemented"
    catalog = b"{}"
    criteria = {
        key: IDENTITY[key]
        for key in ("os", "arch", "model", "auth", "endpoint", "reasoning", "output")
    }
    value = document()
    value["records"][0]["catalog_sha256"] = hashlib.sha256(catalog).hexdigest()
    value["records"].append(value["records"][0].copy())
    inventory = module._inventory_sha256(IDENTITY["files"], IDENTITY["executables"])
    assert (
        match(
            json.dumps(value).encode(),
            criteria,
            "a" * 64,
            "codex-cli 0.165.0",
            catalog,
            inventory,
            NOW,
        )
        is None
    )


def test_snapshot_identity_rejects_same_binary_with_changed_resource_inventory():
    import hashlib

    module = importlib.import_module("llm_council.providers.cli.codex")
    criteria = {
        key: IDENTITY[key]
        for key in ("os", "arch", "model", "auth", "endpoint", "reasoning", "output")
    }
    value = document()
    catalog = b"{}"
    value["records"][0]["catalog_sha256"] = hashlib.sha256(catalog).hexdigest()
    old_inventory = module._inventory_sha256(IDENTITY["files"], IDENTITY["executables"])
    value["records"][0]["files"] = {**IDENTITY["files"], "codex-path/rg": "d" * 64}
    assert (
        module._match_snapshot_identity(
            json.dumps(value).encode(),
            criteria,
            "a" * 64,
            "codex-cli 0.165.0",
            catalog,
            old_inventory,
            NOW,
        )
        is None
    )


def test_compact_records_reuse_one_build_inventory_for_exact_contexts():
    model_fields = {"model", "auth", "endpoint", "reasoning", "output"}
    build = {
        key: value for key, value in document()["records"][0].items() if key not in model_fields
    }
    first = {key: IDENTITY[key] for key in model_fields}
    second = {**first, "model": "another-model", "auth": "chatgpt", "endpoint": "chatgpt"}
    build["contexts"] = [first, second]
    value = {**document(), "schema": 2, "records": [build]}
    payload = json.dumps(value).encode()
    assert validator()(payload, IDENTITY, NOW) == "generation-v1"
    assert validator()(payload, {**IDENTITY, **second}, NOW) == "generation-v1"
    assert validator()(payload, {**IDENTITY, "model": "another-model"}, NOW) is None


@pytest.mark.parametrize("mutation", ["empty", "extra", "invalid_pair", "duplicate"])
def test_compact_records_reject_invalid_or_ambiguous_contexts(mutation):
    fields = {"model", "auth", "endpoint", "reasoning", "output"}
    build = {key: value for key, value in document()["records"][0].items() if key not in fields}
    context = {key: IDENTITY[key] for key in fields}
    build["contexts"] = [context]
    if mutation == "empty":
        build["contexts"] = []
    elif mutation == "extra":
        context["unexpected"] = "x"
    elif mutation == "invalid_pair":
        context["endpoint"] = "chatgpt"
    else:
        build["contexts"].append(context.copy())
    value = {**document(), "schema": 2, "records": [build]}
    assert validator()(json.dumps(value).encode(), IDENTITY, NOW) is None


def test_compact_record_bounds_expanded_contexts():
    module = importlib.import_module("llm_council.providers.cli.codex")
    fields = {"model", "auth", "endpoint", "reasoning", "output"}
    build = {key: value for key, value in document()["records"][0].items() if key not in fields}
    context = {key: IDENTITY[key] for key in fields}
    build["contexts"] = [context] * 1025
    value = {**document(), "schema": 2, "records": [build]}
    assert module._validated_compatibility_records(json.dumps(value).encode(), NOW) is None


@pytest.mark.asyncio
async def test_snapshot_freshness_rechecks_clock_after_record_validation(monkeypatch, tmp_path):
    import asyncio
    from datetime import timedelta

    module = importlib.import_module("llm_council.providers.cli.codex")
    value = document()
    value["expires_at"] = "2026-10-29T12:00:01Z"
    payload = json.dumps(value).encode()
    criteria = {
        key: IDENTITY[key]
        for key in ("os", "arch", "model", "auth", "endpoint", "reasoning", "output")
    }
    calls = 0

    class AdvancingDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            nonlocal calls
            calls += 1
            return NOW if calls == 1 else NOW + timedelta(seconds=2)

    monkeypatch.setattr(module, "datetime", AdvancingDatetime)
    snapshot = module._CodexSnapshot(
        tmp_path / "bin/codex",
        module._SnapshotCleanup(asyncio.get_running_loop()),
        IDENTITY["files"]["bin/codex"],
        module._inventory_sha256(IDENTITY["files"], IDENTITY["executables"]),
    )
    assert not module._snapshot_has_current_candidate(payload, criteria, snapshot)
    assert calls >= 2


def test_compact_record_checks_shared_inventory_once(monkeypatch):
    module = importlib.import_module("llm_council.providers.cli.codex")
    fields = {"model", "auth", "endpoint", "reasoning", "output"}
    build = {key: value for key, value in document()["records"][0].items() if key not in fields}
    context = {key: IDENTITY[key] for key in fields}
    build["contexts"] = [{**context, "model": f"model-{i}"} for i in range(100)]
    value = {**document(), "schema": 2, "records": [build]}
    actual = module.PurePosixPath
    checked = []

    def count_path(name):
        checked.append(name)
        return actual(name)

    monkeypatch.setattr(module, "PurePosixPath", count_path)
    assert module._validated_compatibility_records(json.dumps(value).encode(), NOW) is not None
    assert checked.count("bin/codex") == 1
    assert checked.count("codex-path/rg") == 1


@pytest.mark.asyncio
async def test_dispatch_freshness_rechecks_clock_after_identity_matching(monkeypatch, tmp_path):
    import asyncio
    import hashlib
    from datetime import timedelta

    module = importlib.import_module("llm_council.providers.cli.codex")
    catalog = b"{}"
    value = document()
    value["expires_at"] = "2026-10-29T12:00:01Z"
    value["records"][0]["catalog_sha256"] = hashlib.sha256(catalog).hexdigest()
    payload = json.dumps(value).encode()
    criteria = {
        key: IDENTITY[key]
        for key in ("os", "arch", "model", "auth", "endpoint", "reasoning", "output")
    }
    calls = 0

    class AdvancingDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            nonlocal calls
            calls += 1
            return NOW if calls == 1 else NOW + timedelta(seconds=2)

    monkeypatch.setattr(module, "datetime", AdvancingDatetime)
    snapshot = module._CodexSnapshot(
        tmp_path / "bin/codex",
        module._SnapshotCleanup(asyncio.get_running_loop()),
        IDENTITY["files"]["bin/codex"],
        module._inventory_sha256(IDENTITY["files"], IDENTITY["executables"]),
    )
    admission = module._CodexAdmission(
        snapshot, IDENTITY["version"], catalog, payload, criteria, value["records"][0]
    )
    assert not admission.is_current()
    assert calls >= 2
