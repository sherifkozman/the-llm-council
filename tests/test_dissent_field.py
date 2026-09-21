"""The decision-producing schemas must be able to record a split verdict.

Measured 2026-09-20 on this machine's ledger: 0 of 10 schemas had any field for
disagreement, and all 812 logged council decisions carried none. Run `e349c452` is the
worked example - two seats said "approve with changes", one said "block", the synthesis
recorded `request_changes`, and the decision log shows only the verdict. A reader cannot
tell a 2-1 split from a unanimous call, which is exactly backwards: the dissent is the
part a council exists to produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_council.providers.cli.codex import _prepare_schema_for_codex

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src" / "llm_council" / "schemas"

# critic (458 runs), planner (157), planner --mode assess. `drafter`'s schemas are
# deliberately excluded: they produce artifacts rather than verdicts, and they are the
# heavy ones in the truncation trap.
DECISION_SCHEMAS = ("reviewer.json", "planner.json", "assessor.json")
ARTIFACT_SCHEMAS = ("implementer.json", "architect.json", "test-designer.json")


def load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", DECISION_SCHEMAS)
def test_a_decision_schema_can_record_dissent(name: str) -> None:
    schema = load(name)
    dissent = schema["properties"]["dissent"]

    assert dissent["type"] == "array"
    entry = dissent["items"]["properties"]
    assert {"seat", "position"} <= set(entry), "a dissent needs who, and what they said"
    assert entry["position"]["description"], "the position must be quoted, not paraphrased"
    # Both are REQUIRED per entry, not merely available. An entry without a position is
    # not a record of dissent, and the digest drops it -- the dissent would vanish exactly
    # where it matters, which is the failure this whole field exists to prevent.
    assert {"seat", "position"} <= set(dissent["items"]["required"])


@pytest.mark.parametrize("name", DECISION_SCHEMAS)
def test_dissent_is_never_required(name: str) -> None:
    """The guard that keeps this cheap.

    `_prepare_schema_for_codex` rewrites `required = list(properties)`, so for the codex
    seat every property is mandatory anyway. Adding `dissent` to the schema's own
    `required` list would make it mandatory for *every* seat, on every run, including the
    ones with nothing to report - more output from the seat that sits in the documented
    4000-token truncation trap, for no gain.
    """
    assert "dissent" not in (load(name).get("required") or [])


@pytest.mark.parametrize("name", DECISION_SCHEMAS)
def test_dissent_survives_the_codex_strict_rewrite(name: str) -> None:
    """Codex gets the schema through a strict-mode transform; the field must come out intact."""
    prepared = _prepare_schema_for_codex(load(name))

    assert "dissent" in prepared["properties"]
    assert prepared["properties"]["dissent"]["type"] == "array"


@pytest.mark.parametrize("name", ARTIFACT_SCHEMAS)
def test_artifact_schemas_were_left_alone(name: str) -> None:
    """Scope guard: only the decision schemas were meant to change."""
    assert "dissent" not in load(name)["properties"]


def test_the_synthesizer_is_told_to_fill_it_only_where_the_field_exists() -> None:
    """An optional field nobody is asked for stays empty for ever - but asking always is worse.

    Seven of the ten schemas have no `dissent` property and set `additionalProperties:
    false`, so a blanket instruction would invite a synthesis that fails validation and
    burns a retry. The ask must therefore be guarded by the schema itself.
    """
    source = (SCHEMA_DIR.parents[0] / "engine" / "orchestrator.py").read_text(encoding="utf-8")
    prompt_start = source.index("You are the synthesizer.")
    block = source[prompt_start : prompt_start + 1200]

    assert "do not average them away" in block, "the instruction is gone"
    guard = 'if isinstance(schema, dict) and "dissent" in (schema.get("properties") or {}):'
    assert guard in block, "the instruction must be guarded by the schema having the field"
    assert block.index(guard) < block.index("do not average them away"), "guard must come first"
