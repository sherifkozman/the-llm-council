"""Tests for subagent and schema loading."""

from __future__ import annotations

import pytest

from llm_council.schemas import list_schemas, load_schema
from llm_council.subagents import list_subagents, load_subagent


class TestSubagentLoading:
    """Tests for subagent config loading."""

    def test_list_subagents(self):
        """Test listing available subagents."""
        subagents = list_subagents()
        assert isinstance(subagents, list)
        assert len(subagents) >= 10

        # Check expected subagents are present
        expected = [
            "router",
            "planner",
            "assessor",
            "researcher",
            "architect",
            "implementer",
            "reviewer",
            "test-designer",
            "shipper",
            "red-team",
        ]
        for name in expected:
            assert name in subagents, f"Expected subagent '{name}' not found"

    def test_load_router_subagent(self):
        """Test loading router subagent config."""
        config = load_subagent("router")
        assert config is not None
        assert isinstance(config, dict)
        assert "name" in config or "prompts" in config or "schema" in config

    def test_load_implementer_subagent(self):
        """Test loading implementer subagent config."""
        config = load_subagent("implementer")
        assert config is not None
        assert isinstance(config, dict)

    def test_load_reviewer_subagent(self):
        """Test loading reviewer subagent config."""
        config = load_subagent("reviewer")
        assert config is not None
        assert isinstance(config, dict)

    def test_load_planner_subagent(self):
        """Test loading planner subagent config."""
        config = load_subagent("planner")
        assert config is not None

    def test_load_nonexistent_subagent(self):
        """Test loading nonexistent subagent raises error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_subagent("nonexistent-subagent")

    def test_subagent_has_prompts_section(self):
        """Test subagent configs have prompts section."""
        config = load_subagent("router")
        # Most subagents should have a prompts section
        assert "prompts" in config or "system" in str(config)


class TestSchemaLoading:
    """Tests for JSON schema loading."""

    def test_list_schemas(self):
        """Test listing available schemas."""
        schemas = list_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) >= 10

    def test_load_router_schema(self):
        """Test loading router schema."""
        schema = load_schema("router")
        assert schema is not None
        assert isinstance(schema, dict)
        assert "$schema" in schema or "type" in schema

    def test_load_implementer_schema(self):
        """Test loading implementer schema."""
        schema = load_schema("implementer")
        assert schema is not None
        assert isinstance(schema, dict)
        # Should be a JSON schema
        assert "type" in schema or "properties" in schema

    def test_load_reviewer_schema(self):
        """Test loading reviewer schema."""
        schema = load_schema("reviewer")
        assert schema is not None
        assert "type" in schema or "properties" in schema

    def test_load_nonexistent_schema(self):
        """Test loading nonexistent schema raises error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_schema("nonexistent-schema")

    def test_schema_has_required_fields(self):
        """Test schemas have required fields definition."""
        schema = load_schema("implementer")
        # JSON schemas should have type and properties
        assert "type" in schema
        if schema["type"] == "object":
            assert "properties" in schema or "required" in schema


class TestSubagentSchemaConsistency:
    """Tests for consistency between subagents and schemas."""

    def test_subagents_have_matching_schemas(self):
        """Test that subagents reference existing schemas."""
        subagents = list_subagents()
        schemas = list_schemas()

        for subagent_name in subagents:
            config = load_subagent(subagent_name)
            if "schema" in config:
                schema_name = config["schema"]
                # Schema should exist or be loadable
                assert schema_name in schemas or load_schema(schema_name) is not None

    def test_all_schemas_are_valid_json_schema(self):
        """Test all schemas are valid JSON schema format."""
        schemas = list_schemas()
        for schema_name in schemas:
            schema = load_schema(schema_name)
            # Basic JSON schema validation
            assert isinstance(schema, dict)
            # Should have type or be a reference
            assert "type" in schema or "$ref" in schema or "oneOf" in schema
