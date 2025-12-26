"""
Declarative Tool Registry.

Provides schema-driven tool registration where each Council role declares
its tools with names, parameter schemas, and descriptions in YAML.

This reduces ad-hoc tool wiring and enables config-only tool addition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Parameter definition for a tool."""

    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    required: bool = True
    default: Any = None
    description: str = ""

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class Tool:
    """Tool definition with metadata and parameter schema."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)  # Which agent roles can use this tool
    handler: str | None = None  # Python path to handler function

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """
    Registry for declarative tool definitions.

    Tools are loaded from YAML configuration and can be filtered by role.

    Example config/tool_registry.yaml:
    ```yaml
    tools:
      web_search:
        name: web_search
        description: Search the web for information
        parameters:
          query:
            type: string
            required: true
          limit:
            type: integer
            default: 5
        roles: [drafter, researcher]
    ```
    """

    _instance: ToolRegistry | None = None

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._loaded = False

    @classmethod
    def get_instance(cls) -> ToolRegistry:
        """Get singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_from_yaml(self, config_path: Path) -> None:
        """Load tool definitions from YAML file."""
        if not config_path.exists():
            logger.warning(f"Tool registry config not found: {config_path}")
            return

        try:
            data = yaml.safe_load(config_path.read_text()) or {}
            tools_data = data.get("tools", {})

            for tool_name, tool_config in tools_data.items():
                self.register_tool(self._parse_tool(tool_name, tool_config))

            self._loaded = True
            logger.info(f"Loaded {len(self._tools)} tools from {config_path}")

        except (yaml.YAMLError, OSError) as e:
            logger.error(f"Failed to load tool registry: {e}")

    def _parse_tool(self, name: str, config: dict[str, Any]) -> Tool:
        """Parse tool configuration into Tool object."""
        parameters = []
        params_config = config.get("parameters", {})

        for param_name, param_config in params_config.items():
            if isinstance(param_config, dict):
                parameters.append(
                    ToolParameter(
                        name=param_name,
                        type=param_config.get("type", "string"),
                        required=param_config.get("required", True),
                        default=param_config.get("default"),
                        description=param_config.get("description", ""),
                    )
                )
            else:
                # Simple type shorthand: "query: string"
                parameters.append(
                    ToolParameter(name=param_name, type=str(param_config))
                )

        return Tool(
            name=config.get("name", name),
            description=config.get("description", ""),
            parameters=parameters,
            roles=config.get("roles", []),
            handler=config.get("handler"),
        )

    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools_for_role(self, role: str) -> list[Tool]:
        """Get all tools available to a specific role."""
        return [t for t in self._tools.values() if role in t.roles or not t.roles]

    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def to_openai_tools(self, role: str | None = None) -> list[dict[str, Any]]:
        """Convert tools to OpenAI format, optionally filtered by role."""
        tools = self.get_tools_for_role(role) if role else self.get_all_tools()
        return [t.to_openai_tool() for t in tools]

    def to_anthropic_tools(self, role: str | None = None) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format, optionally filtered by role."""
        tools = self.get_tools_for_role(role) if role else self.get_all_tools()
        return [t.to_anthropic_tool() for t in tools]


def get_tool_registry() -> ToolRegistry:
    """Get the singleton tool registry."""
    return ToolRegistry.get_instance()
