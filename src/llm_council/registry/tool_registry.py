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

DEFAULT_CONFIG_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "config" / "tool_registry.yaml",
    Path.cwd() / "config" / "tool_registry.yaml",
)


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


@dataclass
class CapabilityPack:
    """Named bundle of tools and evidence expectations."""

    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    evidence_requirements: list[str] = field(default_factory=list)


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
        self._capability_packs: dict[str, CapabilityPack] = {}
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
            packs_data = data.get("capability_packs", {})

            for tool_name, tool_config in tools_data.items():
                self.register_tool(self._parse_tool(tool_name, tool_config))
            for pack_name, pack_config in packs_data.items():
                self.register_capability_pack(self._parse_capability_pack(pack_name, pack_config))

            self._loaded = True
            logger.info(
                "Loaded %d tools and %d capability packs from %s",
                len(self._tools),
                len(self._capability_packs),
                config_path,
            )

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
                parameters.append(ToolParameter(name=param_name, type=str(param_config)))

        return Tool(
            name=config.get("name", name),
            description=config.get("description", ""),
            parameters=parameters,
            roles=config.get("roles", []),
            handler=config.get("handler"),
        )

    def _parse_capability_pack(self, name: str, config: dict[str, Any]) -> CapabilityPack:
        """Parse capability pack configuration into CapabilityPack object."""

        return CapabilityPack(
            name=config.get("name", name),
            description=config.get("description", ""),
            tools=list(config.get("tools", [])),
            roles=list(config.get("roles", [])),
            evidence_requirements=list(config.get("evidence_requirements", [])),
        )

    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_capability_pack(self, pack: CapabilityPack) -> None:
        """Register a capability pack in the registry."""

        self._capability_packs[pack.name] = pack
        logger.debug("Registered capability pack: %s", pack.name)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_capability_pack(self, name: str) -> CapabilityPack | None:
        """Get a capability pack by name."""

        return self._capability_packs.get(name)

    def get_tools_for_role(self, role: str) -> list[Tool]:
        """Get all tools available to a specific role."""
        return [t for t in self._tools.values() if role in t.roles or not t.roles]

    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def list_capability_packs(self) -> list[str]:
        """Return the registered capability pack names."""

        return sorted(self._capability_packs.keys())

    def resolve_capability_tools(
        self, pack_names: list[str], role: str | None = None
    ) -> list[Tool]:
        """Resolve tools referenced by capability packs."""

        resolved: list[Tool] = []
        seen: set[str] = set()

        for pack_name in pack_names:
            pack = self.get_capability_pack(pack_name)
            if pack is None:
                logger.debug("Capability pack not found: %s", pack_name)
                continue
            if role and pack.roles and role not in pack.roles:
                continue

            for tool_name in pack.tools:
                tool = self.get_tool(tool_name)
                if tool is None:
                    logger.debug(
                        "Tool %s referenced by capability pack %s is not registered",
                        tool_name,
                        pack_name,
                    )
                    continue
                if role and tool.roles and role not in tool.roles:
                    continue
                if tool.name in seen:
                    continue
                resolved.append(tool)
                seen.add(tool.name)

        return resolved

    def ensure_loaded(self, config_path: Path | None = None) -> None:
        """Load registry data on first use if it has not been loaded yet."""

        if self._loaded:
            return

        if config_path is not None:
            self.load_from_yaml(config_path)
            return

        for candidate in DEFAULT_CONFIG_CANDIDATES:
            if candidate.exists():
                self.load_from_yaml(candidate)
                return

        logger.debug("No default tool registry config found; registry remains empty.")

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
