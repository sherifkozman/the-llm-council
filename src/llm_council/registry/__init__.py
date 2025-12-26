"""
Tool and Agent Registry.

Provides declarative tool registration and thin base agent abstraction.
"""

from llm_council.registry.base_agent import BaseAgent
from llm_council.registry.tool_registry import Tool, ToolRegistry

__all__ = ["Tool", "ToolRegistry", "BaseAgent"]
