"""
Protocol definitions for LLM Council.

Includes Pydantic models for messages, capabilities, and JSON-lines stdio protocol.
"""

from llm_council.protocol.types import (
    CouncilConfig,
    CouncilRequest,
    CouncilResponse,
)

__all__ = ["CouncilRequest", "CouncilResponse", "CouncilConfig"]
