"""A synthesis timeout is not evidence that native JSON schemas are unsupported."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig
from llm_council.providers.base import GenerateResponse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("deadline exceeded"),
        TimeoutError(),
        TimeoutError("operation failed"),
        RuntimeError("wrapper"),
        type("GatewayTimeout", (Exception,), {"code": 504})("gateway failure"),
    ],
)
async def test_timeout_moves_to_next_synthesis_provider_without_inline_retry(error):
    if isinstance(error, RuntimeError):
        error.__cause__ = TimeoutError()
    orch = Orchestrator([], OrchestratorConfig(enable_artifacts=False))
    orch._task = "synthetic"
    orch._subagent_config = {}
    orch._schema = {"type": "object"}
    first = MagicMock()
    second = MagicMock()
    first.supports = AsyncMock(return_value=True)
    second.supports = AsyncMock(return_value=True)
    orch._candidate_providers_for_phase = AsyncMock(
        return_value=[("vertex-ai", first), ("openrouter", second)]
    )
    orch._call_provider = AsyncMock(side_effect=[error, GenerateResponse(text="{}")])
    result, attempts = await orch._run_synthesis({"first": "synthetic draft"}, "synthetic critique")
    assert result.ok is True
    assert attempts == 2
    assert [call.args[0] for call in orch._call_provider.await_args_list] == [
        "vertex-ai",
        "openrouter",
    ]
    assert orch._call_provider.await_args_list[1].args[2].structured_output is not None
