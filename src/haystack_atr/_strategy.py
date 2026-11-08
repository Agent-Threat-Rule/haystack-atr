"""A confirmation strategy that decides tool calls with ATR instead of asking a person."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.hooks.human_in_the_loop.dataclasses import ToolExecutionDecision

from ._scan import Finding, blocking, check_lanes, describe, evaluate, load_engine, tool_call_event

logger = logging.getLogger(__name__)


class ATRToolCallStrategy:
    """
    Check each tool call against Agent Threat Rules before it runs.

    Register it with Haystack's `ConfirmationHook` at the `before_tool` hook point:

    ```python
    from haystack.hooks.human_in_the_loop import ConfirmationHook
    from haystack_atr import ATRToolCallStrategy

    agent = Agent(
        chat_generator=...,
        tools=[...],
        hooks={"before_tool": [ConfirmationHook({"*": ATRToolCallStrategy()})]},
    )
    ```

    A call is rejected when a high or critical match comes from a rule in `block_lane`. The model then receives
    the rejection as the tool result, naming the rules that matched, and the tool does not run. Every other call
    runs with its parameters unchanged.

    :param lane: Rules that are evaluated: "enforce" (stable rules), "alert" (stable and test) or "hunt" (all).
    :param block_lane: Rules that can reject a call; must not be wider than `lane`. `None` only records findings.
    :param on_finding: Called with the findings of each call that had any. Not serialized.
    """

    def __init__(
        self,
        lane: str = "alert",
        block_lane: str | None = "enforce",
        on_finding: Callable[[list[Finding]], None] | None = None,
    ) -> None:
        check_lanes(lane, block_lane)
        self.lane = lane
        self.block_lane = block_lane
        self.on_finding = on_finding
        self._engine = load_engine(lane)

    def run(
        self,
        *,
        tool_name: str,
        tool_description: str,  # noqa: ARG002
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> ToolExecutionDecision:
        findings = evaluate(self._engine, "tool_call", tool_call_event(tool_name, tool_params))
        if findings:
            logger.warning("ATR findings on tool call %s: %s", tool_name, describe(findings))
            if self.on_finding is not None:
                self.on_finding(findings)
        blocked = blocking(findings, self.block_lane)
        if blocked:
            return ToolExecutionDecision(
                tool_name=tool_name,
                execute=False,
                tool_call_id=tool_call_id,
                feedback=f"Tool call blocked by Agent Threat Rules: {describe(blocked)}.",
            )
        return ToolExecutionDecision(
            tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
        )

    async def run_async(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        return self.run(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_params=tool_params,
            tool_call_id=tool_call_id,
            confirmation_strategy_context=confirmation_strategy_context,
        )

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, lane=self.lane, block_lane=self.block_lane)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ATRToolCallStrategy:
        return default_from_dict(cls, data)
