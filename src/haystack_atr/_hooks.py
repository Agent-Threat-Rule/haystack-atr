"""An after_tool hook that keeps tool results carrying an attack away from the model."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole

from ._scan import Finding, blocking, check_lanes, describe, evaluate, load_engine, tool_result_event

logger = logging.getLogger(__name__)


def _result_text(message: ChatMessage) -> Any:
    result = message.tool_call_result.result
    if isinstance(result, str):
        return result
    texts = [part.text for part in result if getattr(part, "text", None)]
    return "\n".join(texts) if texts else ""


class ATRToolResultHook:
    """
    Check the results of the tools that just ran, before the model reads them.

    Register it at the `after_tool` hook point:

    ```python
    from haystack_atr import ATRToolResultHook

    agent = Agent(chat_generator=..., tools=[...], hooks={"after_tool": [ATRToolResultHook()]})
    ```

    When a high or critical match comes from a rule in `block_lane`, the result is replaced with an error
    result naming the rules that matched, so the model never reads the original text. Other results are left
    as they are.

    :param lane: Rules that are evaluated: "enforce" (stable rules), "alert" (stable and test) or "hunt" (all).
    :param block_lane: Rules that can withhold a result; must not be wider than `lane`. `None` only records findings.
    :param on_finding: Called with the findings of each result that had any. Not serialized.
    """

    allowed_hook_points = ("after_tool",)

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

    def _check(self, message: ChatMessage) -> ChatMessage:
        origin = message.tool_call_result.origin
        findings = evaluate(self._engine, "tool_result", tool_result_event(origin.tool_name, _result_text(message)))
        if not findings:
            return message
        logger.warning("ATR findings on result of %s: %s", origin.tool_name, describe(findings))
        if self.on_finding is not None:
            self.on_finding(findings)
        blocked = blocking(findings, self.block_lane)
        if not blocked:
            return message
        return ChatMessage.from_tool(
            f"Tool result withheld by Agent Threat Rules: {describe(blocked)}.",
            origin=origin,
            error=True,
            meta=message.meta,
        )

    def run(self, state: State) -> None:
        messages = list(state.data.get("messages") or [])
        start = len(messages)
        while start > 0 and messages[start - 1].is_from(ChatRole.TOOL):
            start -= 1
        checked = [self._check(m) if m.tool_call_result else m for m in messages[start:]]
        if any(new is not old for new, old in zip(checked, messages[start:])):
            state.set("messages", messages[:start] + checked, handler_override=replace_values)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, lane=self.lane, block_lane=self.block_lane)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ATRToolResultHook:
        return default_from_dict(cls, data)
