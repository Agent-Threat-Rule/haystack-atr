"""A pipeline component that scans chat messages with ATR."""

from __future__ import annotations

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole

from ._scan import Finding, blocking, check_lanes, evaluate, load_engine, tool_result_event, user_event


@component
class ATRMessageScanner:
    """
    Scan user messages and tool results with Agent Threat Rules.

    System and assistant messages are not scanned: they come from the application or the model. Route on
    `blocked` (for example with a `ConditionalRouter`) to stop a pipeline before an attack reaches a model.

    :param lane: Rules that are evaluated: "enforce" (stable rules), "alert" (stable and test) or "hunt" (all).
    :param block_lane: Rules whose high or critical matches set `blocked`; must not be wider than `lane`.
    """

    def __init__(self, lane: str = "alert", block_lane: str | None = "enforce") -> None:
        check_lanes(lane, block_lane)
        self.lane = lane
        self.block_lane = block_lane
        self._engine = load_engine(lane)

    def _findings(self, message: ChatMessage) -> list[Finding]:
        if message.is_from(ChatRole.USER) and message.text:
            return evaluate(self._engine, "user", user_event(message.text))
        found: list[Finding] = []
        for result in message.tool_call_results:
            text = result.result if isinstance(result.result, str) else "\n".join(
                part.text for part in result.result if getattr(part, "text", None)
            )
            found.extend(evaluate(self._engine, "tool_result", tool_result_event(result.origin.tool_name, text)))
        return found

    @component.output_types(messages=list[ChatMessage], findings=list[dict[str, str]], blocked=bool)
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        findings = [finding for message in messages for finding in self._findings(message)]
        return {
            "messages": messages,
            "findings": [finding.to_dict() for finding in findings],
            "blocked": bool(blocking(findings, self.block_lane)),
        }

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, lane=self.lane, block_lane=self.block_lane)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ATRMessageScanner:
        return default_from_dict(cls, data)
