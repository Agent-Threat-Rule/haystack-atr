"""Turn Haystack messages and tool calls into ATR events and evaluate them."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pyatr import AgentEvent, ATREngine, ATRMatch, lane_allows

LANES = ("enforce", "alert", "hunt")
BLOCKING_SEVERITIES = frozenset({"critical", "high"})


@dataclass(frozen=True)
class Finding:
    """One rule match, with the surface it was found on (user, tool_call or tool_result)."""

    rule_id: str
    title: str
    severity: str
    maturity: str
    surface: str
    match: ATRMatch = field(repr=False, compare=False)

    @classmethod
    def from_match(cls, match: ATRMatch, surface: str) -> Finding:
        return cls(
            rule_id=match.rule_id,
            title=match.title,
            severity=str(match.severity).lower(),
            maturity=str(match.maturity),
            surface=surface,
            match=match,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "severity": self.severity,
            "maturity": self.maturity,
            "surface": self.surface,
        }


def check_lanes(lane: str, block_lane: str | None) -> None:
    """Reject unknown lanes, and a block lane wider than the lane that is evaluated."""
    if lane not in LANES:
        raise ValueError(f"lane must be one of {LANES}, got {lane!r}")
    if block_lane is None:
        return
    if block_lane not in LANES:
        raise ValueError(f"block_lane must be one of {LANES} or None, got {block_lane!r}")
    if LANES.index(block_lane) > LANES.index(lane):
        raise ValueError(f"block_lane {block_lane!r} is wider than lane {lane!r}")


def load_engine(lane: str) -> ATREngine:
    """An engine with the rules bundled in pyatr, limited to *lane*."""
    engine = ATREngine(lane=lane)
    if not engine.load_bundled_rules():
        raise RuntimeError("pyatr loaded 0 rules; check the pyatr install")
    return engine


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def tool_call_event(tool_name: str, arguments: Mapping[str, Any]) -> AgentEvent:
    """A tool call as ATR reads it: the tool name plus its arguments as JSON."""
    args = _json(dict(arguments))
    return AgentEvent(
        content=args,
        event_type="tool_call",
        fields={"tool_name": tool_name, "tool_args": args},
    )


def tool_result_event(tool_name: str, result: Any) -> AgentEvent:
    text = result if isinstance(result, str) else _json(result)
    return AgentEvent(
        content=text,
        event_type="tool_response",
        fields={"tool_name": tool_name, "tool_response": text},
    )


def user_event(text: str) -> AgentEvent:
    return AgentEvent(content=text, event_type="llm_input")


def evaluate(engine: ATREngine, surface: str, event: AgentEvent) -> list[Finding]:
    return [Finding.from_match(match, surface) for match in engine.evaluate(event)]


def blocking(findings: Iterable[Finding], block_lane: str | None) -> list[Finding]:
    """High or critical findings whose rule is in *block_lane*; none when blocking is off."""
    if block_lane is None:
        return []
    return [
        finding
        for finding in findings
        if finding.severity in BLOCKING_SEVERITIES and lane_allows(finding.match, block_lane)
    ]


def describe(findings: Iterable[Finding]) -> str:
    return ", ".join(f"{f.rule_id} ({f.severity})" for f in findings)
