"""Reproduce the benign-content numbers in the README.

Usage: python scripts/measure_benign.py <path to an agent-threat-rules checkout> [sample size] [seed]

Samples texts from data/benign-corpus-extended/*.jsonl, checks each one as a tool result with the
rules bundled in the installed pyatr, and counts texts with a high or critical match per blocking lane.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from pyatr import AgentEvent, ATREngine

from haystack_atr._scan import blocking, evaluate


def corpus(repo: Path) -> list[str]:
    texts = []
    for path in sorted((repo / "data" / "benign-corpus-extended").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("content") or row.get("input") or ""
            if isinstance(text, str) and len(text) >= 20:
                texts.append(text)
    return texts


def main() -> None:
    repo = Path(sys.argv[1])
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261108
    texts = corpus(repo)
    sample = random.Random(seed).sample(texts, min(size, len(texts)))
    engine = ATREngine(lane="alert")
    engine.load_bundled_rules()
    blocked = {"enforce": 0, "alert": 0}
    for text in sample:
        event = AgentEvent(content=text, event_type="tool_response", fields={"tool_name": "fetch", "tool_response": text})
        findings = evaluate(engine, "tool_result", event)
        for lane in blocked:
            if blocking(findings, lane):
                blocked[lane] += 1
    print(json.dumps({"corpus_texts": len(texts), "sample": len(sample), "seed": seed, "blocked": blocked}))


if __name__ == "__main__":
    main()
