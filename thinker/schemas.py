from __future__ import annotations

import json
from typing import TypedDict

THINKING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "thesis": {
            "type": "string",
            "description": (
                "Your current thesis — a clear, specific, falsifiable claim. "
                "Write it so a smart non-specialist can understand it on first read. "
                "Use plain language; introduce technical terms only when absolutely needed. "
                "Avoid vague platitudes. If it sounds like a motivational poster, go deeper."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "Your chain of reasoning leading to this thesis. "
                "Be concrete: name assumptions, cite thought experiments, "
                "distinguish logical necessity from empirical likelihood. "
                "Explain in plain language — if a sentence requires a philosophy degree "
                "to parse, rewrite it."
            ),
        },
        "counter_arguments": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Steel-manned counter-arguments against your thesis. "
                "Each should be the strongest version an intelligent opponent would make. "
                "No strawmen."
            ),
        },
        "new_insights": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Non-obvious discoveries from this iteration. "
                "Hidden connections, surprising implications, overlooked distinctions. "
                "Skip anything a well-read undergraduate would already know."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": (
                "Your confidence in the thesis (0-1). "
                "0.9+ means you see no serious unaddressed objections. "
                "Below 0.3 means the thesis needs fundamental rework."
            ),
        },
        "should_continue": {
            "type": "boolean",
            "description": (
                "True if there are promising unexplored directions or unresolved tensions. "
                "False if you've reached a stable, well-defended position "
                "or further iteration would be circular."
            ),
        },
        "next_direction": {
            "type": "string",
            "description": (
                "The most promising direction for the next iteration. "
                "Be specific: name the exact tension, distinction, or angle to explore. "
                "This will be your starting point next round."
            ),
        },
    },
    "required": [
        "thesis",
        "reasoning",
        "counter_arguments",
        "new_insights",
        "confidence",
        "should_continue",
        "next_direction",
    ],
}


class ThinkingResult(TypedDict):
    thesis: str
    reasoning: str
    counter_arguments: list[str]
    new_insights: list[str]
    confidence: float
    should_continue: bool
    next_direction: str


def schema_json() -> str:
    """Return compact JSON string of the schema for CLI --json-schema flag."""
    return json.dumps(THINKING_SCHEMA, separators=(",", ":"))
