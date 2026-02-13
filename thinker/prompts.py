from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thinker.schemas import ThinkingResult

SYSTEM_PROMPT = """\
You are a dialectical philosopher engaged in rigorous iterative thinking.

Core principles:
- Precision over breadth. Go deep on one line of reasoning rather than shallow across many.
- Steel-man every counter-argument — present it in its strongest possible form before responding.
- Name your assumptions explicitly, then challenge them.
- Use concrete examples and thought experiments, not abstractions.
- If something sounds like a motivational poster, dig deeper — you haven't reached insight yet.
- Distinguish the logically necessary from the empirically probable from the merely conventional.
- Track the evolution of your thinking: what changed, what held, and why.

Write for a smart, curious person — not for an academic journal. Use technical terms \
only when they carry meaning that plain language cannot. If you catch yourself writing \
"bivalent property of propositions" instead of "statements are either true or false", \
rewrite. Jargon is a tool, not a credential. The test: could a thoughtful 17-year-old \
follow your reasoning? If not, simplify without losing precision.

You are not here to be balanced or diplomatic. You are here to find what is true, \
even if the answer is uncomfortable, partial, or reveals that the question itself is malformed.\
"""


def build_first_prompt(question: str) -> str:
    return f"""\
Question: {question}

This is iteration 1. Approach the question fresh:

1. Formulate a clear, specific thesis in response to the question.
2. Stress-test your thesis immediately — what's the strongest objection?
3. Look for non-obvious angles: hidden assumptions in the question itself, \
unexpected connections to other domains, edge cases that break intuitions.

Go deep on one line of reasoning rather than surveying the landscape. \
Depth beats breadth.\
"""


def build_iteration_prompt(
    question: str,
    history: list[ThinkingResult],
    iteration: int,
) -> str:
    history_block = _format_history(history)

    return f"""\
Question: {question}

This is iteration {iteration}. Here is the full trajectory of thinking so far:

{history_block}

Your task for this iteration — do ALL of the following:

1. ATTACK your previous position. Find the weakest link in your last thesis and reasoning. \
Where does the argument actually break? Don't be gentle.

2. FOLLOW your own lead. You indicated the next direction should be: \
"{history[-1]['next_direction']}". Explore this seriously.

3. SHIFT perspective. Approach from a different philosophical tradition, discipline, \
or historical era than you used before. Name which one and why it matters here.

4. FIND hidden structure. Are there false dichotomies you've been accepting? \
Collapsed distinctions that should be pulled apart? Category errors?

5. RAISE the stakes. What would change in practice — in how we live, decide, build — \
if your current thesis is wrong? Make the consequences concrete.

Synthesize all of this into an updated position. It's fine to abandon your previous thesis \
entirely if the arguments demand it.\
"""


RECENT_WINDOW = 3  # number of recent iterations to include in full detail


def _format_history(history: list[ThinkingResult]) -> str:
    parts: list[str] = []
    total = len(history)
    cutoff = max(0, total - RECENT_WINDOW)

    # Older iterations: thesis + confidence only
    for i, result in enumerate(history[:cutoff], 1):
        parts.append(
            f"--- Iteration {i} (confidence: {result['confidence']:.2f}) [summary] ---\n"
            f"Thesis: {result['thesis']}"
        )

    # Recent iterations: full detail
    for i, result in enumerate(history[cutoff:], cutoff + 1):
        parts.append(
            f"--- Iteration {i} (confidence: {result['confidence']:.2f}) ---\n"
            f"Thesis: {result['thesis']}\n"
            f"Reasoning: {result['reasoning']}\n"
            f"Counter-arguments: {json.dumps(result['counter_arguments'], ensure_ascii=False)}\n"
            f"New insights: {json.dumps(result['new_insights'], ensure_ascii=False)}\n"
            f"Next direction: {result['next_direction']}"
        )

    return "\n\n".join(parts)
