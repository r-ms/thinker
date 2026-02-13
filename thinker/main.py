from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from thinker.prompts import SYSTEM_PROMPT, build_first_prompt, build_iteration_prompt
from thinker.schemas import ThinkingResult, schema_json


@dataclass
class IterationStats:
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0


@dataclass
class SessionStats:
    iterations: list[IterationStats] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(it.cost_usd for it in self.iterations)

    @property
    def total_input_tokens(self) -> int:
        return sum(it.input_tokens for it in self.iterations)

    @property
    def total_output_tokens(self) -> int:
        return sum(it.output_tokens for it in self.iterations)

    def to_dict(self) -> dict:
        return {
            "total_cost_usd": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "per_iteration": [
                {
                    "cost_usd": it.cost_usd,
                    "input_tokens": it.input_tokens,
                    "output_tokens": it.output_tokens,
                    "duration_ms": it.duration_ms,
                }
                for it in self.iterations
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionStats:
        stats = cls()
        for it in data.get("per_iteration", []):
            stats.iterations.append(
                IterationStats(
                    cost_usd=it.get("cost_usd", 0),
                    input_tokens=it.get("input_tokens", 0),
                    output_tokens=it.get("output_tokens", 0),
                    duration_ms=it.get("duration_ms", 0),
                )
            )
        return stats


def call_claude(prompt: str, model: str | None = None) -> tuple[ThinkingResult, IterationStats]:
    """Call Claude CLI with structured output and return parsed result + stats."""
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        schema_json(),
        "--system-prompt",
        SYSTEM_PROMPT,
        "--no-session-persistence",
        "--tools",
        "",
    ]
    if model:
        cmd.extend(["--model", model])

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"Claude CLI error (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    response = json.loads(result.stdout)
    usage = response.get("usage", {})
    stats = IterationStats(
        cost_usd=response.get("total_cost_usd", 0),
        input_tokens=usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        duration_ms=response.get("duration_ms", 0),
    )
    return response["structured_output"], stats


def print_iteration(iteration: int, result: ThinkingResult, stats: IterationStats) -> None:
    """Print a single iteration result to the terminal."""
    duration_s = stats.duration_ms / 1000
    print(f"\n{'=' * 80}")
    print(
        f"  ITERATION {iteration}  |  confidence: {result['confidence']:.2f}"
        f"  |  ${stats.cost_usd:.4f}  |  {duration_s:.1f}s"
    )
    print(f"{'=' * 80}\n")

    print(f"THESIS:\n{result['thesis']}\n")
    print(f"REASONING:\n{result['reasoning']}\n")

    if result["counter_arguments"]:
        print("COUNTER-ARGUMENTS:")
        for j, ca in enumerate(result["counter_arguments"], 1):
            print(f"  {j}. {ca}")
        print()

    if result["new_insights"]:
        print("NEW INSIGHTS:")
        for j, ins in enumerate(result["new_insights"], 1):
            print(f"  {j}. {ins}")
        print()

    print(f"NEXT DIRECTION:\n{result['next_direction']}\n")

    if not result["should_continue"]:
        print("[Model indicates: no further iteration needed]")


def print_final(result: ThinkingResult, total: int, session_stats: SessionStats) -> None:
    """Print the final summary."""
    print(f"\n{'#' * 80}")
    print(
        f"  FINAL ANSWER  |  {total} iterations"
        f"  |  confidence: {result['confidence']:.2f}"
        f"  |  total: ${session_stats.total_cost:.4f}"
    )
    print(f"{'#' * 80}\n")
    print(result["thesis"])
    print()


def load_session(path: Path) -> dict:
    """Load an existing session from a JSON file."""
    return json.loads(path.read_text())


def save_session(
    question: str,
    model: str | None,
    history: list[ThinkingResult],
    session_stats: SessionStats,
    output_path: Path,
) -> None:
    """Save full session history to a JSON file."""
    session = {
        "question": question,
        "model": model,
        "timestamp": datetime.now(UTC).isoformat(),
        "total_iterations": len(history),
        "final_confidence": history[-1]["confidence"],
        "final_thesis": history[-1]["thesis"],
        "stats": session_stats.to_dict(),
        "iterations": [{"iteration": i, **result} for i, result in enumerate(history, 1)],
    }
    output_path.write_text(json.dumps(session, ensure_ascii=False, indent=2))
    print(f"[Saved to {output_path}]")


def make_default_output_path(question: str) -> Path:
    """Generate a default output file path from the question."""
    slug = re.sub(r"[^\w]+", "-", question.lower()).strip("-")[:40]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"thinker-{slug}-{timestamp}.json")


def run(
    question: str,
    max_iterations: int,
    model: str | None,
    output_path: Path,
    history: list[ThinkingResult],
    session_stats: SessionStats,
) -> None:
    """Main thinking loop."""
    start_iteration = len(history) + 1
    end_iteration = start_iteration + max_iterations - 1

    print(f"Question: {question}")
    print(f"Iterations: {start_iteration} → {end_iteration} ({max_iterations} new)")
    if model:
        print(f"Model: {model}")
    if session_stats.iterations:
        print(f"Prior cost: ${session_stats.total_cost:.4f}")
    print()

    for i in range(start_iteration, end_iteration + 1):
        print(f"[Thinking... iteration {i}/{end_iteration}]")

        if not history:
            prompt = build_first_prompt(question)
        else:
            prompt = build_iteration_prompt(question, history, i)

        result, stats = call_claude(prompt, model)
        history.append(result)
        session_stats.iterations.append(stats)
        print_iteration(i, result, stats)
        save_session(question, model, history, session_stats, output_path)

    print_final(history[-1], len(history), session_stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative philosophical thinker powered by Claude CLI",
    )
    parser.add_argument("question", nargs="?", default=None, help="The question to think about")
    parser.add_argument(
        "-n",
        "--max-iterations",
        type=int,
        default=None,
        help="Number of iterations (default: 5 for new, 1 for --continue-from)",
    )
    parser.add_argument("-m", "--model", default=None, help="Claude model to use")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON file path")
    parser.add_argument(
        "-c",
        "--continue-from",
        type=Path,
        default=None,
        help="Continue from existing session file",
    )

    args = parser.parse_args()

    if args.continue_from:
        session = load_session(args.continue_from)
        question = session["question"]
        history: list[ThinkingResult] = [
            {k: v for k, v in it.items() if k != "iteration"} for it in session["iterations"]
        ]
        session_stats = SessionStats.from_dict(session.get("stats", {}))
        max_iterations = args.max_iterations or 1
        model = args.model or session.get("model")
        output_path = args.output or args.continue_from
        print(f"Resuming session from {args.continue_from} ({len(history)} iterations)\n")
    elif args.question:
        question = args.question
        history = []
        session_stats = SessionStats()
        max_iterations = args.max_iterations or 5
        model = args.model
        output_path = args.output or make_default_output_path(question)
    else:
        parser.error("question is required (unless using --continue-from)")

    run(question, max_iterations, model, output_path, history, session_stats)


if __name__ == "__main__":
    main()
