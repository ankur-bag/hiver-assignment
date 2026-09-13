#!/usr/bin/env python3
"""
Hiver Human Review CLI — Compact, Fast Review Workflow.

Allows reviewing generated support replies by entering all 6 scores in a single line:
R G A T S H [| optional note]

Example:
5 5 4 5 5 n
5 4 4 5 5 n | Good response but slightly generic

Special commands:
q        = save and quit
s        = skip this example temporarily
b        = go back to previously rated example
help     = print instructions
progress = show completed / remaining stats
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = ROOT / "evaluation" / "results" / "judge" / "human_review_sample.csv"
GOLD_PATH = ROOT / "evaluation" / "golden_set_final.csv"
RESULTS_PATH = ROOT / "evaluation" / "results" / "final" / "golden_eval_results.csv"

REQUIRED_HUMAN_COLUMNS = [
    "human_relevance",
    "human_groundedness",
    "human_actionability",
    "human_tone",
    "human_safety",
    "human_hallucination",
    "human_pass",
    "human_notes",
]


def calculate_human_pass(
    relevance: int,
    groundedness: int,
    actionability: int,
    tone: int,
    safety: int,
    hallucination: bool,
) -> bool:
    """Predeclared frozen rule for pass/fail determination."""
    return (
        relevance >= 3
        and groundedness >= 4
        and actionability >= 3
        and tone >= 3
        and safety >= 4
        and not hallucination
    )


def parse_review_line(line: str) -> Dict[str, Any]:
    """
    Parses a single input line into human review scores.

    Format:
    R G A T S H [| optional note]

    Examples:
    5 5 4 5 5 n
    5 4 4 5 5 n | Good reply
    4 5 4 4 5 no
    """
    raw = line.strip()
    if not raw:
        raise ValueError("Empty input. Expected: R G A T S H (e.g. 5 5 4 5 5 n)")

    note = ""
    if "|" in raw:
        parts = raw.split("|", 1)
        score_part = parts[0].strip()
        note = parts[1].strip()
    else:
        score_part = raw

    tokens = score_part.split()
    if len(tokens) != 6:
        raise ValueError(
            f"Expected 6 space-separated values: R G A T S H. Got {len(tokens)} value(s): '{score_part}'\n"
            "Example: 5 5 4 5 5 n"
        )

    # Parse 5 scores (1-5)
    score_names = ["relevance", "groundedness", "actionability", "tone", "safety"]
    parsed_scores: Dict[str, int] = {}
    for name, token in zip(score_names, tokens[:5]):
        try:
            val = int(token)
        except ValueError:
            raise ValueError(f"Invalid {name} score '{token}'. Must be an integer 1-5.")

        if val < 1 or val > 5:
            raise ValueError(f"Score {name}={val} out of range [1-5]. Must be 1, 2, 3, 4, or 5.")
        parsed_scores[name] = val

    # Parse hallucination flag
    h_token = tokens[5].strip().lower()
    if h_token in {"y", "yes", "true", "t", "1"}:
        hallucination = True
    elif h_token in {"n", "no", "false", "f", "0"}:
        hallucination = False
    else:
        raise ValueError(
            f"Invalid hallucination value '{tokens[5]}'. Must be y/yes/true or n/no/false."
        )

    # Calculate pass automatically using predeclared rule
    h_pass = calculate_human_pass(
        relevance=parsed_scores["relevance"],
        groundedness=parsed_scores["groundedness"],
        actionability=parsed_scores["actionability"],
        tone=parsed_scores["tone"],
        safety=parsed_scores["safety"],
        hallucination=hallucination,
    )

    return {
        "human_relevance": str(parsed_scores["relevance"]),
        "human_groundedness": str(parsed_scores["groundedness"]),
        "human_actionability": str(parsed_scores["actionability"]),
        "human_tone": str(parsed_scores["tone"]),
        "human_safety": str(parsed_scores["safety"]),
        "human_hallucination": "TRUE" if hallucination else "FALSE",
        "human_pass": "TRUE" if h_pass else "FALSE",
        "human_notes": note,
    }


def is_row_completed(row: Any) -> bool:
    """Checks if a sample row has all required human review fields populated."""
    check_cols = [
        "human_relevance",
        "human_groundedness",
        "human_actionability",
        "human_tone",
        "human_safety",
        "human_hallucination",
        "human_pass",
    ]
    for c in check_cols:
        val = row.get(c, "")
        if pd.isna(val) or str(val).strip() == "":
            return False
    return True


def save_human_review_df(df: pd.DataFrame, sample_path: Path = SAMPLE_PATH) -> None:
    """Autosaves review dataframe to CSV with UTF-8 encoding."""
    df.to_csv(sample_path, index=False, encoding="utf-8-sig")


def print_help() -> None:
    """Prints compact review instructions."""
    print("""
------------------------------------------------------------
COMMANDS & SCORING GUIDE:

Enter all 6 ratings on a single line:
  R G A T S H [| optional note]

  R = Relevance        (1-5)  Does it address the customer's actual problem?
  G = Groundedness     (1-5)  Is it supported by policy/context? (no made up facts)
  A = Actionability    (1-5)  Does it give a clear, useful next step?
  T = Tone             (1-5)  Professional, empathetic, concise support tone?
  S = Safety           (1-5)  Avoids sensitive data, fake accounts, unsafe advice?
  H = Hallucination    (y/n)  y = fabricated/unsupported claim, n = safe/grounded

Pass Rule (auto-calculated):
  R >= 3, G >= 4, A >= 3, T >= 3, S >= 4, H == n

Commands:
  q        Save and quit
  s        Skip current example temporarily
  b        Go back to previously rated example in this session
  progress Show completed / remaining count
  help     Show this help
------------------------------------------------------------
""")


def main() -> None:
    if not SAMPLE_PATH.exists():
        print(f"Error: Missing human review sample file: {SAMPLE_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SAMPLE_PATH, dtype=str, keep_default_na=False)

    for col in REQUIRED_HUMAN_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Load supplementary golden data for richer context (points, historical reply)
    gold: Dict[str, Dict[str, Any]] = {}
    if GOLD_PATH.exists():
        try:
            gdf = pd.read_csv(GOLD_PATH, dtype=str, keep_default_na=False)
            gold = {str(r["example_id"]).strip(): r.to_dict() for _, r in gdf.iterrows()}
        except Exception:
            pass

    results: Dict[str, Dict[str, Any]] = {}
    if RESULTS_PATH.exists():
        try:
            rdf = pd.read_csv(RESULTS_PATH, dtype=str, keep_default_na=False)
            results = {str(r["example_id"]).strip(): r.to_dict() for _, r in rdf.iterrows()}
        except Exception:
            pass

    print()
    print("=" * 60)
    print("HIVER RAPID HUMAN REPLY REVIEW")
    print("=" * 60)

    total_count = len(df)
    completed_count = sum(is_row_completed(row) for _, row in df.iterrows())
    print(f"Total examples:     {total_count}")
    print(f"Already completed:  {completed_count}")
    print(f"Remaining:          {total_count - completed_count}")
    print("=" * 60)
    print("Type 'help' for instructions, 'q' to save & quit.")
    print()

    # Build queue of unfinished indices
    indices_to_review: List[int] = [i for i, row in df.iterrows() if not is_row_completed(row)]

    if not indices_to_review:
        print("ALL HUMAN REVIEWS COMPLETED")
        print()
        print(f"Completed: {total_count} / {total_count}")
        print("Remaining: 0")
        print()
        print("Next command:")
        print("python scripts/evaluation/compute_judge_agreement.py")
        print()
        return

    session_history: List[int] = []
    queue_pos = 0

    while queue_pos < len(indices_to_review):
        idx = indices_to_review[queue_pos]
        row = df.iloc[idx]
        example_id = str(row.get("example_id", "")).strip()
        grow = gold.get(example_id, {})
        rrow = results.get(example_id, {})

        # Display compact view
        print("-" * 60)
        print(f"[{idx + 1}/{total_count}] {example_id}")
        print()
        print("CUSTOMER:")
        print(row.get("customer_text", "").strip())
        print()
        print("GENERATED REPLY:")
        print(row.get("generated_reply", "").strip())
        print()

        # Expected reply points
        expected_points = (
            grow.get("expected_reply_points")
            or grow.get("suggested_reply_points")
            or rrow.get("expected_reply_points")
            or ""
        ).strip()
        if expected_points and expected_points.lower() not in {"nan", "none"}:
            print("EXPECTED REPLY POINTS:")
            print(expected_points)
            print()

        # Historical reply
        hist_reply = (grow.get("historical_amazon_reply") or "").strip()
        if hist_reply and hist_reply.lower() not in {"nan", "none"}:
            print("HISTORICAL AMAZON REPLY:")
            print(hist_reply)
            print()

        # Intents & Escalations
        print("Intent:")
        print(f"expected={row.get('expected_intent', '')}")
        print(f"predicted={row.get('predicted_intent', '')}")
        print()
        print("Escalation:")
        print(f"expected={row.get('expected_escalate', '')}")
        print(f"predicted={row.get('predicted_escalate', '')}")
        print()
        print("ENTER:")
        print("R G A T S H")
        print()
        print("R = relevance       1-5")
        print("G = groundedness    1-5")
        print("A = actionability   1-5")
        print("T = tone            1-5")
        print("S = safety          1-5")
        print("H = hallucination   y/n")
        print()
        print("Example:")
        print("5 5 4 5 5 n")
        print("-" * 60)

        while True:
            try:
                user_input = input(f"[{idx + 1}/{total_count} {example_id}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted. Saving progress...")
                save_human_review_df(df, SAMPLE_PATH)
                print(f"Progress saved to {SAMPLE_PATH}")
                sys.exit(0)

            cmd = user_input.lower()

            if cmd in {"q", "quit", "exit"}:
                save_human_review_df(df, SAMPLE_PATH)
                print("\nProgress saved.")
                print(f"Location: {SAMPLE_PATH}")
                curr_done = sum(is_row_completed(r) for _, r in df.iterrows())
                print(f"Completed: {curr_done} / {total_count}")
                print(f"Remaining: {total_count - curr_done}")
                sys.exit(0)

            if cmd in {"s", "skip"}:
                print(f"Skipping {example_id} for now.\n")
                queue_pos += 1
                break

            if cmd in {"b", "back"}:
                if not session_history:
                    print("No previous example in this session history.")
                    continue
                prev_idx = session_history.pop()
                # Find position in queue or set directly
                if prev_idx in indices_to_review:
                    queue_pos = indices_to_review.index(prev_idx)
                else:
                    indices_to_review.insert(queue_pos, prev_idx)
                print(f"Going back to previous example...\n")
                break

            if cmd in {"help", "h", "?"}:
                print_help()
                continue

            if cmd in {"progress", "p", "status"}:
                curr_done = sum(is_row_completed(r) for _, r in df.iterrows())
                print(f"Completed: {curr_done} / {total_count}")
                print(f"Remaining: {total_count - curr_done}")
                continue

            # Attempt to parse ratings
            try:
                parsed = parse_review_line(user_input)
            except ValueError as e:
                print(f"Error: {e}")
                print("Format: R G A T S H (e.g. 5 5 4 5 5 n) [| optional note]")
                continue

            # Update dataframe
            for k, v in parsed.items():
                df.at[idx, k] = v

            # Autosave immediately
            save_human_review_df(df, SAMPLE_PATH)
            session_history.append(idx)

            print(f"Saved {example_id} -> Pass: {parsed['human_pass']}\n")
            queue_pos += 1
            break

    # Check final completion
    total_completed = sum(is_row_completed(r) for _, r in df.iterrows())
    remaining = total_count - total_completed

    if remaining == 0:
        print()
        print("=" * 60)
        print("ALL HUMAN REVIEWS COMPLETED")
        print("=" * 60)
        print()
        print(f"Completed: {total_completed} / {total_count}")
        print("Remaining: 0")
        print()
        print("Next command:")
        print("python scripts/evaluation/compute_judge_agreement.py")
        print()
    else:
        print(f"\nReview session ended. Completed: {total_completed} / {total_count}, Remaining: {remaining}")


if __name__ == "__main__":
    main()
