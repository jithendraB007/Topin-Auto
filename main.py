from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from agents.pipeline import MCQPipeline
from schemas import EvaluatedItem, InputSchema
from utils import configure_dspy_from_env, load_json, save_json

# Run GEPA + retry whenever any question is rejected (set < 1.0 to only trigger on heavy failures)
AUTO_OPTIMIZE_THRESHOLD = 1.0


def _build_retry_schema(schema: InputSchema, rejected: list[EvaluatedItem]) -> InputSchema:
    cefr_counts = dict(Counter(r.item.target_cefr for r in rejected))
    return InputSchema(
        subject=schema.subject,
        syllabus_unit=schema.syllabus_unit,
        topic=schema.topic,
        subtopics=schema.subtopics,
        question_type=schema.question_type,
        total_questions=len(rejected),
        cefr_distribution=cefr_counts,
        constraints=schema.constraints,
    )


def _run_gepa(pipeline: MCQPipeline, rejected: list[EvaluatedItem]) -> bool:
    from optimize.gepa_optimize import (
        TRAINSET_PATH,
        append_failures_to_trainset,
        load_trainset,
        optimize_difficulty,
        optimize_rubric,
    )

    added = append_failures_to_trainset(rejected)
    print(f"[AutoOptimize] Added {added} failures to training set.")

    trainset = load_trainset(TRAINSET_PATH)
    if len(trainset) < 2:
        print("[AutoOptimize] Not enough training examples (need >= 2). Skipping GEPA.")
        return False

    print(f"[AutoOptimize] Running GEPA on {len(trainset)} examples...")
    Path("artifacts").mkdir(exist_ok=True)
    optimize_difficulty(trainset)
    optimize_rubric(trainset)
    pipeline.load_optimized_agents()
    print("[AutoOptimize] Optimization complete. Retrying failed questions...")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input schema JSON")
    parser.add_argument("--output", default="data/output_run.json", help="Path to save results")
    parser.add_argument("--no-auto-optimize", action="store_true", help="Disable auto GEPA optimization")
    args = parser.parse_args()

    configure_dspy_from_env()
    raw = load_json(args.input)
    schema = InputSchema(**raw)

    pipeline = MCQPipeline(max_revision_attempts=2)
    pipeline.load_optimized_agents()  # load from artifacts/ if available

    # ── Pass 1 ──────────────────────────────────────────────────
    print(f"Running pipeline (pass 1) - {schema.total_questions} questions...")
    results = pipeline(schema)

    rejected = [r for r in results if not r.accepted]
    accepted_count = len(results) - len(rejected)
    rate = accepted_count / len(results) if results else 1.0
    print(f"Pass 1 -> Accepted: {accepted_count} | Rejected: {len(rejected)} | Rate: {rate:.0%}")

    # ── Auto-optimize + Pass 2 ───────────────────────────────────
    auto_optimize = not args.no_auto_optimize
    if auto_optimize and rejected and rate < AUTO_OPTIMIZE_THRESHOLD:
        print(f"Rate {rate:.0%} below {AUTO_OPTIMIZE_THRESHOLD:.0%} threshold — running GEPA...")
        optimized = _run_gepa(pipeline, rejected)

        if optimized:
            retry_schema = _build_retry_schema(schema, rejected)
            print(f"Retrying {len(rejected)} failed questions (pass 2)...")
            retry_results = pipeline(retry_schema)

            retry_accepted = sum(1 for r in retry_results if r.accepted)
            print(f"Pass 2 → Accepted: {retry_accepted} | Rejected: {len(retry_results) - retry_accepted}")

            # Merge: keep pass-1 accepted + all pass-2 results
            results = [r for r in results if r.accepted] + retry_results

    # ── Save ─────────────────────────────────────────────────────
    serializable = [r.model_dump() for r in results]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, serializable)

    total_accepted = sum(1 for r in results if r.accepted)
    print(f"\nSaved {len(results)} items to {output_path}")
    print(f"Accepted: {total_accepted} | Rejected: {len(results) - total_accepted}")


if __name__ == "__main__":
    main()
