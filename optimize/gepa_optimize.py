from __future__ import annotations

import json
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

from agents.pipeline import DifficultyAgent, RubricAgent
from agents.signatures import DifficultyJudgeSignature, RubricJudgeSignature
from utils import configure_dspy_from_env


# ── Flat wrappers for BootstrapFewShot ───────────────────────────────────────
# BootstrapFewShot calls forward(**example.inputs()) using flat fields.
# The real agents expect an MCQItem object, so we wrap them here.

class _DifficultyFlat(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(DifficultyJudgeSignature)

    def forward(self, stem, options, correct_answer, explanation, target_cefr, target_difficulty):
        return self.judge(
            stem=stem,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            target_cefr=target_cefr,
            target_difficulty=target_difficulty,
        )


class _RubricFlat(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(RubricJudgeSignature)

    def forward(self, stem, options, correct_answer, explanation, target_cefr, target_difficulty):
        return self.judge(
            stem=stem,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            target_cefr=target_cefr,
            target_difficulty=target_difficulty,
            language_variant="British English",
        )


TRAINSET_PATH = Path("data/gepa_train.jsonl")


def _make_reflection_lm() -> dspy.LM:
    load_dotenv()
    return dspy.LM(
        f"openai/{os.getenv('MISTRAL_MODEL', 'open-mistral-nemo')}",
        api_key=os.environ["MISTRAL_API_KEY"],
        api_base=os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1"),
    )


def load_trainset(path: Path = TRAINSET_PATH):
    examples = []
    if not path.exists():
        return examples
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(
                dspy.Example(
                    stem=row["stem"],
                    options=json.dumps(row["options"]),
                    correct_answer=row["correct_answer"],
                    explanation=row["explanation"],
                    target_cefr=row["target_cefr"],
                    target_difficulty=row["target_difficulty"],
                    expected_predicted_cefr=row.get("expected_predicted_cefr", row["target_cefr"]),
                    expected_overall_decision=row.get("expected_overall_decision", "Pass"),
                ).with_inputs(
                    "stem", "options", "correct_answer",
                    "explanation", "target_cefr", "target_difficulty",
                )
            )
    return examples


def append_failures_to_trainset(results: list, path: Path = TRAINSET_PATH) -> int:
    """Append rejected items to the training set, skipping duplicates. Returns count added."""
    existing_stems: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_stems.add(json.loads(line)["stem"].strip())
                except Exception:
                    pass

    added = 0
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            if not r.accepted and r.item.stem.strip() not in existing_stems:
                row = {
                    "stem": r.item.stem,
                    "options": r.item.options,
                    "correct_answer": r.item.correct_answer,
                    "explanation": r.item.explanation,
                    "target_cefr": r.item.target_cefr,
                    "target_difficulty": r.item.target_difficulty,
                    "expected_predicted_cefr": r.item.target_cefr,
                    "expected_overall_decision": "Pass",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing_stems.add(r.item.stem.strip())
                added += 1
    return added


# ── Boolean metrics for BootstrapFewShot ─────────────────────────────────────

def difficulty_metric_bool(gold, pred, trace=None):
    try:
        return json.loads(pred.output_json).get("predicted_cefr") == gold.expected_predicted_cefr
    except Exception:
        return False


def rubric_metric_bool(gold, pred, trace=None):
    try:
        return json.loads(pred.output_json).get("overall_decision") == gold.expected_overall_decision
    except Exception:
        return False


# ── Tuple metrics for GEPA ───────────────────────────────────────────────────

def difficulty_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    try:
        payload = json.loads(pred.output_json)
    except Exception:
        return 0.0, "Output was not valid JSON."
    score = 1.0 if payload.get("predicted_cefr") == gold.expected_predicted_cefr else 0.0
    feedback = (
        f"Expected predicted_cefr={gold.expected_predicted_cefr}, "
        f"got {payload.get('predicted_cefr')}. "
        f"Alignment should reflect the CEFR target more accurately."
    )
    return score, feedback


def rubric_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    try:
        payload = json.loads(pred.output_json)
    except Exception:
        return 0.0, "Output was not valid JSON."
    score = 1.0 if payload.get("overall_decision") == gold.expected_overall_decision else 0.0
    feedback = (
        f"Expected overall_decision={gold.expected_overall_decision}, "
        f"got {payload.get('overall_decision')}. "
        f"Pay special attention to ambiguity, alignment, and option-explanation consistency."
    )
    return score, feedback


# ── BootstrapFewShot (primary optimizer) ─────────────────────────────────────

def optimize_difficulty(trainset):
    """Optimize DifficultyAgent using BootstrapFewShot — adds few-shot demos to the prompt."""
    from dspy.teleprompt import BootstrapFewShot
    student = _DifficultyFlat()
    optimizer = BootstrapFewShot(
        metric=difficulty_metric_bool,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
    )
    optimized = optimizer.compile(student, trainset=trainset)
    optimized.save("artifacts/difficulty_agent_optimized.json")
    print(f"  [Bootstrap] DifficultyAgent optimized.")
    return optimized


def optimize_rubric(trainset):
    """Optimize RubricAgent using BootstrapFewShot — adds few-shot demos to the prompt."""
    from dspy.teleprompt import BootstrapFewShot
    student = _RubricFlat()
    optimizer = BootstrapFewShot(
        metric=rubric_metric_bool,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
    )
    optimized = optimizer.compile(student, trainset=trainset)
    optimized.save("artifacts/rubric_agent_optimized.json")
    print(f"  [Bootstrap] RubricAgent optimized.")
    return optimized


if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    configure_dspy_from_env()
    trainset = load_trainset()
    print(f"Training set: {len(trainset)} examples")
    optimize_difficulty(trainset)
    optimize_rubric(trainset)
    print("Saved optimized modules in artifacts/")
