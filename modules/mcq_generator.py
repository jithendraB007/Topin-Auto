"""
MCQ Generator Module — dspy-cli compatible.

Exposed by `dspy-cli serve` at http://localhost:8000
Web UI auto-generated from forward() parameters.

Usage in web UI:
  1. Fill in topic, subtopic, counts
  2. Paste example_questions_json (JSON array — one example per CEFR level)
  3. Click Run → questions generated and returned as JSON
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import dspy

PROJECT_ROOT = Path(__file__).parent.parent

# Default examples shown in the UI as a starting-point template
_DEFAULT_EXAMPLES = json.dumps([
    {
        "instruction": "Read the sentence and choose the correct word.",
        "question": "She _______ to school every day.",
        "options": ["go", "goes", "going", "gone"],
        "correct_answer": "goes",
        "explanation": "Third-person singular requires -s/-es in simple present.",
        "difficulty": "Easy", "cefr": "A1", "subtopic": ""
    },
    {
        "instruction": "Read the sentence and choose the correct word.",
        "question": "My brother _______ football every weekend.",
        "options": ["play", "plays", "playing", "played"],
        "correct_answer": "plays",
        "explanation": "Third-person singular 'brother' requires the -s form.",
        "difficulty": "Easy", "cefr": "A2", "subtopic": ""
    },
    {
        "instruction": "Spot the error in the sentence.",
        "question": "'They is playing football right now.' — what is wrong?",
        "options": ["They -> It", "is -> are", "playing -> play", "in -> at"],
        "correct_answer": "is -> are",
        "explanation": "Plural subject 'they' requires 'are'.",
        "difficulty": "Medium", "cefr": "B1", "subtopic": ""
    },
    {
        "instruction": "Choose the correct sentence.",
        "question": "Which uses present perfect to show recent completion?",
        "options": [
            "She just finished the report.",
            "She has just finished the report.",
            "She is just finishing the report.",
            "She had just finished the report."
        ],
        "correct_answer": "She has just finished the report.",
        "explanation": "'Has finished' shows recent completion.",
        "difficulty": "Medium", "cefr": "B2", "subtopic": ""
    },
    {
        "instruction": "Choose the correct sentence.",
        "question": "Which shows correct subject-verb agreement?",
        "options": [
            "The quality of the reports are improving.",
            "The quality of the reports is improving.",
            "The quality of the reports have improved.",
            "The quality of the reports were improving."
        ],
        "correct_answer": "The quality of the reports is improving.",
        "explanation": "Head noun 'quality' is singular.",
        "difficulty": "Hard", "cefr": "C1", "subtopic": ""
    },
    {
        "instruction": "Choose the most formal sentence.",
        "question": "Which uses present tense for a scheduled event most formally?",
        "options": [
            "The summit commences at 09:00 tomorrow.",
            "The summit will commence at 09:00 tomorrow.",
            "The summit is commencing at 09:00 tomorrow.",
            "The summit is going to commence at 09:00 tomorrow."
        ],
        "correct_answer": "The summit commences at 09:00 tomorrow.",
        "explanation": "Simple present for scheduled events is the most formal register.",
        "difficulty": "Hard", "cefr": "C2", "subtopic": ""
    }
], indent=2)


class MCQGeneratorModule(dspy.Module):
    """Generate Multiple-Choice Questions (MCQ) for language learners.

    Produces CEFR-levelled MCQ questions validated by difficulty and rubric judges.
    Output is saved to data/mcq/mcq_generator_output.json.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        topic: str,
        subtopic: str,
        easy_count: int,
        medium_count: int,
        hard_count: int,
        example_questions_json: str = _DEFAULT_EXAMPLES,
    ) -> dspy.Prediction:
        """
        Parameters
        ----------
        topic                : Topic (e.g. "English Grammar")
        subtopic             : Subtopic (e.g. "Question Words")
        easy_count           : Total Easy questions (A1 + A2)
        medium_count         : Total Medium questions (B1 + B2)
        hard_count           : Total Hard questions (C1 + C2)
        example_questions_json : JSON array — one example per CEFR level (A1–C2).
                               Edit the subtopic field to match your subtopic.
        """
        # Parse example questions
        try:
            examples = json.loads(example_questions_json)
            for ex in examples:
                if not ex.get("subtopic"):
                    ex["subtopic"] = subtopic
        except json.JSONDecodeError as e:
            return dspy.Prediction(
                status="error",
                message=f"example_questions_json is not valid JSON: {e}",
                generated_questions="",
            )

        # Split easy/medium/hard evenly across CEFR pairs
        config = {
            "type": "mcq",
            "topic": topic,
            "subtopics": [
                {
                    "subtopic": subtopic,
                    "a1_count": easy_count // 2,
                    "a2_count": easy_count - easy_count // 2,
                    "b1_count": medium_count // 2,
                    "b2_count": medium_count - medium_count // 2,
                    "c1_count": hard_count // 2,
                    "c2_count": hard_count - hard_count // 2,
                }
            ],
            "constraints": {
                "questions_per_iteration": 5,
                "max_iterations_per_difficulty": 20,
            },
            "example_questions": examples,
        }

        tmp_cfg = PROJECT_ROOT / "_tmp_mcq_cli.json"
        tmp_cfg.write_text(json.dumps(config, indent=2), encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "generate.py"), "--config", str(tmp_cfg)],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=600,
            )
        finally:
            tmp_cfg.unlink(missing_ok=True)

        if result.returncode != 0:
            return dspy.Prediction(
                status="error",
                message=result.stderr[-2000:] if result.stderr else "Unknown error",
                generated_questions="",
            )

        out_path = PROJECT_ROOT / "data" / "mcq" / "mcq_generator_output.json"
        if not out_path.exists():
            return dspy.Prediction(
                status="error",
                message="Output file not found after generation.",
                generated_questions="",
            )

        output = json.loads(out_path.read_text(encoding="utf-8"))
        summary = output.get("summary", {})
        questions = output.get("questions", {})
        total = sum(
            len(v) for v in questions.values() if isinstance(v, list)
        )

        return dspy.Prediction(
            status="success",
            message=(
                f"Generated {total} questions. "
                f"Easy={len(questions.get('easy', []))} "
                f"Medium={len(questions.get('medium', []))} "
                f"Hard={len(questions.get('hard', []))}"
            ),
            generated_questions=json.dumps(output, indent=2, ensure_ascii=False),
        )
