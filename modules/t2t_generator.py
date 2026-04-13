"""
T2T (Text-to-Text / Open-Answer) Generator Module — dspy-cli compatible.

Exposed by `dspy-cli serve` at http://localhost:8000
Web UI auto-generated from forward() parameters.

Example questions are loaded automatically from the training dataset —
no need to paste them manually.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import dspy

PROJECT_ROOT = Path(__file__).parent.parent


class T2TGeneratorModule(dspy.Module):
    """Generate Text-to-Text (open-answer / writing) questions for language learners.

    Produces CEFR-levelled T2T questions validated by difficulty and rubric judges.
    Output is saved to data/t2t/t2t_generator_output.json.

    Example questions are loaded automatically from the training dataset —
    you do not need to supply them here.
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
    ) -> dspy.Prediction:
        """
        Parameters
        ----------
        topic        : Topic (e.g. "English Language Skills")
        subtopic     : Subtopic (e.g. "Reading and Writing")
        easy_count   : Total Easy questions (A1 + A2)
        medium_count : Total Medium questions (B1 + B2)
        hard_count   : Total Hard questions (C1 + C2)
        """
        config = {
            "type": "t2t",
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
        }

        tmp_cfg = PROJECT_ROOT / "_tmp_t2t_cli.json"
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

        out_path = PROJECT_ROOT / "data" / "t2t" / "t2t_generator_output.json"
        if not out_path.exists():
            return dspy.Prediction(
                status="error",
                message="Output file not found after generation.",
                generated_questions="",
            )

        output = json.loads(out_path.read_text(encoding="utf-8"))
        questions = output.get("questions", {})
        total = sum(len(v) for v in questions.values() if isinstance(v, list))

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
