"""
Image MCQ Generator Module — dspy-cli compatible.

Run: dspy-cli serve --system   (from d:/Topin)
Web UI: http://localhost:8000

Fill in: topic, subtopic, easy_count, medium_count, hard_count
Click Run → questions saved to data/image_mcq/image_mcq_generator_output.json

Each question includes an image_content field — a text description of a
real-world notice/sign. Run generate_review_images.py afterwards to
generate actual images and a review HTML page.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import dspy


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "utils.py").exists():
            return parent
    raise RuntimeError("Cannot find project root (utils.py not found above this file)")


PROJECT_ROOT = _find_project_root()


def _find_venv_python() -> str:
    if platform.system() == "Windows":
        venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


_PYTHON = _find_venv_python()


class ImageMCQGeneratorModule(dspy.Module):
    """Generate Image-based Multiple-Choice Questions (MCQ) for language learners.

    Each question is tied to a real-world notice or sign image description.
    Produces CEFR-levelled questions validated by difficulty and rubric judges.
    Output is saved to data/image_mcq/image_mcq_generator_output.json.

    After generation, run:
        python generate_review_images.py
    to produce actual images and a browser-viewable review page.
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
        topic        : Topic (e.g. "English Language Skills")
        subtopic     : Subtopic (e.g. "Reading Notices")
        easy_count   : Total Easy questions (split across A1 + A2)
        medium_count : Total Medium questions (split across B1 + B2)
        hard_count   : Total Hard questions (split across C1 + C2)

        Output: data/image_mcq/image_mcq_generator_output.json
        Then run: python generate_review_images.py  to create images + review.html
        """
        config = {
            "type": "image_mcq",
            "topic": topic,
            "subtopics": [{
                "subtopic": subtopic,
                "a1_count": easy_count // 2,
                "a2_count": easy_count - easy_count // 2,
                "b1_count": medium_count // 2,
                "b2_count": medium_count - medium_count // 2,
                "c1_count": hard_count // 2,
                "c2_count": hard_count - hard_count // 2,
            }],
            "constraints": {
                "questions_per_iteration": 5,
                "max_iterations_per_difficulty": 20,
            },
        }

        tmp_cfg = PROJECT_ROOT / "_tmp_image_mcq_cli.json"
        tmp_cfg.write_text(json.dumps(config, indent=2), encoding="utf-8")

        try:
            result = subprocess.run(
                [_PYTHON, str(PROJECT_ROOT / "generate.py"), "--config", str(tmp_cfg)],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=600,
            )
        finally:
            tmp_cfg.unlink(missing_ok=True)

        if result.returncode != 0:
            return dspy.Prediction(
                status="error",
                message=result.stderr[-2000:] if result.stderr else "Unknown error",
                generated_questions="",
            )

        out_path = PROJECT_ROOT / "data" / "image_mcq" / "image_mcq_generator_output.json"
        if not out_path.exists():
            return dspy.Prediction(
                status="error",
                message="Output file not found after generation.",
                generated_questions="",
            )

        output    = json.loads(out_path.read_text(encoding="utf-8"))
        questions = output.get("questions", {})
        total     = sum(len(v) for v in questions.values() if isinstance(v, list))

        return dspy.Prediction(
            status="success",
            message=(
                f"Generated {total} image MCQ questions  |  "
                f"Easy={len(questions.get('easy', []))}  "
                f"Medium={len(questions.get('medium', []))}  "
                f"Hard={len(questions.get('hard', []))}  |  "
                f"Next step: run  python generate_review_images.py  to create images"
            ),
            generated_questions=json.dumps(output, indent=2, ensure_ascii=False),
        )
