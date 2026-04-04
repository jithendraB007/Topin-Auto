from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.pipeline import DifficultyAgent, RubricAgent
from schemas import MCQItem
from utils import configure_dspy_from_env


def call_api(prompt: str, options, context):
    configure_dspy_from_env()

    try:
        data = json.loads(prompt)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in prompt: {e}"}

    try:
        item = MCQItem(
            question_number=data.get("question_number", 1),
            topic=data.get("topic", "General"),
            subtopic=data.get("subtopic"),
            target_cefr=data["target_cefr"],
            target_difficulty=data["target_difficulty"],
            stem=data["stem"],
            options=data["options"],
            correct_answer=data["correct_answer"],
            explanation=data["explanation"],
        )
    except Exception as e:
        return {"error": f"Invalid item data: {e}"}

    language_variant = data.get("language_variant", "British English")

    difficulty_agent = DifficultyAgent()
    rubric_agent = RubricAgent()

    diff = difficulty_agent(item)
    rub = rubric_agent(item, language_variant)

    return {
        "output": json.dumps(
            {
                "difficulty": diff.model_dump(),
                "rubric": rub.model_dump(),
            },
            ensure_ascii=False,
        )
    }
