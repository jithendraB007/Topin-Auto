"""MCQ Generator dspy-cli module.

Exposed as a FastAPI endpoint via dspy-cli serve.

Each input field maps directly to a labeled field in the dspy-cli web UI:
  topic                   -- e.g. "English Grammar"
  subtopic                -- e.g. "Present Tense"
  a1_count / a2_count     -- Easy band (A1=beginner, A2=elementary)
  b1_count / b2_count     -- Medium band (B1=intermediate, B2=upper-intermediate)
  c1_count / c2_count     -- Hard band (C1=advanced, C2=mastery)
  questions_per_iteration -- batch size per LLM call (default 5)
  max_iterations          -- max retry loops per CEFR level (default 20)

Output fields:
  questions    -- list of accepted MCQ questions
  summary      -- accepted/rejected counts per difficulty band
  warnings     -- any quota/iteration warnings
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import dspy

from mcq_generator.utils.judges import DifficultyJudgeWrapper, RubricJudgeWrapper
from mcq_generator.utils.models import (
    ExampleQuestion,
    ExampleQuestionSet,
    GenerationConstraints,
    InputSchema,
    SubtopicRequirement,
)
from mcq_generator.utils.orchestrator import MCQGenerationOrchestrator

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[4] / 'data'
_TRAINING_FILE = _DATA_DIR / 'training_dataset_standard.json'


def _load_default_examples() -> ExampleQuestionSet:
    """Load the bundled training dataset as reference examples."""
    if not _TRAINING_FILE.exists():
        return ExampleQuestionSet()
    try:
        with open(_TRAINING_FILE, encoding='utf-8') as f:
            raw = json.load(f)
        items = [
            ExampleQuestion(
                stem=q.get('stem', ''),
                options=q.get('options', []),
                correct_answer=q.get('correct_answer', ''),
                explanation=q.get('explanation', ''),
                difficulty=q.get('target_difficulty'),
                cefr=q.get('target_cefr'),
                subtopic=q.get('subtopic'),
            )
            for q in raw
        ]
        return ExampleQuestionSet(items=items)
    except Exception as e:
        logger.warning(f'Could not load default examples: {e}')
        return ExampleQuestionSet()


class MCQGeneratorCoT(dspy.Module):
    """Schema-driven MCQ generator with difficulty and rubric judges.

    Specify how many questions you want at each CEFR level:
      A1/A2 = Easy  |  B1/B2 = Medium  |  C1/C2 = Hard

    Reference examples are loaded automatically from the bundled training dataset.
    """

    def __init__(self):
        super().__init__()
        self._difficulty_judge = DifficultyJudgeWrapper()
        self._rubric_judge = RubricJudgeWrapper(language_variant='British English')

    def forward(
        self,
        topic: str,
        subtopic: str,
        a1_count: int = 0,
        a2_count: int = 0,
        b1_count: int = 0,
        b2_count: int = 0,
        c1_count: int = 0,
        c2_count: int = 0,
        questions_per_iteration: int = 5,
        max_iterations: int = 20,
    ) -> dspy.Prediction:
        """Generate MCQ questions for the given topic and subtopic.

        Args:
            topic: Subject area (e.g. 'English Grammar')
            subtopic: Specific subtopic (e.g. 'Present Tense')
            a1_count: Number of A1-level questions (Easy / Beginner)
            a2_count: Number of A2-level questions (Easy / Elementary)
            b1_count: Number of B1-level questions (Medium / Intermediate)
            b2_count: Number of B2-level questions (Medium / Upper-Intermediate)
            c1_count: Number of C1-level questions (Hard / Advanced)
            c2_count: Number of C2-level questions (Hard / Mastery)
            questions_per_iteration: Batch size per LLM call (default 5)
            max_iterations: Max retry loops per CEFR level (default 20)

        Returns:
            dspy.Prediction with:
              questions (list[dict]) -- all accepted MCQ questions
              summary   (dict)       -- accepted/rejected counts by difficulty
              warnings  (list[str])  -- quota/iteration warnings if any
        """
        schema = InputSchema(
            topic=topic,
            subtopics=[
                SubtopicRequirement(
                    subtopic=subtopic,
                    a1_count=a1_count,
                    a2_count=a2_count,
                    b1_count=b1_count,
                    b2_count=b2_count,
                    c1_count=c1_count,
                    c2_count=c2_count,
                )
            ],
            constraints=GenerationConstraints(
                questions_per_iteration=questions_per_iteration,
                max_iterations_per_difficulty=max_iterations,
            ),
        )

        example_questions = _load_default_examples()

        orchestrator = MCQGenerationOrchestrator(
            difficulty_judge=self._difficulty_judge,
            rubric_judge=self._rubric_judge,
        )
        result = orchestrator.run(schema=schema, example_questions=example_questions)

        questions = [item.model_dump() for item in result.store.all_items()]
        summary = {
            'easy': {
                'accepted': result.store.count('Easy'),
                'rejected': sum(1 for r in result.rejected if r.get('difficulty') == 'Easy'),
            },
            'medium': {
                'accepted': result.store.count('Medium'),
                'rejected': sum(1 for r in result.rejected if r.get('difficulty') == 'Medium'),
            },
            'hard': {
                'accepted': result.store.count('Hard'),
                'rejected': sum(1 for r in result.rejected if r.get('difficulty') == 'Hard'),
            },
            'total_accepted': len(questions),
            'total_rejected': len(result.rejected),
        }

        return dspy.Prediction(
            questions=questions,
            summary=summary,
            warnings=result.warnings,
        )
