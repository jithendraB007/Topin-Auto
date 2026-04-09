"""Difficulty and Rubric judges for MCQ validation.

Loads optimised DSPy agent weights from artifacts/ when available,
falls back to unoptimised agents otherwise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import dspy
from pydantic import BaseModel

from mcq_generator.utils.models import MCQItem

logger = logging.getLogger(__name__)

# Artifacts live two levels above this file: cli/mcq-generator/artifacts/
_HERE = Path(__file__).resolve()
# Walk up: utils -> mcq_generator -> src -> mcq-generator -> artifacts
_ARTIFACTS_DIR = _HERE.parents[3] / 'artifacts'


# ── DifficultyJudge models ────────────────────────────────────────────────────

class Question(BaseModel):
    question_id: str
    stem: str
    options: list[str]
    correct_answer: str
    explanation: str
    target_cefr: str
    target_difficulty: str


class DifficultyResult(BaseModel):
    question_id: str
    predicted_cefr: Literal['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    predicted_difficulty: Literal['Easy', 'Medium', 'Hard']


class DifficultyOutput(BaseModel):
    results: list[DifficultyResult]


class SimpleDifficultySignature(dspy.Signature):
    """Classify a list of MCQ questions by CEFR level and difficulty.
    For each question analyse vocabulary, grammar, reasoning load, and distractor difficulty.
    A1/A2 -> Easy | B1/B2 -> Medium | C1/C2 -> Hard.
    Return one DifficultyResult per question in the same order.
    """
    questions: list[Question] = dspy.InputField(desc='List of MCQ questions to classify')
    output: DifficultyOutput = dspy.OutputField(desc='Classification results for all questions')


class SimpleDifficultyAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(SimpleDifficultySignature)

    def forward(self, questions: list[Question]) -> dspy.Prediction:
        return self.judge(questions=questions)


# ── RubricJudge models ────────────────────────────────────────────────────────

class RubricQuestion(BaseModel):
    question_id: str
    stem: str
    options: list[str]
    correct_answer: str
    explanation: str
    target_cefr: str
    target_difficulty: str
    language_variant: str


class RubricResult(BaseModel):
    question_id: str
    grammatical_accuracy: Literal['No Issues', 'Minor Issues', 'Major Issues']
    spelling: Literal['No Issues', 'Minor Issues', 'Major Issues']
    ambiguity: Literal['No Issue', 'Minor Issue', 'Major Issue']
    functionality_alignment: Literal['Aligned', 'Partially Aligned', 'Not Aligned']
    instruction_clarity_appropriateness: Literal['Clear', 'Needs Improvement', 'Unclear']
    academic_language_exam_acceptability: Literal['Acceptable', 'Needs Improvement', 'Not Acceptable']
    option_explanation_consistency: Literal['Consistent', 'Inconsistent']
    readability: Literal['Good', 'Needs Improvement', 'Poor']
    formatting_spacing: Literal['No Issues', 'Minor Issues', 'Major Issues']
    punctuation: Literal['No Issues', 'Minor Issues', 'Major Issues']
    british_american_english_consistency: Literal['Consistent', 'Inconsistent']
    overall_decision: Literal['Pass', 'Revise', 'Fail']
    priority_reason: str
    revision_feedback: str


class RubricOutput(BaseModel):
    results: list[RubricResult]


class RubricJudgeSignature(dspy.Signature):
    """Evaluate a list of MCQ questions using the rubric.
    ambiguity is highest priority -- Major Issue forces Fail.
    Return one RubricResult per question in the same order.
    """
    questions: list[RubricQuestion] = dspy.InputField(desc='List of MCQ questions to evaluate')
    output: RubricOutput = dspy.OutputField(desc='Rubric evaluation results for all questions')


class RubricJudgeAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(RubricJudgeSignature)

    def forward(self, questions: list[RubricQuestion]) -> dspy.Prediction:
        return self.judge(questions=questions)


# ── Load optimised agents ─────────────────────────────────────────────────────

def _load_difficulty_agent() -> SimpleDifficultyAgent:
    agent = SimpleDifficultyAgent()
    artifact = _ARTIFACTS_DIR / 'simple_difficulty_optimized.json'
    if artifact.exists():
        agent.load(str(artifact))
        logger.info(f'DifficultyJudge loaded from: {artifact}')
    else:
        logger.info(f'DifficultyJudge: using unoptimised (artifact not found: {artifact})')
    return agent


def _load_rubric_agent() -> RubricJudgeAgent:
    agent = RubricJudgeAgent()
    artifact = _ARTIFACTS_DIR / 'rubric_agent_optimized.json'
    if artifact.exists():
        agent.load(str(artifact))
        logger.info(f'RubricJudge loaded from: {artifact}')
    else:
        logger.info(f'RubricJudge: using unoptimised (artifact not found: {artifact})')
    return agent


# ── Batch judge wrappers ──────────────────────────────────────────────────────

class DifficultyJudgeWrapper:
    """Wraps SimpleDifficultyAgent for batch evaluation of MCQItems."""

    def __init__(self):
        self._agent = _load_difficulty_agent()

    def __call__(
        self,
        *,
        items: list[MCQItem],
        expected_difficulty: str,
    ) -> list[SimpleNamespace]:
        questions = [
            Question(
                question_id=str(item.question_number),
                stem=item.stem,
                options=item.options,
                correct_answer=item.correct_answer,
                explanation=item.explanation,
                target_cefr=item.target_cefr,
                target_difficulty=item.target_difficulty,
            )
            for item in items
        ]
        pred = self._agent(questions=questions)
        return [
            SimpleNamespace(
                passed=res.predicted_difficulty.lower() == expected_difficulty.lower(),
                reason=(
                    f'predicted_cefr={res.predicted_cefr} '
                    f'predicted_difficulty={res.predicted_difficulty}'
                ),
            )
            for res in pred.output.results
        ]


class RubricJudgeWrapper:
    """Wraps RubricJudgeAgent for batch evaluation of MCQItems."""

    def __init__(self, language_variant: str = 'British English'):
        self.language_variant = language_variant
        self._agent = _load_rubric_agent()

    def __call__(self, *, items: list[MCQItem]) -> list[SimpleNamespace]:
        questions = [
            RubricQuestion(
                question_id=str(item.question_number),
                stem=item.stem,
                options=item.options,
                correct_answer=item.correct_answer,
                explanation=item.explanation,
                target_cefr=item.target_cefr,
                target_difficulty=item.target_difficulty,
                language_variant=self.language_variant,
            )
            for item in items
        ]
        pred = self._agent(questions=questions)
        return [
            SimpleNamespace(
                passed=res.overall_decision == 'Pass',
                reason=res.priority_reason,
            )
            for res in pred.output.results
        ]
