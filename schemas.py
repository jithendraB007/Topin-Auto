from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


CEFRLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
DifficultyBand = Literal["Easy", "Medium", "Hard"]
LanguageVariant = Literal["British English", "American English"]


CEFR_TO_DIFFICULTY: Dict[str, str] = {
    "A1": "Easy",
    "A2": "Easy",
    "B1": "Medium",
    "B2": "Medium",
    "C1": "Hard",
    "C2": "Hard",
}


class Constraints(BaseModel):
    options_per_question: int = 4
    single_correct_answer: bool = True
    include_explanation: bool = True
    exam_acceptable_language: bool = True
    avoid_ambiguity: bool = True
    language_variant: LanguageVariant = "British English"
    no_duplicate_questions: bool = True


class InputSchema(BaseModel):
    subject: str
    syllabus_unit: str
    topic: str
    subtopics: List[str] = Field(default_factory=list)
    question_type: Literal["MCQ"] = "MCQ"
    total_questions: int = Field(gt=0)
    cefr_distribution: Dict[CEFRLevel, int]
    constraints: Constraints = Field(default_factory=Constraints)
    sample_questions: List[dict] = Field(
        default_factory=list,
        description="Optional example questions to guide the style and format of generation.",
    )

    @model_validator(mode="after")
    def validate_counts(self) -> "InputSchema":
        total = sum(self.cefr_distribution.values())
        if total != self.total_questions:
            raise ValueError(
                f"Sum of cefr_distribution ({total}) must equal total_questions ({self.total_questions})."
            )
        return self


class PlannedQuestion(BaseModel):
    question_number: int
    question_type: Literal["MCQ"] = "MCQ"
    topic: str
    subtopic: Optional[str] = None
    target_cefr: CEFRLevel
    target_difficulty: DifficultyBand
    angle: Optional[str] = None  # e.g. "fill-in-the-blank", "inference", "error-correction"


class MCQItem(BaseModel):
    question_number: int
    topic: str
    subtopic: Optional[str] = None
    target_cefr: CEFRLevel
    target_difficulty: DifficultyBand
    stem: str
    options: List[str]
    correct_answer: str
    explanation: str


class DifficultyResult(BaseModel):
    predicted_cefr: CEFRLevel
    predicted_difficulty: DifficultyBand
    vocabulary_level: str
    grammar_complexity: str
    reasoning_load: str
    distractor_difficulty: str
    alignment: bool
    justification: str
    revision_feedback: str


class RubricResult(BaseModel):
    grammatical_accuracy: str
    spelling: str
    ambiguity: str
    functionality_alignment: str
    instruction_clarity_appropriateness: str
    academic_language_exam_acceptability: str
    option_explanation_consistency: str
    readability: str
    formatting_spacing: str
    punctuation: str
    british_american_english_consistency: str
    overall_decision: Literal["Pass", "Revise", "Fail"]
    priority_reason: str
    revision_feedback: str


class EvaluatedItem(BaseModel):
    item: MCQItem
    difficulty: DifficultyResult
    rubric: RubricResult
    accepted: bool
    revision_attempts: int = 0
