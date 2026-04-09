"""Pydantic models for input/output of the MCQ generator pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Input Models ──────────────────────────────────────────────────────────────

class SubtopicRequirement(BaseModel):
    """Per-subtopic question counts for each of the 6 CEFR levels.

    CEFR mapping:
        A1, A2  ->  Easy
        B1, B2  ->  Medium
        C1, C2  ->  Hard
    """
    subtopic: str

    a1_count: int = 0
    a2_count: int = 0
    b1_count: int = 0
    b2_count: int = 0
    c1_count: int = 0
    c2_count: int = 0

    @property
    def easy_count(self) -> int:
        return self.a1_count + self.a2_count

    @property
    def medium_count(self) -> int:
        return self.b1_count + self.b2_count

    @property
    def hard_count(self) -> int:
        return self.c1_count + self.c2_count

    @property
    def total(self) -> int:
        return self.easy_count + self.medium_count + self.hard_count


class GenerationConstraints(BaseModel):
    questions_per_iteration: int = 5
    max_iterations_per_difficulty: int = 20


class InputSchema(BaseModel):
    topic: str
    subtopics: list[SubtopicRequirement]
    constraints: GenerationConstraints = Field(default_factory=GenerationConstraints)


class ExampleQuestion(BaseModel):
    stem: str
    options: list[str]
    correct_answer: str
    explanation: str
    difficulty: str | None = None
    cefr: str | None = None
    subtopic: str | None = None


class ExampleQuestionSet(BaseModel):
    items: list[ExampleQuestion] = Field(default_factory=list)

    def filter_examples(
        self,
        *,
        subtopic: str,
        difficulty: str,
        cefr: str | None = None,
    ) -> list[ExampleQuestion]:
        if cefr:
            p1 = [q for q in self.items
                  if q.cefr == cefr and q.subtopic in (None, subtopic)]
            if p1:
                return p1

        p2 = [q for q in self.items
              if q.difficulty in (None, difficulty)
              and q.subtopic in (None, subtopic)]
        if p2:
            return p2

        p3 = [q for q in self.items if q.difficulty in (None, difficulty)]
        if p3:
            return p3

        return self.items


# ── Generator Request/Response Models ─────────────────────────────────────────

class GenerationRequest(BaseModel):
    topic: str
    subtopic: str
    target_cefr: str
    target_difficulty: str
    example_questions: list[ExampleQuestion]
    already_used_stems: list[str]
    batch_size: int


class GeneratedQuestion(BaseModel):
    stem: str
    options: list[str]
    correct_answer: str
    explanation: str


class GenerationBatch(BaseModel):
    questions: list[GeneratedQuestion]


# ── Output Models ─────────────────────────────────────────────────────────────

class MCQItem(BaseModel):
    question_number: int
    topic: str
    subtopic: str | None
    target_cefr: str
    target_difficulty: str
    stem: str
    options: list[str]
    correct_answer: str
    explanation: str


class QuestionStore(BaseModel):
    easy: list[MCQItem] = Field(default_factory=list)
    medium: list[MCQItem] = Field(default_factory=list)
    hard: list[MCQItem] = Field(default_factory=list)

    def add(self, item: MCQItem) -> None:
        difficulty = item.target_difficulty.lower()
        if difficulty == 'easy':
            self.easy.append(item)
        elif difficulty == 'medium':
            self.medium.append(item)
        else:
            self.hard.append(item)

    def count(self, difficulty: str) -> int:
        return len(getattr(self, difficulty.lower(), []))

    def count_by_cefr(self, cefr: str) -> int:
        return sum(
            1 for q in self.easy + self.medium + self.hard
            if q.target_cefr == cefr
        )

    def get_used_stems(self) -> list[str]:
        return [q.stem for q in self.easy + self.medium + self.hard]

    def all_items(self) -> list[MCQItem]:
        return self.easy + self.medium + self.hard


class MCQGenerationResult(BaseModel):
    store: QuestionStore
    rejected: list[dict]
    warnings: list[str]
