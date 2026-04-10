#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCQ Generator CLI
=================
Interactive command-line interface for the MCQ generation pipeline.

Usage:
    python cli/mcq_generator_cli.py                    # interactive prompts
    python cli/mcq_generator_cli.py --schema schema.json  # load schema from file
    python cli/mcq_generator_cli.py --help

Run from the project root (d:/Topin) or from any subdirectory.
"""

import sys
import os
import json
import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Bootstrap: locate project root and inject venv
# ---------------------------------------------------------------------------

def _bootstrap():
    candidate = Path(__file__).resolve().parent.parent
    if not (candidate / "utils.py").exists():
        candidate = Path().resolve()
        if not (candidate / "utils.py").exists():
            if (candidate.parent / "utils.py").exists():
                candidate = candidate.parent
            else:
                raise RuntimeError(
                    "Cannot locate project root. Run from d:/Topin or a subdirectory."
                )
    root = candidate

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    venv_sp = root / ".venv" / "Lib" / "site-packages"
    if not venv_sp.exists():
        venv_sp = root / ".venv" / "lib" / "site-packages"
    if venv_sp.exists() and str(venv_sp) not in sys.path:
        sys.path.insert(0, str(venv_sp))

    return root


PROJECT_ROOT = _bootstrap()
DATA_DIR      = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

import dspy
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class SubtopicRequirement(BaseModel):
    subtopic:  str
    a1_count:  int = 0
    a2_count:  int = 0
    b1_count:  int = 0
    b2_count:  int = 0
    c1_count:  int = 0
    c2_count:  int = 0

    @property
    def easy_count(self) -> int:   return self.a1_count + self.a2_count
    @property
    def medium_count(self) -> int: return self.b1_count + self.b2_count
    @property
    def hard_count(self) -> int:   return self.c1_count + self.c2_count
    @property
    def total(self) -> int:
        return self.easy_count + self.medium_count + self.hard_count


class GenerationConstraints(BaseModel):
    questions_per_iteration:       int = 5
    max_iterations_per_difficulty: int = 20


class InputSchema(BaseModel):
    topic:       str
    subtopics:   list[SubtopicRequirement]
    constraints: GenerationConstraints = Field(default_factory=GenerationConstraints)


class ExampleQuestion(BaseModel):
    stem:           str
    options:        list[str]
    correct_answer: str
    explanation:    str
    difficulty:     str | None = None
    cefr:           str | None = None
    subtopic:       str | None = None


class ExampleQuestionSet(BaseModel):
    items: list[ExampleQuestion] = Field(default_factory=list)

    def filter_examples(self, *, subtopic, difficulty, cefr=None):
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


# ---------------------------------------------------------------------------
# Output Models
# ---------------------------------------------------------------------------

class MCQItem(BaseModel):
    question_number:   int
    topic:             str
    subtopic:          str | None
    target_cefr:       str
    target_difficulty: str
    stem:              str
    options:           list[str]
    correct_answer:    str
    explanation:       str


class QuestionStore(BaseModel):
    easy:   list[MCQItem] = Field(default_factory=list)
    medium: list[MCQItem] = Field(default_factory=list)
    hard:   list[MCQItem] = Field(default_factory=list)

    def add(self, item: MCQItem) -> None:
        d = item.target_difficulty.strip().lower()
        if d == "easy":   self.easy.append(item)
        elif d == "medium": self.medium.append(item)
        elif d == "hard":   self.hard.append(item)
        else: raise ValueError(f"Unknown difficulty: {item.target_difficulty}")

    def get_used_stems(self) -> list[str]:
        return [q.stem for q in self.easy + self.medium + self.hard]

    def count(self, difficulty: str) -> int:
        d = difficulty.strip().lower()
        if d == "easy":   return len(self.easy)
        if d == "medium": return len(self.medium)
        if d == "hard":   return len(self.hard)
        raise ValueError(f"Unknown difficulty: {difficulty}")

    def count_by_cefr(self, cefr: str) -> int:
        return sum(1 for q in self.easy + self.medium + self.hard
                   if q.target_cefr == cefr)

    def all_items(self) -> list[MCQItem]:
        return self.easy + self.medium + self.hard


class MCQGenerationResult(BaseModel):
    store:    QuestionStore
    rejected: list[dict] = Field(default_factory=list)
    warnings: list[str]  = Field(default_factory=list)


# ---------------------------------------------------------------------------
# DSPy Signature + Generator Agent
# ---------------------------------------------------------------------------

class GenerationRequest(BaseModel):
    topic:              str
    subtopic:           str
    target_cefr:        str
    target_difficulty:  str
    example_questions:  list[ExampleQuestion]
    already_used_stems: list[str]
    batch_size:         int


class GeneratedQuestion(BaseModel):
    stem:           str
    options:        list[str]
    correct_answer: str
    explanation:    str


class GenerationBatch(BaseModel):
    questions: list[GeneratedQuestion]


class MCQGeneratorSignature(dspy.Signature):
    """Generate a batch of high-quality MCQ questions for the given topic and difficulty.

    Use the example_questions as a reference for style, vocabulary level, and format.
    Easy examples show A1/A2-level vocabulary; Medium show B1/B2; Hard show C1/C2.

    STRICT RULES:
    - EXACTLY 4 options — no more, no less
    - correct_answer must be copied VERBATIM from one of the options
    - Each stem must be completely different from all stems in already_used_stems
    - Only one answer can be correct; the other 3 are plausible but clearly wrong
    - Align vocabulary and grammar complexity to the target_cefr level
    - Avoid ambiguity — only one option can be correct

    Return exactly batch_size questions in the output.
    """
    request: GenerationRequest = dspy.InputField(
        desc="Batch generation parameters: topic, CEFR level, difficulty, reference examples, used stems"
    )
    output: GenerationBatch = dspy.OutputField(
        desc="Batch of generated MCQ questions, one per item in questions list"
    )


class MCQGeneratorAgent(dspy.Module):
    def __init__(self, *, store, difficulty_judge, rubric_judge):
        super().__init__()
        self.generate         = dspy.ChainOfThought(MCQGeneratorSignature)
        self.store            = store
        self.difficulty_judge = difficulty_judge
        self.rubric_judge     = rubric_judge

    def forward(self, *, schema, example_questions, topic, subtopic,
                target_cefr, target_difficulty, target_count, rejected, warnings):
        if target_count <= 0:
            return

        iteration = 0
        q_number  = len(self.store.all_items()) + 1

        print(f"\n  [{target_difficulty} / {target_cefr}]  target={target_count}  subtopic={subtopic!r}")

        while self.store.count_by_cefr(target_cefr) < target_count:
            iteration += 1
            if iteration > schema.constraints.max_iterations_per_difficulty:
                msg = (f"{target_difficulty}/{target_cefr}: max iterations "
                       f"({schema.constraints.max_iterations_per_difficulty}) reached. "
                       f"Accepted {self.store.count_by_cefr(target_cefr)}/{target_count}.")
                warnings.append(msg)
                print(f"    [WARNING] {msg}")
                break

            needed   = target_count - self.store.count_by_cefr(target_cefr)
            batch_sz = min(schema.constraints.questions_per_iteration, needed)
            relevant = example_questions.filter_examples(
                subtopic=subtopic or "",
                difficulty=target_difficulty,
                cefr=target_cefr,
            )
            request = GenerationRequest(
                topic=topic,
                subtopic=subtopic or "",
                target_cefr=target_cefr,
                target_difficulty=target_difficulty,
                example_questions=relevant,
                already_used_stems=self.store.get_used_stems(),
                batch_size=batch_sz,
            )

            # 1. Generate
            print(f"    [Iter {iteration}] Generating {batch_sz} questions...", flush=True)
            try:
                pred  = self.generate(request=request)
                batch = pred.output.questions
            except Exception as e:
                rejected.append({"stage": "generation", "cefr": target_cefr,
                                  "difficulty": target_difficulty,
                                  "iteration": iteration, "error": str(e)})
                print(f"    [Gen Error] {e}")
                continue

            # 2. Hard-validate
            valid_items = []
            for q in batch:
                errs = hard_validate(q)
                if errs:
                    rejected.append({"stage": "hard_validate", "cefr": target_cefr,
                                      "difficulty": target_difficulty,
                                      "iteration": iteration, "stem": q.stem[:60], "errors": errs})
                    print(f"    [Hard Fail] {errs}")
                    continue
                valid_items.append(MCQItem(
                    question_number=q_number, topic=topic, subtopic=subtopic,
                    target_cefr=target_cefr, target_difficulty=target_difficulty,
                    stem=q.stem, options=q.options,
                    correct_answer=q.correct_answer, explanation=q.explanation,
                ))
                q_number += 1

            if not valid_items:
                print(f"    [Skip] All {len(batch)} failed hard validation.")
                continue
            print(f"    [Hard]  {len(valid_items)}/{len(batch)} passed.")

            # 3. DifficultyJudge
            try:
                diff_results = self.difficulty_judge(
                    items=valid_items, expected_difficulty=target_difficulty)
            except Exception as e:
                rejected.append({"stage": "difficulty_judge_error", "error": str(e)})
                print(f"    [Diff Error] {e}")
                continue

            diff_passed = []
            for item, diff in zip(valid_items, diff_results):
                if diff.passed:
                    diff_passed.append(item)
                else:
                    rejected.append({"stage": "difficulty", "q": item.question_number,
                                      "cefr": target_cefr, "difficulty": target_difficulty,
                                      "reason": diff.reason, "stem": item.stem[:60]})
                    print(f"    [Diff Fail] Q{item.question_number}: {diff.reason}")

            print(f"    [Diff]  {len(diff_passed)}/{len(valid_items)} passed.")
            if not diff_passed:
                continue

            # 4. RubricJudge
            try:
                rub_results = self.rubric_judge(items=diff_passed)
            except Exception as e:
                rejected.append({"stage": "rubric_judge_error", "error": str(e)})
                print(f"    [Rub Error] {e}")
                continue

            accepted_count = 0
            for item, rub in zip(diff_passed, rub_results):
                if not rub.passed:
                    rejected.append({"stage": "rubric", "q": item.question_number,
                                      "cefr": target_cefr, "difficulty": target_difficulty,
                                      "reason": rub.reason, "stem": item.stem[:60]})
                    print(f"    [Rub  Fail] Q{item.question_number}: {rub.reason}")
                    continue
                self.store.add(item)
                accepted_count += 1
                total = self.store.count_by_cefr(target_cefr)
                print(f"    [Accepted]  Q{item.question_number} "
                      f"({target_cefr}/{target_difficulty}) {total}/{target_count}")
                if total >= target_count:
                    break

            print(f"    [Rubric] {accepted_count}/{len(diff_passed)} passed.")


# ---------------------------------------------------------------------------
# Hard Validate
# ---------------------------------------------------------------------------

def hard_validate(q: GeneratedQuestion) -> list:
    errors = []
    if not isinstance(q.options, list) or len(q.options) != 4:
        errors.append(f"Expected 4 options, got {len(q.options) if isinstance(q.options, list) else type(q.options).__name__}")
    if q.correct_answer not in q.options:
        errors.append(f'correct_answer "{q.correct_answer}" not in options')
    if not q.explanation or not q.explanation.strip():
        errors.append("explanation is empty")
    if not q.stem or not q.stem.strip():
        errors.append("stem is empty")
    return errors


# ---------------------------------------------------------------------------
# Judge Models + Wrappers
# ---------------------------------------------------------------------------

class Question(BaseModel):
    question_id: str; stem: str; options: list[str]
    correct_answer: str; explanation: str
    target_cefr: str; target_difficulty: str

class DifficultyResult(BaseModel):
    question_id:          str
    predicted_cefr:       Literal["A1","A2","B1","B2","C1","C2"]
    predicted_difficulty: Literal["Easy","Medium","Hard"]

class DifficultyOutput(BaseModel):
    results: list[DifficultyResult]

class SimpleDifficultySignature(dspy.Signature):
    """Classify MCQ questions by CEFR level and difficulty.
    A1/A2 -> Easy | B1/B2 -> Medium | C1/C2 -> Hard.
    Return one DifficultyResult per question in the same order.
    """
    questions: list[Question]   = dspy.InputField(desc="MCQ questions to classify")
    output:    DifficultyOutput = dspy.OutputField(desc="Classification results")

class SimpleDifficultyAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(SimpleDifficultySignature)
    def forward(self, questions):
        return self.judge(questions=questions)


class RubricQuestion(BaseModel):
    question_id: str; stem: str; options: list[str]
    correct_answer: str; explanation: str
    target_cefr: str; target_difficulty: str; language_variant: str

class RubricResult(BaseModel):
    question_id:                          str
    grammatical_accuracy:                 Literal["No Issues","Minor Issues","Major Issues"]
    spelling:                             Literal["No Issues","Minor Issues","Major Issues"]
    ambiguity:                            Literal["No Issue","Minor Issue","Major Issue"]
    functionality_alignment:              Literal["Aligned","Partially Aligned","Not Aligned"]
    instruction_clarity_appropriateness:  Literal["Clear","Needs Improvement","Unclear"]
    academic_language_exam_acceptability: Literal["Acceptable","Needs Improvement","Not Acceptable"]
    option_explanation_consistency:       Literal["Consistent","Inconsistent"]
    readability:                          Literal["Good","Needs Improvement","Poor"]
    formatting_spacing:                   Literal["No Issues","Minor Issues","Major Issues"]
    punctuation:                          Literal["No Issues","Minor Issues","Major Issues"]
    british_american_english_consistency: Literal["Consistent","Inconsistent"]
    overall_decision:                     Literal["Pass","Revise","Fail"]
    priority_reason:                      str
    revision_feedback:                    str

class RubricOutput(BaseModel):
    results: list[RubricResult]

class RubricJudgeSignature(dspy.Signature):
    """Evaluate MCQ questions using the rubric.
    ambiguity is highest priority — Major Issue forces Fail.
    Return one RubricResult per question in the same order.
    """
    questions: list[RubricQuestion] = dspy.InputField(desc="MCQ questions to evaluate")
    output:    RubricOutput         = dspy.OutputField(desc="Rubric evaluation results")

class RubricJudgeAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(RubricJudgeSignature)
    def forward(self, questions):
        return self.judge(questions=questions)


class DifficultyJudgeWrapper:
    def __call__(self, *, items, expected_difficulty):
        questions = [
            Question(question_id=str(item.question_number), stem=item.stem,
                     options=item.options, correct_answer=item.correct_answer,
                     explanation=item.explanation, target_cefr=item.target_cefr,
                     target_difficulty=item.target_difficulty)
            for item in items
        ]
        pred = _diff_agent(questions=questions)
        return [
            SimpleNamespace(
                passed=res.predicted_difficulty.lower() == expected_difficulty.lower(),
                reason=f"predicted_cefr={res.predicted_cefr} predicted_difficulty={res.predicted_difficulty}",
            )
            for res in pred.output.results
        ]


class RubricJudgeWrapper:
    def __init__(self, language_variant="British English"):
        self.language_variant = language_variant

    def __call__(self, *, items):
        questions = [
            RubricQuestion(question_id=str(item.question_number), stem=item.stem,
                           options=item.options, correct_answer=item.correct_answer,
                           explanation=item.explanation, target_cefr=item.target_cefr,
                           target_difficulty=item.target_difficulty,
                           language_variant=self.language_variant)
            for item in items
        ]
        pred = _rub_agent(questions=questions)
        return [
            SimpleNamespace(passed=res.overall_decision == "Pass", reason=res.priority_reason)
            for res in pred.output.results
        ]


# ---------------------------------------------------------------------------
# Per-Difficulty Agents + Orchestrator
# ---------------------------------------------------------------------------

_CEFR_LEVELS = ["A1","A2","B1","B2","C1","C2"]
_CEFR_TO_DIFFICULTY = {
    "A1":"Easy","A2":"Easy",
    "B1":"Medium","B2":"Medium",
    "C1":"Hard","C2":"Hard",
}

class BaseDifficultyAgent:
    difficulty_name: str = ""
    default_cefr:    str = ""

    def __init__(self, *, generator):
        self.generator = generator

    def generate_questions(self, *, schema, example_questions, topic,
                           subtopic, target_cefr, target_count, rejected, warnings):
        self.generator(
            schema=schema, example_questions=example_questions,
            topic=topic, subtopic=subtopic, target_cefr=target_cefr,
            target_difficulty=self.difficulty_name, target_count=target_count,
            rejected=rejected, warnings=warnings,
        )


class EasyMCQAgent(BaseDifficultyAgent):
    difficulty_name = "Easy";  default_cefr = "A2"

class MediumMCQAgent(BaseDifficultyAgent):
    difficulty_name = "Medium"; default_cefr = "B1"

class HardMCQAgent(BaseDifficultyAgent):
    difficulty_name = "Hard";  default_cefr = "B2"


class MCQGenerationOrchestrator:
    def __init__(self, *, difficulty_judge, rubric_judge):
        self.store     = QuestionStore()
        self.generator = MCQGeneratorAgent(
            store=self.store,
            difficulty_judge=difficulty_judge,
            rubric_judge=rubric_judge,
        )
        self.easy_agent   = EasyMCQAgent(generator=self.generator)
        self.medium_agent = MediumMCQAgent(generator=self.generator)
        self.hard_agent   = HardMCQAgent(generator=self.generator)
        self._agents = {
            "Easy":   self.easy_agent,
            "Medium": self.medium_agent,
            "Hard":   self.hard_agent,
        }

    def run(self, *, schema, example_questions):
        rejected = []
        warnings = []
        for req in schema.subtopics:
            SEP = "=" * 60
            print(f"\n{SEP}")
            print(f"  Topic: {schema.topic}  |  Subtopic: {req.subtopic}")
            print(f"  Easy:   A1={req.a1_count}  A2={req.a2_count}")
            print(f"  Medium: B1={req.b1_count}  B2={req.b2_count}")
            print(f"  Hard:   C1={req.c1_count}  C2={req.c2_count}")
            print(SEP)
            for cefr in _CEFR_LEVELS:
                target_count = getattr(req, cefr.lower() + "_count")
                if target_count == 0:
                    continue
                difficulty = _CEFR_TO_DIFFICULTY[cefr]
                self._agents[difficulty].generate_questions(
                    schema=schema, example_questions=example_questions,
                    topic=schema.topic, subtopic=req.subtopic,
                    target_cefr=cefr, target_count=target_count,
                    rejected=rejected, warnings=warnings,
                )
        return MCQGenerationResult(store=self.store, rejected=rejected, warnings=warnings)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _prompt(label, default=None, cast=str):
    """Print a prompt and return cast value; use default on empty input."""
    if default is not None:
        disp = f"  {label} [{default}]: "
    else:
        disp = f"  {label}: "
    while True:
        raw = input(disp).strip()
        if not raw:
            if default is not None:
                return cast(default)
            print("    (required — please enter a value)")
            continue
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"    Expected {cast.__name__}, got: {raw!r}")


def _prompt_int(label, default=0):
    return _prompt(label, default=default, cast=int)


def _build_schema_interactive() -> tuple[InputSchema, ExampleQuestionSet]:
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  MCQ Generator — Input Schema Builder")
    print(SEP)

    topic = _prompt("Topic (e.g. 'English Grammar')")

    subtopics = []
    while True:
        print()
        st_name = _prompt("  Subtopic name (e.g. 'Present Tense')")
        print(f"  Enter question counts for each CEFR level (0 = skip):")
        req = SubtopicRequirement(
            subtopic=st_name,
            a1_count=_prompt_int("    A1 (Easy / beginner)"),
            a2_count=_prompt_int("    A2 (Easy / elementary)"),
            b1_count=_prompt_int("    B1 (Medium / intermediate)"),
            b2_count=_prompt_int("    B2 (Medium / upper-intermediate)"),
            c1_count=_prompt_int("    C1 (Hard / advanced)"),
            c2_count=_prompt_int("    C2 (Hard / proficient)"),
        )
        subtopics.append(req)
        print(f"  Total for '{st_name}': {req.total} questions")
        again = _prompt("  Add another subtopic? (y/n)", default="n").lower()
        if again != "y":
            break

    print()
    batch_size  = _prompt_int("  Questions per batch (batch_size)", default=5)
    max_iters   = _prompt_int("  Max iterations per CEFR level", default=10)

    schema = InputSchema(
        topic=topic,
        subtopics=subtopics,
        constraints=GenerationConstraints(
            questions_per_iteration=batch_size,
            max_iterations_per_difficulty=max_iters,
        ),
    )

    # Example questions — load from training dataset or skip
    example_questions = ExampleQuestionSet()
    train_path = DATA_DIR / "mcq" / "training_dataset_standard.json"
    if train_path.exists():
        use_examples = _prompt(
            f"  Load reference examples from data/mcq/training_dataset_standard.json? (y/n)",
            default="y",
        ).lower()
        if use_examples == "y":
            DIFF_MAP = {"A1":"Easy","A2":"Easy","B1":"Medium","B2":"Medium","C1":"Hard","C2":"Hard"}
            rows = json.loads(train_path.read_text(encoding="utf-8"))
            items = []
            for r in rows:
                cefr = r.get("target_cefr","")
                items.append(ExampleQuestion(
                    stem=r["stem"],
                    options=r.get("options", []),
                    correct_answer=r.get("correct_answer",""),
                    explanation=r.get("explanation",""),
                    difficulty=DIFF_MAP.get(cefr,""),
                    cefr=cefr,
                    subtopic=None,
                ))
            example_questions = ExampleQuestionSet(items=items)
            print(f"  Loaded {len(items)} reference examples.")

    return schema, example_questions


def _load_schema_from_file(path: str) -> tuple[InputSchema, ExampleQuestionSet]:
    """Load schema from a JSON file.

    Expected format:
    {
      "topic": "English Grammar",
      "subtopics": [
        {"subtopic": "Present Tense", "a1_count": 2, "b1_count": 3, ...}
      ],
      "constraints": {"questions_per_iteration": 5, "max_iterations_per_difficulty": 10},
      "example_questions": [
        {"stem": "...", "options": [...], "correct_answer": "...", "explanation": "..."}
      ]
    }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = InputSchema(
        topic=data["topic"],
        subtopics=[SubtopicRequirement(**s) for s in data.get("subtopics", [])],
        constraints=GenerationConstraints(**data.get("constraints", {})),
    )
    raw_examples = data.get("example_questions", [])
    example_questions = ExampleQuestionSet(
        items=[ExampleQuestion(**e) for e in raw_examples]
    )
    return schema, example_questions


def _print_results(result: MCQGenerationResult, schema: InputSchema):
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  GENERATION COMPLETE")
    print(SEP)

    for req in schema.subtopics:
        print(f"\n  Subtopic: {req.subtopic}")
        for cefr in _CEFR_LEVELS:
            target = getattr(req, cefr.lower() + "_count")
            if target == 0:
                continue
            accepted = result.store.count_by_cefr(cefr)
            status   = "OK" if accepted >= target else "PARTIAL"
            diff     = _CEFR_TO_DIFFICULTY[cefr]
            print(f"    {cefr} ({diff:<6}): {accepted}/{target}  [{status}]")

    total = len(result.store.all_items())
    print(f"\n  Total accepted : {total}")
    print(f"  Total rejected : {len(result.rejected)}")
    if result.warnings:
        print("  Warnings:")
        for w in result.warnings:
            print(f"    {w}")

    if total == 0:
        return

    print(f"\n  {'Q#':<4} {'CEFR':<6} {'Difficulty':<10} Stem preview")
    print("  " + "-" * 72)
    for item in result.store.all_items():
        stem_preview = item.stem.replace("\n", " ")[:52]
        print(f"  {item.question_number:<4} {item.target_cefr:<6} "
              f"{item.target_difficulty:<10} {stem_preview}...")

    # Full questions
    print(f"\n{SEP}")
    print("  FULL QUESTIONS")
    print(SEP)
    for item in result.store.all_items():
        print(f"\n  Q{item.question_number}. [{item.target_cefr} / {item.target_difficulty}]")
        print(f"  {item.stem}")
        for j, opt in enumerate(item.options):
            marker = "*" if opt == item.correct_answer else " "
            print(f"    {marker} {chr(65+j)}) {opt}")
        print(f"  Correct: {item.correct_answer}")
        print(f"  Explanation: {item.explanation}")


def _save_results(result: MCQGenerationResult, schema: InputSchema, out_path: Path):
    output = {
        "schema": schema.model_dump(),
        "summary": {
            cefr: {
                "accepted": result.store.count_by_cefr(cefr),
                "target": sum(
                    getattr(req, cefr.lower() + "_count")
                    for req in schema.subtopics
                ),
            }
            for cefr in _CEFR_LEVELS
        },
        "questions": {
            "easy":   [q.model_dump() for q in result.store.easy],
            "medium": [q.model_dump() for q in result.store.medium],
            "hard":   [q.model_dump() for q in result.store.hard],
        },
        "rejected": result.rejected,
        "warnings": result.warnings,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Output saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MCQ Generator CLI — generate MCQ questions from a schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for first-time use)
  python cli/mcq_generator_cli.py

  # Load schema from a JSON file
  python cli/mcq_generator_cli.py --schema my_schema.json

  # Specify custom output path
  python cli/mcq_generator_cli.py --output data/mcq/my_output.json
""",
    )
    parser.add_argument(
        "--schema", type=str, default=None,
        help="Path to a JSON schema file (skips interactive prompts)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: data/mcq/mcq_generator_output.json)",
    )
    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  MCQ Generator CLI  (DSPy + Mistral)")
    print(SEP)
    print(f"  Project root : {PROJECT_ROOT}")

    # ── Configure DSPy ────────────────────────────────────────────────────────
    print("\n  Connecting to Mistral...")
    from utils import configure_dspy_from_env
    lm = configure_dspy_from_env()
    print(f"  Active LM    : {lm}")

    # ── Load judges ───────────────────────────────────────────────────────────
    global _diff_agent, _rub_agent

    _diff_agent = SimpleDifficultyAgent()
    diff_art = ARTIFACTS_DIR / "simple_difficulty_optimized.json"
    if diff_art.exists():
        _diff_agent.load(str(diff_art))
        print(f"  DifficultyJudge : loaded from artifacts/simple_difficulty_optimized.json")
    else:
        print(f"  DifficultyJudge : using unoptimised agent")

    _rub_agent = RubricJudgeAgent()
    rub_art = ARTIFACTS_DIR / "rubric_agent_optimized.json"
    if rub_art.exists():
        _rub_agent.load(str(rub_art))
        print(f"  RubricJudge     : loaded from artifacts/rubric_agent_optimized.json")
    else:
        print(f"  RubricJudge     : using unoptimised agent")

    # ── Get schema ────────────────────────────────────────────────────────────
    if args.schema:
        print(f"\n  Loading schema from: {args.schema}")
        schema, example_questions = _load_schema_from_file(args.schema)
        print(f"  Topic    : {schema.topic}")
        print(f"  Subtopics: {[s.subtopic for s in schema.subtopics]}")
    else:
        schema, example_questions = _build_schema_interactive()

    total_requested = sum(s.total for s in schema.subtopics)
    print(f"\n  Total questions requested: {total_requested}")

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"\n  Starting generation...")
    orchestrator = MCQGenerationOrchestrator(
        difficulty_judge=DifficultyJudgeWrapper(),
        rubric_judge=RubricJudgeWrapper(),
    )
    result = orchestrator.run(schema=schema, example_questions=example_questions)

    # ── Print results ─────────────────────────────────────────────────────────
    _print_results(result, schema)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else DATA_DIR / "mcq" / "mcq_generator_output.json"
    _save_results(result, schema, out_path)


if __name__ == "__main__":
    main()
