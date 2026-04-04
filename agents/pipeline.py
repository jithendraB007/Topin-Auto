from __future__ import annotations

import json
from typing import List

import dspy

from schemas import (
    CEFR_TO_DIFFICULTY,
    DifficultyResult,
    EvaluatedItem,
    InputSchema,
    MCQItem,
    PlannedQuestion,
    RubricResult,
)
from agents.signatures import (
    DifficultyJudgeSignature,
    MCQGeneratorSignature,
    PlannerSignature,
    RevisionSignature,
    RubricJudgeSignature,
)


def _loads_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


_DIFFICULTY_NORM = {
    "very easy": "Easy",
    "easy": "Easy",
    "medium": "Medium",
    "moderate": "Medium",
    "intermediate": "Medium",
    "hard": "Hard",
    "difficult": "Hard",
    "very hard": "Hard",
    "very difficult": "Hard",
}


def _normalize_difficulty(row: dict) -> dict:
    raw = row.get("target_difficulty", "")
    normalized = _DIFFICULTY_NORM.get(raw.lower().strip())
    if normalized:
        row = {**row, "target_difficulty": normalized}
    # question_type must always be "MCQ" — LLM sometimes puts the angle here
    if row.get("question_type", "MCQ") != "MCQ":
        # salvage: if angle is missing, move it; always reset question_type
        if not row.get("angle"):
            row = {**row, "angle": row["question_type"]}
        row = {**row, "question_type": "MCQ"}
    return row


_FALSY_ALIGNMENT = {"false", "no", "0", "not aligned", "fail", "misaligned", "poor", "bad", "low"}


def _normalize_alignment(row: dict) -> dict:
    val = row.get("alignment")
    if isinstance(val, bool):
        return row
    if isinstance(val, str):
        aligned = val.lower().strip() not in _FALSY_ALIGNMENT
        return {**row, "alignment": aligned}
    return row


def _normalize_overall_decision(row: dict) -> dict:
    val = row.get("overall_decision", "")
    v = val.lower().strip()
    if "fail" in v:
        decision = "Fail"
    elif "revis" in v:
        decision = "Revise"
    else:
        decision = "Pass"
    return {**row, "overall_decision": decision}


def hard_validate_item(item: MCQItem) -> list[str]:
    errors: list[str] = []
    if len(item.options) != 4:
        errors.append("MCQ must contain exactly 4 options.")
    if item.correct_answer not in item.options:
        errors.append("Correct answer must be one of the options.")
    if not item.explanation.strip():
        errors.append("Explanation must not be empty.")
    return errors


class PlannerAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.ChainOfThought(PlannerSignature)

    def forward(self, schema: InputSchema) -> List[PlannedQuestion]:
        result = self.plan(
            subject=schema.subject,
            syllabus_unit=schema.syllabus_unit,
            topic=schema.topic,
            subtopics=json.dumps(schema.subtopics),
            total_questions=str(schema.total_questions),
            cefr_distribution=json.dumps(schema.cefr_distribution),
            constraints=schema.constraints.model_dump_json(),
        )
        plan = _loads_json(result.plan_json)
        planned = [PlannedQuestion(**_normalize_difficulty(row)) for row in plan]
        if len(planned) != schema.total_questions:
            raise ValueError("Planner returned the wrong number of questions.")
        return planned

    def _enforce_unique_angles(self, planned: List[PlannedQuestion]) -> List[PlannedQuestion]:
        """Post-process: ensure no two questions share the same angle."""
        _ALL_ANGLES = [
            "fill-in-the-blank", "sentence-completion", "error-correction", "inference",
            "vocabulary-in-context", "question-formation", "conversation-completion",
            "word-order", "concept-identification", "real-world-application",
            "paraphrase-selection", "affirmative-negative-transformation",
        ]
        seen: set[str] = set()
        angle_pool = list(_ALL_ANGLES)
        result = []
        for q in planned:
            angle = (q.angle or "").lower().strip()
            if not angle or angle in seen:
                # pick first unused angle from pool
                for candidate in angle_pool:
                    if candidate not in seen:
                        angle = candidate
                        break
                else:
                    angle = f"variant-{len(seen)}"
            seen.add(angle)
            result.append(q.model_copy(update={"angle": angle}))
        return result


class MCQGeneratorAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(MCQGeneratorSignature)

    def forward(
        self,
        plan: PlannedQuestion,
        schema: InputSchema,
        used_stems: list[str] | None = None,
    ) -> MCQItem:
        result = self.generate(
            topic=plan.topic,
            subtopic=plan.subtopic or "",
            target_cefr=plan.target_cefr,
            target_difficulty=plan.target_difficulty,
            angle=plan.angle or "fill-in-the-blank",
            already_used_stems=json.dumps(used_stems or []),
            sample_questions=json.dumps(schema.sample_questions or []),
            constraints=schema.constraints.model_dump_json(),
        )
        raw = _loads_json(result.output_json)
        return MCQItem(
            question_number=plan.question_number,
            topic=plan.topic,
            subtopic=plan.subtopic,
            target_cefr=plan.target_cefr,
            target_difficulty=plan.target_difficulty,
            stem=raw["stem"],
            options=raw["options"],
            correct_answer=raw["correct_answer"],
            explanation=raw["explanation"],
        )


class DifficultyAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(DifficultyJudgeSignature)

    def forward(self, item: MCQItem) -> DifficultyResult:
        result = self.judge(
            stem=item.stem,
            options=json.dumps(item.options),
            correct_answer=item.correct_answer,
            explanation=item.explanation,
            target_cefr=item.target_cefr,
            target_difficulty=item.target_difficulty,
        )
        return DifficultyResult(**_normalize_alignment(_loads_json(result.output_json)))


class RubricAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(RubricJudgeSignature)

    def forward(self, item: MCQItem, language_variant: str) -> RubricResult:
        result = self.judge(
            stem=item.stem,
            options=json.dumps(item.options),
            correct_answer=item.correct_answer,
            explanation=item.explanation,
            target_cefr=item.target_cefr,
            target_difficulty=item.target_difficulty,
            language_variant=language_variant,
        )
        rubric = RubricResult(**_normalize_overall_decision(_loads_json(result.output_json)))
        if rubric.ambiguity.lower() == "major issue":
            rubric.overall_decision = "Fail"
        # "Revise" with no real reason is a false positive — treat as Pass
        elif rubric.overall_decision == "Revise":
            trivial = {"none", "no major issues", "no major issues found", "n/a", ""}
            reason = rubric.priority_reason.lower().strip()
            if any(t in reason for t in trivial):
                rubric.overall_decision = "Pass"
        return rubric


class RevisionAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.revise = dspy.ChainOfThought(RevisionSignature)

    def forward(
        self,
        item: MCQItem,
        difficulty_feedback: str,
        rubric_feedback: str,
        schema: InputSchema,
        angle: str = "",
        used_stems: list[str] | None = None,
    ) -> MCQItem:
        result = self.revise(
            topic=item.topic,
            subtopic=item.subtopic or "",
            target_cefr=item.target_cefr,
            target_difficulty=item.target_difficulty,
            angle=angle or "fill-in-the-blank",
            stem=item.stem,
            options=json.dumps(item.options),
            correct_answer=item.correct_answer,
            explanation=item.explanation,
            difficulty_feedback=difficulty_feedback,
            rubric_feedback=rubric_feedback,
            already_used_stems=json.dumps(used_stems or []),
            constraints=schema.constraints.model_dump_json(),
        )
        raw = _loads_json(result.output_json)
        return item.model_copy(
            update={
                "stem": raw["stem"],
                "options": raw["options"],
                "correct_answer": raw["correct_answer"],
                "explanation": raw["explanation"],
            }
        )


class MCQPipeline(dspy.Module):
    def __init__(self, max_revision_attempts: int = 2):
        super().__init__()
        self.max_revision_attempts = max_revision_attempts
        self.planner = PlannerAgent()
        self.generator = MCQGeneratorAgent()
        self.difficulty = DifficultyAgent()
        self.rubric = RubricAgent()
        self.revision = RevisionAgent()

    def load_optimized_agents(self) -> None:
        from pathlib import Path
        diff_path = Path("artifacts/difficulty_agent_optimized.json")
        rub_path = Path("artifacts/rubric_agent_optimized.json")
        if diff_path.exists():
            self.difficulty.load(str(diff_path))
            print("[Pipeline] Loaded optimized DifficultyAgent.")
        if rub_path.exists():
            self.rubric.load(str(rub_path))
            print("[Pipeline] Loaded optimized RubricAgent.")

    def _is_accepted(self, item: MCQItem, diff: DifficultyResult, rub: RubricResult) -> bool:
        hard_errors = hard_validate_item(item)
        if hard_errors:
            return False
        if not diff.alignment:
            return False
        if rub.overall_decision != "Pass":
            return False
        return True

    # Common English stop words — excluded from content-word overlap check
    _STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "on",
        "at", "by", "for", "with", "from", "as", "or", "and", "but", "not",
        "this", "that", "these", "those", "it", "its", "he", "she", "they",
        "we", "you", "i", "my", "your", "his", "her", "our", "their",
        "which", "who", "what", "how", "when", "where", "why",
    }

    def _is_duplicate_stem(self, stem: str, used_stems: list[str]) -> bool:
        """Return True if stem is essentially identical to an already-accepted stem."""
        s = stem.lower().strip()
        for u in used_stems:
            u_norm = u.lower().strip()
            if s == u_norm:
                return True
            # Content-word overlap: only flag when >=90% of meaningful words match
            sw = {w for w in s.split() if w not in self._STOP_WORDS and len(w) > 2}
            uw = {w for w in u_norm.split() if w not in self._STOP_WORDS and len(w) > 2}
            if sw and uw and len(sw & uw) / max(len(sw), len(uw)) >= 0.90:
                return True
        return False

    def forward(self, schema: InputSchema) -> list[EvaluatedItem]:
        plan = self.planner(schema)
        # Enforce unique angles across the plan
        plan = self.planner._enforce_unique_angles(plan)

        results: list[EvaluatedItem] = []
        accepted_stems: list[str] = []

        for planned in plan:
            angle = planned.angle or "fill-in-the-blank"

            # Generator retry: attempt up to 3 times if hard validation fails or stem is duplicate
            for gen_attempt in range(3):
                item = self.generator(planned, schema, used_stems=accepted_stems)
                hard_errors = hard_validate_item(item)
                if hard_errors:
                    print(f"[Generator] Attempt {gen_attempt + 1} structural errors: {hard_errors}. Retrying...")
                    continue
                if self._is_duplicate_stem(item.stem, accepted_stems):
                    print(f"[Generator] Attempt {gen_attempt + 1} duplicate stem detected. Retrying...")
                    continue
                break

            revision_attempts = 0
            while True:
                diff = self.difficulty(item)
                rub = self.rubric(item, schema.constraints.language_variant)
                hard_errors = hard_validate_item(item)
                is_dup = self._is_duplicate_stem(item.stem, accepted_stems)
                accepted = self._is_accepted(item, diff, rub) and not is_dup

                if accepted:
                    accepted_stems.append(item.stem)
                    results.append(
                        EvaluatedItem(
                            item=item,
                            difficulty=diff,
                            rubric=rub,
                            accepted=True,
                            revision_attempts=revision_attempts,
                        )
                    )
                    break

                if revision_attempts >= self.max_revision_attempts:
                    reasons = []
                    if hard_errors:
                        reasons.append(f"structural: {'; '.join(hard_errors)}")
                    if is_dup:
                        reasons.append("duplicate stem")
                    if not diff.alignment:
                        reasons.append(f"CEFR mismatch: predicted={diff.predicted_cefr} target={item.target_cefr}")
                    if rub.overall_decision != "Pass":
                        reasons.append(f"rubric={rub.overall_decision}: {rub.priority_reason}")
                    print(f"  [REJECTED] Q{item.question_number} — {' | '.join(reasons) or 'unknown'}")
                    results.append(
                        EvaluatedItem(
                            item=item,
                            difficulty=diff,
                            rubric=rub,
                            accepted=False,
                            revision_attempts=revision_attempts,
                        )
                    )
                    break

                # Prepend structural errors to rubric feedback so reviser fixes them first
                rub_feedback = rub.revision_feedback
                if hard_errors:
                    rub_feedback = (
                        f"CRITICAL STRUCTURAL ERRORS (fix these first): {'; '.join(hard_errors)}. "
                        f"You MUST produce exactly 4 options and the correct_answer must be one of them. "
                        f"{rub_feedback}"
                    )
                if is_dup:
                    rub_feedback = f"DUPLICATE STEM DETECTED — rewrite the question completely. {rub_feedback}"

                item = self.revision(
                    item,
                    difficulty_feedback=diff.revision_feedback,
                    rubric_feedback=rub_feedback,
                    schema=schema,
                    angle=angle,
                    used_stems=accepted_stems,
                )
                revision_attempts += 1

        return results
