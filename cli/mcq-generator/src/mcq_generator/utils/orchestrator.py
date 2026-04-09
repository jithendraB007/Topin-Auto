"""MCQ generation orchestrator — wires all components and runs the quota loop."""

from __future__ import annotations

import logging

import dspy

from mcq_generator.signatures.mcq_generator import MCQGeneratorSignature
from mcq_generator.utils.judges import DifficultyJudgeWrapper, RubricJudgeWrapper
from mcq_generator.utils.models import (
    ExampleQuestionSet,
    GenerationRequest,
    InputSchema,
    MCQGenerationResult,
    MCQItem,
    QuestionStore,
)
from mcq_generator.utils.validators import hard_validate

logger = logging.getLogger(__name__)

_CEFR_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
_CEFR_TO_DIFFICULTY = {
    'A1': 'Easy', 'A2': 'Easy',
    'B1': 'Medium', 'B2': 'Medium',
    'C1': 'Hard', 'C2': 'Hard',
}


class MCQGeneratorAgent(dspy.Module):
    """Generates MCQs and runs the quota loop inside forward().

    Dependencies injected at construction:
      self.store            -- QuestionStore accumulating accepted questions
      self.difficulty_judge -- DifficultyJudgeWrapper (batch)
      self.rubric_judge     -- RubricJudgeWrapper (batch)
    """

    def __init__(
        self,
        *,
        store: QuestionStore,
        difficulty_judge: DifficultyJudgeWrapper,
        rubric_judge: RubricJudgeWrapper,
    ):
        super().__init__()
        self.generate = dspy.ChainOfThought(MCQGeneratorSignature)
        self.store = store
        self.difficulty_judge = difficulty_judge
        self.rubric_judge = rubric_judge

    def forward(
        self,
        *,
        schema: InputSchema,
        example_questions: ExampleQuestionSet,
        topic: str,
        subtopic: str | None,
        target_cefr: str,
        target_difficulty: str,
        target_count: int,
        rejected: list,
        warnings: list,
    ) -> None:
        if target_count <= 0:
            return

        iteration = 0
        q_number = len(self.store.all_items()) + 1

        logger.info(
            f'[{target_difficulty}/{target_cefr}] target={target_count} subtopic={subtopic!r}'
        )
        print(f'\n[{target_difficulty} / {target_cefr}]  '
              f'target={target_count}  subtopic={subtopic!r}')

        while self.store.count_by_cefr(target_cefr) < target_count:
            iteration += 1
            if iteration > schema.constraints.max_iterations_per_difficulty:
                msg = (
                    f'{target_difficulty}/{target_cefr}: max iterations '
                    f'({schema.constraints.max_iterations_per_difficulty}) reached. '
                    f'Accepted {self.store.count_by_cefr(target_cefr)}/{target_count}.'
                )
                warnings.append(msg)
                print(f'  [WARNING] {msg}')
                break

            needed = target_count - self.store.count_by_cefr(target_cefr)
            batch_sz = min(schema.constraints.questions_per_iteration, needed)
            relevant = example_questions.filter_examples(
                subtopic=subtopic or '',
                difficulty=target_difficulty,
                cefr=target_cefr,
            )
            request = GenerationRequest(
                topic=topic,
                subtopic=subtopic or '',
                target_cefr=target_cefr,
                target_difficulty=target_difficulty,
                example_questions=relevant,
                already_used_stems=self.store.get_used_stems(),
                batch_size=batch_sz,
            )

            # 1. Generate batch
            print(f'  [Iter {iteration}] Generating {batch_sz} questions...')
            try:
                pred = self.generate(request=request)
                batch = pred.output.questions
            except Exception as e:
                rejected.append({
                    'stage': 'generation', 'cefr': target_cefr,
                    'difficulty': target_difficulty,
                    'iteration': iteration, 'error': str(e),
                })
                print(f'  [Gen Error] iter={iteration}: {e}')
                continue

            # 2. Hard-validate entire batch
            valid_items: list[MCQItem] = []
            for q in batch:
                errors = hard_validate(q)
                if errors:
                    rejected.append({
                        'stage': 'hard_validate', 'cefr': target_cefr,
                        'difficulty': target_difficulty,
                        'iteration': iteration, 'stem': q.stem[:60], 'errors': errors,
                    })
                    print(f'  [Hard Fail] {errors}')
                    continue
                valid_items.append(MCQItem(
                    question_number=q_number,
                    topic=topic,
                    subtopic=subtopic,
                    target_cefr=target_cefr,
                    target_difficulty=target_difficulty,
                    stem=q.stem,
                    options=q.options,
                    correct_answer=q.correct_answer,
                    explanation=q.explanation,
                ))
                q_number += 1

            if not valid_items:
                print(f'  [Skip] All {len(batch)} failed hard validation.')
                continue
            print(f'  [Hard]  {len(valid_items)}/{len(batch)} passed.')

            # 3. DifficultyJudge — one batch call
            try:
                diff_results = self.difficulty_judge(
                    items=valid_items,
                    expected_difficulty=target_difficulty,
                )
            except Exception as e:
                rejected.append({'stage': 'difficulty_judge_error', 'error': str(e)})
                print(f'  [Diff Error] {e}')
                continue

            diff_passed: list[MCQItem] = []
            for item, diff in zip(valid_items, diff_results):
                if diff.passed:
                    diff_passed.append(item)
                else:
                    rejected.append({
                        'stage': 'difficulty', 'q': item.question_number,
                        'cefr': target_cefr, 'difficulty': target_difficulty,
                        'reason': diff.reason, 'stem': item.stem[:60],
                    })
                    print(f'  [Diff Fail] Q{item.question_number}: {diff.reason}')

            print(f'  [Diff]  {len(diff_passed)}/{len(valid_items)} passed.')
            if not diff_passed:
                continue

            # 4. RubricJudge — one batch call
            try:
                rub_results = self.rubric_judge(items=diff_passed)
            except Exception as e:
                rejected.append({'stage': 'rubric_judge_error', 'error': str(e)})
                print(f'  [Rub Error] {e}')
                continue

            accepted_count = 0
            for item, rub in zip(diff_passed, rub_results):
                if not rub.passed:
                    rejected.append({
                        'stage': 'rubric', 'q': item.question_number,
                        'cefr': target_cefr, 'difficulty': target_difficulty,
                        'reason': rub.reason, 'stem': item.stem[:60],
                    })
                    print(f'  [Rub  Fail] Q{item.question_number}: {rub.reason}')
                    continue

                # 5. Save to store
                self.store.add(item)
                accepted_count += 1
                total = self.store.count_by_cefr(target_cefr)
                print(f'  [Accepted]  Q{item.question_number} '
                      f'({target_cefr}/{target_difficulty}) '
                      f'{total}/{target_count}')
                if total >= target_count:
                    break

            print(f'  [Rubric] {accepted_count}/{len(diff_passed)} passed.')


class BaseDifficultyAgent:
    difficulty_name: str = ''

    def __init__(self, *, generator: MCQGeneratorAgent):
        self.generator = generator

    def generate_questions(
        self,
        *,
        schema: InputSchema,
        example_questions: ExampleQuestionSet,
        topic: str,
        subtopic: str | None,
        target_cefr: str,
        target_count: int,
        rejected: list,
        warnings: list,
    ) -> None:
        self.generator(
            schema=schema,
            example_questions=example_questions,
            topic=topic,
            subtopic=subtopic,
            target_cefr=target_cefr,
            target_difficulty=self.difficulty_name,
            target_count=target_count,
            rejected=rejected,
            warnings=warnings,
        )


class EasyMCQAgent(BaseDifficultyAgent):
    difficulty_name = 'Easy'


class MediumMCQAgent(BaseDifficultyAgent):
    difficulty_name = 'Medium'


class HardMCQAgent(BaseDifficultyAgent):
    difficulty_name = 'Hard'


class MCQGenerationOrchestrator:
    """Wires all components and runs the generation loop."""

    def __init__(
        self,
        *,
        difficulty_judge: DifficultyJudgeWrapper,
        rubric_judge: RubricJudgeWrapper,
    ):
        self.store = QuestionStore()
        self.generator = MCQGeneratorAgent(
            store=self.store,
            difficulty_judge=difficulty_judge,
            rubric_judge=rubric_judge,
        )
        self.easy_agent = EasyMCQAgent(generator=self.generator)
        self.medium_agent = MediumMCQAgent(generator=self.generator)
        self.hard_agent = HardMCQAgent(generator=self.generator)
        self._agents = {
            'Easy': self.easy_agent,
            'Medium': self.medium_agent,
            'Hard': self.hard_agent,
        }

    def run(
        self,
        *,
        schema: InputSchema,
        example_questions: ExampleQuestionSet,
    ) -> MCQGenerationResult:
        rejected: list = []
        warnings: list = []

        for req in schema.subtopics:
            sep = '=' * 60
            print(f'\n{sep}')
            print(f'Topic: {schema.topic}  |  Subtopic: {req.subtopic}')
            print(f'  Easy:   A1={req.a1_count}  A2={req.a2_count}')
            print(f'  Medium: B1={req.b1_count}  B2={req.b2_count}')
            print(f'  Hard:   C1={req.c1_count}  C2={req.c2_count}')
            print(sep)

            for cefr in _CEFR_LEVELS:
                count_attr = cefr.lower() + '_count'
                target_count = getattr(req, count_attr)
                if target_count == 0:
                    continue

                difficulty = _CEFR_TO_DIFFICULTY[cefr]
                agent = self._agents[difficulty]
                agent.generate_questions(
                    schema=schema,
                    example_questions=example_questions,
                    topic=schema.topic,
                    subtopic=req.subtopic,
                    target_cefr=cefr,
                    target_count=target_count,
                    rejected=rejected,
                    warnings=warnings,
                )

        return MCQGenerationResult(
            store=self.store,
            rejected=rejected,
            warnings=warnings,
        )
