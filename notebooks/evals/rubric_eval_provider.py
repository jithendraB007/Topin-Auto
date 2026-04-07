from __future__ import annotations
import json, os, sys
from pathlib import Path
from typing import Literal
from pydantic import BaseModel

EVALS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_venv_sp = PROJECT_ROOT / '.venv' / 'Lib' / 'site-packages'
if not _venv_sp.exists():
    _venv_sp = PROJECT_ROOT / '.venv' / 'lib' / 'site-packages'
if _venv_sp.exists() and str(_venv_sp) not in sys.path:
    sys.path.insert(0, str(_venv_sp))

from utils import configure_dspy_from_env
import dspy


class RubricQuestion(BaseModel):
    question_id:       str
    stem:              str
    options:           list[str]
    correct_answer:    str
    explanation:       str
    target_cefr:       str
    target_difficulty: str
    language_variant:  str


class RubricResult(BaseModel):
    question_id:                          str
    grammatical_accuracy:                 Literal['No Issues', 'Minor Issues', 'Major Issues']
    spelling:                             Literal['No Issues', 'Minor Issues', 'Major Issues']
    ambiguity:                            Literal['No Issue', 'Minor Issue', 'Major Issue']
    functionality_alignment:              Literal['Aligned', 'Partially Aligned', 'Not Aligned']
    instruction_clarity_appropriateness:  Literal['Clear', 'Needs Improvement', 'Unclear']
    academic_language_exam_acceptability: Literal['Acceptable', 'Needs Improvement', 'Not Acceptable']
    option_explanation_consistency:       Literal['Consistent', 'Inconsistent']
    readability:                          Literal['Good', 'Needs Improvement', 'Poor']
    formatting_spacing:                   Literal['No Issues', 'Minor Issues', 'Major Issues']
    punctuation:                          Literal['No Issues', 'Minor Issues', 'Major Issues']
    british_american_english_consistency: Literal['Consistent', 'Inconsistent']
    overall_decision:                     Literal['Pass', 'Revise', 'Fail']
    priority_reason:                      str
    revision_feedback:                    str


class RubricOutput(BaseModel):
    results: list[RubricResult]


class RubricJudgeSignature(dspy.Signature):
    """Evaluate a list of MCQ questions using the rubric.
    ambiguity is highest priority — Major Issue forces Fail.
    Return one RubricResult per question in the same order.
    """
    questions: list[RubricQuestion] = dspy.InputField(desc='List of MCQ questions to be evaluated')
    output:    RubricOutput         = dspy.OutputField(desc='Rubric evaluation results for all questions')


class RubricJudgeAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(RubricJudgeSignature)

    def forward(self, questions):
        return self.judge(questions=questions)


_OPTIMIZED = PROJECT_ROOT / "artifacts" / "rubric_agent_optimized.json"
_agent = None


def call_api(prompt: str, options, context):
    global _agent
    configure_dspy_from_env()

    if _agent is None:
        _agent = RubricJudgeAgent()
        if _OPTIMIZED.exists():
            _agent.load(str(_OPTIMIZED))

    try:
        data = json.loads(prompt)
        if isinstance(data, dict):
            data = [data]
        questions = [RubricQuestion(**q) for q in data]
    except Exception as e:
        return {"error": f"Invalid input: {e}"}

    try:
        pred    = _agent(questions=questions)
        results = [r.model_dump() for r in pred.output.results]
        return {"output": json.dumps(results)}
    except Exception as e:
        return {"error": str(e)}
