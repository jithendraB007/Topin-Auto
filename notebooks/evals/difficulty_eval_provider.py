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


class Question(BaseModel):
    question_id:       str
    stem:              str
    options:           list[str]
    correct_answer:    str
    explanation:       str
    target_cefr:       str
    target_difficulty: str


class DifficultyResult(BaseModel):
    question_id:          str
    predicted_cefr:       Literal['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    predicted_difficulty: Literal['Easy', 'Medium', 'Hard']


class DifficultyOutput(BaseModel):
    results: list[DifficultyResult]


class SimpleDifficultySignature(dspy.Signature):
    """Classify a list of MCQ questions by CEFR level and difficulty.
    Map: A1/A2 -> Easy | B1/B2 -> Medium | C1/C2 -> Hard.
    Return one DifficultyResult per question in the same order.
    """
    questions: list[Question]   = dspy.InputField(desc='List of MCQ questions to be classified')
    output:    DifficultyOutput = dspy.OutputField(desc='Classification results for all questions')


class SimpleDifficultyAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(SimpleDifficultySignature)

    def forward(self, questions: list[Question]) -> dspy.Prediction:
        return self.judge(questions=questions)


_OPTIMIZED = PROJECT_ROOT / "artifacts" / "simple_difficulty_optimized.json"
_agent = None


def call_api(prompt: str, options, context):
    global _agent
    configure_dspy_from_env()

    if _agent is None:
        _agent = SimpleDifficultyAgent()
        if _OPTIMIZED.exists():
            _agent.load(str(_OPTIMIZED))

    try:
        data = json.loads(prompt)
        if isinstance(data, dict):
            data = [data]
        questions = [Question(**q) for q in data]
    except Exception as e:
        return {"error": f"Invalid input: {e}"}

    try:
        pred    = _agent(questions=questions)
        results = [r.model_dump() for r in pred.output.results]
        return {"output": json.dumps(results)}
    except Exception as e:
        return {"error": str(e)}
