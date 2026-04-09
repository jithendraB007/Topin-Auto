"""Hard structural validation for generated MCQ questions."""

from __future__ import annotations

from mcq_generator.utils.models import GeneratedQuestion


def hard_validate(q: GeneratedQuestion) -> list[str]:
    """Returns a list of error strings. Empty list = valid."""
    errors = []
    if not isinstance(q.options, list) or len(q.options) != 4:
        got = len(q.options) if isinstance(q.options, list) else type(q.options).__name__
        errors.append(f'Expected 4 options, got {got}')
    if q.correct_answer not in q.options:
        errors.append(f'correct_answer "{q.correct_answer}" not found in options')
    if not q.explanation or not q.explanation.strip():
        errors.append('explanation is empty')
    if not q.stem or not q.stem.strip():
        errors.append('stem is empty')
    return errors
