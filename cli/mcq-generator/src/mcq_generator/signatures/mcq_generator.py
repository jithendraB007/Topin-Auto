"""MCQ Generator DSPy signature."""

import dspy

from mcq_generator.utils.models import GenerationRequest, GenerationBatch


class MCQGeneratorSignature(dspy.Signature):
    """Generate a batch of high-quality MCQ questions for the given topic and difficulty.

    Use the example_questions as a reference for style, vocabulary level, and format.
    Easy examples show A1/A2-level vocabulary; Medium show B1/B2; Hard show C1/C2.

    STRICT RULES -- every question in the batch must follow these:
    - EXACTLY 4 options -- no more, no less
    - correct_answer must be copied VERBATIM from one of the options
    - Each stem must be completely different from all stems in already_used_stems
    - Only one answer can be correct; the other 3 are plausible but clearly wrong
    - Align vocabulary and grammar complexity to the target_cefr level
    - Avoid ambiguity -- only one option can be correct

    Return exactly batch_size questions in the output.
    """
    request: GenerationRequest = dspy.InputField(
        desc='Batch generation parameters: topic, CEFR level, difficulty, reference examples, used stems'
    )
    output: GenerationBatch = dspy.OutputField(
        desc='Batch of generated MCQ questions, one per item in questions list'
    )
