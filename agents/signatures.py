import dspy


class PlannerSignature(dspy.Signature):
    """Create a balanced MCQ generation plan from a schema.

    Return a JSON array. Each element must contain:
    question_number, question_type (ALWAYS "MCQ"), topic, subtopic, target_cefr, target_difficulty, angle.

    RULES:
    - question_type must ALWAYS be the string "MCQ" — never put the angle here.
    - Keep the counts exactly aligned with the requested CEFR distribution.
    - Every question must have a UNIQUE angle — no two questions may share the same angle.
    - Spread angles across the full variety:
      fill-in-the-blank, sentence-completion, error-correction, inference,
      vocabulary-in-context, question-formation, conversation-completion,
      word-order, concept-identification, real-world-application,
      paraphrase-selection, affirmative-negative-transformation.
    - Spread subtopics as evenly as possible — avoid giving the same subtopic to adjacent slots.
    - Each question must test a different grammar point or concept even within the same subtopic.
    """

    subject = dspy.InputField()
    syllabus_unit = dspy.InputField()
    topic = dspy.InputField()
    subtopics = dspy.InputField()
    total_questions = dspy.InputField()
    cefr_distribution = dspy.InputField()
    constraints = dspy.InputField()
    plan_json = dspy.OutputField()


class MCQGeneratorSignature(dspy.Signature):
    """Generate one high-quality MCQ.

    STRICT RULES — violating any of these will cause rejection:
    - options array must contain EXACTLY 4 items, no more, no less
    - correct_answer must be copied EXACTLY as it appears in the options array
    - exactly 1 correct answer; the other 3 are plausible but clearly wrong distractors
    - explanation must justify only the correct answer
    - align vocabulary and grammar complexity to the target CEFR level
    - avoid any ambiguity — only one option can be correct
    - use the requested English variant consistently throughout
    - follow the question angle strictly (fill-in-the-blank, error-correction, inference, etc.)
    - the stem must be COMPLETELY DIFFERENT from any question in the already_used_stems list
    - if sample_questions are provided, match their style, vocabulary level, and question format

    Return valid JSON with exactly these keys: stem, options, correct_answer, explanation.
    The options value must be a JSON array of exactly 4 strings.
    """

    topic = dspy.InputField()
    subtopic = dspy.InputField()
    target_cefr = dspy.InputField()
    target_difficulty = dspy.InputField()
    angle = dspy.InputField(desc="The specific question type/angle to use, e.g. fill-in-the-blank, inference, error-correction")
    already_used_stems = dspy.InputField(desc="JSON array of question stems already generated — your new stem must be completely different")
    sample_questions = dspy.InputField(desc="JSON array of example questions showing the desired style and format. May be empty.")
    constraints = dspy.InputField()
    output_json = dspy.OutputField()


class DifficultyJudgeSignature(dspy.Signature):
    """Judge whether the MCQ matches the intended CEFR level and derived difficulty.

    Return valid JSON with keys:
    predicted_cefr, predicted_difficulty, vocabulary_level, grammar_complexity,
    reasoning_load, distractor_difficulty, alignment, justification, revision_feedback.
    """

    stem = dspy.InputField()
    options = dspy.InputField()
    correct_answer = dspy.InputField()
    explanation = dspy.InputField()
    target_cefr = dspy.InputField()
    target_difficulty = dspy.InputField()
    output_json = dspy.OutputField()


class RubricJudgeSignature(dspy.Signature):
    """Evaluate the MCQ using the rubric.

    Ambiguity is the highest-priority criterion.
    If ambiguity is a major issue, the item must not pass.

    Return valid JSON with keys:
    grammatical_accuracy, spelling, ambiguity, functionality_alignment,
    instruction_clarity_appropriateness, academic_language_exam_acceptability,
    option_explanation_consistency, readability, formatting_spacing, punctuation,
    british_american_english_consistency, overall_decision, priority_reason,
    revision_feedback.
    """

    stem = dspy.InputField()
    options = dspy.InputField()
    correct_answer = dspy.InputField()
    explanation = dspy.InputField()
    target_cefr = dspy.InputField()
    target_difficulty = dspy.InputField()
    language_variant = dspy.InputField()
    output_json = dspy.OutputField()


class RevisionSignature(dspy.Signature):
    """Revise the MCQ using judge feedback.

    STRICT RULES — must be followed exactly:
    - options array must contain EXACTLY 4 items, no more, no less
    - correct_answer must be copied EXACTLY as it appears in the revised options array
    - preserve the topic and target CEFR level
    - fix ambiguity first, then alignment, then language issues
    - do not introduce new ambiguity while fixing other issues
    - the revised stem must be completely different from all stems in already_used_stems

    Return valid JSON with keys: stem, options, correct_answer, explanation.
    The options value must be a JSON array of exactly 4 strings.
    """

    topic = dspy.InputField()
    subtopic = dspy.InputField()
    target_cefr = dspy.InputField()
    target_difficulty = dspy.InputField()
    angle = dspy.InputField(desc="The question angle to preserve, e.g. fill-in-the-blank, inference")
    stem = dspy.InputField()
    options = dspy.InputField()
    correct_answer = dspy.InputField()
    explanation = dspy.InputField()
    difficulty_feedback = dspy.InputField()
    rubric_feedback = dspy.InputField()
    already_used_stems = dspy.InputField(desc="JSON array of stems already accepted — do not repeat any of these")
    constraints = dspy.InputField()
    output_json = dspy.OutputField()
