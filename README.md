# Topin — AI Question Generator

Generates exam-quality questions for language learners using **DSPy** and **Mistral AI**.
Three question types are supported, each with its own Jupyter notebook.

---

## Project Structure

```
Topin/
├── notebooks/
│   ├── mcq_generator.ipynb          # Multiple-Choice Questions (text-based)
│   ├── text_to_text_generator.ipynb # Open-answer / writing questions (T2T)
│   ├── image_mcq_generator.ipynb    # MCQ questions based on a notice/sign description
│   ├── difficulty_judge.ipynb       # Standalone CEFR difficulty classifier
│   └── rubric_judge.ipynb           # Standalone quality rubric evaluator
├── data/
│   ├── mcq/
│   │   ├── training_dataset_standard.json
│   │   └── eval_dataset_standard.json
│   ├── t2t/
│   │   ├── training_dataset_clean_full.json
│   │   └── eval_dataset_clean.json
│   └── image_mcq/
│       ├── training_dataset_24_clean.json
│       └── eval_dataset_24_clean.json
├── artifacts/
│   ├── simple_difficulty_optimized.json   # Optimised DifficultyJudge weights
│   └── rubric_agent_optimized.json        # Optimised RubricJudge weights
├── utils.py                               # configure_dspy_from_env()
└── .env                                   # MISTRAL_API_KEY, MISTRAL_MODEL
```

---

## Question Formats

### MCQ (Multiple-Choice Question)

```json
{
  "question_id": "Q1",
  "instruction": "Read the sentence carefully and answer the question.",
  "question": "The boy is carrying a red backpack.\n\nWhat is the boy carrying?",
  "options": ["A red backpack", "A blue suitcase", "A lunch box", "A football"],
  "correct_answer": "A red backpack",
  "explanation": "The sentence clearly states the boy is carrying a red backpack.",
  "target_cefr": "A1",
  "target_difficulty": null
}
```

### T2T (Text-to-Text / Open Answer)

```json
{
  "question_id": "Q1",
  "instruction": "Read the sentence carefully and answer the question.",
  "question": "The girl is reading beside the window.\n\nWhere is the girl reading?",
  "expected_answer": "Beside the window",
  "explanation": "The sentence clearly states beside the window.",
  "target_cefr": "A1",
  "target_difficulty": null
}
```

### Image MCQ

```json
{
  "question_id": "Q1",
  "instruction": "Read the notice and choose the correct answer.",
  "image_content": "A swimming pool safety notice says swimmers must wear a swimming cap.",
  "question": "What should you wear?",
  "options": ["A jacket.", "A swimming cap.", "A scarf.", "A pair of gloves."],
  "correct_answer": "A swimming cap.",
  "explanation": "The notice states that swimmers must wear a swimming cap.",
  "target_cefr": "A2",
  "target_difficulty": "Easy"
}
```

> **Field guide**
> - `instruction` — the reading task direction shown to the learner (e.g. *"Read the passage and choose the correct answer."*)
> - `question` — the passage text + the actual comprehension question
> - `image_content` *(Image MCQ only)* — text description of the real-world notice/sign; always a **separate top-level key**, never merged into `question` or `instruction`

---

## CEFR to Difficulty Mapping

| CEFR | Difficulty |
|------|-----------|
| A1   | Easy      |
| A2   | Easy      |
| B1   | Medium    |
| B2   | Medium    |
| C1   | Hard      |
| C2   | Hard      |

---

## Process Flow

```
InputSchema
  topic, subtopics [ { subtopic, a1_count, a2_count, b1_count, b2_count, c1_count, c2_count } ]
  constraints { questions_per_iteration, max_iterations_per_difficulty }
        |
        v
Orchestrator.run()
  iterates CEFR levels: A1 -> A2 -> B1 -> B2 -> C1 -> C2
        |
        v  (for each CEFR level with count > 0)
GeneratorAgent.forward()   <-- quota loop
        |
        |  while store.count_by_cefr(cefr) < target_count:
        |
        |    Step 1: Generate batch
        |            ChainOfThought LLM call
        |            Input:  topic, subtopic, target_cefr, example_questions, batch_size
        |            Output: list of GeneratedQuestion
        |
        |    Step 2: hard_validate()
        |            Check: instruction not empty
        |            Check: question not empty
        |            Check: exactly 4 options (MCQ only)
        |            Check: correct_answer is one of the options (MCQ only)
        |            Check: explanation not empty
        |
        |    Step 3: DifficultyJudge  (batch LLM call)
        |            Classifies each question's CEFR level
        |            Rejects if predicted difficulty != target band
        |
        |    Step 4: RubricJudge  (batch LLM call)
        |            Evaluates quality criteria
        |            overall_decision must be "Pass" to accept
        |
        |    Step 5: store.add(accepted)
        |
        v
GenerationResult
  store.easy    [ accepted Easy questions ]
  store.medium  [ accepted Medium questions ]
  store.hard    [ accepted Hard questions ]
  rejected      [ list of failed attempts with stage + reason ]
  warnings      [ quota-not-met messages ]
```

### RubricJudge Quality Criteria

| Criterion | Description |
|-----------|-------------|
| `grammatical_accuracy` | No grammar errors in instruction, question, options, explanation |
| `spelling` | No spelling errors |
| `ambiguity` | Only one option can be correct — Major Issue forces **Fail** |
| `functionality_alignment` | Question tests what the instruction says |
| `instruction_clarity_appropriateness` | Instruction is clear and grade-appropriate |
| `academic_language_exam_acceptability` | Suitable for a formal language exam |
| `option_explanation_consistency` | Explanation matches the correct answer |
| `readability` | Natural, fluent reading experience |
| `formatting_spacing` | Consistent formatting |
| `punctuation` | Correct punctuation throughout |

**Image MCQ only — highest priority criterion:**

| `image_content_coherence` | Effect |
|--------------------------|--------|
| `Incoherent` | Forces **Fail** — image_content does not logically support the question |
| `Partially Coherent` | Forces **Revise** |
| `Coherent` | Normal evaluation continues |

---

## Running a Notebook

### Prerequisites

1. Create `.env` in the project root:

   ```
   MISTRAL_API_KEY=your_key_here
   MISTRAL_MODEL=mistral-small-latest
   MISTRAL_API_BASE=https://api.mistral.ai/v1
   ```

2. Install dependencies (once):

   ```bash
   pip install dspy-ai pydantic python-dotenv
   ```

3. Open Jupyter from `d:/Topin` or `d:/Topin/notebooks`.

### Cell Execution Order

Run cells **top to bottom**. Each cell depends on the previous.

| Cell | Content |
|------|---------|
| 1 | Setup — imports, project root detection, venv site-packages injection |
| 2 | Configure DSPy — loads `.env`, connects to Mistral AI |
| 3 | Load datasets — reads training/eval JSON into `train_rows`, `eval_rows` |
| 4 | Input models — `SubtopicRequirement`, `InputSchema`, `ExampleQuestionSet` |
| 5 | Output models — `MCQItem`/`T2TItem`/`ImageMCQItem`, `QuestionStore` |
| 6 | Generator signature + `GeneratorAgent` (contains the full quota loop) |
| 7 | Judges — `DifficultyJudge` + `RubricJudge` (loads optimised artifacts if found) |
| 8 | `hard_validate()` — fast structural checks before LLM judging |
| 9 | Per-difficulty agents — `Easy`, `Medium`, `Hard` wrappers that delegate to GeneratorAgent |
| 10 | `Orchestrator` — wires QuestionStore + GeneratorAgent + difficulty agents |
| **11** | **Edit this cell** — define the schema and run generation |
| 12 | Save output JSON to `data/<type>/<type>_generator_output.json` |

### Editing Cell 11 (Schema Definition)

```python
schema = InputSchema(
    topic='Reading Notices and Signs',
    subtopics=[
        SubtopicRequirement(
            subtopic='Public Notices',
            a1_count=5,    # 5 A1-level (Easy) questions
            a2_count=5,    # 5 A2-level (Easy) questions
            b1_count=3,    # 3 B1-level (Medium) questions
            b2_count=3,
            c1_count=2,    # 2 C1-level (Hard) questions
            c2_count=0,    # skip C2
        )
    ],
    constraints=GenerationConstraints(
        questions_per_iteration=5,       # LLM batch size per call
        max_iterations_per_difficulty=20, # retry limit per CEFR level
    ),
)
```

Set any count to `0` to skip that CEFR level entirely.

---

## Output File Format

```json
{
  "schema": { "topic": "...", "subtopics": [ ... ] },
  "summary": {
    "easy":   { "accepted": 10, "rejected": 3 },
    "medium": { "accepted": 6,  "rejected": 1 },
    "hard":   { "accepted": 2,  "rejected": 4 },
    "total_accepted": 18,
    "total_rejected": 8
  },
  "questions": {
    "easy":   [ { "question_id": "Q1", "instruction": "...", "question": "...", ... } ],
    "medium": [ ... ],
    "hard":   [ ... ]
  },
  "rejected": [
    { "stage": "hard_validate", "question": "...", "errors": ["instruction is empty"] },
    { "stage": "difficulty",    "reason": "predicted Easy, expected Medium" },
    { "stage": "rubric",        "reason": "ambiguity: Major Issue" }
  ],
  "warnings": [
    "A1/Easy: max iterations (20) reached. Accepted 3/5."
  ]
}
```

---

## Artifacts

Optimised judge weights are saved by `difficulty_judge.ipynb` and `rubric_judge.ipynb`.
Each generator notebook loads them automatically at startup.

| File | Used by |
|------|---------|
| `artifacts/simple_difficulty_optimized.json` | MCQ + T2T difficulty judge |
| `artifacts/rubric_agent_optimized.json` | MCQ + T2T rubric judge |
| `artifacts/image_mcq_difficulty_optimized.json` | Image MCQ difficulty judge |
| `artifacts/image_mcq_rubric_optimized.json` | Image MCQ rubric judge |

If an artifact is not found the judge falls back to an unoptimised `ChainOfThought` agent automatically — no error is raised.
