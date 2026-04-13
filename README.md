# Topin — AI Question Generator

Generates exam-quality questions for language learners using **DSPy** and **Mistral AI**.
Three generator types are supported: **MCQ**, **T2T** (open-answer), and **Image MCQ**.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Three Ways to Run](#three-ways-to-run)
3. [Method 1 — Jupyter Notebook](#method-1--jupyter-notebook)
4. [Method 2 — Terminal CLI (`generate.py`)](#method-2--terminal-cli-generatepy)
5. [Method 3 — dspy-cli Web UI](#method-3--dspy-cli-web-ui)
6. [Generator 1 — MCQ](#generator-1--mcq-multiple-choice-questions)
7. [Generator 2 — T2T](#generator-2--t2t-text-to-text--open-answer)
8. [Generator 3 — Image MCQ](#generator-3--image-mcq)
9. [Output Format Reference](#output-format-reference)
10. [CEFR to Difficulty Mapping](#cefr-to-difficulty-mapping)
11. [Full Generation Pipeline](#full-generation-pipeline)
12. [Judge Notebooks](#judge-notebooks-train-once-use-always)
13. [Artifacts](#artifacts)
14. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
pip install dspy-ai pydantic python-dotenv openai google-genai Pillow jupyter nbconvert

# 2. Create .env in the project root
MISTRAL_API_KEY=your_mistral_key
MISTRAL_MODEL=mistral-small-latest
MISTRAL_API_BASE=https://api.mistral.ai/v1

# 3. Open Jupyter
cd d:/Topin
jupyter notebook
```

---

## Three Ways to Run

| Method | Who | How |
|--------|-----|-----|
| **Jupyter Notebook** | Developers / analysts with Python setup | Open `.ipynb`, edit input cell, run all cells |
| **Terminal CLI** | Any team member with Python | `python generate.py --config configs/my_topic.json` |
| **dspy-cli Web UI** | Team members (browser-based, no coding) | `dspy-cli serve --system` → open `http://localhost:8000` |

---

## Method 1 — Jupyter Notebook

### Pattern (all generators)

```
Open notebook in Jupyter
       ↓
Run all cells from top to the INPUT CELL
       ↓
Edit the INPUT CELL — set topic, subtopics, counts
       ↓
Run the INPUT CELL → generation starts automatically
       ↓
Run the SAVE CELL (last cell) → output written to data/
```

See [Generator 1 — MCQ](#generator-1--mcq-multiple-choice-questions), [Generator 2 — T2T](#generator-2--t2t-text-to-text--open-answer), and [Generator 3 — Image MCQ](#generator-3--image-mcq) for the exact input cell content for each notebook.

---

## Method 2 — Terminal CLI (`generate.py`)

### Command

```bash
python generate.py --config configs/my_topic.json
```

### Workflow

```
1. Copy a template config:
     cp configs/template_mcq.json        configs/my_mcq_topic.json
     cp configs/template_t2t.json        configs/my_t2t_topic.json
     cp configs/template_image_mcq.json  configs/my_image_mcq_topic.json

2. Edit the JSON file — change topic, subtopic, and counts

3. Run:
     python generate.py --config configs/my_mcq_topic.json

4. Output is saved to:
     data/mcq/mcq_generator_output.json
     data/t2t/t2t_generator_output.json
     data/image_mcq/image_mcq_generator_output.json
```

### MCQ Config File Format

```json
{
  "type": "mcq",
  "topic": "English Grammar",
  "subtopics": [
    {
      "subtopic": "Question Words",
      "easy_count":   10,
      "medium_count": 10,
      "hard_count":    5
    }
  ],
  "questions": [
    {
      "instruction": "Choose the correct question word.",
      "question": "_______ is your teacher?",
      "options": ["Who", "What", "Where", "Why"],
      "correct_answer": "Who",
      "explanation": "Use 'Who' to ask about a person.",
      "difficulty": "Easy",
      "cefr": "A1"
    }
  ]
}
```

> **`questions`** — example questions used to guide the model (see full template at `configs/template_mcq.json`).
> Provide at least one example per CEFR level you are generating.
> `easy_count/medium_count/hard_count` are automatically split across the two CEFR levels in each band:
> Easy → A1 + A2, Medium → B1 + B2, Hard → C1 + C2.

### T2T Config File Format

```json
{
  "type": "t2t",
  "topic": "English Language Skills",
  "subtopics": [
    {
      "subtopic": "Reading and Writing",
      "a1_count": 5,
      "a2_count": 5,
      "b1_count": 3,
      "b2_count": 3,
      "c1_count": 2,
      "c2_count": 0
    }
  ],
  "constraints": {
    "questions_per_iteration": 5,
    "max_iterations_per_difficulty": 20
  }
}
```

> No `questions` (example questions) needed — T2T loads examples automatically from the training dataset.

### Image MCQ Config File Format

```json
{
  "type": "image_mcq",
  "topic": "Reading Notices and Signs",
  "subtopics": [
    {
      "subtopic": "Public Notices",
      "a1_count": 5,
      "a2_count": 5,
      "b1_count": 3,
      "b2_count": 3,
      "c1_count": 2,
      "c2_count": 0
    }
  ],
  "constraints": {
    "questions_per_iteration": 5,
    "max_iterations_per_difficulty": 20
  }
}
```

> No `questions` needed — Image MCQ loads examples automatically from the training dataset.

### Multiple Subtopics in One Config

```json
"subtopics": [
  { "subtopic": "Present Tense", "easy_count": 10, "medium_count": 10, "hard_count": 5 },
  { "subtopic": "Past Tense",    "easy_count": 10, "medium_count": 10, "hard_count": 5 }
]
```

---

## Method 3 — dspy-cli Web UI

### One-Time Setup

```bash
# Install dspy-cli globally (requires Python ≥ 3.11)
uv tool install dspy-cli
```

### Start the Server

```bash
cd d:/Topin
dspy-cli serve --system
```

Open your browser at `http://localhost:8000`.

### MCQ Generator — Web UI Fields

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `topic` | text | `English Grammar` | Topic name |
| `subtopic` | text | `Question Words` | Subtopic name |
| `easy_count` | number | `10` | Total Easy questions (split A1 + A2) |
| `medium_count` | number | `10` | Total Medium questions (split B1 + B2) |
| `hard_count` | number | `5` | Total Hard questions (split C1 + C2) |

> Example questions are loaded automatically from `configs/example_questions_mcq.json` — no upload needed.
> To change the example questions, edit that file directly.

### T2T Generator — Web UI Fields

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `topic` | text | `English Language Skills` | Topic name |
| `subtopic` | text | `Reading and Writing` | Subtopic name |
| `easy_count` | number | `10` | Total Easy questions |
| `medium_count` | number | `6` | Total Medium questions |
| `hard_count` | number | `2` | Total Hard questions |

---

## Generator 1 — MCQ (Multiple-Choice Questions)

**Notebook:** `notebooks/mcq_generator.ipynb`
**Output:** `data/mcq/mcq_generator_output.json`

### Notebook Input Cell

```python
schema = InputSchema(
    topic='English Grammar',          # ← your topic
    subtopics=[
        SubtopicRequirement(
            subtopic='Present Tense', # ← your subtopic
            a1_count=5,   # A1-level Easy questions
            a2_count=5,   # A2-level Easy questions
            b1_count=3,   # B1-level Medium questions
            b2_count=3,   # B2-level Medium questions
            c1_count=2,   # C1-level Hard questions
            c2_count=0,   # set to 0 to skip
        )
    ],
    constraints=GenerationConstraints(
        questions_per_iteration=5,        # batch size per LLM call
        max_iterations_per_difficulty=20, # retry limit per CEFR level
    ),
)
```

**Multiple subtopics:**

```python
subtopics=[
    SubtopicRequirement(subtopic='Present Tense', a1_count=5, a2_count=5, b1_count=3, b2_count=3, c1_count=2, c2_count=0),
    SubtopicRequirement(subtopic='Past Tense',    a1_count=5, a2_count=5, b1_count=3, b2_count=3, c1_count=2, c2_count=0),
]
```

### MCQ Output Format

```json
{
  "schema": {
    "topic": "English Grammar",
    "subtopics": [
      { "subtopic": "Present Tense", "a1_count": 5, "a2_count": 5, "b1_count": 3, "b2_count": 3, "c1_count": 2, "c2_count": 0 }
    ]
  },
  "summary": {
    "easy":   { "accepted": 10, "rejected": 3 },
    "medium": { "accepted": 6,  "rejected": 1 },
    "hard":   { "accepted": 2,  "rejected": 4 }
  },
  "questions": {
    "easy": [
      {
        "question_number": 1,
        "topic": "English Grammar",
        "subtopic": "Present Tense",
        "target_cefr": "A1",
        "target_difficulty": "Easy",
        "instruction": "Read the sentence and choose the correct word.",
        "question": "She _______ to school every day.",
        "options": ["go", "goes", "going", "gone"],
        "correct_answer": "goes",
        "explanation": "Third-person singular subjects require -s/-es in simple present."
      }
    ],
    "medium": [ ... ],
    "hard":   [ ... ]
  },
  "rejected": [
    { "stage": "hard_validate", "errors": ["instruction is empty"] },
    { "stage": "difficulty",    "reason": "predicted Easy, expected Medium" },
    { "stage": "rubric",        "reason": "ambiguity: Major Issue" }
  ],
  "warnings": ["A1/Easy: max iterations (20) reached. Accepted 3/5."]
}
```

**Summary keys:** `easy`, `medium`, `hard` — each has `accepted` and `rejected` counts.

---

## Generator 2 — T2T (Text-to-Text / Open Answer)

**Notebook:** `notebooks/text_to_text_generator.ipynb`
**Output:** `data/t2t/t2t_generator_output.json`

### Notebook Input Cell

```python
schema = InputSchema(
    topic='English Language Skills',  # ← your topic
    subtopics=[
        SubtopicRequirement(
            subtopic='Reading and Writing',  # ← your subtopic
            a1_count=5,   # short comprehension questions
            a2_count=5,   # word-reorder / one-sentence replies
            b1_count=3,   # paragraph from notes (35–45 words)
            b2_count=3,
            c1_count=2,
            c2_count=0,
        )
    ],
    constraints=GenerationConstraints(
        questions_per_iteration=5,
        max_iterations_per_difficulty=20,
    ),
)
```

### T2T Output Format

```json
{
  "schema": {
    "topic": "English Language Skills",
    "subtopics": [
      { "subtopic": "Reading and Writing", "a1_count": 5, "a2_count": 5, "b1_count": 3, "b2_count": 3, "c1_count": 2, "c2_count": 0 }
    ]
  },
  "summary": {
    "A1": { "accepted": 5, "target": 5 },
    "A2": { "accepted": 5, "target": 5 },
    "B1": { "accepted": 3, "target": 3 },
    "B2": { "accepted": 3, "target": 3 },
    "C1": { "accepted": 2, "target": 2 },
    "C2": { "accepted": 0, "target": 0 }
  },
  "questions": {
    "easy": [
      {
        "question_number": 1,
        "topic": "English Language Skills",
        "subtopic": "Reading and Writing",
        "target_cefr": "A1",
        "target_difficulty": "Easy",
        "question_type": "comprehension",
        "instruction": "Read the sentence carefully and answer the question.",
        "question": "The girl is reading beside the window.\n\nWhere is the girl reading?",
        "expected_answer": "Beside the window",
        "explanation": "The sentence clearly states beside the window."
      }
    ],
    "medium": [ ... ],
    "hard":   [ ... ]
  },
  "rejected": [ ... ],
  "warnings": [ ... ]
}
```

> **T2T differences from MCQ:**
> - Has `expected_answer` instead of `options` + `correct_answer`
> - Has an extra `question_type` field (e.g. `comprehension`, `word_reorder`, `sentence_completion`)
> - Summary keys are CEFR levels (`A1`–`C2`) with a `target` count, not `easy/medium/hard`

---

## Generator 3 — Image MCQ

**Notebook:** `notebooks/image_mcq_generator.ipynb`
**Output:** `data/image_mcq/image_mcq_generator_output.json`

### Notebook Input Cell

```python
schema = InputSchema(
    topic='Reading Notices and Signs',  # ← your topic
    subtopics=[
        SubtopicRequirement(
            subtopic='Public Notices',  # ← your subtopic
            a1_count=5,
            a2_count=5,
            b1_count=3,
            b2_count=3,
            c1_count=2,
            c2_count=0,
        )
    ],
    constraints=GenerationConstraints(
        questions_per_iteration=5,
        max_iterations_per_difficulty=20,
    ),
)
```

### Image MCQ Output Format

```json
{
  "schema": {
    "topic": "Reading Notices and Signs",
    "subtopics": [
      { "subtopic": "Public Notices", "a1_count": 5, "a2_count": 5, "b1_count": 3, "b2_count": 3, "c1_count": 2, "c2_count": 0 }
    ]
  },
  "summary": {
    "easy":           { "accepted": 10, "rejected": 3 },
    "medium":         { "accepted": 6,  "rejected": 2 },
    "hard":           { "accepted": 2,  "rejected": 1 },
    "total_accepted": 18,
    "total_rejected": 6
  },
  "questions": {
    "easy": [
      {
        "question_number": 1,
        "topic": "Reading Notices and Signs",
        "subtopic": "Public Notices",
        "target_cefr": "A2",
        "target_difficulty": "Easy",
        "instruction": "Read the notice and choose the correct answer.",
        "image_content": "A swimming pool safety notice says swimmers must wear a swimming cap.",
        "question": "What should you wear?",
        "options": ["A jacket.", "A swimming cap.", "A scarf.", "A pair of gloves."],
        "correct_answer": "A swimming cap.",
        "explanation": "The notice states that swimmers must wear a swimming cap."
      }
    ],
    "medium": [ ... ],
    "hard":   [ ... ]
  },
  "rejected": [ ... ],
  "warnings": [ ... ]
}
```

> **Image MCQ differences from MCQ:**
> - Has an extra `image_content` field — always a separate key, never merged into `question`
> - Summary includes `total_accepted` and `total_rejected` (MCQ does not)

---

## Output Format Reference

### Field Reference — All Question Types

| Field | MCQ | T2T | Image MCQ | Description |
|-------|:---:|:---:|:---------:|-------------|
| `question_number` | ✓ | ✓ | ✓ | Auto-incrementing integer (not a string ID) |
| `topic` | ✓ | ✓ | ✓ | Carried from input schema |
| `subtopic` | ✓ | ✓ | ✓ | Carried from input schema |
| `target_cefr` | ✓ | ✓ | ✓ | `A1`, `A2`, `B1`, `B2`, `C1`, or `C2` |
| `target_difficulty` | ✓ | ✓ | ✓ | `Easy`, `Medium`, or `Hard` |
| `instruction` | ✓ | ✓ | ✓ | Task direction shown to the learner |
| `question` | ✓ | ✓ | ✓ | The actual question text |
| `options` | ✓ | — | ✓ | Array of exactly 4 answer choices |
| `correct_answer` | ✓ | — | ✓ | Must be one of the values in `options` |
| `expected_answer` | — | ✓ | — | Model answer for open-answer questions |
| `explanation` | ✓ | ✓ | ✓ | Why the correct answer is correct |
| `question_type` | — | ✓ | — | `comprehension`, `word_reorder`, `sentence_completion`, etc. |
| `image_content` | — | — | ✓ | Text description of the notice or sign |

### Summary Format by Generator

| Generator | Summary Keys |
|-----------|-------------|
| MCQ | `easy.accepted`, `easy.rejected`, `medium.accepted`, `medium.rejected`, `hard.accepted`, `hard.rejected` |
| T2T | `A1.accepted`, `A1.target`, `A2.accepted`, `A2.target`, ... `C2.accepted`, `C2.target` |
| Image MCQ | Same as MCQ + `total_accepted`, `total_rejected` |

### Rejected Item Format

Each item in the `rejected` array has:

```json
{
  "stage": "hard_validate",
  "errors": ["instruction is empty", "correct_answer not in options"]
}
```

or

```json
{
  "stage": "difficulty",
  "reason": "predicted Easy, expected Medium"
}
```

or

```json
{
  "stage": "rubric",
  "reason": "ambiguity: Major Issue"
}
```

Possible `stage` values: `hard_validate`, `difficulty`, `rubric`.

---

## CEFR to Difficulty Mapping

| CEFR | Difficulty | Notebook field | Config field (MCQ CLI) | Config field (T2T/Image CLI) |
|------|-----------|---------------|----------------------|---------------------------|
| A1   | Easy      | `a1_count`    | `easy_count` ÷ 2     | `a1_count` |
| A2   | Easy      | `a2_count`    | `easy_count` ÷ 2     | `a2_count` |
| B1   | Medium    | `b1_count`    | `medium_count` ÷ 2   | `b1_count` |
| B2   | Medium    | `b2_count`    | `medium_count` ÷ 2   | `b2_count` |
| C1   | Hard      | `c1_count`    | `hard_count` ÷ 2     | `c1_count` |
| C2   | Hard      | `c2_count`    | `hard_count` ÷ 2     | `c2_count` |

> When using `easy_count/medium_count/hard_count` in the MCQ CLI config, `generate.py` automatically splits them evenly across the two CEFR levels in each band.
> Set any count to `0` to skip that CEFR level entirely.

---

## Full Generation Pipeline

```
InputSchema
  topic, subtopics [ { subtopic, a1_count … c2_count } ]
  constraints { questions_per_iteration, max_iterations_per_difficulty }
        │
        ▼
Orchestrator.run()
  iterates CEFR levels: A1 → A2 → B1 → B2 → C1 → C2
        │
        ▼  (for each CEFR level with count > 0)
GeneratorAgent.forward()              ← quota loop
        │
        │  while store.count_by_cefr(cefr) < target_count:
        │
        │    Step 1: Generate batch
        │            ChainOfThought LLM call (Mistral)
        │            Input:  topic, subtopic, target_cefr, example_questions, batch_size
        │            Output: list of GeneratedQuestion
        │
        │    Step 2: hard_validate()
        │            ✓ instruction not empty
        │            ✓ question not empty
        │            ✓ exactly 4 options (MCQ / Image MCQ only)
        │            ✓ correct_answer is one of the options (MCQ / Image MCQ only)
        │            ✓ explanation not empty
        │            ✓ image_content not empty (Image MCQ only)
        │
        │    Step 3: DifficultyJudge  (Mistral LLM call)
        │            Classifies each question's CEFR level
        │            Rejects if predicted CEFR ≠ target band
        │
        │    Step 4: RubricJudge  (Mistral LLM call)
        │            Evaluates quality criteria (10 criteria for MCQ/T2T, 11 for Image MCQ)
        │            overall_decision must be "Pass" to accept
        │
        │    Step 5: store.add(accepted)
        │
        ▼
GenerationResult
  store.easy    [ accepted Easy questions ]
  store.medium  [ accepted Medium questions ]
  store.hard    [ accepted Hard questions ]
  rejected      [ failed attempts with stage + reason ]
  warnings      [ quota-not-met messages ]
        │
        ▼
Save cell writes output JSON to data/<type>/<type>_generator_output.json
```

---

## Judge Notebooks (Train Once, Use Always)

The three judge notebooks are standalone training notebooks. Run each once to produce optimised artifacts that all generator notebooks load automatically.

```
difficulty_judge.ipynb  →  artifacts/simple_difficulty_optimized.json
                                     ↑ loaded by mcq_generator.ipynb
                                              text_to_text_generator.ipynb

rubric_judge.ipynb      →  artifacts/rubric_agent_optimized.json
                                     ↑ loaded by mcq_generator.ipynb
                                              text_to_text_generator.ipynb

image_judge.ipynb       →  artifacts/image_judge_optimized.json
(trains Pixtral vision judge)        ↑ loaded by image_mcq_generator.ipynb
```

If an artifact is not found the judge falls back to an unoptimised `ChainOfThought` automatically — no error is raised.

### Running `difficulty_judge.ipynb` or `rubric_judge.ipynb`

Fully self-contained. Run all cells top-to-bottom — no input to edit.

### Running `image_judge.ipynb`

**Additional requirements:**

```bash
pip install google-genai Pillow
```

**Additional `.env` keys needed:**

```
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
```

**What each cell does:**

| Cell | Action |
|------|--------|
| 1    | Setup — paths, venv injection |
| 2    | Configure DSPy (Pixtral LM) + DALL-E + Gemini clients |
| 3    | Define Pydantic models + DSPy signature + agent |
| 4    | Define metric (`aligned`=1.0, `not_aligned`=0.0) |
| 5    | Load `training_dataset_24_judge_ready.json` (24 questions, Q1–Q24) |
| 6    | **Generate 96 training images** — 48 DALL-E + 48 Gemini (skips existing files) |
| 7    | Build DSPy examples from generated images |
| 8    | Baseline evaluation (before optimization) |
| 9    | Run GEPA optimizer → BootstrapFewShot fallback → save artifact |
| 10   | Load optimized agent + post-optimization evaluation |
| 11   | Write Promptfoo provider |
| 12   | Build Promptfoo test cases (Q21–Q24) |
| 13   | Write Promptfoo config YAML |
| 14   | Run Promptfoo eval |
| 15   | Display results |

> Cell 6 makes ~96 API calls. Run it once — it skips files that already exist on re-runs.

---

## RubricJudge Quality Criteria

### MCQ and T2T (10 criteria)

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

### Image MCQ (11 criteria — adds highest-priority check)

| Criterion | Effect |
|-----------|--------|
| `image_content_coherence` = `Incoherent` | Forces **Fail** immediately |
| `image_content_coherence` = `Partially Coherent` | Forces **Revise** |
| `image_content_coherence` = `Coherent` | Normal evaluation continues |

---

## Project Structure

```
Topin/
├── notebooks/
│   ├── mcq_generator.ipynb           # Generator 1 — MCQ
│   ├── text_to_text_generator.ipynb  # Generator 2 — T2T
│   ├── image_mcq_generator.ipynb     # Generator 3 — Image MCQ
│   ├── difficulty_judge.ipynb        # Judge training (run once)
│   ├── rubric_judge.ipynb            # Judge training (run once)
│   └── image_judge.ipynb             # Judge training (run once)
├── configs/
│   ├── template_mcq.json             # MCQ config template (copy + edit)
│   ├── template_t2t.json             # T2T config template (copy + edit)
│   ├── template_image_mcq.json       # Image MCQ config template (copy + edit)
│   └── example_questions_mcq.json    # Example questions auto-loaded by MCQ generator
├── data/
│   ├── mcq/
│   │   ├── mcq_generator_output.json         ← generated output
│   │   ├── training_dataset_standard.json
│   │   └── eval_dataset_standard.json
│   ├── t2t/
│   │   ├── t2t_generator_output.json         ← generated output
│   │   ├── training_dataset_clean_full.json
│   │   └── eval_dataset_clean.json
│   └── image_mcq/
│       ├── image_mcq_generator_output.json   ← generated output
│       ├── training_dataset_24_clean.json
│       ├── eval_dataset_24_clean.json
│       └── training_dataset_24_judge_ready.json
├── artifacts/
│   ├── simple_difficulty_optimized.json
│   ├── rubric_agent_optimized.json
│   └── image_judge_optimized.json
├── src/
│   └── topin/
│       └── modules/
│           ├── mcq_generator.py      # dspy-cli MCQ module
│           └── t2t_generator.py      # dspy-cli T2T module
├── generate.py                       # CLI runner
├── dspy.config.yaml                  # dspy-cli configuration
├── utils.py
└── .env
```

---

## Artifacts

| File | Created by | Used by |
|------|-----------|---------|
| `artifacts/simple_difficulty_optimized.json` | `difficulty_judge.ipynb` | `mcq_generator.ipynb`, `text_to_text_generator.ipynb` |
| `artifacts/rubric_agent_optimized.json` | `rubric_judge.ipynb` | `mcq_generator.ipynb`, `text_to_text_generator.ipynb` |
| `artifacts/image_judge_optimized.json` | `image_judge.ipynb` | `image_mcq_generator.ipynb` |

---

## Environment Variables

Create a `.env` file in `d:/Topin/`:

```
MISTRAL_API_KEY=your_mistral_key       # required for all generators
MISTRAL_MODEL=mistral-small-latest
MISTRAL_API_BASE=https://api.mistral.ai/v1

OPENAI_API_KEY=your_openai_key         # required for image_judge.ipynb (DALL-E 3)
GOOGLE_API_KEY=your_google_key         # required for image_judge.ipynb (Gemini Imagen 3)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: dspy` | Run `pip install dspy-ai` inside your venv |
| `KeyError: MISTRAL_API_KEY` | Check `.env` exists at `d:/Topin/.env` with the correct key |
| `No module named jupyter` | Run `pip install jupyter nbconvert` in the project venv |
| `dspy-cli is not installed in your project virtual environment` | Use `dspy-cli serve --system` (bypasses the venv check) |
| `Not a valid DSPy project directory` | Run from `d:/Topin/` — the directory must contain `dspy.config.yaml` and `src/` |
| `Configuration must contain 'models' section` | Check `dspy.config.yaml` has a `models.registry` block |
| Questions rejected at difficulty stage | Improve the example questions in `configs/example_questions_mcq.json` or increase `max_iterations_per_difficulty` |
| Questions rejected at rubric stage | Run `rubric_judge.ipynb` to optimise the rubric judge first |
| Cell 6 in `image_judge.ipynb` fails for Gemini | Check `GOOGLE_API_KEY` in `.env`; run `pip install google-genai` |
| Output has warnings about quota not met | Increase `max_iterations_per_difficulty` or decrease question counts |
