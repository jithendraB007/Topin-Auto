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
9. [Image Review Pipeline](#image-review-pipeline)
10. [Output Format Reference](#output-format-reference)
11. [CEFR to Difficulty Mapping](#cefr-to-difficulty-mapping)
12. [Full Generation Pipeline](#full-generation-pipeline)
13. [Judge Notebooks](#judge-notebooks-train-once-use-always)
14. [Project Structure](#project-structure)
15. [Artifacts](#artifacts)
16. [Environment Variables](#environment-variables)
17. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
pip install dspy-ai pydantic python-dotenv openai google-genai Pillow jupyter nbconvert requests mistralai

# 2. Create .env in the project root
MISTRAL_API_KEY=your_mistral_key
MISTRAL_MODEL=mistral-small-latest
MISTRAL_API_BASE=https://api.mistral.ai/v1
HF_TOKEN=hf_...        # free — for image generation (get at huggingface.co/settings/tokens)

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

### Available Modules in the Web UI

| Module | Description |
|--------|-------------|
| **MCQ Generator** | Generates multiple-choice questions |
| **T2T Generator** | Generates open-answer / writing questions |
| **Image MCQ Generator** | Generates image-based MCQ questions (with `image_content` descriptions) |

### MCQ Generator — Web UI Fields

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `topic` | text | `English Grammar` | Topic name |
| `subtopic` | text | `Question Words` | Subtopic name |
| `easy_count` | number | `10` | Total Easy questions (split A1 + A2) |
| `medium_count` | number | `10` | Total Medium questions (split B1 + B2) |
| `hard_count` | number | `5` | Total Hard questions (split C1 + C2) |

**Example questions** are loaded automatically from `configs/example_questions_mcq.json`.
To use different example questions, edit that file **before** clicking Run.
The file format is a JSON array — see [Example Questions File Format](#example-questions-file-format).

### T2T Generator — Web UI Fields

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `topic` | text | `English Language Skills` | Topic name |
| `subtopic` | text | `Reading and Writing` | Subtopic name |
| `easy_count` | number | `10` | Total Easy questions |
| `medium_count` | number | `6` | Total Medium questions |
| `hard_count` | number | `2` | Total Hard questions |

### Image MCQ Generator — Web UI Fields

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `topic` | text | `Reading Notices and Signs` | Topic name |
| `subtopic` | text | `Public Notices` | Subtopic name |
| `easy_count` | number | `2` | Total Easy questions (split A1 + A2) |
| `medium_count` | number | `2` | Total Medium questions (split B1 + B2) |
| `hard_count` | number | `1` | Total Hard questions (split C1 + C2) |

After generation completes, run the [Image Review Pipeline](#image-review-pipeline) to generate actual images and quality scores.

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
        "question_number": "Q1",
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
    "medium": [ { "question_number": "Q3", "...": "..." } ],
    "hard":   [ { "question_number": "Q4", "...": "..." } ]
  },
  "rejected": [
    { "stage": "hard_validate", "errors": ["instruction is empty"] },
    { "stage": "difficulty",    "reason": "predicted Easy, expected Medium" },
    { "stage": "rubric",        "reason": "ambiguity: Major Issue" }
  ],
  "warnings": ["A1/Easy: max iterations (20) reached. Accepted 3/5."]
}
```

**Question numbering:** `Q1`, `Q2`, `Q3` … sequentially across all difficulty buckets (Easy first, then Medium, then Hard).

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
        "question_number": "Q1",
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
    "medium": [ { "question_number": "Q3", "...": "..." } ],
    "hard":   [ { "question_number": "Q4", "...": "..." } ]
  }
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
  "questions": {
    "easy": [
      {
        "question_number": "Q1",
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
    "medium": [ { "question_number": "Q3", "...": "..." } ],
    "hard":   [ { "question_number": "Q4", "...": "..." } ]
  }
}
```

> **Key field:** `image_content` — a text description of the real-world notice or sign.
> Run [Image Review Pipeline](#image-review-pipeline) to turn these descriptions into actual images.

---

## Image Review Pipeline

After generating Image MCQ questions, use these two scripts to generate actual images and evaluate their quality.

### Step 1 — Generate Images

```bash
python generate_review_images.py
```

Reads `data/image_mcq/image_mcq_generator_output.json`, generates one image per question using **FLUX.1-schnell** (Hugging Face, free), and produces a browser-viewable HTML review page.

**Outputs:**
- `data/image_mcq/review_images/1.png`, `2.png`, … (one per question)
- `data/image_mcq/review.html` — open in browser to review all questions with images

**Requires:** `HF_TOKEN` in `.env` (free — get at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

**Options:**

```bash
python generate_review_images.py                          # auto-detect provider
python generate_review_images.py --provider hf            # force Hugging Face (free)
python generate_review_images.py --provider dalle         # force DALL-E 3 (paid, needs OPENAI_API_KEY)
python generate_review_images.py --provider gemini        # force Gemini Imagen (paid, needs GOOGLE_API_KEY)
python generate_review_images.py --input path/to/file.json  # custom input file
```

**Provider priority (auto mode):** Hugging Face → DALL-E 3 → Gemini Imagen

Already-generated images are skipped automatically on re-runs.

### Step 2 — Judge Image Quality

```bash
python judge_images.py
```

Sends each generated image to **Pixtral-12B** (Mistral vision model) for quality evaluation using your existing `MISTRAL_API_KEY`.

**Outputs:**
- `data/image_mcq/image_judge_output.json` — full rubric scores per question
- `data/image_mcq/review_with_scores.html` — open in browser to see images with scores

**Rubric (4 criteria, scored Excellent / Good / Poor):**

| Criterion | What it checks |
|-----------|---------------|
| `relevance_to_description` | Does the image match the `image_content` description? |
| `visual_quality` | Is the image realistic and well-rendered? |
| `text_legibility` | Is any text in the image readable? |
| `contextual_fit` | Does it look like a genuine real-world notice or sign? |

Plus an `overall_score` (1–10) and brief `reasoning`.

**Requires:** `MISTRAL_API_KEY` in `.env` (already configured for generation).

### Full Image Workflow

```
dspy-cli / generate.py / Jupyter
          │
          ▼
data/image_mcq/image_mcq_generator_output.json
   (questions with image_content text descriptions)
          │
          ▼  python generate_review_images.py
data/image_mcq/review_images/1.png, 2.png, ...
data/image_mcq/review.html            ← open to review
          │
          ▼  python judge_images.py
data/image_mcq/image_judge_output.json
data/image_mcq/review_with_scores.html  ← open to review with scores
```

---

## Example Questions File Format

The file `configs/example_questions_mcq.json` guides the model on style, vocabulary level, and structure. Edit it before running the MCQ generator.

The module accepts **two formats** — use whichever is easier for your team:

**Format A — plain JSON array (minimal):**

```json
[
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
```

**Format B — full config object** (the module reads from the `"questions"` key automatically):

```json
{
  "type": "mcq",
  "topic": "English Grammar",
  "subtopics": [ { "subtopic": "Conditional Sentences", "easy_count": 10, "medium_count": 8, "hard_count": 4 } ],
  "questions": [
    { "instruction": "...", "question": "...", "options": ["..."], "correct_answer": "...", "explanation": "...", "difficulty": "Easy", "cefr": "A1" }
  ]
}
```

**Field rules:**

| Field | Rule |
|-------|------|
| `instruction` | Short task direction shown to the learner |
| `question` | The MCQ question text (use `_______` for blanks) |
| `options` | Exactly **4** items |
| `correct_answer` | Copied **verbatim** from one of the options |
| `explanation` | Why the correct answer is correct |
| `difficulty` | `Easy`, `Medium`, or `Hard` only |
| `cefr` | `A1`, `A2`, `B1`, `B2`, `C1`, or `C2` only |

---

## Output Format Reference

### Field Reference — All Question Types

| Field | MCQ | T2T | Image MCQ | Description |
|-------|:---:|:---:|:---------:|-------------|
| `question_number` | ✓ | ✓ | ✓ | Sequential string ID: `"Q1"`, `"Q2"`, `"Q3"` … (Easy → Medium → Hard order) |
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
generate.py reformats question_number → Q1, Q2, Q3 …
        │
        ▼
Output JSON saved to data/<type>/<type>_generator_output.json
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

Fully self-contained. Run all cells top-to-bottom — no input to edit.

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
│   │   └── mcq_generator_output.json
│   ├── t2t/
│   │   └── t2t_generator_output.json
│   └── image_mcq/
│       ├── image_mcq_generator_output.json   ← question generator output
│       ├── review_images/                    ← generated PNGs (1.png, 2.png, …)
│       ├── review.html                       ← open in browser (images + questions)
│       ├── image_judge_output.json           ← Pixtral quality scores
│       └── review_with_scores.html           ← open in browser (images + scores)
├── artifacts/
│   ├── simple_difficulty_optimized.json
│   ├── rubric_agent_optimized.json
│   └── image_judge_optimized.json
├── src/
│   └── topin/
│       └── modules/
│           ├── mcq_generator.py        # dspy-cli MCQ module
│           ├── t2t_generator.py        # dspy-cli T2T module
│           └── image_mcq_generator.py  # dspy-cli Image MCQ module
├── generate.py                         # CLI runner (all 3 generators)
├── generate_review_images.py           # Image generation script (HF / DALL-E / Gemini)
├── judge_images.py                     # Image quality judge (Pixtral)
├── dspy.config.yaml                    # dspy-cli configuration
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
# Required — all generators use Mistral AI
MISTRAL_API_KEY=your_mistral_key
MISTRAL_MODEL=mistral-small-latest
MISTRAL_API_BASE=https://api.mistral.ai/v1

# Required for image generation (free)
HF_TOKEN=hf_...              # get at https://huggingface.co/settings/tokens

# Optional — alternative image generation providers (paid)
OPENAI_API_KEY=your_openai_key    # for DALL-E 3
GOOGLE_API_KEY=your_google_key    # for Gemini Imagen
```

**Image provider priority** (auto mode in `generate_review_images.py`):
1. `HF_TOKEN` → FLUX.1-schnell (Hugging Face, **free**)
2. `OPENAI_API_KEY` → DALL-E 3 (paid)
3. `GOOGLE_API_KEY` → Gemini Imagen (paid)

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
| `HF FAILED: HTTP 410` | Old HF endpoint — update URL to `router.huggingface.co` in `generate_review_images.py` |
| `HF FAILED: HTTP 503` | Model is loading on HF servers — the script retries automatically; wait ~1 min and re-run |
| Gemini image generation: `limit: 0` quota error | Gemini image models require billing. Use `--provider hf` (free) instead |
| `cannot import name 'Mistral' from 'mistralai'` | Use `from mistralai.client import Mistral` — the installed SDK version uses this path |
| Questions rejected at difficulty stage | Improve the example questions in `configs/example_questions_mcq.json` or increase `max_iterations_per_difficulty` |
| Questions rejected at rubric stage | Run `rubric_judge.ipynb` to optimise the rubric judge first |
| Output has warnings about quota not met | Increase `max_iterations_per_difficulty` or decrease question counts |
| `review_with_scores.html` shows "No image" | Run `generate_review_images.py` first before `judge_images.py` |
