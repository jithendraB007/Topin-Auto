# Topin — AI Question Generator

Generates exam-quality questions for language learners using **DSPy** and **Mistral AI**.
Three generator types: **MCQ** (multiple-choice), **T2T** (open-answer), **Image MCQ** (notice/sign based).

---

## Table of Contents

1. [First-Time Setup](#first-time-setup)
2. [Environment Variables (.env)](#environment-variables-env)
3. [Three Ways to Run](#three-ways-to-run)
4. [Method 1 — Jupyter Notebook](#method-1--jupyter-notebook)
5. [Method 2 — Terminal CLI (generate.py)](#method-2--terminal-cli-generatepy)
6. [Method 3 — dspy-cli Web UI](#method-3--dspy-cli-web-ui)
7. [Generator 1 — MCQ](#generator-1--mcq-multiple-choice-questions)
8. [Generator 2 — T2T](#generator-2--t2t-text-to-text--open-answer)
9. [Generator 3 — Image MCQ](#generator-3--image-mcq)
10. [Image Review Pipeline](#image-review-pipeline)
11. [Example Questions File (MCQ)](#example-questions-file-mcq)
12. [Output Format Reference](#output-format-reference)
13. [CEFR to Difficulty Mapping](#cefr-to-difficulty-mapping)
14. [Full Generation Pipeline (Internal)](#full-generation-pipeline-internal)
15. [Judge Notebooks (Train Once)](#judge-notebooks-train-once)
16. [Project Structure](#project-structure)
17. [Artifacts](#artifacts)
18. [Troubleshooting](#troubleshooting)

---

## First-Time Setup

### 1. Create and activate virtual environment

```bash
cd d:/Topin

# Create venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

### 2. Install all dependencies

```bash
pip install dspy-ai pydantic python-dotenv openai google-genai Pillow jupyter nbconvert requests mistralai nbclient nbformat google-genai
```

### 3. Create the .env file

Create a file named `.env` in `d:/Topin/` with the content shown in the [Environment Variables](#environment-variables-env) section below.

### 4. Install dspy-cli (for Web UI — one time only)

```bash
# Requires Python ≥ 3.11 and uv installed globally
uv tool install dspy-cli
```

---

## Environment Variables (.env)

Create `d:/Topin/.env` with the following keys:

```
# ── Required for all generators ───────────────────────────────────────────────
MISTRAL_API_KEY=your_mistral_key
MISTRAL_MODEL=open-mistral-nemo
MISTRAL_API_BASE=https://api.mistral.ai/v1

# ── Required for image generation (free) ──────────────────────────────────────
HF_TOKEN=hf_...
# Get a free token at: https://huggingface.co/settings/tokens
# Click "New token" → Role: Read → Create → copy the hf_... value

# ── Optional — paid image generation providers ─────────────────────────────────
OPENAI_API_KEY=sk-...       # for DALL-E 3
GOOGLE_API_KEY=AIza...      # for Gemini Imagen
```

**Image provider priority** (auto mode): HF (free) → DALL-E → Gemini

---

## Three Ways to Run

| Method | Command | Best for |
|--------|---------|---------|
| **Jupyter Notebook** | `jupyter notebook` | Developers — full visibility into each step |
| **Terminal CLI** | `python generate.py --config configs/file.json` | Any team member — config-driven batch generation |
| **dspy-cli Web UI** | `dspy-cli serve --system` → `http://localhost:8000` | Non-technical users — browser form, no coding |

---

## Method 1 — Jupyter Notebook

### Start Jupyter

```bash
cd d:/Topin
.venv/Scripts/activate     # activate venv first
jupyter notebook
```

Then open a notebook from the `notebooks/` folder in the browser.

### How to use each notebook

```
1. Open the notebook in Jupyter
2. Run ALL cells from the top (Kernel → Restart & Run All)
   — OR — run cells one by one top to bottom
3. When you reach the INPUT CELL (labelled in the notebook):
       Edit the schema — change topic, subtopic, counts
4. Run the INPUT CELL — generation starts automatically
5. Wait for all cells to complete
6. The last cell saves output to data/<type>/<type>_generator_output.json
```

### Which notebook for which generator?

| Generator | Notebook file |
|-----------|--------------|
| MCQ | `notebooks/mcq_generator.ipynb` |
| T2T | `notebooks/text_to_text_generator.ipynb` |
| Image MCQ | `notebooks/image_mcq_generator.ipynb` |

---

## Method 2 — Terminal CLI (generate.py)

### List available generator types

```bash
python generate.py --list-types
```

Output:

```
Available generator types:
  mcq          mcq_generator.ipynb          [OK]
  t2t          text_to_text_generator.ipynb [OK]
  image_mcq    image_mcq_generator.ipynb    [OK]
```

### Run a generator from a config file

```bash
python generate.py --config configs/my_config.json
```

### Full workflow — step by step

```bash
# Step 1 — Copy the right template
cp configs/template_mcq.json       configs/my_mcq.json         # for MCQ
cp configs/template_t2t.json       configs/my_t2t.json         # for T2T
cp configs/template_image_mcq.json configs/my_image_mcq.json   # for Image MCQ

# Step 2 — Edit the config (change topic, subtopic, counts)
# Open configs/my_mcq.json in any editor

# Step 3 — Run
python generate.py --config configs/my_mcq.json

# Step 4 — Find your output
# MCQ      → data/mcq/mcq_generator_output.json
# T2T      → data/t2t/t2t_generator_output.json
# Image MCQ → data/image_mcq/image_mcq_generator_output.json
```

### MCQ config format

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
  "constraints": {
    "questions_per_iteration": 5,
    "max_iterations_per_difficulty": 20
  },
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

> `questions` = example questions that guide the model's style and difficulty level.
> `easy_count/medium_count/hard_count` are split evenly: Easy → A1+A2, Medium → B1+B2, Hard → C1+C2.

### T2T config format

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

> No `questions` key needed — T2T loads example questions automatically from the training dataset.

### Image MCQ config format

```json
{
  "type": "image_mcq",
  "topic": "Reading Notices and Signs",
  "subtopics": [
    {
      "subtopic": "Public Notices",
      "a1_count": 2,
      "a2_count": 2,
      "b1_count": 2,
      "b2_count": 1,
      "c1_count": 1,
      "c2_count": 0
    }
  ],
  "constraints": {
    "questions_per_iteration": 5,
    "max_iterations_per_difficulty": 20
  }
}
```

> No `questions` key needed — Image MCQ loads examples automatically from the training dataset.

### Multiple subtopics in one run

```json
"subtopics": [
  { "subtopic": "Present Tense", "easy_count": 10, "medium_count": 10, "hard_count": 5 },
  { "subtopic": "Past Tense",    "easy_count": 10, "medium_count": 10, "hard_count": 5 }
]
```

Both subtopics are generated in a single run and combined in the same output file.

---

## Method 3 — dspy-cli Web UI

### Start the server

```bash
cd d:/Topin
dspy-cli serve --system
```

Then open `http://localhost:8000` in your browser.

### Stop the server

Press `Ctrl + C` in the terminal.

### Available modules in the Web UI

| Module name | What it generates | Output file |
|-------------|-------------------|-------------|
| **MCQ Generator** | Multiple-choice questions | `data/mcq/mcq_generator_output.json` |
| **T2T Generator** | Open-answer / writing questions | `data/t2t/t2t_generator_output.json` |
| **Image MCQ Generator** | Image-based MCQs (with `image_content`) | `data/image_mcq/image_mcq_generator_output.json` |

### Web UI fields — all three modules

All three modules share the same 5 fields:

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `topic` | text | `English Grammar` | The overall topic |
| `subtopic` | text | `Question Words` | The specific subtopic |
| `easy_count` | number | `4` | Total Easy questions (auto-split A1 + A2) |
| `medium_count` | number | `4` | Total Medium questions (auto-split B1 + B2) |
| `hard_count` | number | `2` | Total Hard questions (auto-split C1 + C2) |

### MCQ — example questions

For MCQ, example questions are loaded automatically from `configs/example_questions_mcq.json`.
Edit that file **before clicking Run** to change what examples the model sees.
See [Example Questions File (MCQ)](#example-questions-file-mcq) for the format.

### After generating Image MCQ questions from the Web UI

The Web UI only generates the question text (with `image_content` descriptions).
To get actual images, run the Image Review Pipeline after the Web UI finishes:

```bash
python generate_review_images.py   # Step 1 — generate images from image_content
python judge_images.py             # Step 2 — score images with Pixtral
```

---

## Generator 1 — MCQ (Multiple-Choice Questions)

**Notebook:** `notebooks/mcq_generator.ipynb`
**Output:** `data/mcq/mcq_generator_output.json`

### Notebook input cell

```python
schema = InputSchema(
    topic='English Grammar',
    subtopics=[
        SubtopicRequirement(
            subtopic='Present Tense',
            a1_count=5,   # Easy A1
            a2_count=5,   # Easy A2
            b1_count=3,   # Medium B1
            b2_count=3,   # Medium B2
            c1_count=2,   # Hard C1
            c2_count=0,   # Hard C2  (0 = skip)
        )
    ],
    constraints=GenerationConstraints(
        questions_per_iteration=5,
        max_iterations_per_difficulty=20,
    ),
)

example_questions = ExampleQuestionSet(items=[
    ExampleQuestion(
        instruction='Read the sentence and choose the correct word.',
        question='She _______ to school every day.',
        options=['go', 'goes', 'going', 'gone'],
        correct_answer='goes',
        explanation='Third-person singular requires -s/-es in simple present.',
        difficulty='Easy',
        cefr='A1',
        subtopic='Present Tense',
    ),
    # add more examples ...
])
```

### MCQ output format

```json
{
  "schema": { "topic": "English Grammar", "subtopics": [ { "subtopic": "Present Tense", "a1_count": 5, "..." : "..." } ] },
  "summary": {
    "easy":   { "accepted": 10, "rejected": 3 },
    "medium": { "accepted": 6,  "rejected": 1 },
    "hard":   { "accepted": 2,  "rejected": 4 }
  },
  "questions": {
    "easy": [
      {
        "question_number":   "Q1",
        "topic":             "English Grammar",
        "subtopic":          "Present Tense",
        "target_cefr":       "A1",
        "target_difficulty": "Easy",
        "instruction":       "Read the sentence and choose the correct word.",
        "question":          "She _______ to school every day.",
        "options":           ["go", "goes", "going", "gone"],
        "correct_answer":    "goes",
        "explanation":       "Third-person singular requires -s/-es in simple present."
      },
      { "question_number": "Q2", "target_cefr": "A2", "...": "..." }
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

**`question_number`** is formatted as `Q1`, `Q2`, `Q3` … sequentially: Easy first, then Medium, then Hard.

---

## Generator 2 — T2T (Text-to-Text / Open Answer)

**Notebook:** `notebooks/text_to_text_generator.ipynb`
**Output:** `data/t2t/t2t_generator_output.json`

### Notebook input cell

```python
schema = InputSchema(
    topic='English Language Skills',
    subtopics=[
        SubtopicRequirement(
            subtopic='Reading and Writing',
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

> No `example_questions` needed — T2T loads them automatically from the training dataset.

### T2T output format

```json
{
  "schema": { "topic": "English Language Skills", "...": "..." },
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
        "question_number":   "Q1",
        "topic":             "English Language Skills",
        "subtopic":          "Reading and Writing",
        "target_cefr":       "A1",
        "target_difficulty": "Easy",
        "question_type":     "comprehension",
        "instruction":       "Read the sentence carefully and answer the question.",
        "question":          "The girl is reading beside the window.\n\nWhere is the girl reading?",
        "expected_answer":   "Beside the window",
        "explanation":       "The sentence clearly states beside the window."
      }
    ],
    "medium": [ { "question_number": "Q3", "...": "..." } ],
    "hard":   [ { "question_number": "Q4", "...": "..." } ]
  }
}
```

> **T2T differences from MCQ:**
> - `expected_answer` instead of `options` + `correct_answer`
> - `question_type` field: `comprehension`, `word_reorder`, `sentence_completion`, etc.
> - Summary uses CEFR keys (`A1`–`C2`) with `target` count, not `easy/medium/hard`

---

## Generator 3 — Image MCQ

**Notebook:** `notebooks/image_mcq_generator.ipynb`
**Output:** `data/image_mcq/image_mcq_generator_output.json`

### How it works

```
Generator creates questions where each question has an "image_content" field —
a text description of a real-world notice or sign, e.g.:
  "A library notice says the building will be closed for maintenance tomorrow."

This description is later used by generate_review_images.py to produce an
actual image of that notice using AI image generation.
```

### Notebook input cell

```python
schema = InputSchema(
    topic='Reading Notices and Signs',
    subtopics=[
        SubtopicRequirement(
            subtopic='Public Notices',
            a1_count=2,
            a2_count=2,
            b1_count=2,
            b2_count=1,
            c1_count=1,
            c2_count=0,
        )
    ],
    constraints=GenerationConstraints(
        questions_per_iteration=5,
        max_iterations_per_difficulty=20,
    ),
)
```

> No `example_questions` needed — Image MCQ loads them automatically from the training dataset.

### Image MCQ output format

```json
{
  "schema": { "topic": "Reading Notices and Signs", "...": "..." },
  "summary": {
    "easy":           { "accepted": 4,  "rejected": 1 },
    "medium":         { "accepted": 3,  "rejected": 2 },
    "hard":           { "accepted": 1,  "rejected": 0 },
    "total_accepted": 8,
    "total_rejected": 3
  },
  "questions": {
    "easy": [
      {
        "question_number":   "Q1",
        "topic":             "Reading Notices and Signs",
        "subtopic":          "Public Notices",
        "target_cefr":       "A2",
        "target_difficulty": "Easy",
        "instruction":       "Read the notice and choose the correct answer.",
        "image_content":     "A library notice says the building will be closed for maintenance tomorrow.",
        "question":          "Why will the library be closed tomorrow?",
        "options":           ["For a holiday", "For maintenance", "For a special event", "For a meeting"],
        "correct_answer":    "For maintenance",
        "explanation":       "The notice states the building will be closed for maintenance."
      }
    ],
    "medium": [ { "question_number": "Q3", "...": "..." } ],
    "hard":   [ { "question_number": "Q4", "...": "..." } ]
  }
}
```

> **Key field:** `image_content` — a plain-English description of the notice/sign.
> This is what `generate_review_images.py` reads to generate the actual image.

---

## Image Review Pipeline

This pipeline runs **after** Image MCQ questions are generated.
It turns the `image_content` text descriptions into actual images and scores them.

### Complete step-by-step

```
Step 1 — Generate Image MCQ questions
   (use Jupyter notebook, generate.py CLI, or dspy-cli Web UI)
   Output: data/image_mcq/image_mcq_generator_output.json
   Each question has an "image_content" description field

        ↓

Step 2 — Generate actual images
   Command: python generate_review_images.py
   Reads:   data/image_mcq/image_mcq_generator_output.json  (the image_content fields)
   Writes:  data/image_mcq/review_images/1.png, 2.png, 3.png, ...
            data/image_mcq/review.html   ← open in browser to review

        ↓

Step 3 — Judge image quality
   Command: python judge_images.py
   Reads:   data/image_mcq/review_images/1.png, 2.png, ...
            data/image_mcq/image_mcq_generator_output.json
   Writes:  data/image_mcq/image_judge_output.json
            data/image_mcq/review_with_scores.html  ← open in browser for scores
```

---

### Step 2 — generate_review_images.py

#### Basic usage (auto-detect provider)

```bash
python generate_review_images.py
```

#### All options

```bash
# Specify a custom input file
python generate_review_images.py --input data/image_mcq/image_mcq_generator_output.json

# Force a specific image provider
python generate_review_images.py --provider hf          # Hugging Face FLUX.1-schnell (free)
python generate_review_images.py --provider gptimage    # GPT Image 1 — best quality (paid)
python generate_review_images.py --provider dalle       # DALL-E 3 (needs OPENAI_API_KEY, paid)
python generate_review_images.py --provider gemini      # Gemini Imagen (needs GOOGLE_API_KEY, paid)
```

#### Provider comparison

| Provider | Key needed | Cost | Model | Quality |
|----------|-----------|------|-------|---------|
| `hf` (default) | `HF_TOKEN` | **Free** | FLUX.1-schnell | Good |
| `gptimage` | `OPENAI_API_KEY` | Paid | GPT Image 1 (`gpt-image-1`) | **Best** |
| `dalle` | `OPENAI_API_KEY` | Paid | DALL-E 3 | Excellent |
| `gemini` | `GOOGLE_API_KEY` | Paid | Gemini Imagen | Excellent |

#### What it does internally

```
For each question in image_mcq_generator_output.json:
  1. Reads the "image_content" field
  2. Builds an image prompt:
       "A realistic photograph of a printed public notice or sign.
        {image_content}. Professional printed notice board, clean typography..."
  3. Sends the prompt to the image generation API
  4. Downloads and saves the image as data/image_mcq/review_images/{N}.png
  5. Skips if the file already exists (safe to re-run)

After all images:
  6. Builds review.html — embeds all images as base64 + question text,
     options (correct answer highlighted in green ✓), and explanations
```

#### Output files

| File | Description |
|------|-------------|
| `data/image_mcq/review_images/1.png` | Generated image for question 1 |
| `data/image_mcq/review_images/2.png` | Generated image for question 2 |
| `data/image_mcq/review_images/N.png` | One per question |
| `data/image_mcq/review.html` | Self-contained HTML — open in browser to review all |

---

### Step 3 — judge_images.py

#### Basic usage

```bash
python judge_images.py
```

#### What it does internally

```
For each question:
  1. Reads the corresponding image from data/image_mcq/review_images/{N}.png
  2. Encodes image as base64
  3. Sends to Pixtral-12B (Mistral vision model) with this rubric prompt:
       "Evaluate this image. The intended notice content is: {image_content}
        Score on 4 criteria: relevance, visual quality, text legibility,
        contextual fit. Overall score 1–10. Return JSON only."
  4. Parses the JSON response
  5. Prints the score + brief reasoning to the terminal

After all questions:
  6. Saves image_judge_output.json with full rubric for every question
  7. Builds review_with_scores.html — images + question text + score badges
```

#### Rubric criteria

| Criterion | What is checked | Score |
|-----------|----------------|-------|
| `relevance_to_description` | Does image match the `image_content` description? | Excellent / Good / Poor |
| `visual_quality` | Is the image realistic and well-rendered? | Excellent / Good / Poor |
| `text_legibility` | Is text in the image readable? | Excellent / Good / Poor |
| `contextual_fit` | Does it look like a genuine real-world notice/sign? | Excellent / Good / Poor |
| `overall_score` | Combined quality rating | 1 – 10 |

#### Output files

| File | Description |
|------|-------------|
| `data/image_mcq/image_judge_output.json` | Full rubric scores + reasoning per question |
| `data/image_mcq/review_with_scores.html` | Self-contained HTML — open in browser for scored review |

#### image_judge_output.json format

```json
{
  "metadata": {
    "total_questions": 5,
    "judge_model":     "pixtral-12b-2409",
    "generator":       "FLUX.1-schnell (HuggingFace)",
    "rubric_criteria": ["relevance_to_description", "visual_quality", "text_legibility", "contextual_fit"]
  },
  "results": [
    {
      "question_number":   1,
      "difficulty":        "easy",
      "image_content":     "A library notice says the building will be closed for maintenance tomorrow.",
      "question":          "Why will the library be closed tomorrow?",
      "options":           ["For a holiday", "For maintenance", "For a special event", "For a meeting"],
      "correct_answer":    "For maintenance",
      "target_cefr":       "A2",
      "target_difficulty": "Easy",
      "image_path":        "1.png",
      "rubric": {
        "relevance_to_description": "Good",
        "visual_quality":           "Excellent",
        "text_legibility":          "Poor",
        "contextual_fit":           "Good",
        "overall_score":            6,
        "reasoning":                "Image shows a notice board but text is not clearly legible."
      },
      "error": null
    }
  ]
}
```

---

## Example Questions File (MCQ)

File: `configs/example_questions_mcq.json`

This file guides the MCQ generator on style, vocabulary level, and structure.
Edit it before running the MCQ generator (via any method).

### Format A — plain JSON array

```json
[
  {
    "instruction": "Choose the correct question word.",
    "question":    "_______ is your teacher?",
    "options":     ["Who", "What", "Where", "Why"],
    "correct_answer": "Who",
    "explanation": "Use 'Who' to ask about a person.",
    "difficulty":  "Easy",
    "cefr":        "A1"
  },
  {
    "instruction": "Choose the correct question word.",
    "question":    "_______ do you go to school?",
    "options":     ["When", "Where", "Who", "Which"],
    "correct_answer": "When",
    "explanation": "Use 'When' to ask about time.",
    "difficulty":  "Easy",
    "cefr":        "A2"
  }
]
```

### Format B — full config object

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

The module auto-detects both formats and reads from `questions` or `example_questions` key.

### Field rules

| Field | Rule |
|-------|------|
| `instruction` | Short task direction shown to the learner |
| `question` | The MCQ question text (`_______` for blanks) |
| `options` | Exactly **4** items |
| `correct_answer` | Copied **verbatim** from one of the options |
| `explanation` | Why the correct answer is correct |
| `difficulty` | `Easy`, `Medium`, or `Hard` only |
| `cefr` | `A1`, `A2`, `B1`, `B2`, `C1`, or `C2` only |

Provide at least one example per CEFR level you are generating (A1 for easy, B1 for medium, C1 for hard at minimum).

---

## Output Format Reference

### Field comparison — all three generators

| Field | MCQ | T2T | Image MCQ | Notes |
|-------|:---:|:---:|:---------:|-------|
| `question_number` | ✓ | ✓ | ✓ | `"Q1"`, `"Q2"`, `"Q3"` — sequential, Easy→Medium→Hard |
| `topic` | ✓ | ✓ | ✓ | Copied from input |
| `subtopic` | ✓ | ✓ | ✓ | Copied from input |
| `target_cefr` | ✓ | ✓ | ✓ | `A1` / `A2` / `B1` / `B2` / `C1` / `C2` |
| `target_difficulty` | ✓ | ✓ | ✓ | `Easy` / `Medium` / `Hard` |
| `instruction` | ✓ | ✓ | ✓ | Task direction shown to learner |
| `question` | ✓ | ✓ | ✓ | The actual question text |
| `options` | ✓ | — | ✓ | Array of exactly 4 answer choices |
| `correct_answer` | ✓ | — | ✓ | Must match one of the options exactly |
| `expected_answer` | — | ✓ | — | Model answer for open-answer questions |
| `explanation` | ✓ | ✓ | ✓ | Why the answer is correct |
| `question_type` | — | ✓ | — | `comprehension`, `word_reorder`, etc. |
| `image_content` | — | — | ✓ | Text description of the notice/sign image |

### Summary format by generator

| Generator | Summary keys |
|-----------|-------------|
| MCQ | `easy.accepted`, `easy.rejected`, `medium.accepted`, `medium.rejected`, `hard.accepted`, `hard.rejected` |
| T2T | `A1.accepted`, `A1.target`, `A2.accepted`, `A2.target`, … `C2.accepted`, `C2.target` |
| Image MCQ | Same as MCQ + `total_accepted`, `total_rejected` |

---

## CEFR to Difficulty Mapping

| CEFR | Difficulty | Notebook field | easy/medium/hard_count (CLI) |
|------|-----------|---------------|------------------------------|
| A1   | Easy      | `a1_count`    | `easy_count ÷ 2` (rounded down) |
| A2   | Easy      | `a2_count`    | `easy_count ÷ 2` (remainder) |
| B1   | Medium    | `b1_count`    | `medium_count ÷ 2` (rounded down) |
| B2   | Medium    | `b2_count`    | `medium_count ÷ 2` (remainder) |
| C1   | Hard      | `c1_count`    | `hard_count ÷ 2` (rounded down) |
| C2   | Hard      | `c2_count`    | `hard_count ÷ 2` (remainder) |

Set any count to `0` to skip that CEFR level entirely.

---

## Full Generation Pipeline (Internal)

What happens inside each generator when you click Run or call `generate.py`:

```
InputSchema (topic, subtopics, constraints)
        │
        ▼
Orchestrator loops over CEFR levels: A1 → A2 → B1 → B2 → C1 → C2
        │
        ▼  (for each level where count > 0)
GeneratorAgent quota loop:
  WHILE accepted < target_count AND iterations < max_iterations:
    │
    ├─ Step 1: GENERATE BATCH
    │         LLM call (Mistral) with ChainOfThought
    │         Input:  topic, subtopic, target_cefr, example_questions, batch_size
    │         Output: list of draft questions
    │
    ├─ Step 2: HARD VALIDATE (instant, no LLM)
    │         ✓ instruction not empty
    │         ✓ question not empty
    │         ✓ exactly 4 options (MCQ / Image MCQ)
    │         ✓ correct_answer is in options (MCQ / Image MCQ)
    │         ✓ explanation not empty
    │         ✓ image_content not empty (Image MCQ only)
    │         ✗ fails → added to rejected[stage="hard_validate"]
    │
    ├─ Step 3: DIFFICULTY JUDGE (LLM call — Mistral)
    │         Classifies the question's CEFR level
    │         ✗ predicted CEFR ≠ target band → rejected[stage="difficulty"]
    │
    ├─ Step 4: RUBRIC JUDGE (LLM call — Mistral)
    │         Evaluates 10 quality criteria (11 for Image MCQ)
    │         overall_decision must be "Pass"
    │         ✗ fails → rejected[stage="rubric"]
    │
    └─ Step 5: STORE — add accepted questions
        │
        ▼
GenerationResult assembled:
  easy   [ accepted Easy questions ]
  medium [ accepted Medium questions ]
  hard   [ accepted Hard questions ]
  rejected [ all failed questions with stage + reason ]
  warnings [ levels where quota was not fully met ]
        │
        ▼
generate.py post-processing:
  question_number reformatted → Q1, Q2, Q3 ...
        │
        ▼
Saved to data/<type>/<type>_generator_output.json
```

---

## Judge Notebooks (Train Once)

These notebooks train the quality judges used inside the generators. Run each once.
The trained judges are saved as artifacts and loaded automatically on every generation run.

| Notebook | Trains | Artifact produced | Used by |
|----------|--------|-------------------|---------|
| `notebooks/difficulty_judge.ipynb` | Difficulty classifier | `artifacts/simple_difficulty_optimized.json` | MCQ, T2T generators |
| `notebooks/rubric_judge.ipynb` | Rubric quality judge | `artifacts/rubric_agent_optimized.json` | MCQ, T2T generators |
| `notebooks/image_judge.ipynb` | Pixtral image judge | `artifacts/image_judge_optimized.json` | Image MCQ generator |

### How to run a judge notebook

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/difficulty_judge.ipynb
# Click: Kernel → Restart & Run All
# Wait for completion — artifact is saved automatically
```

Repeat for `rubric_judge.ipynb` and `image_judge.ipynb`.

> If an artifact file is missing, the generator falls back to an unoptimised judge automatically — no error is raised. Run the judge notebooks to improve generation quality.

---

## Project Structure

```
Topin/
├── notebooks/
│   ├── mcq_generator.ipynb             # Generator 1 — MCQ
│   ├── text_to_text_generator.ipynb    # Generator 2 — T2T
│   ├── image_mcq_generator.ipynb       # Generator 3 — Image MCQ
│   ├── difficulty_judge.ipynb          # Trains difficulty judge (run once)
│   ├── rubric_judge.ipynb              # Trains rubric judge (run once)
│   └── image_judge.ipynb               # Trains image judge (run once)
│
├── configs/
│   ├── template_mcq.json               # MCQ config template — copy and edit
│   ├── template_t2t.json               # T2T config template — copy and edit
│   ├── template_image_mcq.json         # Image MCQ config template — copy and edit
│   └── example_questions_mcq.json      # MCQ example questions — edit before running
│
├── data/
│   ├── mcq/
│   │   ├── mcq_generator_output.json           ← MCQ generator output
│   │   ├── training_dataset_standard.json
│   │   └── eval_dataset_standard.json
│   ├── t2t/
│   │   ├── t2t_generator_output.json           ← T2T generator output
│   │   ├── training_dataset_clean_full.json
│   │   └── eval_dataset_clean.json
│   └── image_mcq/
│       ├── image_mcq_generator_output.json     ← Image MCQ generator output
│       ├── review_images/                      ← generated PNGs (1.png, 2.png …)
│       ├── review.html                         ← open in browser: images + questions
│       ├── image_judge_output.json             ← Pixtral quality scores
│       ├── review_with_scores.html             ← open in browser: images + scores
│       ├── training_dataset_24_clean.json
│       └── training_dataset_24_judge_ready.json
│
├── artifacts/
│   ├── simple_difficulty_optimized.json        ← from difficulty_judge.ipynb
│   ├── rubric_agent_optimized.json             ← from rubric_judge.ipynb
│   └── image_judge_optimized.json              ← from image_judge.ipynb
│
├── src/
│   └── topin/
│       └── modules/
│           ├── mcq_generator.py        # dspy-cli MCQ module
│           ├── t2t_generator.py        # dspy-cli T2T module
│           └── image_mcq_generator.py  # dspy-cli Image MCQ module
│
├── generate.py                         # CLI runner — runs any generator from config
├── generate_review_images.py           # Generates images from image_content descriptions
├── judge_images.py                     # Scores generated images with Pixtral vision
├── dspy.config.yaml                    # dspy-cli server configuration
├── utils.py
└── .env                                # API keys (never commit this file)
```

---

## Artifacts

| File | Created by | Loaded by | Purpose |
|------|-----------|-----------|---------|
| `artifacts/simple_difficulty_optimized.json` | `difficulty_judge.ipynb` | `mcq_generator.ipynb`, `text_to_text_generator.ipynb` | Optimised difficulty classifier |
| `artifacts/rubric_agent_optimized.json` | `rubric_judge.ipynb` | `mcq_generator.ipynb`, `text_to_text_generator.ipynb` | Optimised rubric judge |
| `artifacts/image_judge_optimized.json` | `image_judge.ipynb` | `image_mcq_generator.ipynb` | Optimised image content judge |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: dspy` | Package not installed | `pip install dspy-ai` inside venv |
| `KeyError: MISTRAL_API_KEY` | `.env` missing or wrong path | Check `.env` exists at `d:/Topin/.env` |
| `No module named jupyter` | Jupyter not in venv | `pip install jupyter nbconvert nbclient` |
| `dspy-cli is not installed in your project virtual environment` | dspy-cli venv check | Use `dspy-cli serve --system` (bypasses the check) |
| `Not a valid DSPy project directory` | Wrong working directory | Run from `d:/Topin/` — must contain `dspy.config.yaml` |
| `Configuration must contain 'models' section` | Bad config file | Check `dspy.config.yaml` has a `models.registry` block |
| `HF FAILED: HTTP 410` | Old HF endpoint deprecated | Already fixed — uses `router.huggingface.co` now |
| `HF FAILED: HTTP 503` | HF model is loading cold | Script retries automatically — wait 1 min and re-run |
| Gemini `limit: 0` quota error | Gemini image models require billing | Use `--provider hf` (free) instead |
| `cannot import name 'Mistral' from 'mistralai'` | SDK version uses different path | `from mistralai.client import Mistral` |
| `review_with_scores.html` shows "No image" | Images not generated yet | Run `python generate_review_images.py` first |
| Questions rejected at difficulty stage | Model generating wrong CEFR level | Improve `configs/example_questions_mcq.json` or increase `max_iterations_per_difficulty` |
| Questions rejected at rubric stage | Quality criteria not met | Run `rubric_judge.ipynb` to optimise the rubric judge |
| Output warnings: quota not met | Not enough questions generated | Increase `max_iterations_per_difficulty` or reduce counts |
| `_tmp_generate_*.ipynb` file left behind | Notebook execution crashed | Safe to delete — it is recreated on next run |
