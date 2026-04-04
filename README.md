# Topin — Automated MCQ Generation Pipeline

An end-to-end, AI-driven Multiple Choice Question (MCQ) generation system built with **DSPy**, **Mistral AI**, and **Promptfoo**. The pipeline plans, generates, judges, revises, and auto-optimises questions — all from a single JSON input file.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [Core Concepts](#4-core-concepts)
5. [Data Models (Schemas)](#5-data-models-schemas)
6. [Agent Architecture](#6-agent-architecture)
7. [Complete Pipeline Flow](#7-complete-pipeline-flow)
8. [Auto-Optimization Loop](#8-auto-optimization-loop)
9. [Evaluation System (Promptfoo)](#9-evaluation-system-promptfoo)
10. [File Reference](#10-file-reference)
11. [Configuration](#11-configuration)
12. [Running the Project](#12-running-the-project)
13. [Input Schema Reference](#13-input-schema-reference)
14. [Output Schema Reference](#14-output-schema-reference)
15. [Validation Rules](#15-validation-rules)
16. [Extending the Pipeline](#16-extending-the-pipeline)

---

## 1. Project Overview

Topin takes a structured JSON input (subject, topic, CEFR distribution, sample questions) and automatically produces a validated, deduplicated set of MCQs at the correct difficulty level. Every generated question goes through two independent LLM judges before being accepted. If questions fail, the pipeline:

- Revises them using structured feedback (up to 2 attempts)
- Collects failures into a training set
- Runs BootstrapFewShot optimisation to improve the judges
- Retries the failed questions with the improved judges

The result is a self-improving system — the more it runs, the better its judges become.

---

## 2. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM Framework | [DSPy](https://dspy.ai) | Structured LLM calls with typed signatures, chain-of-thought, and optimisers |
| LLM Provider | [Mistral AI](https://mistral.ai) | Model inference via OpenAI-compatible API |
| Default Model | `open-mistral-nemo` | Free-tier Mistral model (7B) |
| Data Validation | [Pydantic v2](https://docs.pydantic.dev) | Schema validation for all inputs and outputs |
| Evaluation | [Promptfoo](https://promptfoo.dev) | LLM-as-judge evals and structural assertion testing |
| Optimiser | DSPy BootstrapFewShot | Injects few-shot examples into judge prompts using past failures |
| Env Management | python-dotenv | Loads API keys from `.env` |

---

## 3. Project Structure

```
d:\Topin\
│
├── main.py                        # Entry point — orchestrates the full pipeline + auto-optimize
├── schemas.py                     # All Pydantic data models
├── utils.py                       # DSPy config, JSON helpers
│
├── agents/
│   ├── pipeline.py                # All 5 DSPy agents + MCQPipeline orchestrator
│   └── signatures.py              # DSPy Signature classes (LLM prompt contracts)
│
├── optimize/
│   └── gepa_optimize.py           # BootstrapFewShot optimiser for judge agents
│
├── evals/
│   ├── provider.py                # Promptfoo provider — runs full pipeline as eval
│   ├── judge_provider.py          # Promptfoo provider — runs judges only on a single question
│   ├── promptfooconfig.yaml       # Pipeline-level eval config
│   └── judge_promptfooconfig.yaml # Judge-level eval config (good + flawed questions)
│
├── data/
│   ├── sample_input.json          # Example input with 15 questions + sample_questions
│   ├── output_run.json            # Last pipeline run output (auto-written)
│   ├── gepa_train.jsonl           # Growing training set (appended from failures)
│   ├── gold_good.jsonl            # Hand-labelled good questions (expected: Pass)
│   └── gold_flawed.jsonl          # Hand-labelled flawed questions (expected: Fail)
│
├── artifacts/
│   ├── difficulty_agent_optimized.json   # Saved few-shot demos for DifficultyAgent
│   └── rubric_agent_optimized.json       # Saved few-shot demos for RubricAgent
│
├── .env                           # API keys (not committed)
├── .env.example                   # Template for .env
└── requirements.txt               # Python dependencies
```

---

## 4. Core Concepts

### CEFR Levels
The Common European Framework of Reference for Languages defines six proficiency levels:

| CEFR | Description | Difficulty Band |
|------|-------------|----------------|
| A1 | Beginner | Easy |
| A2 | Elementary | Easy |
| B1 | Intermediate | Medium |
| B2 | Upper-Intermediate | Medium |
| C1 | Advanced | Hard |
| C2 | Mastery | Hard |

Every question is planned and judged against its target CEFR level.

### DSPy Signatures
A DSPy `Signature` is a typed prompt contract. It declares named input fields and output fields. DSPy compiles these into structured LLM calls with Chain-of-Thought reasoning. Instead of writing raw prompts, you define what information goes in and what structured output comes out.

### BootstrapFewShot
A DSPy optimiser that improves a module by injecting worked examples (few-shot demonstrations) directly into its prompt. It uses past training examples to find demonstrations where the module succeeded, then stores these permanently in the module's JSON artifact. Each subsequent run loads these artifacts and benefits from the improved few-shot context.

---

## 5. Data Models (Schemas)

All models are defined in [schemas.py](schemas.py) using Pydantic v2.

### InputSchema — What you provide

```json
{
  "subject": "English",
  "syllabus_unit": "Unit 3",
  "topic": "Present Tense",
  "subtopics": ["Simple Present", "Present Continuous", "Adverbs of Frequency", "Subject-Verb Agreement"],
  "question_type": "MCQ",
  "total_questions": 15,
  "cefr_distribution": { "A1": 3, "A2": 4, "B1": 4, "B2": 4 },
  "constraints": {
    "options_per_question": 4,
    "single_correct_answer": true,
    "include_explanation": true,
    "exam_acceptable_language": true,
    "avoid_ambiguity": true,
    "language_variant": "British English",
    "no_duplicate_questions": true
  },
  "sample_questions": [
    {
      "stem": "She _______ to school every day.",
      "options": ["go", "goes", "going", "gone"],
      "correct_answer": "goes",
      "explanation": "Third-person singular subjects require -s/-es in simple present.",
      "target_cefr": "A1"
    }
  ]
}
```

> **Validation rule**: `sum(cefr_distribution.values())` must equal `total_questions` — otherwise the schema is rejected before any LLM call is made.

### PlannedQuestion — Internal planning output

```python
question_number: int        # 1-based index
question_type:   "MCQ"      # always MCQ
topic:           str        # inherited from input
subtopic:        str        # one of the input subtopics
target_cefr:     CEFRLevel  # A1 / A2 / B1 / B2 / C1 / C2
target_difficulty: DifficultyBand  # Easy / Medium / Hard
angle:           str        # unique question format, e.g. "error-correction"
```

### MCQItem — A generated question

```python
question_number: int
topic:           str
subtopic:        str
target_cefr:     CEFRLevel
target_difficulty: DifficultyBand
stem:            str        # the question text
options:         List[str]  # exactly 4 options
correct_answer:  str        # must be one of the options verbatim
explanation:     str        # justification of the correct answer
```

### DifficultyResult — Output of DifficultyAgent

```python
predicted_cefr:       CEFRLevel   # what the judge thinks the CEFR is
predicted_difficulty: DifficultyBand
vocabulary_level:     str
grammar_complexity:   str
reasoning_load:       str
distractor_difficulty: str
alignment:            bool        # True = predicted matches target
justification:        str
revision_feedback:    str         # what to fix if misaligned
```

### RubricResult — Output of RubricAgent

```python
grammatical_accuracy:                  str
spelling:                              str
ambiguity:                             str   # "Major Issue" triggers auto-Fail
functionality_alignment:               str
instruction_clarity_appropriateness:   str
academic_language_exam_acceptability:  str
option_explanation_consistency:        str
readability:                           str
formatting_spacing:                    str
punctuation:                           str
british_american_english_consistency:  str
overall_decision:  "Pass" | "Revise" | "Fail"
priority_reason:   str
revision_feedback: str
```

### EvaluatedItem — Final output per question

```python
item:               MCQItem
difficulty:         DifficultyResult
rubric:             RubricResult
accepted:           bool
revision_attempts:  int
```

---

## 6. Agent Architecture

The pipeline uses five DSPy agents, each wrapping a `ChainOfThought` LLM call through a typed `Signature`.

```
┌─────────────────────────────────────────────────────────┐
│                   5 DSPy Agents                         │
│                                                         │
│  PlannerAgent       → decides what to generate          │
│  MCQGeneratorAgent  → writes the question               │
│  DifficultyAgent    → judges CEFR alignment             │
│  RubricAgent        → judges quality & language         │
│  RevisionAgent      → rewrites based on feedback        │
└─────────────────────────────────────────────────────────┘
```

### PlannerAgent

**File**: [agents/pipeline.py](agents/pipeline.py) — `PlannerAgent`  
**Signature**: [agents/signatures.py](agents/signatures.py) — `PlannerSignature`

Takes the full `InputSchema` and produces a plan: one `PlannedQuestion` per slot. The LLM is instructed to:
- Assign a unique `angle` (question format) to every slot — no two questions may share the same angle
- Spread subtopics evenly across slots
- Respect the CEFR distribution exactly

Post-processing in `_enforce_unique_angles()` guarantees uniqueness even if the LLM produces duplicates, by cycling through 12 predefined angle types:

```
fill-in-the-blank, sentence-completion, error-correction, inference,
vocabulary-in-context, question-formation, conversation-completion,
word-order, concept-identification, real-world-application,
paraphrase-selection, affirmative-negative-transformation
```

### MCQGeneratorAgent

**File**: [agents/pipeline.py](agents/pipeline.py) — `MCQGeneratorAgent`  
**Signature**: [agents/signatures.py](agents/signatures.py) — `MCQGeneratorSignature`

Generates a single MCQ given a `PlannedQuestion`. Key inputs beyond basic topic/CEFR:

| Input | Purpose |
|-------|---------|
| `angle` | Forces a specific question format |
| `already_used_stems` | JSON list of stems already accepted — model must not repeat them |
| `sample_questions` | User-provided examples to guide style, vocabulary level, and format |

Outputs raw JSON with keys: `stem`, `options`, `correct_answer`, `explanation`.

### DifficultyAgent

**File**: [agents/pipeline.py](agents/pipeline.py) — `DifficultyAgent`  
**Signature**: [agents/signatures.py](agents/signatures.py) — `DifficultyJudgeSignature`

Evaluates whether the question's vocabulary, grammar complexity, and reasoning load match the target CEFR level. Returns `alignment: bool` — the single most important field for acceptance.

Can be optimised with BootstrapFewShot. Loads from `artifacts/difficulty_agent_optimized.json` if it exists.

### RubricAgent

**File**: [agents/pipeline.py](agents/pipeline.py) — `RubricAgent`  
**Signature**: [agents/signatures.py](agents/signatures.py) — `RubricJudgeSignature`

Evaluates 11 quality dimensions and returns `overall_decision`: Pass / Revise / Fail.

Special rules applied after the LLM call:
- If `ambiguity == "Major Issue"` → force `overall_decision = "Fail"` regardless of LLM output
- If `overall_decision == "Revise"` but `priority_reason` contains only trivial phrases ("none", "no major issues", "n/a") → upgrade to `"Pass"` (false positive suppression)

Can be optimised with BootstrapFewShot. Loads from `artifacts/rubric_agent_optimized.json` if it exists.

### RevisionAgent

**File**: [agents/pipeline.py](agents/pipeline.py) — `RevisionAgent`  
**Signature**: [agents/signatures.py](agents/signatures.py) — `RevisionSignature`

Rewrites a failing question using structured feedback from both judges. Receives:
- The original `stem`, `options`, `correct_answer`, `explanation`
- `difficulty_feedback` from DifficultyAgent
- `rubric_feedback` from RubricAgent (prepended with structural errors if any)
- `angle` — must preserve the same question format
- `already_used_stems` — must not repeat any accepted stem

---

## 7. Complete Pipeline Flow

### Step-by-step walkthrough

```
 User runs:
 .venv/Scripts/python main.py --input data/sample_input.json
```

---

#### Step 1 — Configuration (`utils.py`)

`configure_dspy_from_env()` reads `.env`, creates a `dspy.LM` pointing to the Mistral API using the OpenAI-compatible endpoint, and calls `dspy.configure(lm=lm)`. All subsequent DSPy calls use this globally configured model.

```
.env
  MISTRAL_API_KEY  = your-key
  MISTRAL_MODEL    = open-mistral-nemo
  MISTRAL_API_BASE = https://api.mistral.ai/v1
```

---

#### Step 2 — Load Input & Pre-built Artifacts (`main.py`)

```python
schema = InputSchema(**load_json(args.input))   # validates input
pipeline = MCQPipeline(max_revision_attempts=2)
pipeline.load_optimized_agents()                # loads artifacts/ if they exist
```

If `artifacts/difficulty_agent_optimized.json` or `artifacts/rubric_agent_optimized.json` exist from a previous run, the judges are loaded with their few-shot demonstrations already injected — they are immediately better than a cold start.

---

#### Step 3 — Planning (`PlannerAgent`)

One LLM call. Returns a list of 15 `PlannedQuestion` objects (for the default sample input).

Example output for a B1/Medium slot:
```json
{
  "question_number": 7,
  "question_type": "MCQ",
  "topic": "Present Tense",
  "subtopic": "Subject-Verb Agreement",
  "target_cefr": "B1",
  "target_difficulty": "Medium",
  "angle": "error-correction"
}
```

`_enforce_unique_angles()` then post-processes the entire plan to guarantee no two slots share the same angle string.

---

#### Step 4 — Generation Loop (per question)

For each `PlannedQuestion`, the pipeline runs this inner loop:

```
GENERATION ATTEMPT (up to 3 times)
  ├── Call MCQGeneratorAgent
  │     inputs: topic, subtopic, cefr, difficulty, angle,
  │             already_used_stems, sample_questions, constraints
  │
  ├── hard_validate_item()         ← deterministic checks
  │     - exactly 4 options?
  │     - correct_answer literally in options?
  │     - explanation non-empty?
  │
  └── _is_duplicate_stem()         ← similarity check
        - exact match with any accepted stem?
        - >=90% content-word overlap with any accepted stem?
        → if duplicate: retry generation
```

If all 3 generation attempts still fail structural checks, the item proceeds anyway (it will fail the judge loop and be counted as rejected).

---

#### Step 5 — Judge + Revise Loop (per question)

```
JUDGE LOOP (runs until accepted or max_revision_attempts reached)
  │
  ├── DifficultyAgent (LLM)
  │     → predicted_cefr, alignment (bool), revision_feedback
  │
  ├── RubricAgent (LLM)
  │     → overall_decision (Pass/Revise/Fail), revision_feedback
  │
  ├── hard_validate_item()   ← re-check after any revision
  ├── _is_duplicate_stem()   ← re-check after any revision
  │
  ├── ACCEPTED if ALL of:
  │     alignment == True
  │     overall_decision == "Pass"
  │     no structural errors
  │     stem is not a duplicate
  │
  ├── IF accepted:
  │     append stem to accepted_stems[]
  │     append EvaluatedItem(accepted=True) to results
  │     BREAK
  │
  ├── IF revision_attempts >= 2:
  │     print rejection reason
  │     append EvaluatedItem(accepted=False) to results
  │     BREAK
  │
  └── ELSE: call RevisionAgent
        inputs: item + difficulty_feedback + rubric_feedback
                + structural errors prepended if any
                + "DUPLICATE STEM DETECTED" prepended if duplicate
                + angle + already_used_stems
        → revised MCQItem
        revision_attempts += 1
        LOOP BACK to judges
```

---

#### Step 6 — Results Collection

After all 15 questions complete, `results` is a list of `EvaluatedItem` objects. Each contains the question, both judge outputs, whether it was accepted, and how many revision attempts were needed.

---

#### Step 7 — Auto-Optimize + Pass 2 (`main.py`)

```python
rejected = [r for r in results if not r.accepted]
rate = accepted_count / len(results)

if rejected and rate < AUTO_OPTIMIZE_THRESHOLD:   # default threshold = 1.0 (any failure triggers)
    _run_gepa(pipeline, rejected)
    retry_results = pipeline(retry_schema)         # Pass 2: only the failed questions
    results = [r for r in results if r.accepted] + retry_results
```

See [Section 8](#8-auto-optimization-loop) for full detail.

---

#### Step 8 — Save Output

```python
save_json("data/output_run.json", [r.model_dump() for r in results])
```

Every field of every accepted and rejected question, with full judge commentary, is written to the output file.

---

## 8. Auto-Optimization Loop

The self-improvement mechanism runs automatically whenever any question is rejected.

```
Rejected questions
       │
       ▼
append_failures_to_trainset()
  ├── reads existing data/gepa_train.jsonl
  ├── deduplicates by stem
  └── appends new failed items with:
        expected_predicted_cefr = target_cefr
        expected_overall_decision = "Pass"
       │
       ▼
load_trainset()
  └── returns list of dspy.Example objects
      (needs >= 2 examples to proceed)
       │
       ▼
optimize_difficulty(trainset)
  ├── creates _DifficultyFlat (flat-arg wrapper around DifficultyAgent)
  ├── runs BootstrapFewShot(metric=difficulty_metric_bool, max_bootstrapped_demos=3)
  └── saves artifacts/difficulty_agent_optimized.json

optimize_rubric(trainset)
  ├── creates _RubricFlat (flat-arg wrapper around RubricAgent)
  ├── runs BootstrapFewShot(metric=rubric_metric_bool, max_bootstrapped_demos=3)
  └── saves artifacts/rubric_agent_optimized.json
       │
       ▼
pipeline.load_optimized_agents()
  ├── self.difficulty.load("artifacts/difficulty_agent_optimized.json")
  └── self.rubric.load("artifacts/rubric_agent_optimized.json")
       │
       ▼
Pass 2: pipeline(retry_schema)
  └── retry_schema has only the rejected questions' CEFR counts
      the pipeline runs again with the same 5-agent flow
      but judges now have few-shot examples → better decisions
       │
       ▼
Merge results:
  results = [pass-1 accepted] + [pass-2 all results]
```

### Why BootstrapFewShot instead of GEPA?

GEPA (Gradient-based Efficient Prompt Adaptation) rewrites the instruction text of a prompt. This requires a strong reflection LLM (GPT-4 class). With Mistral `open-mistral-nemo` as the reflection LLM, GEPA produces unchanged instructions — it does not improve anything.

BootstrapFewShot instead adds concrete worked examples to the prompt. It does not rewrite instructions — it just finds examples where the student module succeeded on the training set and injects those as demonstrations. This works reliably with smaller models.

### Training Set Growth

```
Run 1: 2 failures → gepa_train.jsonl has 2 entries
Run 2: 1 failure  → gepa_train.jsonl has 3 entries (deduplicated)
Run 3: 0 failures → no new entries, optimization skipped
```

The training set never shrinks. Over many runs, the judges become progressively more calibrated.

---

## 9. Evaluation System (Promptfoo)

Two independent eval configs allow testing the pipeline and the judges separately.

### Pipeline Eval — `evals/promptfooconfig.yaml`

Tests the full end-to-end pipeline. The provider (`evals/provider.py`) runs `main.py` as a subprocess with the sample input and returns the JSON output.

Assertions check:
- All questions have exactly 4 options
- Every `correct_answer` appears verbatim in its `options` array
- No two questions share the same stem
- Questions span at least 3 different CEFR levels
- LLM-as-judge verifies output is well-formed and pedagogically sound

Run:
```bash
npx promptfoo@latest eval -c evals/promptfooconfig.yaml
```

### Judge Eval — `evals/judge_promptfooconfig.yaml`

Tests only the DifficultyAgent + RubricAgent against a gold dataset of known-good and known-flawed questions. The provider (`evals/judge_provider.py`) accepts a single question JSON, runs both judges, and returns their combined output.

Gold dataset:
- `data/gold_good.jsonl` — well-formed questions expected to receive `overall_decision: Pass`
- `data/gold_flawed.jsonl` — intentionally broken questions expected to receive `overall_decision: Fail`

Flaws tested include:
- Only 3 options instead of 4
- Correct answer not present in options
- Answer contradicts explanation
- Swapped explanations between questions
- Factually wrong correct answer

Run:
```bash
npx promptfoo@latest eval -c evals/judge_promptfooconfig.yaml
```

---

## 10. File Reference

| File | Role |
|------|------|
| [main.py](main.py) | CLI entry point. Parses args, runs pipeline, handles auto-optimize, saves output |
| [schemas.py](schemas.py) | All Pydantic models: InputSchema, PlannedQuestion, MCQItem, DifficultyResult, RubricResult, EvaluatedItem |
| [utils.py](utils.py) | `configure_dspy_from_env()`, `load_json()`, `save_json()` |
| [agents/signatures.py](agents/signatures.py) | DSPy Signature classes — the typed prompt contracts for all 5 agents |
| [agents/pipeline.py](agents/pipeline.py) | All 5 agent classes + MCQPipeline orchestrator. Contains all normalisation logic and validation |
| [optimize/gepa_optimize.py](optimize/gepa_optimize.py) | BootstrapFewShot optimiser. Flat wrappers, training set loader, metric functions, optimize functions |
| [evals/provider.py](evals/provider.py) | Promptfoo provider for full pipeline eval |
| [evals/judge_provider.py](evals/judge_provider.py) | Promptfoo provider for judge-only eval |
| [evals/promptfooconfig.yaml](evals/promptfooconfig.yaml) | Pipeline eval configuration |
| [evals/judge_promptfooconfig.yaml](evals/judge_promptfooconfig.yaml) | Judge eval configuration with 23 test cases |
| [data/sample_input.json](data/sample_input.json) | Default input: 15 English Present Tense questions across A1-B2 |
| [data/output_run.json](data/output_run.json) | Auto-written after every run |
| [data/gepa_train.jsonl](data/gepa_train.jsonl) | Growing training set. Each line is one question with expected judge output |
| [data/gold_good.jsonl](data/gold_good.jsonl) | Hand-labelled good questions for eval |
| [data/gold_flawed.jsonl](data/gold_flawed.jsonl) | Hand-labelled flawed questions for eval |
| [artifacts/](artifacts/) | Auto-generated. Stores optimised agent JSON (few-shot demos). Committed to speed up cold starts |

---

## 11. Configuration

### `.env` file

```env
MISTRAL_API_KEY=your-api-key-here
MISTRAL_MODEL=open-mistral-nemo
MISTRAL_API_BASE=https://api.mistral.ai/v1
```

### Model options (Mistral free tier)

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `open-mistral-nemo` | Fast | Basic | Default. Good for prototyping |
| `mistral-small-latest` | Medium | Better | Recommended for production |
| `mistral-medium-latest` | Slower | Best free | Best question diversity |

Change `MISTRAL_MODEL` in `.env` — no code changes needed.

### Pipeline tuning (in `main.py`)

```python
AUTO_OPTIMIZE_THRESHOLD = 1.0   # 1.0 = optimize on any failure, 0.8 = only when <80% accepted
MCQPipeline(max_revision_attempts=2)  # how many times to revise before rejecting
```

### Generator tuning (in `utils.py`)

```python
dspy.LM(
    temperature=0.2,    # lower = more consistent, higher = more creative
    max_tokens=2000,
)
```

---

## 12. Running the Project

### Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

### Generate questions

```bash
# Basic run — generates 15 questions from sample input
.venv/Scripts/python main.py --input data/sample_input.json

# Custom output path
.venv/Scripts/python main.py --input data/sample_input.json --output data/my_output.json

# Skip auto-optimization (faster, useful for debugging)
.venv/Scripts/python main.py --input data/sample_input.json --no-auto-optimize
```

### Run optimiser manually

```bash
# Only needed if you want to pre-train judges before running the pipeline
.venv/Scripts/python optimize/gepa_optimize.py
```

### Run evals

```bash
# Full pipeline eval
npx promptfoo@latest eval -c evals/promptfooconfig.yaml

# Judge-only eval against gold dataset
npx promptfoo@latest eval -c evals/judge_promptfooconfig.yaml

# View results in browser
npx promptfoo@latest view
```

---

## 13. Input Schema Reference

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `subject` | string | The academic subject, e.g. "English" |
| `syllabus_unit` | string | Unit label, e.g. "Unit 3" |
| `topic` | string | Main topic, e.g. "Present Tense" |
| `subtopics` | string[] | List of sub-topics to cover |
| `total_questions` | int | Must equal sum of all CEFR counts |
| `cefr_distribution` | object | Keys: A1/A2/B1/B2/C1/C2, values: count |

### Optional fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question_type` | "MCQ" | "MCQ" | Only MCQ supported currently |
| `constraints.options_per_question` | int | 4 | Number of answer choices |
| `constraints.single_correct_answer` | bool | true | Exactly one correct option |
| `constraints.include_explanation` | bool | true | Each question has an explanation |
| `constraints.exam_acceptable_language` | bool | true | Formal, exam-appropriate register |
| `constraints.avoid_ambiguity` | bool | true | Only one defensible correct answer |
| `constraints.language_variant` | string | "British English" | "British English" or "American English" |
| `constraints.no_duplicate_questions` | bool | true | Enforced programmatically |
| `sample_questions` | object[] | [] | Example questions to guide style and format |

### `sample_questions` format

Each object in the array should have:
```json
{
  "stem": "question text",
  "options": ["A", "B", "C", "D"],
  "correct_answer": "B",
  "explanation": "why B is correct",
  "target_cefr": "B1"
}
```

Providing one sample per CEFR level gives the best results.

---

## 14. Output Schema Reference

`data/output_run.json` is a JSON array. Each element:

```json
{
  "item": {
    "question_number": 1,
    "topic": "Present Tense",
    "subtopic": "Simple Present",
    "target_cefr": "A1",
    "target_difficulty": "Easy",
    "stem": "She _______ to school every day.",
    "options": ["go", "goes", "going", "gone"],
    "correct_answer": "goes",
    "explanation": "Third-person singular subjects require -s/-es."
  },
  "difficulty": {
    "predicted_cefr": "A1",
    "predicted_difficulty": "Easy",
    "vocabulary_level": "Simple",
    "grammar_complexity": "Simple",
    "reasoning_load": "Low",
    "distractor_difficulty": "Low",
    "alignment": true,
    "justification": "...",
    "revision_feedback": "None"
  },
  "rubric": {
    "grammatical_accuracy": "No issues found",
    "spelling": "No issues found",
    "ambiguity": "No ambiguity found",
    "functionality_alignment": "No issues found",
    "instruction_clarity_appropriateness": "Clear and appropriate",
    "academic_language_exam_acceptability": "Acceptable",
    "option_explanation_consistency": "Consistent",
    "readability": "Easy to read",
    "formatting_spacing": "No issues found",
    "punctuation": "No issues found",
    "british_american_english_consistency": "Consistent with British English",
    "overall_decision": "Pass",
    "priority_reason": "No major issues found",
    "revision_feedback": "None"
  },
  "accepted": true,
  "revision_attempts": 0
}
```

---

## 15. Validation Rules

### Deterministic (code-level) — always enforced

| Rule | Where checked | On failure |
|------|--------------|------------|
| `len(options) == 4` | `hard_validate_item()` | Retry generator (3x), then flag in revision |
| `correct_answer in options` | `hard_validate_item()` | Same |
| `explanation` is non-empty | `hard_validate_item()` | Same |
| Stem not identical to accepted stem | `_is_duplicate_stem()` exact match | Retry generator, flag in revision |
| Stem not >=90% content-word overlap with accepted stem | `_is_duplicate_stem()` overlap | Same |
| CEFR distribution sums to total_questions | `InputSchema.validate_counts()` | Schema rejected, pipeline never starts |

### LLM-based — judges make these decisions

| Rule | Who checks | On failure |
|------|-----------|------------|
| Vocabulary matches target CEFR | DifficultyAgent | `alignment=False` → revise or reject |
| Grammar complexity matches target CEFR | DifficultyAgent | Same |
| No grammatical errors in question text | RubricAgent | `overall_decision=Revise/Fail` |
| Only one defensibly correct answer | RubricAgent (ambiguity field) | If "Major Issue" → force Fail |
| British/American English consistency | RubricAgent | Flags in rubric, may trigger Revise |
| Options consistent with explanation | RubricAgent | May trigger Revise |

### Override rules (post-LLM, in code)

| Condition | Action |
|-----------|--------|
| `ambiguity == "Major Issue"` | Force `overall_decision = "Fail"` |
| `overall_decision == "Revise"` AND `priority_reason` is trivial | Upgrade to `"Pass"` (false positive suppression) |
| `question_type` is not "MCQ" in planner output | Assume LLM put the angle there — salvage it and reset `question_type = "MCQ"` |
| `target_difficulty` is "Very Hard", "Moderate", etc. | Normalise to "Easy" / "Medium" / "Hard" |
| `alignment` is "High", "True", "Yes" (string) | Normalise to `True` (bool) |

---

## 16. Extending the Pipeline

### Add a new subject or topic

Just change `data/sample_input.json`. No code changes needed. Add relevant `sample_questions` for the best results.

### Increase question variety

Provide richer `sample_questions` — one per CEFR level, covering different angles. The generator uses these as style guides.

### Improve acceptance rate

1. Add more examples to `data/gepa_train.jsonl` (manually or by running the pipeline more)
2. Run `optimize/gepa_optimize.py` manually to build the artifact
3. Use a more capable model (`mistral-small-latest` or better)

### Add a new question type (e.g. True/False)

1. Add `"TrueFalse"` to the `question_type` literal in `schemas.py`
2. Create a new `TrueFalseGeneratorSignature` in `agents/signatures.py`
3. Add a `TrueFalseGeneratorAgent` in `agents/pipeline.py`
4. Route by `question_type` in `MCQPipeline.forward()`

### Add a new judge dimension

1. Add a new field to `RubricResult` in `schemas.py`
2. Update `RubricJudgeSignature` docstring to instruct the LLM to output it
3. Add any special override logic in `RubricAgent.forward()`

### Run on a schedule (batch mode)

```bash
# Generate questions from multiple input files
for f in data/inputs/*.json; do
    .venv/Scripts/python main.py --input "$f" --output "data/outputs/$(basename $f)"
done
```

Each run feeds failures back into `gepa_train.jsonl`, progressively improving the judges across all batches.
