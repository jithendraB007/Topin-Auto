# MCQ Generator — dspy-cli Service

Serves the MCQ generation pipeline as a FastAPI HTTP endpoint using [dspy-cli](https://github.com/cmpnd-ai/dspy-cli).

## Setup

```bash
# 1. Install dspy-cli globally
uv tool install dspy-cli

# 2. Go to the project folder and sync dependencies
cd cli/mcq-generator
uv sync

# 3. Copy .env.template -> .env and add your Mistral API key
cp .env.template .env
# Edit .env: MISTRAL_API_KEY=your_key_here
```

## Run the server

```bash
cd cli/mcq-generator
dspy-cli serve
```

Server starts at `http://localhost:8000`.  
Interactive web UI available at `http://localhost:8000`.

## API

### `POST /MCQGeneratorCoT`

**Request body:**

```json
{
  "topic": "English Grammar",
  "subtopic": "Present Tense",
  "a1_count": 2,
  "a2_count": 2,
  "b1_count": 2,
  "b2_count": 2,
  "c1_count": 1,
  "c2_count": 0,
  "questions_per_iteration": 5,
  "max_iterations": 20
}
```

| Field | Type | Description |
|-------|------|-------------|
| `topic` | string | Subject area (e.g. "English Grammar") |
| `subtopic` | string | Specific subtopic (e.g. "Present Tense") |
| `a1_count` | int | Easy/Beginner questions (use 0 to skip) |
| `a2_count` | int | Easy/Elementary questions (use 0 to skip) |
| `b1_count` | int | Medium/Intermediate questions (use 0 to skip) |
| `b2_count` | int | Medium/Upper-Intermediate questions (use 0 to skip) |
| `c1_count` | int | Hard/Advanced questions (use 0 to skip) |
| `c2_count` | int | Hard/Mastery questions (use 0 to skip) |
| `questions_per_iteration` | int | Batch size per LLM call (recommended: 5) |
| `max_iterations` | int | Max retry loops per CEFR level (recommended: 20) |

**Response:**
```json
{
  "questions": [
    {
      "question_number": 1,
      "topic": "English Grammar",
      "subtopic": "Present Tense",
      "target_cefr": "A1",
      "target_difficulty": "Easy",
      "stem": "...",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "A",
      "explanation": "..."
    }
  ],
  "summary": {
    "easy":   {"accepted": 4, "rejected": 1},
    "medium": {"accepted": 3, "rejected": 2},
    "hard":   {"accepted": 1, "rejected": 3},
    "total_accepted": 8,
    "total_rejected": 6
  },
  "warnings": []
}
```

### curl Example

```bash
curl -X POST http://localhost:8000/MCQGeneratorCoT \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "English Grammar",
    "subtopic": "Present Tense",
    "a1_count": 2,
    "a2_count": 0,
    "b1_count": 2,
    "b2_count": 0,
    "c1_count": 1,
    "c2_count": 0,
    "questions_per_iteration": 5,
    "max_iterations": 10
  }'
```

## Project Structure

```
cli/mcq-generator/
├── dspy.config.yaml          # Model configuration (Mistral AI)
├── pyproject.toml
├── .env                      # API keys (not committed)
├── artifacts/                # Optimised DSPy judge weights
│   ├── simple_difficulty_optimized.json
│   └── rubric_agent_optimized.json
├── data/
│   └── training_dataset_standard.json  # Default reference examples
└── src/mcq_generator/
    ├── signatures/
    │   └── mcq_generator.py  # MCQGeneratorSignature
    ├── modules/
    │   └── mcq_generator_cot.py  # MCQGeneratorCoT (main endpoint)
    └── utils/
        ├── models.py         # Pydantic data models
        ├── validators.py     # hard_validate()
        ├── judges.py         # DifficultyJudge + RubricJudge wrappers
        └── orchestrator.py   # MCQGenerationOrchestrator + quota loop
```

## Models

Switch models by editing `dspy.config.yaml`:

- `mistral:nemo` — `open-mistral-nemo` (default, fastest)
- `mistral:small` — `mistral-small-latest` (better quality)
- `mistral:large` — `mistral-large-latest` (best quality)

Change the default:
```yaml
models:
  default: mistral:small  # was mistral:nemo
```
