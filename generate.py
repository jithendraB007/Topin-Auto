#!/usr/bin/env python3
"""
generate.py — CLI runner for Topin question generators.

Usage
-----
  python generate.py --config configs/english_grammar_mcq.json
  python generate.py --config configs/reading_notices.json
  python generate.py --list-types

How it works
------------
  1. Reads a JSON config file (topic, subtopics, counts, example questions).
  2. Builds the Python schema code from the config.
  3. Patches the generator notebook's input cell with the new schema.
  4. Runs the patched notebook with `jupyter nbconvert --execute`.
  5. Output is saved automatically to data/<type>/<type>_generator_output.json.

Config format: see configs/template_mcq.json (or template_t2t.json / template_image_mcq.json).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient  # pip install nbclient (bundled with jupyter)

# Windows: ZMQ requires the selector event loop, not the default Proactor loop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

PROJECT_ROOT = Path(__file__).parent

NOTEBOOK_MAP = {
    "mcq":       PROJECT_ROOT / "notebooks" / "mcq_generator.ipynb",
    "t2t":       PROJECT_ROOT / "notebooks" / "text_to_text_generator.ipynb",
    "image_mcq": PROJECT_ROOT / "notebooks" / "image_mcq_generator.ipynb",
}

OUTPUT_MAP = {
    "mcq":       PROJECT_ROOT / "data" / "mcq"       / "mcq_generator_output.json",
    "t2t":       PROJECT_ROOT / "data" / "t2t"       / "t2t_generator_output.json",
    "image_mcq": PROJECT_ROOT / "data" / "image_mcq" / "image_mcq_generator_output.json",
}


def _reformat_question_numbers(output_path: Path) -> None:
    """Reformat question_number from int (1, 2, 3) to Q-string (Q1, Q2, Q3)."""
    if not output_path.exists():
        return
    data = json.loads(output_path.read_text(encoding="utf-8"))
    counter = 1
    for bucket in ("easy", "medium", "hard"):
        for q in data.get("questions", {}).get(bucket, []):
            q["question_number"] = f"Q{counter}"
            counter += 1
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Schema code builders ───────────────────────────────────────────────────────

def _normalise_subtopic(st: dict) -> dict:
    """Accept both a1_count/a2_count/... and the simpler easy_count/medium_count/hard_count."""
    if "easy_count" in st or "medium_count" in st or "hard_count" in st:
        easy   = st.get("easy_count",   0)
        medium = st.get("medium_count", 0)
        hard   = st.get("hard_count",   0)
        return {
            "subtopic":  st.get("subtopic", ""),
            "a1_count":  easy // 2,
            "a2_count":  easy - easy // 2,
            "b1_count":  medium // 2,
            "b2_count":  medium - medium // 2,
            "c1_count":  hard // 2,
            "c2_count":  hard - hard // 2,
        }
    return st


def _build_subtopics_code(subtopics: list) -> str:
    lines = ["    subtopics=["]
    for raw_st in subtopics:
        st = _normalise_subtopic(raw_st)
        lines.append("        SubtopicRequirement(")
        lines.append(f"            subtopic={st['subtopic']!r},")
        for lvl in ("a1", "a2", "b1", "b2", "c1", "c2"):
            lines.append(f"            {lvl}_count={st.get(lvl + '_count', 0)},")
        lines.append("        ),")
    lines.append("    ],")
    return "\n".join(lines)


def _build_constraints_code(constraints: dict) -> str:
    batch  = constraints.get("questions_per_iteration", 5)
    max_it = constraints.get("max_iterations_per_difficulty", 20)
    return (
        "    constraints=GenerationConstraints(\n"
        f"        questions_per_iteration={batch},\n"
        f"        max_iterations_per_difficulty={max_it},\n"
        "    ),"
    )


def _build_mcq_examples_code(examples: list) -> str:
    """Build ExampleQuestionSet code for MCQ generator."""
    lines = ["\nexample_questions = ExampleQuestionSet(items=["]
    for ex in examples:
        opts = ex.get("options", [])
        lines.append("    ExampleQuestion(")
        lines.append(f"        instruction={ex.get('instruction', '')!r},")
        lines.append(f"        question={ex.get('question', '')!r},")
        lines.append(f"        options={opts!r},")
        lines.append(f"        correct_answer={ex.get('correct_answer', '')!r},")
        lines.append(f"        explanation={ex.get('explanation', '')!r},")
        lines.append(f"        difficulty={ex.get('difficulty', 'Easy')!r},")
        lines.append(f"        cefr={ex.get('cefr', 'A1')!r},")
        lines.append(f"        subtopic={ex.get('subtopic', '')!r},")
        lines.append("    ),")
    lines.append("])")
    return "\n".join(lines)


def build_schema_code(config: dict, gen_type: str) -> str:
    """Build the Python code that defines schema (+ example_questions for MCQ)."""
    topic       = config["topic"]
    subtopics   = config.get("subtopics", [])
    constraints = config.get("constraints", {})

    schema_code = (
        f"schema = InputSchema(\n"
        f"    topic={topic!r},\n"
        f"{_build_subtopics_code(subtopics)}\n"
        f"{_build_constraints_code(constraints)}\n"
        f")"
    )

    if gen_type == "mcq":
        # Accept both "example_questions" and "questions" as the examples key
        examples = config.get("example_questions") or config.get("questions", [])
        schema_code += "\n" + _build_mcq_examples_code(examples)
    # T2T and Image MCQ: example_questions are loaded from the dataset inside the notebook —
    # no need to inject them here.

    return schema_code


# ── Notebook patching ──────────────────────────────────────────────────────────

def _patch_input_cell(cell_source: str | list, new_schema_code: str) -> str:
    """Replace the schema definition (up to the orchestrator line) with new_schema_code."""
    if isinstance(cell_source, list):
        source = "".join(cell_source)
    else:
        source = cell_source

    lines      = source.splitlines(keepends=True)
    before     = []
    after      = []
    state      = "before_schema"

    for line in lines:
        stripped = line.strip()

        if state == "before_schema":
            if stripped.startswith("schema = InputSchema("):
                state = "in_schema"
                # Don't add this line — it will be replaced by new_schema_code
            else:
                before.append(line)

        elif state == "in_schema":
            # Skip lines until we hit the orchestrator creation
            if stripped.startswith("orchestrator = "):
                state = "after_schema"
                after.append(line)

        elif state == "after_schema":
            after.append(line)

    if state == "before_schema":
        raise ValueError("Could not find 'schema = InputSchema(' in the input cell.")

    result = "".join(before) + new_schema_code + "\n\n" + "".join(after)
    return result


def inject_and_run(config_path: str) -> int:
    config    = json.loads(Path(config_path).read_text(encoding="utf-8"))
    gen_type  = config.get("type", "").lower()

    if gen_type not in NOTEBOOK_MAP:
        print(f"ERROR: 'type' must be one of: {', '.join(NOTEBOOK_MAP)}")
        print(f"       Got: {gen_type!r}")
        return 1

    nb_path = NOTEBOOK_MAP[gen_type]
    if not nb_path.exists():
        print(f"ERROR: Notebook not found: {nb_path}")
        return 1

    # Load notebook
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Build schema code from config
    new_schema_code = build_schema_code(config, gen_type)

    # Find and patch input cell (the one containing 'schema = InputSchema(')
    patched = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "schema = InputSchema(" in src:
            try:
                new_src = _patch_input_cell(cell["source"], new_schema_code)
                cell["source"] = new_src
                patched = True
                break
            except ValueError as e:
                print(f"ERROR patching notebook: {e}")
                return 1

    if not patched:
        print(f"ERROR: Could not find input cell in {nb_path.name}")
        return 1

    # Fix forward-reference NameErrors: make all annotations lazy in the fresh kernel.
    # MCQGeneratorAgent (cell 10) references DifficultyJudgeWrapper (defined later in cell 12).
    # This is safe — from __future__ import annotations must be the first statement.
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if src.strip() and "from __future__ import annotations" not in src:
                cell["source"] = "from __future__ import annotations\n" + src
            break

    # Write modified notebook to a temp file (never overwrites original)
    tmp_nb = PROJECT_ROOT / f"_tmp_generate_{gen_type}.ipynb"
    tmp_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    topic = config.get("topic", "unknown")
    subs  = [s.get("subtopic", "") for s in config.get("subtopics", [])]
    print(f"Generator : {gen_type}")
    print(f"Topic     : {topic}")
    print(f"Subtopics : {', '.join(subs)}")
    print(f"Notebook  : {nb_path.name}")
    print()
    print("Running... (this may take several minutes)")
    print()

    try:
        nb_obj = nbformat.read(str(tmp_nb), as_version=4)
        client = NotebookClient(
            nb_obj,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
        )
        client.execute()
        # Reformat question_number: 1 → Q1, 2 → Q2, ...
        _reformat_question_numbers(OUTPUT_MAP[gen_type])
        print()
        print("Done. Output saved to data/")
        return 0
    except Exception as e:
        print()
        msg = str(e).encode("ascii", errors="replace").decode("ascii")
        print(f"Notebook execution failed: {msg}")
        return 1
    finally:
        if tmp_nb.exists():
            tmp_nb.unlink()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run a Topin question generator from a JSON config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --config configs/english_grammar_mcq.json
  python generate.py --config configs/reading_notices.json
  python generate.py --list-types

Config templates:
  configs/template_mcq.json        MCQ generator
  configs/template_t2t.json        T2T (open-answer) generator
  configs/template_image_mcq.json  Image MCQ generator
""",
    )
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument(
        "--list-types", action="store_true",
        help="List available generator types and exit",
    )
    args = parser.parse_args()

    if args.list_types:
        print("Available generator types:")
        for name, path in NOTEBOOK_MAP.items():
            exists = "OK" if path.exists() else "MISSING"
            print(f"  {name:<12} {path.name}  [{exists}]")
        return

    if not args.config:
        parser.print_help()
        sys.exit(1)

    sys.exit(inject_and_run(args.config))


if __name__ == "__main__":
    main()
