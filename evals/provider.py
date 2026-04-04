from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent


def call_api(prompt: str, options, context):
    input_path = ROOT / "data/sample_input.json"
    output_path = ROOT / "data/promptfoo_eval_output.json"

    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(ROOT))
    if result.returncode != 0:
        return {"error": result.stderr or result.stdout}

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return {"output": json.dumps(payload, ensure_ascii=False)}
