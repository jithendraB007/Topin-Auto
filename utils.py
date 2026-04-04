from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import dspy
from dotenv import load_dotenv


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def configure_dspy_from_env() -> dspy.LM:
    load_dotenv()
    api_key = os.environ["MISTRAL_API_KEY"]
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    api_base = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")

    # DSPy documents the generic LM setup via dspy.LM(...).
    # This starter uses the OpenAI-compatible chat-completions shape exposed by Mistral's API.
    lm = dspy.LM(
        f"openai/{model}",
        api_key=api_key,
        api_base=api_base,
        temperature=0.2,
        max_tokens=2000,
    )
    dspy.configure(lm=lm)
    return lm
