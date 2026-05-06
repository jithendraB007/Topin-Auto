#!/usr/bin/env python3
"""
generate_review_images.py — Generate review images for Image MCQ questions.

Reads:  data/image_mcq/image_mcq_generator_output.json
Writes: data/image_mcq/review_images/<qnum>.png          ← winning image per question
        data/image_mcq/review_images/<qnum>_candidate_1.png  ← azure provider only
        data/image_mcq/review_images/<qnum>_candidate_2.png  ← azure provider only
        data/image_mcq/review.html                       ← open in browser
        data/image_mcq/image_selection_log.json          ← azure: DSPy judge log

Providers:
    azure    — GPT Image 2 (×2 candidates) + DSPy GPT-5.4-Mini judge (recommended)
    hf       — Hugging Face FLUX.1-schnell (free)
    gptimage — GPT Image 2 via OpenAI API
    dalle    — DALL-E 3
    gemini   — Gemini Imagen

Usage:
    python generate_review_images.py                       # auto-detect provider
    python generate_review_images.py --provider azure      # Azure GPT Image 2 + DSPy judge
    python generate_review_images.py --provider hf         # HuggingFace (free)
    python generate_review_images.py --provider gptimage   # GPT Image 2
    python generate_review_images.py --provider dalle      # DALL-E 3
    python generate_review_images.py --provider gemini     # Gemini Imagen
"""
import argparse
import base64
import io
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_INPUT  = PROJECT_ROOT / "data" / "image_mcq" / "image_mcq_generator_output.json"
IMAGES_DIR     = PROJECT_ROOT / "data" / "image_mcq" / "review_images"
REVIEW_HTML    = PROJECT_ROOT / "data" / "image_mcq" / "review.html"
SELECTION_LOG  = PROJECT_ROOT / "data" / "image_mcq" / "image_selection_log.json"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── Azure config ──────────────────────────────────────────────────────────────
AZURE_KEY            = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_ENDPOINT       = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_VERSION    = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
AZURE_IMAGE_MODEL_1  = os.environ.get("AZURE_IMAGE_DEPLOYMENT_1", "gpt-5.4-2026-03-05")
AZURE_IMAGE_MODEL_2  = os.environ.get("AZURE_IMAGE_DEPLOYMENT_2", "gpt-5.4-mini-2026-03-17")
AZURE_JUDGE_MODEL    = os.environ.get("AZURE_JUDGE_DEPLOYMENT",   "gpt-5.4-mini-2026-03-17")

# HuggingFace config (kept for --provider hf fallback)
HF_MODEL   = "black-forest-labs/FLUX.1-schnell"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

# Pollinations.ai config (free, no API key, FLUX model)
POLLINATIONS_MODEL = "pollinations/flux"
POLLINATIONS_BASE  = "https://image.pollinations.ai/prompt"


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_image_prompt(image_content: str) -> str:
    return (
        f"A realistic photograph of a printed public notice or sign displayed on a wall. "
        f"{image_content} "
        f"Professional printed notice board, clean typography, white or light background, "
        f"easy to read text, no people in frame."
    )

def _build_image_prompt_b(image_content: str) -> str:
    """Alternate prompt style — more formal / institutional."""
    return (
        f"A close-up of a neatly printed official notice board mounted on a wall. "
        f"{image_content} "
        f"Clear, legible text, formal institutional layout, real-world photograph, "
        f"high contrast text on light background."
    )


# ── Image saver helper ─────────────────────────────────────────────────────────

def _save_openai_image(image_data, out_path: Path) -> None:
    """Save OpenAI image response (URL or b64_json) to disk."""
    if getattr(image_data, "url", None):
        urllib.request.urlretrieve(image_data.url, out_path)
    elif getattr(image_data, "b64_json", None):
        out_path.write_bytes(base64.b64decode(image_data.b64_json))
    else:
        raise ValueError("Image response has neither url nor b64_json")


# ── Azure GPT Image 2 + DSPy Judge (main provider) ───────────────────────────

def _azure_generate_image(_azure_client, _model: str, prompt: str, out_path: Path) -> bool:
    """Generate one image for the azure pipeline via HuggingFace FLUX.1-schnell.
    (Azure gpt-5.4 deployments are text/vision models and do not support image generation.)
    """
    return _generate_hf(prompt, out_path)


def _generate_azure_dspy(image_content: str, qnum: str, out_path: Path, force: bool = False) -> dict:
    """
    Generate 2 candidate images via HuggingFace FLUX.1-schnell:
      Candidate 1 — photographic notice style prompt
      Candidate 2 — formal institutional style prompt
    Evaluate both with Azure GPT-5.4-Mini via DSPy vision judge. Save winner.
    Returns a log dict with scores, winner, reasoning.
    """
    import dspy

    c1_path = IMAGES_DIR / f"{qnum}_candidate_1.png"
    c2_path = IMAGES_DIR / f"{qnum}_candidate_2.png"

    log = {
        "question_number":    qnum,
        "image_content":      image_content,
        "image_gen_model":    POLLINATIONS_MODEL,
        "candidate_1":        c1_path.name,
        "candidate_1_prompt": "photographic notice style",
        "candidate_2":        c2_path.name,
        "candidate_2_prompt": "formal institutional style",
        "judge_model":        AZURE_JUDGE_MODEL,
        "winner":             None,
        "winner_file":        None,
        "score_1":            None,
        "score_2":            None,
        "reasoning":          None,
        "error":              None,
    }

    if force:
        for p in (c1_path, c2_path, out_path):
            p.unlink(missing_ok=True)

    prompt   = _build_image_prompt(image_content)
    prompt_b = _build_image_prompt_b(image_content)

    # ── Step 1: Candidate 1 — photographic notice style ───────────────────────
    if not c1_path.exists():
        ok1 = _generate_pollinations(prompt, c1_path)
        if not ok1:
            log["error"] = "candidate_1: Pollinations generation failed"
    else:
        print(f"  [Skip gen] {c1_path.name} already exists")

    # ── Step 2: Candidate 2 — formal institutional style ─────────────────────
    if not c2_path.exists():
        ok2 = _generate_pollinations(prompt_b, c2_path)
        if not ok2:
            log["error"] = (log.get("error") or "") + " | candidate_2: Pollinations generation failed"
    else:
        print(f"  [Skip gen] {c2_path.name} already exists")

    # ── Step 3: DSPy image judge ───────────────────────────────────────────────
    c1_ok = c1_path.exists()
    c2_ok = c2_path.exists()

    if c1_ok and c2_ok:
        try:
            azure_lm = dspy.LM(
                f"azure/{AZURE_JUDGE_MODEL}",
                api_key=AZURE_KEY,
                api_base=AZURE_ENDPOINT,
                api_version=AZURE_API_VERSION,
            )

            class ImageJudge(dspy.Signature):
                """Compare two images of a public notice/sign for educational MCQ use.
                Select the image that best matches the intended content with clear,
                legible text and realistic presentation."""
                image_content: str    = dspy.InputField(desc="Intended notice/sign content")
                image_1:       dspy.Image = dspy.InputField(desc="Candidate 1 — photographic notice style")
                image_2:       dspy.Image = dspy.InputField(desc="Candidate 2 — formal institutional style")
                score_1:       int    = dspy.OutputField(desc="Score for image 1 (1-10): relevance + text legibility + visual quality")
                score_2:       int    = dspy.OutputField(desc="Score for image 2 (1-10): relevance + text legibility + visual quality")
                winner:        str    = dspy.OutputField(desc="'image_1' or 'image_2' — whichever scores higher")
                reasoning:     str    = dspy.OutputField(desc="One sentence explaining the choice")

            with dspy.context(lm=azure_lm):
                judge  = dspy.Predict(ImageJudge)
                result = judge(
                    image_content=image_content,
                    image_1=dspy.Image(str(c1_path)),
                    image_2=dspy.Image(str(c2_path)),
                )

            log["score_1"]   = result.score_1
            log["score_2"]   = result.score_2
            log["winner"]    = result.winner
            log["reasoning"] = result.reasoning

            winner_path = c1_path if result.winner == "image_1" else c2_path
            shutil.copy2(winner_path, out_path)   # overwrites existing winner with correct one
            log["winner_file"] = out_path.name

            print(f"  [DSPy Judge] Winner: {result.winner}  "
                  f"(score: {result.score_1} vs {result.score_2})")
            print(f"  [DSPy Judge] {result.reasoning[:90]}")

        except Exception as e:
            log["error"] = (log.get("error") or "") + f" | judge: {e}"
            print(f"  [DSPy Judge FAILED]: {e}")
            # Fall back: use whichever candidate exists
            fallback = c1_path if c1_ok else c2_path
            shutil.copy2(fallback, out_path)
            log["winner_file"] = out_path.name
            print(f"  [Fallback] Copied {fallback.name} as winner")

    elif c1_ok:
        shutil.copy2(c1_path, out_path)
        log["winner_file"] = out_path.name
        print(f"  [Fallback] Only candidate 1 available — used as winner")

    elif c2_ok:
        shutil.copy2(c2_path, out_path)
        log["winner_file"] = out_path.name
        print(f"  [Fallback] Only candidate 2 available — used as winner")

    else:
        log["error"] = "Both candidates failed to generate"
        print(f"  [Azure FAILED] Both candidates failed for {qnum}")

    return log


# ── Pollinations.ai FLUX (free, no API key) ──────────────────────────────────

def _generate_pollinations(prompt: str, out_path: Path) -> bool:
    if out_path.exists():
        print(f"  [Skip] {out_path.name} already exists")
        return True
    try:
        import urllib.parse
        encoded = urllib.parse.quote(prompt)
        url = f"{POLLINATIONS_BASE}/{encoded}?width=1024&height=1024&model=flux&nologo=true"
        for attempt in range(3):
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            out_path.write_bytes(resp.content)
            print(f"  [Pollinations] Generated: {out_path.name}")
            return True
        raise RuntimeError("No response after 3 attempts")
    except Exception as e:
        print(f"  [Pollinations FAILED] {out_path.name}: {e}")
        return False


# ── HuggingFace FLUX.1-schnell (free) ────────────────────────────────────────

_GPT_BILLING_LIMIT_HIT = False


def _generate_hf(prompt: str, out_path: Path) -> bool:
    if out_path.exists():
        print(f"  [Skip] {out_path.name} already exists")
        return True
    try:
        token   = os.environ["HF_TOKEN"]
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": prompt}
        for attempt in range(3):
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 503:
                print(f"  [HF] Model loading, waiting 20s... ({attempt+1}/3)")
                time.sleep(20)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            out_path.write_bytes(resp.content)
            print(f"  [HF] Generated: {out_path.name}")
            return True
        raise RuntimeError("Model still loading after 3 attempts")
    except Exception as e:
        print(f"  [HF FAILED] {out_path.name}: {e}")
        return False


# ── GPT Image 2 (OpenAI API) ─────────────────────────────────────────────────

def _generate_gpt_image(prompt: str, out_path: Path) -> bool:
    global _GPT_BILLING_LIMIT_HIT
    if out_path.exists():
        print(f"  [Skip] {out_path.name} already exists")
        return True
    if _GPT_BILLING_LIMIT_HIT:
        return _generate_hf(prompt, out_path)
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp   = client.images.generate(
            model="gpt-image-2", prompt=prompt, n=1,
            size="1024x1024", quality="high",
        )
        _save_openai_image(resp.data[0], out_path)
        print(f"  [GPT-Image-2] Generated: {out_path.name}")
        return True
    except Exception as e:
        if "billing_hard_limit_reached" in str(e) or "billing_limit" in str(e):
            _GPT_BILLING_LIMIT_HIT = True
            print("  [GPT-Image-2] Daily quota reached — switching to HuggingFace")
            return _generate_hf(prompt, out_path)
        print(f"  [GPT-Image-2 FAILED] {out_path.name}: {e}")
        return False


# ── DALL-E 3 ─────────────────────────────────────────────────────────────────

def _generate_dalle(prompt: str, out_path: Path) -> bool:
    if out_path.exists():
        print(f"  [Skip] {out_path.name} already exists")
        return True
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp   = client.images.generate(
            model="dall-e-3", prompt=prompt, size="1024x1024",
            quality="standard", n=1, response_format="url",
        )
        urllib.request.urlretrieve(resp.data[0].url, out_path)
        print(f"  [DALL-E] Generated: {out_path.name}")
        return True
    except Exception as e:
        print(f"  [DALL-E FAILED] {out_path.name}: {e}")
        return False


# ── Gemini Imagen ─────────────────────────────────────────────────────────────

def _generate_gemini(prompt: str, out_path: Path) -> bool:
    if out_path.exists():
        print(f"  [Skip] {out_path.name} already exists")
        return True
    try:
        from PIL import Image as PILImage
        from google import genai as google_genai
        from google.genai import types as google_types

        client   = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=prompt,
            config=google_types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        img_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img_bytes = part.inline_data.data
                break
        if img_bytes is None:
            raise ValueError("No image data returned")
        PILImage.open(io.BytesIO(img_bytes)).save(out_path, format="PNG")
        print(f"  [Gemini] Generated: {out_path.name}")
        return True
    except Exception as e:
        print(f"  [Gemini FAILED] {out_path.name}: {e}")
        return False


# ── Provider selector ─────────────────────────────────────────────────────────

def _get_simple_generator(provider: str):
    """Return a simple (prompt, path) → bool generator for non-azure providers."""
    has_hf     = bool(os.environ.get("HF_TOKEN"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_google = bool(os.environ.get("GOOGLE_API_KEY"))

    if provider == "pollinations":
        print("Using Pollinations.ai (FLUX) — free, no API key")
        return _generate_pollinations

    if provider == "hf":
        if not has_hf:
            raise RuntimeError("HF_TOKEN not set in .env")
        print("Using Hugging Face (FLUX.1-schnell) — free tier")
        return _generate_hf

    if provider == "gptimage":
        if not has_openai:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        print("Using GPT Image 2 (gpt-image-2)")
        return _generate_gpt_image

    if provider == "dalle":
        if not has_openai:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        print("Using DALL-E 3")
        return _generate_dalle

    if provider == "gemini":
        if not has_google:
            raise RuntimeError("GOOGLE_API_KEY not set in .env")
        print("Using Gemini Imagen")
        return _generate_gemini

    # auto: Pollinations first (free, no key needed)
    print("Using Pollinations.ai (FLUX) — free, no API key")
    return _generate_pollinations

    raise RuntimeError(
        "No image generation API key found.\n"
        "Add to .env:  HF_TOKEN=hf_...   (free)\n"
        "           or AZURE_OPENAI_KEY=...  (Azure GPT Image 2 + DSPy judge)\n"
        "           or OPENAI_API_KEY=sk-... (GPT Image 2)"
    )


# ── HTML review builder ───────────────────────────────────────────────────────

def _build_review_html(questions: list[dict], images_dir: Path, output_path: Path) -> None:
    def _img_tag(qnum: str) -> str:
        img_path = images_dir / f"{qnum}.png"
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            return (f'<img src="data:image/png;base64,{b64}" alt="{qnum}" '
                    f'style="max-width:400px;border-radius:8px;box-shadow:0 2px 8px #0002;">')
        return ('<div style="width:400px;height:300px;background:#f0f0f0;display:flex;'
                'align-items:center;justify-content:center;border-radius:8px;color:#999;">'
                'Image not generated</div>')

    cards = []
    for q in questions:
        qnum     = str(q.get("question_number", ""))
        opts_html = "".join(
            f'<li style="{"color:#1a7a1a;font-weight:600;" if o == q.get("correct_answer") else ""}">'
            f'{o}{"  ✓" if o == q.get("correct_answer") else ""}</li>'
            for o in q.get("options", [])
        )
        cards.append(f"""
        <div style="background:#fff;border-radius:12px;box-shadow:0 2px 12px #0001;padding:24px;margin-bottom:32px;display:flex;gap:24px;align-items:flex-start;">
          <div style="flex-shrink:0">{_img_tag(qnum)}</div>
          <div style="flex:1">
            <div style="font-size:13px;color:#888;margin-bottom:4px;">
              {qnum} &nbsp;|&nbsp; {q.get('target_cefr','')} &nbsp;|&nbsp; {q.get('target_difficulty','')}
            </div>
            <div style="font-size:12px;color:#aaa;margin-bottom:8px;font-style:italic;">{q.get('image_content','')}</div>
            <div style="font-size:13px;color:#555;margin-bottom:6px;">{q.get('instruction','')}</div>
            <div style="font-size:16px;font-weight:600;margin-bottom:12px;color:#222;">{q.get('question','')}</div>
            <ol type="A" style="margin:0;padding-left:20px;color:#444;">{opts_html}</ol>
            <div style="margin-top:12px;font-size:12px;color:#666;background:#f8f8f8;padding:8px 12px;border-radius:6px;">
              <strong>Explanation:</strong> {q.get('explanation','')}
            </div>
          </div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Image MCQ Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background:#f5f5f5; margin:0; padding:32px; }}
    h1 {{ color:#222; margin-bottom:8px; }}
    .subtitle {{ color:#888; font-size:14px; margin-bottom:32px; }}
  </style>
</head>
<body>
  <h1>Image MCQ Review</h1>
  <div class="subtitle">Total questions: {len(questions)} &nbsp;|&nbsp; Generated by Topin</div>
  {"".join(cards)}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\nReview page saved : {output_path}")
    print(f"Open in browser   : file:///{output_path.as_posix()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate review images for Image MCQ questions")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to image_mcq_generator_output.json")
    parser.add_argument("--provider", default="auto",
                        choices=["auto", "azure", "pollinations", "hf", "gptimage", "dalle", "gemini"],
                        help="Image provider (default: auto). 'azure' uses Pollinations gen + Azure DSPy judge")
    parser.add_argument("--force", action="store_true",
                        help="Delete and regenerate all images even if they already exist")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    data      = json.loads(input_path.read_text(encoding="utf-8"))
    questions = []
    if isinstance(data, list):
        # Flat array format: eval_dataset_24_clean.json style
        for q in data:
            # Normalise question_id → question_number so the rest of the code is uniform
            if "question_id" in q and "question_number" not in q:
                q = dict(q, question_number=q["question_id"])
            questions.append(q)
    else:
        # Nested format: image_mcq_generator_output.json style
        for bucket in ("easy", "medium", "hard"):
            questions.extend(data.get("questions", {}).get(bucket, []))

    if not questions:
        print("No questions found in the output file.")
        return 1

    print(f"Found {len(questions)} questions")
    print(f"Images will be saved to: {IMAGES_DIR}")
    print()

    # ── Azure provider — two-image + DSPy judge flow ──────────────────────────
    if args.provider == "azure" or (args.provider == "auto" and AZURE_KEY):
        if not AZURE_KEY:
            print("ERROR: AZURE_OPENAI_KEY not set in .env")
            return 1
        print(f"Using Pollinations.ai FLUX (×2 candidates) + Azure GPT-5.4-Mini Judge")
        print(f"  Image gen   : Pollinations.ai FLUX (free, no API key)")
        print(f"  Judge model : {AZURE_JUDGE_MODEL} (Azure GPT-5.4-Mini vision)")
        print()

        ok, failed  = 0, 0
        selection_log = []

        for q in questions:
            qnum          = str(q.get("question_number", f"Q{questions.index(q)+1}"))
            image_content = q.get("image_content", "")
            out_path      = IMAGES_DIR / f"{qnum}.png"

            print(f"[{qnum}] {image_content[:70]}")
            log = _generate_azure_dspy(image_content, qnum, out_path, force=args.force)
            selection_log.append(log)

            if out_path.exists():
                ok += 1
            else:
                failed += 1

        # Save DSPy selection log
        SELECTION_LOG.write_text(
            json.dumps({
                "provider":       "pollinations+azure-dspy",
                "image_gen_model": POLLINATIONS_MODEL,
                "judge_model":    AZURE_JUDGE_MODEL,
                "results":        selection_log,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\nDone. Generated={ok}  Failed={failed}")
        print(f"DSPy selection log: {SELECTION_LOG}")

    # ── Simple providers (HF / GPT Image 2 / DALL-E / Gemini) ────────────────
    else:
        try:
            generate = _get_simple_generator(args.provider)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 1

        ok, failed = 0, 0
        for q in questions:
            qnum          = str(q.get("question_number", f"Q{questions.index(q)+1}"))
            image_content = q.get("image_content", "")
            out_path      = IMAGES_DIR / f"{qnum}.png"
            prompt        = _build_image_prompt(image_content)

            if args.force:
                out_path.unlink(missing_ok=True)
            print(f"[{qnum}] {image_content[:70]}")
            if generate(prompt, out_path):
                ok += 1
            else:
                failed += 1

        print(f"\nDone. Generated={ok}  Failed={failed}")

    # ── Build HTML review page ────────────────────────────────────────────────
    _build_review_html(questions, IMAGES_DIR, REVIEW_HTML)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
