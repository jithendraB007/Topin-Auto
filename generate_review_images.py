#!/usr/bin/env python3
"""
generate_review_images.py — Generate review images for Image MCQ questions.

Reads:  data/image_mcq/image_mcq_generator_output.json
Writes: data/image_mcq/review_images/Q1.png, Q2.png, ...
        data/image_mcq/review.html   ← open in browser to review all questions + images

Image generation uses Hugging Face (free), DALL-E 3, or Gemini Imagen.
Add at least one key to your .env file:
    HF_TOKEN=hf_...             for Hugging Face (free) ← recommended
    OPENAI_API_KEY=sk-...       for DALL-E 3 (paid)
    GOOGLE_API_KEY=AIza...      for Gemini Imagen (paid)

Usage:
    python generate_review_images.py
    python generate_review_images.py --input data/image_mcq/image_mcq_generator_output.json
    python generate_review_images.py --provider hf        # force Hugging Face (free)
    python generate_review_images.py --provider dalle     # force DALL-E
    python generate_review_images.py --provider gemini    # force Gemini
"""
import argparse
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_INPUT  = PROJECT_ROOT / "data" / "image_mcq" / "image_mcq_generator_output.json"
IMAGES_DIR     = PROJECT_ROOT / "data" / "image_mcq" / "review_images"
REVIEW_HTML    = PROJECT_ROOT / "data" / "image_mcq" / "review.html"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face model — FLUX.1-schnell: fast, high quality, free
HF_MODEL = "black-forest-labs/FLUX.1-schnell"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"


# ── Image prompt builder ──────────────────────────────────────────────────────

def _build_image_prompt(image_content: str) -> str:
    """Turn an image_content description into a detailed image generation prompt."""
    return (
        f"A realistic photograph of a printed public notice or sign displayed on a wall. "
        f"{image_content} "
        f"Professional printed notice board, clean typography, white or light background, "
        f"easy to read text, no people in frame."
    )


# ── Hugging Face generator (free) ─────────────────────────────────────────────

def _generate_hf(prompt: str, out_path: Path) -> bool:
    if out_path.exists():
        print(f"  [Skip] {out_path.name} already exists")
        return True
    try:
        token = os.environ["HF_TOKEN"]
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": prompt}

        # HF may return 503 while model loads — retry up to 3 times
        for attempt in range(3):
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 503:
                wait = 20
                print(f"  [HF] Model loading, waiting {wait}s... (attempt {attempt+1}/3)")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            # Response is raw image bytes
            out_path.write_bytes(resp.content)
            print(f"  [HF] Generated: {out_path.name}")
            return True

        raise RuntimeError("Model still loading after 3 attempts — try again in a minute")
    except Exception as e:
        print(f"  [HF FAILED] {out_path.name}: {e}")
        return False


# ── DALL-E 3 generator ────────────────────────────────────────────────────────

def _generate_dalle(prompt: str, out_path: Path) -> bool:
    if out_path.exists():
        print(f"  [Skip] {out_path.name} already exists")
        return True
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url",
        )
        urllib.request.urlretrieve(resp.data[0].url, out_path)
        print(f"  [DALL-E] Generated: {out_path.name}")
        return True
    except Exception as e:
        print(f"  [DALL-E FAILED] {out_path.name}: {e}")
        return False


# ── Gemini Imagen generator ───────────────────────────────────────────────────

def _generate_gemini(prompt: str, out_path: Path) -> bool:
    if out_path.exists():
        print(f"  [Skip] {out_path.name} already exists")
        return True
    try:
        from PIL import Image as PILImage
        from google import genai as google_genai
        from google.genai import types as google_types

        client = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=prompt,
            config=google_types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        img_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img_bytes = part.inline_data.data
                break
        if img_bytes is None:
            raise ValueError("No image data returned in response")
        PILImage.open(io.BytesIO(img_bytes)).save(out_path, format="PNG")
        print(f"  [Gemini] Generated: {out_path.name}")
        return True
    except Exception as e:
        print(f"  [Gemini FAILED] {out_path.name}: {e}")
        return False


# ── Provider selector ─────────────────────────────────────────────────────────

def _get_generator(provider: str):
    """Return the image generation function based on available API keys."""
    has_hf     = bool(os.environ.get("HF_TOKEN"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_google = bool(os.environ.get("GOOGLE_API_KEY"))

    if provider == "hf":
        if not has_hf:
            raise RuntimeError("HF_TOKEN not set in .env — get a free token at https://huggingface.co/settings/tokens")
        print(f"Using Hugging Face (FLUX.1-schnell) — free tier")
        return _generate_hf

    if provider == "dalle":
        if not has_openai:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        return _generate_dalle

    if provider == "gemini":
        if not has_google:
            raise RuntimeError("GOOGLE_API_KEY not set in .env")
        return _generate_gemini

    # auto: prefer HF (free), then DALL-E, then Gemini
    if has_hf:
        print(f"Using Hugging Face (FLUX.1-schnell) — free tier")
        return _generate_hf
    if has_openai:
        print("Using DALL-E 3 (OPENAI_API_KEY found)")
        return _generate_dalle
    if has_google:
        print("Using Gemini Imagen (GOOGLE_API_KEY found)")
        return _generate_gemini

    raise RuntimeError(
        "No image generation API key found.\n"
        "Add one of these to your .env file:\n"
        "  HF_TOKEN=hf_...             for Hugging Face (free) ← recommended\n"
        "  OPENAI_API_KEY=sk-...       for DALL-E 3\n"
        "  GOOGLE_API_KEY=AIza...      for Gemini Imagen"
    )


# ── HTML review builder ───────────────────────────────────────────────────────

def _build_review_html(questions: list[dict], images_dir: Path, output_path: Path) -> None:
    """Generate a self-contained HTML review page."""
    import base64

    def _img_tag(qnum: str) -> str:
        img_path = images_dir / f"{qnum}.png"
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            return f'<img src="data:image/png;base64,{b64}" alt="{qnum}" style="max-width:400px;border-radius:8px;box-shadow:0 2px 8px #0002;">'
        return f'<div style="width:400px;height:300px;background:#f0f0f0;display:flex;align-items:center;justify-content:center;border-radius:8px;color:#999;">Image not generated</div>'

    cards = []
    for q in questions:
        qnum = str(q.get("question_number", ""))
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
    print(f"\nReview page saved: {output_path}")
    print(f"Open in browser:  file:///{output_path.as_posix()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate review images for Image MCQ questions")
    parser.add_argument("--input",    default=str(DEFAULT_INPUT), help="Path to image_mcq_generator_output.json")
    parser.add_argument("--provider", default="auto", choices=["auto", "hf", "dalle", "gemini"],
                        help="Image generation provider (default: auto — prefers HF free tier)")
    args = parser.parse_args()

    # Load questions
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    data      = json.loads(input_path.read_text(encoding="utf-8"))
    questions = []
    for bucket in ("easy", "medium", "hard"):
        questions.extend(data.get("questions", {}).get(bucket, []))

    if not questions:
        print("No questions found in the output file.")
        return 1

    print(f"Found {len(questions)} questions")
    print(f"Images will be saved to: {IMAGES_DIR}")
    print()

    # Get image generator
    try:
        generate = _get_generator(args.provider)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    # Generate one image per question
    ok, failed = 0, 0
    for q in questions:
        qnum          = str(q.get("question_number", f"Q{questions.index(q)+1}"))
        image_content = q.get("image_content", "")
        out_path      = IMAGES_DIR / f"{qnum}.png"
        prompt        = _build_image_prompt(image_content)

        print(f"[{qnum}] {image_content[:70]}")
        success = generate(prompt, out_path)
        if success:
            ok += 1
        else:
            failed += 1

    print(f"\nDone. Generated={ok}  Failed={failed}")

    # Build HTML review page
    _build_review_html(questions, IMAGES_DIR, REVIEW_HTML)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
