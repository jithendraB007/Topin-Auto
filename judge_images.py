#!/usr/bin/env python3
"""
judge_images.py — Judge generated review images using Pixtral (Mistral vision model).

Reads:  data/image_mcq/review_images/1.png, 2.png, ...
        data/image_mcq/image_mcq_generator_output.json
Writes: data/image_mcq/image_judge_output.json
        data/image_mcq/review_with_scores.html

Uses MISTRAL_API_KEY from .env (already configured).

Usage:
    python judge_images.py
"""
import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

IMAGES_DIR   = PROJECT_ROOT / "data" / "image_mcq" / "review_images"
INPUT_JSON   = PROJECT_ROOT / "data" / "image_mcq" / "image_mcq_generator_output.json"
OUTPUT_JSON  = PROJECT_ROOT / "data" / "image_mcq" / "image_judge_output.json"
OUTPUT_HTML  = PROJECT_ROOT / "data" / "image_mcq" / "review_with_scores.html"

PIXTRAL_MODEL = "pixtral-12b-2409"

JUDGE_PROMPT = """\
You are an image quality evaluator for educational content.

You are shown an AI-generated image that is supposed to represent a real-world public notice or sign.
The intended notice content is: "{image_content}"

Evaluate the image on these 4 criteria:

1. relevance_to_description — Does the image match the described notice/sign content?
2. visual_quality — Is the image realistic, clear, and well-rendered?
3. text_legibility — Is any text in the image readable/legible?
4. contextual_fit — Does it look like a genuine real-world notice or sign (not a cartoon, etc.)?

Score each criterion as: "Excellent", "Good", or "Poor"
Give an overall_score from 1 to 10.
Give a brief reasoning (1-2 sentences).

Respond ONLY with valid JSON, no markdown, no extra text:
{{
  "relevance_to_description": "Excellent|Good|Poor",
  "visual_quality": "Excellent|Good|Poor",
  "text_legibility": "Excellent|Good|Poor",
  "contextual_fit": "Excellent|Good|Poor",
  "overall_score": <1-10>,
  "reasoning": "<brief explanation>"
}}"""


def _image_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def judge_image(client: Mistral, image_path: Path, image_content: str) -> dict:
    """Send image to Pixtral and return parsed rubric scores."""
    b64 = _image_to_b64(image_path)
    prompt = JUDGE_PROMPT.format(image_content=image_content)

    response = client.chat.complete(
        model=PIXTRAL_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _score_label(overall: int) -> str:
    if overall >= 8: return "Excellent"
    if overall >= 6: return "Good"
    if overall >= 4: return "Fair"
    return "Poor"


def _badge_color(overall: int) -> str:
    if overall >= 8: return "#1a7a1a"
    if overall >= 6: return "#b07800"
    if overall >= 4: return "#c05000"
    return "#cc0000"


def _criterion_color(val: str) -> str:
    return {"Excellent": "#1a7a1a", "Good": "#b07800", "Poor": "#cc0000"}.get(val, "#555")


def build_html(results: list[dict], output_path: Path) -> None:
    """Generate self-contained HTML review page with rubric scores."""
    cards = []
    for r in results:
        qnum    = r["question_number"]
        rubric  = r.get("rubric", {})
        overall = rubric.get("overall_score", 0)
        error   = r.get("error")

        # Image tag (base64 embedded)
        img_path = IMAGES_DIR / f"{qnum}.png"
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            img_tag = f'<img src="data:image/png;base64,{b64}" style="max-width:380px;border-radius:8px;box-shadow:0 2px 8px #0002;">'
        else:
            img_tag = '<div style="width:380px;height:280px;background:#f0f0f0;display:flex;align-items:center;justify-content:center;color:#999;border-radius:8px;">No image</div>'

        # Options
        opts_html = "".join(
            f'<li style="{"color:#1a7a1a;font-weight:600;" if o == r.get("correct_answer") else ""}">'
            f'{o}{"  ✓" if o == r.get("correct_answer") else ""}</li>'
            for o in r.get("options", [])
        )

        # Rubric block
        if error:
            rubric_html = f'<div style="color:#cc0000;font-size:12px;">Judge error: {error}</div>'
        else:
            criteria = ["relevance_to_description", "visual_quality", "text_legibility", "contextual_fit"]
            labels   = ["Relevance", "Visual Quality", "Text Legibility", "Contextual Fit"]
            rows = "".join(
                f'<tr><td style="padding:3px 8px 3px 0;color:#555;font-size:12px;">{lbl}</td>'
                f'<td style="padding:3px 0;font-size:12px;font-weight:600;color:{_criterion_color(rubric.get(c,""))};">'
                f'{rubric.get(c,"—")}</td></tr>'
                for c, lbl in zip(criteria, labels)
            )
            rubric_html = f"""
            <div style="margin-top:12px;">
              <span style="font-size:22px;font-weight:700;color:{_badge_color(overall)};">{overall}/10</span>
              <span style="font-size:12px;color:{_badge_color(overall)};margin-left:6px;font-weight:600;">{_score_label(overall)}</span>
              <table style="margin-top:8px;border-collapse:collapse;">{rows}</table>
              <div style="margin-top:8px;font-size:12px;color:#666;background:#f8f8f8;padding:8px 12px;border-radius:6px;">
                <strong>Judge reasoning:</strong> {rubric.get('reasoning','')}
              </div>
            </div>"""

        cards.append(f"""
        <div style="background:#fff;border-radius:12px;box-shadow:0 2px 12px #0001;padding:24px;margin-bottom:32px;display:flex;gap:24px;align-items:flex-start;">
          <div style="flex-shrink:0">{img_tag}</div>
          <div style="flex:1">
            <div style="font-size:13px;color:#888;margin-bottom:4px;">
              Q{qnum} &nbsp;|&nbsp; {r.get('target_cefr','')} &nbsp;|&nbsp; {r.get('target_difficulty','')}
            </div>
            <div style="font-size:12px;color:#aaa;margin-bottom:8px;font-style:italic;">{r.get('image_content','')}</div>
            <div style="font-size:13px;color:#555;margin-bottom:6px;">{r.get('instruction','')}</div>
            <div style="font-size:16px;font-weight:600;margin-bottom:12px;color:#222;">{r.get('question','')}</div>
            <ol type="A" style="margin:0;padding-left:20px;color:#444;">{opts_html}</ol>
            <div style="margin-top:10px;font-size:12px;color:#666;background:#f8f8f8;padding:8px 12px;border-radius:6px;">
              <strong>Explanation:</strong> {r.get('explanation','')}
            </div>
            {rubric_html}
          </div>
        </div>""")

    # Summary stats
    scored = [r for r in results if "rubric" in r and "overall_score" in r["rubric"]]
    avg = round(sum(r["rubric"]["overall_score"] for r in scored) / len(scored), 1) if scored else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Image MCQ Review with Scores</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background:#f5f5f5; margin:0; padding:32px; }}
    h1 {{ color:#222; margin-bottom:8px; }}
    .subtitle {{ color:#888; font-size:14px; margin-bottom:32px; }}
  </style>
</head>
<body>
  <h1>Image MCQ Review — Quality Scores</h1>
  <div class="subtitle">
    Total questions: {len(results)} &nbsp;|&nbsp;
    Average score: <strong>{avg}/10</strong> &nbsp;|&nbsp;
    Judge: Pixtral-12B &nbsp;|&nbsp; Generator: FLUX.1-schnell (HuggingFace)
  </div>
  {"".join(cards)}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


def main():
    # Load questions
    data      = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    questions = []
    for bucket in ("easy", "medium", "hard"):
        for q in data.get("questions", {}).get(bucket, []):
            q["_difficulty"] = bucket
            questions.append(q)

    print(f"Questions to judge : {len(questions)}")
    print(f"Model              : {PIXTRAL_MODEL}")
    print()

    client  = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    results = []

    for q in questions:
        qnum          = q.get("question_number")
        image_content = q.get("image_content", "")
        img_path      = IMAGES_DIR / f"{qnum}.png"

        print(f"[Q{qnum}] {image_content[:65]}")

        result = {
            "question_number":  qnum,
            "difficulty":       q.get("_difficulty", ""),
            "instruction":      q.get("instruction", ""),
            "image_content":    image_content,
            "question":         q.get("question", ""),
            "options":          q.get("options", []),
            "correct_answer":   q.get("correct_answer", ""),
            "explanation":      q.get("explanation", ""),
            "target_cefr":      q.get("target_cefr", ""),
            "target_difficulty":q.get("target_difficulty", ""),
            "image_path":       f"{qnum}.png",
            "rubric":           {},
            "error":            None,
        }

        if not img_path.exists():
            print(f"  [SKIP] Image not found: {img_path.name}")
            result["error"] = "image_not_found"
            results.append(result)
            continue

        try:
            rubric = judge_image(client, img_path, image_content)
            result["rubric"] = rubric
            score  = rubric.get("overall_score", "?")
            label  = _score_label(score) if isinstance(score, int) else ""
            print(f"  Score: {score}/10 ({label}) — {rubric.get('reasoning','')[:80]}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            result["error"] = str(e)

        results.append(result)

    # Save JSON
    output = {
        "metadata": {
            "total_questions": len(results),
            "judge_model":     PIXTRAL_MODEL,
            "generator":       "FLUX.1-schnell (HuggingFace)",
            "rubric_criteria": ["relevance_to_description", "visual_quality", "text_legibility", "contextual_fit"],
        },
        "results": results,
    }
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJudge output saved: {OUTPUT_JSON}")

    # Build HTML
    build_html(results, OUTPUT_HTML)
    print(f"Review page saved : {OUTPUT_HTML}")
    print(f"Open in browser   : file:///{OUTPUT_HTML.as_posix()}")

    # Print summary
    scored = [r for r in results if r.get("rubric", {}).get("overall_score")]
    if scored:
        avg = sum(r["rubric"]["overall_score"] for r in scored) / len(scored)
        print(f"\nSummary: {len(scored)}/{len(results)} judged  |  Average score: {avg:.1f}/10")


if __name__ == "__main__":
    main()
