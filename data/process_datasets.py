import json, sys

# === CEFR assignments for each question ID ===
TRAIN_CEFR = {
    "Q3":  ("A2","Easy"),   # girl decides to stay home - direct dialogue recall
    "Q4":  ("B1","Medium"), # cat sick - longer dialogue inference
    "Q5":  ("A2","Easy"),   # David started at 12 - factual recall
    "Q20": ("A2","Easy"),   # David eats cereal & fruit - factual recall
    "Q48": ("B1","Medium"), # Farida sweater - single-sentence inference
    "Q49": ("A2","Easy"),   # Health camp 10am = morning - fact match
    "Q50": ("B1","Medium"), # 'explicit' vocabulary in context
    "Q51": ("B1","Medium"), # Nisha charger - single-sentence inference
    "Q52": ("A2","Easy"),   # Science exhibition 11am = morning - fact match
    "Q53": ("B1","Medium"), # Conversation completion - who prepared draft
    "Q54": ("B2","Medium"), # Passive voice identification
    "Q55": ("B2","Medium"), # neither/nor parallel structure
    "Q56": ("B1","Medium"), # Priya alarm - short passage inference
    "Q59": ("B2","Medium"), # Passive voice identification (she)
    "Q66": ("B2","Medium"), # not only/but also parallel structure
    "Q67": ("B1","Medium"), # Conversation completion - projector
}

GOOD_CEFR = {
    "Q1":  ("A2","Easy"),   # Assembly indoors - factual recall
    "Q2":  ("A2","Easy"),   # Bakery sold out - inference
    "Q14": ("A2","Easy"),   # Anna interview - what sport
    "Q57": ("B2","Medium"), # Text completion - connective + relative pronoun
    "Q60": ("B2","Medium"), # 'unanimous' vocabulary
    "Q63": ("B1","Medium"), # Preposition fill-in - around/after
    "Q68": ("B1","Medium"), # Conversation completion - Rahul injured
    "Q69": ("B2","Medium"), # Passive voice rewrite - receipt
    "Q70": ("B2","Medium"), # Passive voice rewrite - Lalita
}

FLAWED_CEFR = {
    "Q13": ("A2","Easy"),   # FLAW: "All of the above" not supported by dialogue
    "Q15": ("A2","Easy"),   # FLAW: correct_answer "7:00" not in options [4:00,5:00,6:00]
    "Q17": ("A2","Easy"),   # FLAW: correct_answer "A cup" not in options [medal,trophy,holiday]
    "Q21": ("B1","Medium"), # FLAW: correct_answer contradicts the conversation
    "Q58": ("B2","Medium"), # FLAW: marked answer "approving,schedule" wrong (should be approve,schedule)
}

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return {q["question_id"]: q for q in json.load(f)}

def make_record(q, cefr, diff, decision):
    # Skip questions with empty options (open-ended/speaking tasks)
    if not q.get("options"):
        return None
    exp = q.get("explanation") or ""
    return {
        "question_id": q["question_id"],
        "stem": q["stem"],
        "options": q["options"],
        "correct_answer": q["correct_answer"],
        "explanation": exp,
        "target_cefr": cefr,
        "target_difficulty": diff,
        "expected_predicted_cefr": cefr,
        "expected_overall_decision": decision,
    }

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written {len(records)} records -> {path}")

train_src = load_json("d:/Topin/data/training_dataset_standard.json")
eval_src  = load_json("d:/Topin/data/eval_dataset_standard.json")

# gepa_train.jsonl
train_records = []
for qid, (cefr, diff) in TRAIN_CEFR.items():
    r = make_record(train_src[qid], cefr, diff, "Pass")
    if r: train_records.append(r)
write_jsonl("d:/Topin/data/gepa_train.jsonl", train_records)

# gold_good.jsonl
good_records = []
for qid, (cefr, diff) in GOOD_CEFR.items():
    r = make_record(eval_src[qid], cefr, diff, "Pass")
    if r: good_records.append(r)
write_jsonl("d:/Topin/data/gold_good.jsonl", good_records)

# gold_flawed.jsonl
flawed_records = []
for qid, (cefr, diff) in FLAWED_CEFR.items():
    r = make_record(eval_src[qid], cefr, diff, "Fail")
    if r: flawed_records.append(r)
write_jsonl("d:/Topin/data/gold_flawed.jsonl", flawed_records)

print("\nCEFR distribution (train):", {})
from collections import Counter
print("  Train:", dict(sorted(Counter(r["target_cefr"] for r in train_records).items())))
print("  Good eval:", dict(sorted(Counter(r["target_cefr"] for r in good_records).items())))
print("  Flawed eval:", dict(sorted(Counter(r["target_cefr"] for r in flawed_records).items())))
