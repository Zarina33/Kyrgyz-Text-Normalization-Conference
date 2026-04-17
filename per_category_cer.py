"""
Per-category CER breakdown for each normalization type.
Uses the same category logic as dataset_analysis.py.
"""

import json
import re


# ── helpers ──────────────────────────────────────────────────────────────────

def cer(ref: str, hyp: str) -> float:
    """Character Error Rate via Levenshtein distance."""
    r, h = list(ref), list(hyp)
    n = len(r)
    if n == 0:
        return 0.0 if len(h) == 0 else 1.0
    # DP table
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev = d[:]
        d[0] = i
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + cost)
    return d[len(h)] / n


def categorize(inp: str, tgt: str):
    """Return set of normalization categories for this example."""
    cats = set()
    if tgt and inp and tgt[0].isupper() and inp[0].islower():
        cats.add("Capitalization")
    if sum(1 for c in tgt if c in '.,!?;:') > sum(1 for c in inp if c in '.,!?;:'):
        cats.add("Punctuation")
    if sum(1 for c in inp if c.isupper()) > len(inp) * 0.3:
        cats.add("All-Caps")
    if re.search(r'\d+[а-яёөүңкгжшщ]', inp, re.IGNORECASE):
        cats.add("Digit-Word")
    if inp.strip() == tgt.strip():
        cats.add("No Change")
    return cats


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ── main ─────────────────────────────────────────────────────────────────────

SYSTEMS = {
    "Rule-Based":     "eval_results/rulebased_formal.jsonl",
    "Gemma 4":        "eval_results/gemma4_formal.jsonl",
    "FT":             "eval_results/finetuned_formal.jsonl",
    "Pre-Train+FT":   "eval_results/pretrained_ft_formal.jsonl",
}

CATEGORIES = ["Punctuation", "Capitalization", "All-Caps", "Digit-Word", "No Change"]

BASE = "./"

# Load all prediction files
data = {name: load_jsonl(BASE + path) for name, path in SYSTEMS.items()}

# All systems should have the same inputs/targets — use FT as reference
ref_data = data["FT"]

# Assign categories to each example
example_cats = [categorize(ex["input"], ex["target"]) for ex in ref_data]

# Compute per-category CER for each system
results = {}
counts = {}
for cat in CATEGORIES:
    indices = [i for i, cats in enumerate(example_cats) if cat in cats]
    counts[cat] = len(indices)
    results[cat] = {}
    for sname, preds in data.items():
        if not indices:
            results[cat][sname] = float("nan")
            continue
        total_cer = sum(cer(preds[i]["target"], preds[i]["pred"]) for i in indices)
        results[cat][sname] = total_cer / len(indices)

# ── print table ──────────────────────────────────────────────────────────────

systems = list(SYSTEMS.keys())
header = f"{'Category':<18} {'N':>5}  " + "  ".join(f"{s:>14}" for s in systems)
print(header)
print("-" * len(header))
for cat in CATEGORIES:
    row = f"{cat:<18} {counts[cat]:>5}  "
    row += "  ".join(f"{results[cat][s]:>14.4f}" for s in systems)
    print(row)

# ── also dump as JSON for LaTeX ───────────────────────────────────────────────
out = {"categories": CATEGORIES, "systems": systems, "counts": counts, "cer": results}
with open(BASE + "per_category_cer.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved to per_category_cer.json")
