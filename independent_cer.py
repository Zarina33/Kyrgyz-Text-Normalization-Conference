"""
Compute CER of each system's predictions against TWO references:
  (a) the original Gemini-generated target (in-domain reference)
  (b) my hand-written independent reference (50 examples, idx 100-149)

This quantifies how much of the reported CER is "fitting Gemini" vs.
"genuinely producing human-quality normalization".
"""

import json
import os
from independent_reference import independent_reference

BASE = os.path.dirname(os.path.abspath(__file__))

SYSTEMS = {
    "Rule-Based":   "eval_results/rulebased_formal.jsonl",
    "Gemma 4":      "eval_results/gemma4_formal.jsonl",
    "FT":           "eval_results/finetuned_formal.jsonl",
    "Pre-Train+FT": "eval_results/pretrained_ft_formal.jsonl",
    "Zero-Shot":    "eval_results/zeroshot_formal.jsonl",
}


def cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    n = len(r)
    if n == 0:
        return 0.0 if len(h) == 0 else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev = d[:]
        d[0] = i
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + cost)
    return d[len(h)] / n


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


indices = sorted(independent_reference.keys())

# Load predictions for each system at the selected indices
system_preds = {}
for sname, path in SYSTEMS.items():
    full = load_jsonl(os.path.join(BASE, path))
    system_preds[sname] = {i: full[i] for i in indices}

# Sanity-check: the input at a given index matches across files
ref_inputs = {i: system_preds["FT"][i]["input"] for i in indices}
for sname, preds in system_preds.items():
    for i in indices:
        assert preds[i]["input"] == ref_inputs[i], f"Input mismatch at {i} in {sname}"

# ── CER computations ─────────────────────────────────────────────────────────

results = {}
for sname, preds in system_preds.items():
    cer_vs_gemini = [cer(preds[i]["target"], preds[i]["pred"]) for i in indices]
    cer_vs_mine   = [cer(independent_reference[i], preds[i]["pred"]) for i in indices]
    results[sname] = {
        "cer_vs_gemini_target": sum(cer_vs_gemini) / len(cer_vs_gemini),
        "cer_vs_independent":   sum(cer_vs_mine) / len(cer_vs_mine),
    }

# Also: agreement between Gemini target and my independent reference
gemini_vs_mine = [
    cer(independent_reference[i], system_preds["FT"][i]["target"])
    for i in indices
]
reference_agreement_cer = sum(gemini_vs_mine) / len(gemini_vs_mine)

# ── report ───────────────────────────────────────────────────────────────────

print(f"N examples: {len(indices)}  (indices {indices[0]}-{indices[-1]})\n")
print(f"{'System':<14} {'CER vs Gemini':>16} {'CER vs Independent':>22} {'Δ':>10}")
print("-" * 66)
for sname, r in results.items():
    delta = r["cer_vs_independent"] - r["cer_vs_gemini_target"]
    print(f"{sname:<14} {r['cer_vs_gemini_target']:>16.4f} {r['cer_vs_independent']:>22.4f} {delta:>+10.4f}")

print()
print(f"Reference-agreement CER  (Gemini target vs independent ref): {reference_agreement_cer:.4f}")
print("  (lower = the two references agree more; this is the noise floor)")

# Save for the paper
out = {
    "n_examples": len(indices),
    "index_range": [indices[0], indices[-1]],
    "per_system": results,
    "reference_agreement_cer": reference_agreement_cer,
}
out_path = os.path.join(BASE, "independent_cer.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
