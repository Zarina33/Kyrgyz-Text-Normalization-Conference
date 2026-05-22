"""Compute inter-annotator reliability and per-system confidence intervals
for the human evaluation in `human_eval/eval_200_annotated.csv`.

Reproduces Tables 4 and 6 of the camera-ready paper:
  - Wilson 95% CI on per-system mean human score (pooled across both annotators)
  - Cohen's kappa, PABAK, Gwet's AC1 per system
  - Pooled IAA across all four systems
  - Count of fine-tuned outputs that BOTH annotators rated correct but that
    differ character-by-character from the Gemini reference (the 162/199
    statistic cited in section 5.3)

Run:
    python reliability_metrics.py
"""
import csv
from math import sqrt
from pathlib import Path

CSV_PATH = Path(__file__).parent / "human_eval" / "eval_200_annotated.csv"

SYSTEMS = [
    ("rb", "Rule-Based",  "rule_based"),
    ("g4", "Gemma 4",     "gemma4"),
    ("ft", "mT5 FT",      "mt5_finetuned"),
    ("pt", "mT5 PT+FT",   "mt5_pretrained_ft"),
]


def cohen_kappa(a, b):
    n = len(a)
    p_obs = sum(1 for x, y in zip(a, b) if x == y) / n
    p1 = sum(a) / n
    p2 = sum(b) / n
    p_exp = p1 * p2 + (1 - p1) * (1 - p2)
    if p_exp == 1.0:
        return 1.0 if p_obs == 1.0 else float("nan")
    return (p_obs - p_exp) / (1 - p_exp)


def pabak(a, b):
    """Prevalence-and-bias-adjusted kappa = 2 * p_obs - 1."""
    n = len(a)
    p_obs = sum(1 for x, y in zip(a, b) if x == y) / n
    return 2 * p_obs - 1


def gwet_ac1(a, b):
    """Gwet's AC1 for binary ratings, more robust than kappa under prevalence skew."""
    n = len(a)
    p_obs = sum(1 for x, y in zip(a, b) if x == y) / n
    p1 = sum(a) / n
    p2 = sum(b) / n
    pi = (p1 + p2) / 2
    p_e = 2 * pi * (1 - pi)
    if p_e == 1.0:
        return float("nan")
    return (p_obs - p_e) / (1 - p_e)


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    print(f"Loaded {len(rows)} examples from {CSV_PATH.name}\n")

    header = f"{'System':<14}{'Score':>7}{'95% Wilson CI':>22}{'kappa':>9}{'PABAK':>9}{'AC1':>9}{'%agree':>10}"
    print(header)
    print("-" * len(header))

    overall_a1, overall_a2 = [], []
    for key, name, _col in SYSTEMS:
        a1 = [int(r[f"annotator1_{key}"]) for r in rows]
        a2 = [int(r[f"annotator2_{key}"]) for r in rows]
        n = len(a1)

        correct = sum(a1) + sum(a2)
        total = 2 * n
        score = correct / total
        lo, hi = wilson_ci(correct, total)

        agree = sum(1 for x, y in zip(a1, a2) if x == y) / n

        print(f"{name:<14}{score:>7.3f}   [{lo:.3f}, {hi:.4f}]   "
              f"{cohen_kappa(a1, a2):>5.3f}   {pabak(a1, a2):>5.3f}   "
              f"{gwet_ac1(a1, a2):>5.3f}   {agree:>6.1%}")

        overall_a1.extend(a1)
        overall_a2.extend(a2)

    print("-" * len(header))
    n_all = len(overall_a1)
    correct_all = sum(overall_a1) + sum(overall_a2)
    score_all = correct_all / (2 * n_all)
    lo, hi = wilson_ci(correct_all, 2 * n_all)
    print(f"{'Pooled':<14}{score_all:>7.3f}   [{lo:.3f}, {hi:.4f}]   "
          f"{cohen_kappa(overall_a1, overall_a2):>5.3f}   "
          f"{pabak(overall_a1, overall_a2):>5.3f}   "
          f"{gwet_ac1(overall_a1, overall_a2):>5.3f}   "
          f"{sum(1 for x, y in zip(overall_a1, overall_a2) if x == y) / n_all:>6.1%}")

    # Correct-but-differs-from-reference (paper section 5.3, Table 5).
    both_correct = 0
    differ = 0
    for r in rows:
        if int(r["annotator1_ft"]) == 1 and int(r["annotator2_ft"]) == 1:
            both_correct += 1
            if r["mt5_finetuned"].strip() != r["reference"].strip():
                differ += 1
    print(f"\nFine-tuned mT5 outputs rated correct by BOTH annotators: {both_correct}")
    print(f"  ...of which differ from reference at the character level: "
          f"{differ} ({differ / both_correct:.1%})")


if __name__ == "__main__":
    main()
