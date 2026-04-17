"""
Bootstrap significance test for CER differences between systems.
"""

import json
import numpy as np
def compute_cer(ref, hyp):
    ref, hyp = ref.strip(), hyp.strip()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    r, h = list(ref), list(hyp)
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)] / len(r)

EVAL_DIR = "./eval_results"
N_BOOTSTRAP = 10000
SEED = 42

def load_preds(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def bootstrap_cer(refs, hyps, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(refs)
    cer_scores = np.array([compute_cer(r, h) for r, h in zip(refs, hyps)])
    means = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        means.append(cer_scores[idx].mean())
    return np.array(means)

def p_value(boot_a, boot_b):
    """p-value that system A is better than system B (A < B in CER)."""
    diff = boot_b - boot_a
    return (diff < 0).mean()

def main():
    ft  = load_preds(f"{EVAL_DIR}/finetuned_formal.jsonl")
    pt  = load_preds(f"{EVAL_DIR}/pretrained_ft_formal.jsonl")
    rb  = load_preds(f"{EVAL_DIR}/rulebased_formal.jsonl")
    g4  = load_preds(f"{EVAL_DIR}/gemma4_formal.jsonl")

    refs = [r["target"] for r in ft]

    print("Running bootstrap significance tests (10,000 iterations)...")
    boot_ft = bootstrap_cer(refs, [r["pred"] for r in ft])
    boot_pt = bootstrap_cer(refs, [r["pred"] for r in pt])
    boot_rb = bootstrap_cer(refs, [r["pred"] for r in rb])
    boot_g4 = bootstrap_cer(refs, [r["pred"] for r in g4])

    print(f"\nMean CER:")
    print(f"  mT5 Fine-Tuned:    {boot_ft.mean():.4f} ± {boot_ft.std():.4f}")
    print(f"  mT5 Pre-Train+FT:  {boot_pt.mean():.4f} ± {boot_pt.std():.4f}")
    print(f"  Gemma 4:           {boot_g4.mean():.4f} ± {boot_g4.std():.4f}")
    print(f"  Rule-Based:        {boot_rb.mean():.4f} ± {boot_rb.std():.4f}")

    print(f"\nSignificance tests (p-value, H0: no difference):")
    p1 = p_value(boot_ft, boot_rb)
    p2 = p_value(boot_ft, boot_g4)
    p3 = p_value(boot_ft, boot_pt)
    p4 = p_value(boot_g4, boot_rb)

    print(f"  FT < Rule-Based:   p = {p1:.4f} {'***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else 'n.s.'}")
    print(f"  FT < Gemma 4:      p = {p2:.4f} {'***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'n.s.'}")
    print(f"  FT < Pre-Train+FT: p = {p3:.4f} {'***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else 'n.s.'}")
    print(f"  Gemma 4 < Rule-Based: p = {p4:.4f} {'***' if p4 < 0.001 else '**' if p4 < 0.01 else '*' if p4 < 0.05 else 'n.s.'}")

if __name__ == "__main__":
    main()
