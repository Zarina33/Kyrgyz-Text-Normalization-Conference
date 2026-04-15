"""
Error analysis: why mT5 Pre-trained+FT underperforms mT5 Fine-tuned.
Compares predictions from both models and categorizes differences.
"""

import json
from pathlib import Path
from evaluate import compute_cer
from jiwer import wer as compute_wer

FINETUNED_PREDS    = "/home/zarina/Work/RESEARCH/eval_results/finetuned_formal.jsonl"
PRETRAINED_PREDS   = "/home/zarina/Work/RESEARCH/eval_results/pretrained_ft_formal.jsonl"


def load_preds(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def categorize_error(inp, pred, ref):
    """Categorize what type of error the prediction has vs reference."""
    errors = []

    # Capitalization
    if pred and ref and pred[0].lower() == ref[0].lower() and pred[0] != ref[0]:
        errors.append("capitalization")
    elif pred and ref and pred[0] != ref[0]:
        if ref[0].isupper() and pred[0].islower():
            errors.append("capitalization")

    # Punctuation (check end of sentences)
    ref_punct  = sum(1 for c in ref  if c in '.,!?;:')
    pred_punct = sum(1 for c in pred if c in '.,!?;:')
    if abs(ref_punct - pred_punct) > 1:
        errors.append("punctuation")

    # Word-level changes (model changed words it shouldn't have)
    ref_words  = ref.lower().split()
    pred_words = pred.lower().split()
    inp_words  = inp.lower().split()

    # Words in pred that are NOT in input and NOT in ref = hallucination
    inp_set = set(inp_words)
    ref_set = set(ref_words)
    extra_words = [w for w in pred_words if w not in inp_set and w not in ref_set]
    if extra_words:
        errors.append("hallucination")

    # Words from input that were incorrectly changed
    wrong_changes = 0
    for w in inp_words:
        if w in ref_set and w not in set(pred_words):
            wrong_changes += 1
    if wrong_changes > 1:
        errors.append("over-correction")

    if not errors:
        errors.append("other")

    return errors


def main():
    ft_rows = load_preds(FINETUNED_PREDS)
    pt_rows = load_preds(PRETRAINED_PREDS)

    assert len(ft_rows) == len(pt_rows)

    # Find examples where pre-trained+FT is worse than fine-tuned
    ft_wins = []   # fine-tuned is better
    pt_wins = []   # pre-trained+FT is better
    ties    = []   # similar

    for ft, pt in zip(ft_rows, pt_rows):
        ref = ft["target"]
        inp = ft["input"]

        ft_cer = compute_cer(ref, ft["pred"])
        pt_cer = compute_cer(ref, pt["pred"])

        diff = pt_cer - ft_cer
        if diff > 0.05:
            ft_wins.append({"input": inp, "ref": ref, "ft_pred": ft["pred"],
                            "pt_pred": pt["pred"], "ft_cer": ft_cer, "pt_cer": pt_cer,
                            "diff": diff})
        elif diff < -0.05:
            pt_wins.append({"input": inp, "ref": ref, "ft_pred": ft["pred"],
                            "pt_pred": pt["pred"], "ft_cer": ft_cer, "pt_cer": pt_cer,
                            "diff": diff})
        else:
            ties.append(diff)

    print(f"Total examples: {len(ft_rows)}")
    print(f"Fine-tuned better:      {len(ft_wins)} ({100*len(ft_wins)/len(ft_rows):.1f}%)")
    print(f"Pre-trained+FT better:  {len(pt_wins)} ({100*len(pt_wins)/len(ft_rows):.1f}%)")
    print(f"Similar (diff < 0.05):  {len(ties)}   ({100*len(ties)/len(ft_rows):.1f}%)")

    # Error categorization for cases where pre-trained+FT is worse
    print("\n" + "=" * 70)
    print("ERROR CATEGORIES (where Pre-trained+FT is worse):")
    print("=" * 70)

    category_counts = {}
    for ex in ft_wins:
        cats = categorize_error(ex["input"], ex["pt_pred"], ex["ref"])
        for c in cats:
            category_counts[c] = category_counts.get(c, 0) + 1

    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<20} {count:>4} ({100*count/len(ft_wins):.1f}%)")

    # Show worst 10 examples where pre-trained+FT fails
    ft_wins_sorted = sorted(ft_wins, key=lambda x: -x["diff"])
    print("\n" + "=" * 70)
    print("TOP 10 EXAMPLES WHERE PRE-TRAINED+FT IS WORSE:")
    print("=" * 70)
    for ex in ft_wins_sorted[:10]:
        print(f"\nINPUT:    {ex['input'][:100]}")
        print(f"REF:      {ex['ref'][:100]}")
        print(f"FT pred:  {ex['ft_pred'][:100]}  (CER={ex['ft_cer']:.3f})")
        print(f"PT pred:  {ex['pt_pred'][:100]}  (CER={ex['pt_cer']:.3f})")
        print(f"Diff:     +{ex['diff']:.3f} (PT worse)")

    # Show examples where pre-trained+FT actually wins
    pt_wins_sorted = sorted(pt_wins, key=lambda x: x["diff"])
    print("\n" + "=" * 70)
    print("TOP 5 EXAMPLES WHERE PRE-TRAINED+FT IS BETTER:")
    print("=" * 70)
    for ex in pt_wins_sorted[:5]:
        print(f"\nINPUT:    {ex['input'][:100]}")
        print(f"REF:      {ex['ref'][:100]}")
        print(f"FT pred:  {ex['ft_pred'][:100]}  (CER={ex['ft_cer']:.3f})")
        print(f"PT pred:  {ex['pt_pred'][:100]}  (CER={ex['pt_cer']:.3f})")
        print(f"Diff:     {ex['diff']:.3f} (PT better)")


if __name__ == "__main__":
    main()
