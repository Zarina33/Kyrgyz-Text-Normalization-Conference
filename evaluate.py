"""
Evaluation script for Kyrgyz text normalization.
Runs 4 systems on the formal test set and computes CER, WER, Exact Match.
"""

import json
import csv
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from jiwer import wer as compute_wer
from rule_based import correct as rule_based_correct

# ===================== CONFIG =====================
FINETUNED_MODEL    = "/home/zarina/Work/RESEARCH/checkpoints_new/mt5-kyrgyz-mixed/final"
PRETRAINED_FT_MODEL = "/home/zarina/Work/RESEARCH/checkpoints_new/mt5-kyrgyz-pretrained-finetuned/final"
ZEROSHOT_MODEL     = "google/mt5-small"

TEST_FORMAL   = "/home/zarina/Work/RESEARCH/dataset/test_set_formal.jsonl"
TEST_ASR      = "/home/zarina/Work/RESEARCH/annotation_todo.csv"

BATCH_SIZE = 16
MAX_LEN    = 256
# ==================================================


def load_formal_test(path):
    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            examples.append({"input": d["input"], "target": d["target"]})
    return examples


def load_asr_test(path):
    examples = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["is_valid"] == "1" and row["asr_output"].strip() and row["corrected"].strip():
                examples.append({"input": row["asr_output"], "target": row["corrected"]})
    return examples


def compute_cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip(), hyp.strip()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    # Levenshtein at character level
    r, h = list(ref), list(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(r)][len(h)] / len(r)


def compute_metrics(references, hypotheses):
    assert len(references) == len(hypotheses)
    cer_scores = [compute_cer(r, h) for r, h in zip(references, hypotheses)]
    wer_scores = [compute_wer(r, h) for r, h in zip(references, hypotheses)]
    em_scores  = [1.0 if r.strip() == h.strip() else 0.0 for r, h in zip(references, hypotheses)]
    return {
        "CER": round(sum(cer_scores) / len(cer_scores), 4),
        "WER": round(sum(wer_scores) / len(wer_scores), 4),
        "EM":  round(sum(em_scores)  / len(em_scores),  4),
    }


class MT5System:
    def __init__(self, model_name, label):
        self.label = label
        print(f"Loading {label} from {model_name}...")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict_batch(self, texts):
        inputs = ["correct: " + t for t in texts]
        enc = self.tokenizer(
            inputs, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LEN
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **enc, max_length=MAX_LEN, num_beams=4, early_stopping=True
            )
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    def predict_all(self, examples):
        preds = []
        for i in range(0, len(examples), BATCH_SIZE):
            batch = [e["input"] for e in examples[i:i+BATCH_SIZE]]
            preds.extend(self.predict_batch(batch))
            if (i // BATCH_SIZE) % 5 == 0:
                print(f"  {self.label}: {min(i+BATCH_SIZE, len(examples))}/{len(examples)}")
        return preds


class RuleBasedSystem:
    label = "Rule-based"

    def predict_all(self, examples):
        return [rule_based_correct(e["input"]) for e in examples]


def evaluate_system(system, examples, set_name):
    preds = system.predict_all(examples)
    refs  = [e["target"] for e in examples]
    metrics = compute_metrics(refs, preds)
    print(f"  [{set_name}] {system.label}: CER={metrics['CER']:.4f} | WER={metrics['WER']:.4f} | EM={metrics['EM']:.4f}")
    return metrics, preds


def print_table(results):
    print("\n" + "=" * 70)
    print(f"{'System':<35} {'CER':>6} {'WER':>6} {'EM':>6}")
    print("-" * 70)
    for set_name, system_name, metrics in results:
        print(f"{system_name + ' [' + set_name + ']':<35} {metrics['CER']:>6.4f} {metrics['WER']:>6.4f} {metrics['EM']:>6.4f}")
    print("=" * 70)


def save_predictions(preds, examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for e, p in zip(examples, preds):
            f.write(json.dumps({"input": e["input"], "target": e["target"], "pred": p}, ensure_ascii=False) + "\n")
    print(f"  Saved predictions → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["formal", "asr", "both"], default="both")
    parser.add_argument("--systems", nargs="+",
                        choices=["zeroshot", "rulebased", "finetuned", "pretrained_ft", "all"],
                        default=["all"])
    args = parser.parse_args()

    run_all = "all" in args.systems
    run = lambda name: run_all or name in args.systems

    # Load test sets
    formal_examples = load_formal_test(TEST_FORMAL) if args.test in ("formal", "both") else []
    asr_examples    = load_asr_test(TEST_ASR)       if args.test in ("asr", "both")    else []

    print(f"Formal test: {len(formal_examples)} examples")
    print(f"ASR test:    {len(asr_examples)} examples\n")

    results = []
    output_dir = Path("/home/zarina/Work/RESEARCH/eval_results")
    output_dir.mkdir(exist_ok=True)

    # Rule-based
    if run("rulebased"):
        sys = RuleBasedSystem()
        for examples, set_name in [(formal_examples, "formal"), (asr_examples, "asr")]:
            if examples:
                m, preds = evaluate_system(sys, examples, set_name)
                results.append((set_name, sys.label, m))
                save_predictions(preds, examples, output_dir / f"rulebased_{set_name}.jsonl")

    # Zero-shot
    if run("zeroshot"):
        sys = MT5System(ZEROSHOT_MODEL, "Zero-shot mT5")
        for examples, set_name in [(formal_examples, "formal"), (asr_examples, "asr")]:
            if examples:
                m, preds = evaluate_system(sys, examples, set_name)
                results.append((set_name, sys.label, m))
                save_predictions(preds, examples, output_dir / f"zeroshot_{set_name}.jsonl")
        del sys
        torch.cuda.empty_cache()

    # Fine-tuned
    if run("finetuned"):
        sys = MT5System(FINETUNED_MODEL, "mT5 Fine-tuned")
        for examples, set_name in [(formal_examples, "formal"), (asr_examples, "asr")]:
            if examples:
                m, preds = evaluate_system(sys, examples, set_name)
                results.append((set_name, sys.label, m))
                save_predictions(preds, examples, output_dir / f"finetuned_{set_name}.jsonl")
        del sys
        torch.cuda.empty_cache()

    # Pre-trained + Fine-tuned
    if run("pretrained_ft"):
        sys = MT5System(PRETRAINED_FT_MODEL, "mT5 Pre-trained+FT")
        for examples, set_name in [(formal_examples, "formal"), (asr_examples, "asr")]:
            if examples:
                m, preds = evaluate_system(sys, examples, set_name)
                results.append((set_name, sys.label, m))
                save_predictions(preds, examples, output_dir / f"pretrained_ft_{set_name}.jsonl")
        del sys
        torch.cuda.empty_cache()

    print_table(results)

    # Save summary
    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump([{"set": s, "system": n, "metrics": m} for s, n, m in results], f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
