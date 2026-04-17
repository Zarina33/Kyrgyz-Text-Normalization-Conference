"""
Prepare CSV for human evaluation.
200 examples, outputs from all systems, 3 annotators.
"""

import json
import csv
import random

EVAL_DIR  = "./eval_results"
OUTPUT    = "./human_eval/human_eval_200.csv"
N         = 200
SEED      = 42

def load_jsonl(path):
    rows = {}
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            rows[i] = d
    return rows

def main():
    import os
    os.makedirs("./human_eval", exist_ok=True)

    rb  = load_jsonl(f"{EVAL_DIR}/rulebased_formal.jsonl")
    zs  = load_jsonl(f"{EVAL_DIR}/zeroshot_formal.jsonl")
    ft  = load_jsonl(f"{EVAL_DIR}/finetuned_formal.jsonl")
    pt  = load_jsonl(f"{EVAL_DIR}/pretrained_ft_formal.jsonl")
    g4  = load_jsonl(f"{EVAL_DIR}/gemma4_formal.jsonl")

    random.seed(SEED)
    indices = random.sample(range(len(ft)), N)

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "input", "reference",
            "rule_based", "mt5_finetuned", "mt5_pretrained_ft", "gemma4",
            "annotator1_rb", "annotator1_ft", "annotator1_pt", "annotator1_g4",
            "annotator2_rb", "annotator2_ft", "annotator2_pt", "annotator2_g4",
            "annotator3_rb", "annotator3_ft", "annotator3_pt", "annotator3_g4",
        ])
        for i, idx in enumerate(indices):
            writer.writerow([
                i + 1,
                ft[idx]["input"],
                ft[idx]["target"],
                rb[idx]["pred"],
                ft[idx]["pred"],
                pt[idx]["pred"],
                g4[idx]["pred"],
                "", "", "", "",
                "", "", "", "",
                "", "", "", "",
            ])

    print(f"Saved → {OUTPUT}")
    print(f"200 examples, 5 systems, 3 annotators")
    print(f"\nИнструкция:")
    print(f"  Оценить каждый выход системы: 1 = корректно, 0 = некорректно")
    print(f"  Заполнить колонки annotator1_*, annotator2_*, annotator3_*")

if __name__ == "__main__":
    main()
