"""
Dataset analysis for the paper's Dataset section.
"""

import json
import re
from collections import Counter

TRAIN_PATH = "/home/zarina/Work/RESEARCH/dataset/mixed_train.jsonl"
TEST_PATH  = "/home/zarina/Work/RESEARCH/dataset/test_set_formal.jsonl"
SAMPLE     = 50000  # analyze first 50k for speed


def analyze(path, label, max_rows=None):
    inputs, targets = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            inputs.append(d["input"])
            targets.append(d["target"])

    total = len(inputs)

    # Lengths
    inp_lens  = [len(t) for t in inputs]
    tgt_lens  = [len(t) for t in targets]

    # Types of normalization needed
    needs_capitalization = sum(1 for i, t in zip(inputs, targets)
                               if t and i and t[0].isupper() and i[0].islower())

    needs_punctuation = sum(1 for i, t in zip(inputs, targets)
                            if sum(1 for c in t if c in '.,!?;:') >
                               sum(1 for c in i if c in '.,!?;:'))

    has_uppercase_input = sum(1 for i in inputs
                              if sum(1 for c in i if c.isupper()) > len(i) * 0.3)

    has_digit_words = sum(1 for i in inputs if re.search(r'\d+[а-яёөүңкгжшщ]', i, re.IGNORECASE))

    different = sum(1 for i, t in zip(inputs, targets) if i.strip() != t.strip())

    print(f"\n{'='*60}")
    print(f"Dataset: {label} ({total:,} examples)")
    print(f"{'='*60}")
    print(f"Avg input length (chars):   {sum(inp_lens)/total:.1f}")
    print(f"Avg target length (chars):  {sum(tgt_lens)/total:.1f}")
    print(f"Min/Max input length:       {min(inp_lens)} / {max(inp_lens)}")
    print(f"\nNormalization types (% of sample):")
    print(f"  Needs capitalization:     {100*needs_capitalization/total:.1f}%")
    print(f"  Needs punctuation:        {100*needs_punctuation/total:.1f}%")
    print(f"  Has ALLCAPS segments:     {100*has_uppercase_input/total:.1f}%")
    print(f"  Has digit+word (e.g. 8жыл): {100*has_digit_words/total:.1f}%")
    print(f"  Input != Target (changed):  {100*different/total:.1f}%")


def main():
    print("Analyzing dataset...")
    analyze(TEST_PATH, "Test set (formal)", max_rows=None)
    analyze(TRAIN_PATH, "Train set (sample)", max_rows=SAMPLE)


if __name__ == "__main__":
    main()
