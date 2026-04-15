"""
Evaluate Gemma 4 (via Ollama) on Kyrgyz text normalization task.
"""

import json
import requests
from pathlib import Path
from evaluate import load_formal_test, compute_metrics, save_predictions

TEST_FORMAL = "/home/zarina/Work/RESEARCH/dataset/test_set_formal.jsonl"
OUTPUT_DIR  = Path("/home/zarina/Work/RESEARCH/eval_results")
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "gemma4:e4b"

PROMPT_TEMPLATE = """You are a Kyrgyz language text normalization assistant.
Your task: normalize the given Kyrgyz text by fixing capitalization, punctuation, and spelling errors.
Return ONLY the normalized text, nothing else.

Input: {text}
Output:"""


def predict_ollama(text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(text=text)
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    })
    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    examples = load_formal_test(TEST_FORMAL)
    print(f"Test examples: {len(examples)}")
    print(f"Model: {MODEL_NAME}\n")

    preds = []
    for i, ex in enumerate(examples):
        pred = predict_ollama(ex["input"])
        preds.append(pred)
        print(f"  [{i+1}/{len(examples)}] {ex['input'][:60]!r} → {pred[:60]!r}")

    refs = [e["target"] for e in examples]
    metrics = compute_metrics(refs, preds)

    print(f"\nGemma4 zero-shot: CER={metrics['CER']:.4f} | WER={metrics['WER']:.4f} | EM={metrics['EM']:.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_predictions(preds, examples, OUTPUT_DIR / "gemma4_formal.jsonl")

    # Append to summary
    summary_path = OUTPUT_DIR / "results_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = []

    summary.append({"set": "formal", "system": "Gemma4 zero-shot", "metrics": metrics})
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary updated → {summary_path}")


if __name__ == "__main__":
    main()
