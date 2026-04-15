"""
Interactive inference for the fine-tuned Kyrgyz text normalization model.

Usage:
    python inference.py --model_path path/to/model
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_path: str, device: torch.device):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


def correct(text: str, tokenizer, model, device: torch.device) -> str:
    prefix = "correct: "
    inputs = tokenizer(
        prefix + text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Kyrgyz text normalization inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory (local) or HuggingFace model ID",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (default: use CUDA if available)",
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    tokenizer, model = load_model(args.model_path, device)

    print("=" * 60)
    print("Kyrgyz Text Normalization — Interactive Mode")
    print("Type 'exit' or press Ctrl+C to quit.")
    print("=" * 60)

    while True:
        try:
            text = input("\nINPUT:  ").strip()
            if text.lower() == "exit":
                break
            if not text:
                continue
            result = correct(text, tokenizer, model, device)
            print(f"OUTPUT: {result}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
