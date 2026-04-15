# Kyrgyz Text Normalization

Code and resources for training, evaluating, and analyzing five text normalization systems for the Kyrgyz language:

| System | Description |
|---|---|
| Rule-Based | Capitalization + punctuation heuristics |
| Zero-Shot mT5 | `google/mt5-small` without fine-tuning |
| mT5 Fine-Tuned | mT5-small fine-tuned on 1.67M pairs |
| mT5 Pre-Train+FT | Continual pre-training on Kyrgyz corpus, then fine-tuning |
| Gemma 4 Zero-Shot | Gemma 4 (9.6B, 4-bit) via Ollama |

**Key result:** Fine-tuned mT5-small (300M params) achieves CER 0.0796, outperforming Gemma 4 (CER 0.1620) — a model 30x larger.

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.0, CUDA 12.1.

For Gemma 4 evaluation, [Ollama](https://ollama.com) must be installed and running:
```bash
ollama pull gemma4:e4b
```

## Project Structure

```
├── train_mt5.py          # Fine-tune mT5-small on normalization dataset
├── pretrain_mt5.py       # Continual pre-training with span corruption
├── rule_based.py         # Rule-based baseline
├── inference.py          # Interactive inference with fine-tuned model
├── evaluate.py           # Automatic evaluation (CER / WER / Exact Match)
├── evaluate_llm.py       # Gemma 4 zero-shot evaluation via Ollama
├── significance_test.py  # Bootstrap significance tests (n=10,000)
├── error_analysis.py     # Error analysis: FT vs Pre-Train+FT
├── dataset_analysis.py   # Dataset statistics and normalization type analysis
├── plot_normalization.py # Figure: distribution of normalization types
├── prepare_datasets.py   # Prepare train/test splits from raw data
├── prepare_human_eval.py # Generate CSV for human evaluation
├── requirements.txt
└── human_eval/
    └── eval_200_annotated.csv
```

## Training

### Step 1 — Fine-tune mT5-small

```bash
python train_mt5.py
```

Key hyperparameters (edit in script):
- Effective batch size: 64 (physical 4 x gradient accumulation 16)
- Learning rate: 3e-4 with cosine schedule
- Warmup steps: 500
- Epochs: 5
- Max sequence length: 256

Hardware: NVIDIA RTX 5080 (16 GB VRAM), ~8h per epoch.

### Step 2 (optional) — Continual pre-training + fine-tune

```bash
# Pre-train on Kyrgyz news/books corpus (3 epochs, span corruption)
python pretrain_mt5.py

# Then fine-tune the pre-trained model
# Set MODEL_NAME in train_mt5.py to the pre-trained checkpoint path
python train_mt5.py
```

## Evaluation

```bash
# Automatic metrics (CER, WER, Exact Match) for all systems
python evaluate.py

# Gemma 4 zero-shot (requires Ollama running)
python evaluate_llm.py

# Bootstrap significance tests
python significance_test.py
```

## Inference

```bash
python inference.py --model_path path/to/finetuned/model
```

Example:
```
INPUT:  барды жакшы болсун коркунучту жерлерди тазалаш керек
OUTPUT: Барды жакшы болсун. Коркунучтуу жерлерди тазалаш керек.
```

## Analysis

```bash
# Error analysis: why does continual pre-training underperform?
python error_analysis.py

# Dataset statistics and normalization type breakdown
python dataset_analysis.py

# Reproduce the normalization type figure
python plot_normalization.py
```

## Results

### Automatic Evaluation (test set, 1,000 examples)

| System | CER | WER | EM |
|---|---|---|---|
| Rule-Based | 0.2029 | 0.5659 | 0.004 |
| Zero-Shot mT5-small | 0.9887 | 0.9981 | 0.000 |
| Gemma 4 Zero-Shot | 0.1620 | 0.4320 | 0.015 |
| **mT5-small Fine-Tuned** | **0.0796** | **0.1978** | **0.186** |
| mT5-small Pre-Train+FT | 0.0825 | 0.2017 | 0.184 |

### Human Evaluation (200 examples, 2 native annotators)

| System | Human Score |
|---|---|
| Rule-Based | 0.500 |
| Gemma 4 Zero-Shot | 0.793 |
| mT5-small Pre-Train+FT | **0.998** |
| mT5-small Fine-Tuned | **0.998** |

## License

Code: MIT License.
