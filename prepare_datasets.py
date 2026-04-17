"""
Подготовка датасетов для обучения mT5.

Из mt5_datasetLAST.jsonl:
  - 1000 строк → test_set_formal.jsonl (тест, не трогаем при обучении)
  - остаток → смешиваем с asr_conversational_dataset.jsonl → mixed_train.jsonl

Итоговый формат всех файлов: {"input": ..., "target": ...}
"""

import json
import random
from pathlib import Path

random.seed(42)

LAST_JSONL     = Path("./data/raw/mt5_datasetLAST.jsonl")
ASR_JSONL      = Path("./data/raw/asr_conversational_dataset.jsonl")
OUT_DIR        = Path("./dataset")
TEST_SIZE      = 1000


def load_normalize(path: Path) -> list[dict]:
    """Читает jsonl и нормализует в {"input": ..., "target": ...}"""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # mt5_datasetLAST формат
            if "wrong_text" in obj:
                records.append({"input": obj["wrong_text"], "target": obj["corrected_text"]})
            # asr_conversational формат
            elif "noncorrected" in obj:
                records.append({"input": obj["noncorrected"], "target": obj["corrected"]})
            else:
                print(f"  Неизвестный формат: {list(obj.keys())} — пропускаем")
    return records


def save_jsonl(records: list[dict], path: Path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Сохранено {len(records):,} строк → {path.name}")


def main():
    print("Загружаем mt5_datasetLAST.jsonl...")
    last = load_normalize(LAST_JSONL)
    print(f"  {len(last):,} строк")

    print("Загружаем asr_conversational_dataset.jsonl...")
    asr = load_normalize(ASR_JSONL)
    print(f"  {len(asr):,} строк")

    # Перемешиваем last и отделяем тест
    random.shuffle(last)
    test_set  = last[:TEST_SIZE]
    train_last = last[TEST_SIZE:]

    print(f"\nРазбивка mt5_datasetLAST:")
    print(f"  train: {len(train_last):,}")
    print(f"  test:  {len(test_set):,}")

    # Смешанный трейн
    mixed_train = train_last + asr
    random.shuffle(mixed_train)

    print(f"\nСмешанный train: {len(mixed_train):,} строк")
    print(f"  из mt5_datasetLAST: {len(train_last):,}")
    print(f"  из asr_conv:        {len(asr):,}")

    print("\nСохраняем...")
    save_jsonl(test_set,    OUT_DIR / "test_set_formal.jsonl")
    save_jsonl(mixed_train, OUT_DIR / "mixed_train.jsonl")

    print("\nГотово!")
    print(f"  test_set_formal.jsonl  → {TEST_SIZE} строк (не трогать при обучении)")
    print(f"  mixed_train.jsonl      → {len(mixed_train):,} строк")


if __name__ == "__main__":
    main()
