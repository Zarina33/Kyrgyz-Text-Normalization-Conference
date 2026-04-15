import sys
import random
import torch
import mlflow
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# ===================== CONFIG =====================
MODEL_NAME      = "google/mt5-small"
CORPUS_PATH     = "/home/zarina/Work/RESEARCH/kyrgyz_corpus.txt"
OUTPUT_DIR      = "/home/zarina/Work/RESEARCH/checkpoints_new/mt5-kyrgyz-pretrained"
MAX_LEN         = 256
BATCH_SIZE      = 4
GRAD_ACCUM      = 16         # effective batch = 64
EPOCHS          = 3
LR              = 1e-4
WARMUP_STEPS    = 1000
SAVE_STEPS      = 5000
LOG_STEPS       = 500
MASK_RATE       = 0.15       # доля токенов для маскировки
MEAN_SPAN_LEN   = 3          # средняя длина маскируемого спана
EXPERIMENT_NAME = "kyrgyz-continual-pretrain"
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def span_corruption(token_ids: list[int], tokenizer, mask_rate=0.15, mean_span=3) -> tuple[list[int], list[int]]:
    """
    T5-style span corruption.
    Возвращает (input_ids, target_ids) с сентинел-токенами.
    """
    n = len(token_ids)
    num_to_mask = max(1, int(n * mask_rate))

    # Генерируем спаны для маскировки
    masked = [False] * n
    masked_count = 0
    attempts = 0
    while masked_count < num_to_mask and attempts < n * 2:
        attempts += 1
        span_len = max(1, int(random.gauss(mean_span, 1)))
        start = random.randint(0, n - 1)
        end = min(start + span_len, n)
        for i in range(start, end):
            if not masked[i]:
                masked[i] = True
                masked_count += 1

    # Сентинел-токены: <extra_id_0>, <extra_id_1>, ...
    sentinel_ids = [tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)]

    input_ids = []
    target_ids = []
    sentinel_idx = 0
    in_span = False

    for i, tok in enumerate(token_ids):
        if masked[i]:
            if not in_span:
                if sentinel_idx < len(sentinel_ids):
                    input_ids.append(sentinel_ids[sentinel_idx])
                    target_ids.append(sentinel_ids[sentinel_idx])
                    sentinel_idx += 1
                in_span = True
            target_ids.append(tok)
        else:
            in_span = False
            input_ids.append(tok)

    target_ids.append(tokenizer.eos_token_id)
    return input_ids, target_ids


class KyrgyzCorpusDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, mask_rate, mean_span):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_rate = mask_rate
        self.mean_span = mean_span
        self.chunks = []

        print("Читаем корпус и нарезаем на чанки...")
        buffer = []
        with open(path, encoding="utf-8") as f:
            for line in tqdm(f, desc="Загрузка"):
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line, add_special_tokens=False)
                buffer.extend(ids)
                while len(buffer) >= max_len:
                    self.chunks.append(buffer[:max_len])
                    buffer = buffer[max_len:]

        if buffer:
            self.chunks.append(buffer)

        print(f"Чанков: {len(self.chunks):,}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        token_ids = self.chunks[idx]
        input_ids, target_ids = span_corruption(
            token_ids, self.tokenizer, self.mask_rate, self.mean_span
        )

        # Паддинг / обрезка
        input_ids  = input_ids[:self.max_len]
        target_ids = target_ids[:self.max_len]

        pad = self.tokenizer.pad_token_id

        def pad_seq(seq, length):
            return seq + [pad] * (length - len(seq))

        input_ids  = pad_seq(input_ids,  self.max_len)
        target_ids = pad_seq(target_ids, self.max_len)

        labels = [t if t != pad else -100 for t in target_ids]

        return {
            "input_ids":      torch.tensor(input_ids,  dtype=torch.long),
            "attention_mask": torch.tensor([1 if t != pad else 0 for t in input_ids], dtype=torch.long),
            "labels":         torch.tensor(labels,     dtype=torch.long),
        }


def find_latest_checkpoint(output_dir):
    output_path = Path(output_dir)
    checkpoints = sorted(
        [d for d in output_path.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[1])
    )
    return checkpoints[-1] if checkpoints else None


def main():
    mlflow.set_tracking_uri("file:///home/zarina/Work/RESEARCH/mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    latest_ckpt = find_latest_checkpoint(OUTPUT_DIR)
    resume_step = 0
    resume_epoch = 0

    if latest_ckpt:
        resume_step = int(latest_ckpt.name.split("-")[1])
        print(f"Resuming from checkpoint: {latest_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(str(latest_ckpt))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(latest_ckpt)).to(device)
    else:
        print("Загружаем базовый mt5-small...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    dataset = KyrgyzCorpusDataset(CORPUS_PATH, tokenizer, MAX_LEN, MASK_RATE, MEAN_SPAN_LEN)

    val_size   = int(len(dataset) * 0.02)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    if resume_step > 0:
        resume_epoch = resume_step // steps_per_epoch
        for _ in range(resume_step):
            scheduler.step()

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_path / "logs"))

    print(f"Train: {train_size:,} | Val: {val_size:,} | Total steps: {total_steps:,}")

    with mlflow.start_run(run_name="mt5-kyrgyz-pretrained"):
        mlflow.log_params({
            "model": MODEL_NAME,
            "corpus": Path(CORPUS_PATH).name,
            "chunks": len(dataset),
            "max_len": MAX_LEN,
            "mask_rate": MASK_RATE,
            "mean_span": MEAN_SPAN_LEN,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM,
            "epochs": EPOCHS,
            "lr": LR,
            "total_steps": total_steps,
        })

        global_step = resume_step
        best_val_loss = float("inf")

        for epoch in range(resume_epoch, EPOCHS):
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout)

            for step, batch in enumerate(pbar):
                steps_done = (resume_step - resume_epoch * steps_per_epoch) if epoch == resume_epoch else 0
                if epoch == resume_epoch and step < steps_done * GRAD_ACCUM:
                    continue

                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRAD_ACCUM
                loss.backward()
                running_loss += loss.item()

                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    current_loss = loss.item() * GRAD_ACCUM
                    current_lr   = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}", step=f"{global_step}/{total_steps}")
                    writer.add_scalar("train/loss", current_loss, global_step)
                    writer.add_scalar("train/lr",   current_lr,   global_step)
                    mlflow.log_metrics({"train_loss": current_loss, "lr": current_lr}, step=global_step)

                    if global_step % LOG_STEPS == 0:
                        avg = running_loss / LOG_STEPS
                        print(f"\nEpoch {epoch+1} | Step {global_step}/{total_steps} | Loss: {avg:.4f} | LR: {current_lr:.2e}")
                        sys.stdout.flush()
                        running_loss = 0.0

                    if global_step % SAVE_STEPS == 0:
                        ckpt = output_path / f"checkpoint-{global_step}"
                        model.save_pretrained(ckpt)
                        tokenizer.save_pretrained(ckpt)
                        torch.save({
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "global_step": global_step,
                            "epoch": epoch,
                        }, ckpt / "training_state.pt")
                        print(f"Checkpoint saved: {ckpt}")
                        sys.stdout.flush()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    val_loss += outputs.loss.item()
            val_loss /= len(val_loader)

            writer.add_scalar("val/loss", val_loss, epoch)
            mlflow.log_metric("val_loss", val_loss, step=global_step)
            print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_path / "best"
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                print(f"Best model saved! Val Loss: {val_loss:.4f}")

        final_path = output_path / "final"
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        writer.close()
        print(f"\nPre-training done. Модель → {final_path}")
        print(f"Следующий шаг: запусти train_mt5.py с MODEL_NAME = '{final_path}'")


if __name__ == "__main__":
    main()
