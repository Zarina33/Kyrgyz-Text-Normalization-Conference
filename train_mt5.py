import json
import sys
import torch
import mlflow
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# ===================== CONFIG =====================
MODEL_NAME = "./checkpoints_new/mt5-kyrgyz-pretrained/final"
DATA_PATH = "./dataset/mixed_train.jsonl"
OUTPUT_DIR = "./checkpoints_new/mt5-kyrgyz-pretrained-finetuned"
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8  # effective batch = 64
EPOCHS = 5
LR = 3e-4
WARMUP_STEPS = 500
SAVE_STEPS = 5000
LOG_STEPS = 500
PREFIX = "normalize: "
EXPERIMENT_NAME = "kyrgyz-text-normalization"
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class NormalizationDataset(Dataset):
    def __init__(self, path, tokenizer, max_input_len, max_target_len, prefix):
        self.samples = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                src = r.get("input", r.get("wrong_text", r.get("noncorrected")))
                tgt = r.get("target", r.get("corrected_text", r.get("corrected")))
                self.samples.append((src, tgt))
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.prefix = prefix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        input_enc = self.tokenizer(
            self.prefix + src,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            tgt,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in output_dir for resume."""
    output_path = Path(output_dir)
    checkpoints = sorted(
        [d for d in output_path.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[1])
    )
    return checkpoints[-1] if checkpoints else None


def main():
    # MLflow setup
    mlflow.set_tracking_uri(f"file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Check for resume
    latest_ckpt = find_latest_checkpoint(OUTPUT_DIR)
    resume_step = 0
    resume_epoch = 0

    if latest_ckpt:
        resume_step = int(latest_ckpt.name.split("-")[1])
        print(f"Found checkpoint: {latest_ckpt} (step {resume_step})")
        print("Resuming training...")
        tokenizer = AutoTokenizer.from_pretrained(str(latest_ckpt))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(latest_ckpt)).to(device)
    else:
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    print("Loading dataset...")
    dataset = NormalizationDataset(DATA_PATH, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN, PREFIX)
    print(f"Total samples: {len(dataset)}")

    # 95/5 split
    val_size = int(len(dataset) * 0.05)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    # Fast-forward scheduler if resuming
    if resume_step > 0:
        resume_epoch = resume_step // steps_per_epoch
        for _ in range(resume_step):
            scheduler.step()
        print(f"Resuming from epoch {resume_epoch + 1}, step {resume_step}/{total_steps}")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_path / "logs"))

    # Determine run name from OUTPUT_DIR
    run_name = Path(OUTPUT_DIR).name

    with mlflow.start_run(run_name=run_name):
        # Log params
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "dataset": Path(DATA_PATH).name,
            "dataset_size": len(dataset),
            "train_size": train_size,
            "val_size": val_size,
            "max_input_len": MAX_INPUT_LEN,
            "max_target_len": MAX_TARGET_LEN,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "epochs": EPOCHS,
            "lr": LR,
            "warmup_steps": WARMUP_STEPS,
            "total_steps": total_steps,
            "resumed_from_step": resume_step,
        })

        global_step = resume_step
        best_val_loss = float("inf")

        for epoch in range(resume_epoch, EPOCHS):
            model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout)
            for step, batch in enumerate(pbar):
                # Skip steps already done in this epoch when resuming
                steps_done_in_epoch = resume_step - (resume_epoch * steps_per_epoch) if epoch == resume_epoch else 0
                if epoch == resume_epoch and step < steps_done_in_epoch * GRAD_ACCUM_STEPS:
                    continue

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()

                running_loss += loss.item()

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    current_loss = loss.item() * GRAD_ACCUM_STEPS
                    current_lr = scheduler.get_last_lr()[0]

                    pbar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}", step=f"{global_step}/{total_steps}")

                    writer.add_scalar("train/loss", current_loss, global_step)
                    writer.add_scalar("train/lr", current_lr, global_step)
                    mlflow.log_metrics({"train_loss": current_loss, "lr": current_lr}, step=global_step)

                    if global_step % LOG_STEPS == 0:
                        avg_loss = running_loss / LOG_STEPS
                        print(f"\nEpoch {epoch+1}/{EPOCHS} | Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
                        sys.stdout.flush()
                        running_loss = 0.0

                    if global_step % SAVE_STEPS == 0:
                        ckpt_path = output_path / f"checkpoint-{global_step}"
                        model.save_pretrained(ckpt_path)
                        tokenizer.save_pretrained(ckpt_path)
                        # Save optimizer & scheduler for resume
                        torch.save({
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "global_step": global_step,
                            "epoch": epoch,
                        }, ckpt_path / "training_state.pt")
                        print(f"Saved checkpoint: {ckpt_path}")
                        sys.stdout.flush()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
                mlflow.log_metric("best_val_loss", best_val_loss, step=global_step)
                print(f"New best model saved! Val Loss: {val_loss:.4f}")

        # Save final
        final_path = output_path / "final"
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        writer.close()

        mlflow.log_metric("final_val_loss", val_loss, step=global_step)
        mlflow.log_artifact(str(output_path / "best"), "best_model")

        print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
