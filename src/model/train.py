"""
train.py
========
Fine-tunes MiniLM on the custom patient-aware
hallucination detection dataset.

Model architecture:
  MiniLM encoder
    → [CLS] embedding
    → Dropout (0.3)
    → Classification head (hidden→2)  is_hallucinated
    → Regression head    (hidden→1)   trust_score

Input format:
  [CLS] patient: {context} [SEP] question: {q} [SEP]
  answer: {a} [SEP]

Loss:
  total = 0.7 × CrossEntropy + 0.3 × MSE

Saves best checkpoint to outputs/best_model/
Logs epoch metrics to logs/training_log.json
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error
)

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SPLITS_DIR  = os.path.join(PROJECT_ROOT, "data", "splits")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR    = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

# ── Config ────────────────────────────────────────────────────
MODEL_NAME    = "microsoft/MiniLM-L12-H384-uncased"
MODEL_LABEL   = "MiniLM-L12-H384"
MAX_LEN       = 384
BATCH_SIZE    = 2
EPOCHS        = 2
LR            = 2e-5
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
DROPOUT       = 0.3
ALPHA         = 0.7    # weight for classification loss
RANDOM_SEED   = 42
PATIENT_CHARS = 500    # chars of patient context to include
HEAD_HIDDEN   = 128

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def log(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")


def format_duration(seconds):
    """Format seconds as h:mm:ss for progress logs."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def to_python_types(obj):
    """Recursively convert NumPy scalars to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ─────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────
class HallucinationDataset(Dataset):
    """
    Tokenises each record into:
      [CLS] patient: {context} [SEP]
            question: {question} [SEP]
            answer: {answer} [SEP]

    Labels:
      is_hallucinated : int   (0 or 1)
      trust_score     : float (0.3 / 0.6 / 0.9 / 1.0)
    """

    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Truncate patient context to keep input manageable
        patient = str(
            row.get("patient_context", "")
        )[:PATIENT_CHARS]
        question = str(row.get("question", ""))
        answer   = str(row.get("answer", ""))

        # Build combined input text
        text = (
            f"patient: {patient} "
            f"question: {question} "
            f"answer: {answer}"
        )

        # Tokenise
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids"      : encoding["input_ids"].squeeze(),
            "attention_mask" : encoding["attention_mask"].squeeze(),
            "label"          : torch.tensor(
                int(row["is_hallucinated"]),
                dtype=torch.long
            ),
            "trust_score"    : torch.tensor(
                float(row.get("trust_score", 0.5)),
                dtype=torch.float
            ),
        }


# ─────────────────────────────────────────────────────────────
# DUAL-HEAD MODEL
# ─────────────────────────────────────────────────────────────
class PatientAwareHallucinationDetector(nn.Module):
    """
    MiniLM encoder with two output heads:
      1. Classification head → is_hallucinated (binary)
      2. Regression head     → trust_score (continuous)
    """

    def __init__(self, model_name, dropout=DROPOUT):
        super().__init__()

        # Base encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size
        head_hidden  = min(HEAD_HIDDEN, hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Head 1: classification (hallucinated vs correct)
        self.classifier = nn.Linear(hidden_size, 2)

        # Head 2: regression (trust score 0→1)
        self.regressor  = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
            nn.Sigmoid()          # output in [0, 1]
        )

    def forward(self, input_ids, attention_mask, return_features=False):
        # Encode — take [CLS] token (index 0)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Head 1: classification logits
        logits = self.classifier(cls_output)

        # Head 2: trust score
        trust  = self.regressor(cls_output).squeeze(-1)

        if return_features:
            return logits, trust, cls_output

        return logits, trust


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    log("Loading dataset splits...")

    train_df = pd.read_json(
        os.path.join(SPLITS_DIR, "train.jsonl"),
        lines=True
    )
    val_df = pd.read_json(
        os.path.join(SPLITS_DIR, "val.jsonl"),
        lines=True
    )

    print(f"  Train : {len(train_df):,} rows")
    print(f"  Val   : {len(val_df):,} rows")
    print(f"  Device: {DEVICE}")
    print(f"  Model : {MODEL_NAME}")

    # Verify required columns exist
    required = ["patient_context", "question",
                "answer", "is_hallucinated", "trust_score"]
    for col in required:
        if col not in train_df.columns:
            raise ValueError(
                f"Missing column '{col}' in train.jsonl. "
                f"Re-run build_combined.py"
            )

    return train_df, val_df


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def compute_metrics(all_labels, all_preds,
                    all_trusts, all_trust_preds):
    """Compute all metrics for one epoch."""
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds,
                    average="binary", zero_division=0)
    prec = precision_score(all_labels, all_preds,
                           average="binary", zero_division=0)
    rec  = recall_score(all_labels, all_preds,
                        average="binary", zero_division=0)
    mse  = mean_squared_error(all_trusts, all_trust_preds)
    mae  = mean_absolute_error(all_trusts, all_trust_preds)

    return {
        "accuracy"   : round(acc,  4),
        "f1"         : round(f1,   4),
        "precision"  : round(prec, 4),
        "recall"     : round(rec,  4),
        "trust_mse"  : round(mse,  4),
        "trust_mae"  : round(mae,  4),
    }


# ─────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer,
                scheduler, ce_loss_fn, mse_loss_fn,
                epoch, total_epochs):

    model.train()
    total_loss   = 0
    all_labels   = []
    all_preds    = []
    all_trusts   = []
    all_trust_preds = []
    n_batches    = len(loader)
    epoch_start  = time.time()

    for batch_idx, batch in enumerate(loader):
        input_ids   = batch["input_ids"].to(DEVICE)
        attn_mask   = batch["attention_mask"].to(DEVICE)
        labels      = batch["label"].to(DEVICE)
        trust_scores = batch["trust_score"].to(DEVICE)

        optimizer.zero_grad()

        logits, trust_pred = model(input_ids, attn_mask)

        # Combined loss
        ce_loss  = ce_loss_fn(logits, labels)
        mse_loss = mse_loss_fn(trust_pred, trust_scores)
        loss     = ALPHA * ce_loss + (1 - ALPHA) * mse_loss

        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Collect predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        all_trusts.extend(trust_scores.cpu().numpy())
        all_trust_preds.extend(
            trust_pred.detach().cpu().numpy()
        )

        # Progress every 50 batches and at the end of the epoch
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            elapsed = time.time() - epoch_start
            avg_batch_time = elapsed / (batch_idx + 1)
            remaining_batches = n_batches - (batch_idx + 1)
            eta = remaining_batches * avg_batch_time
            print(f"  Epoch {epoch}/{total_epochs} "
                  f"batch {batch_idx+1}/{n_batches} "
                  f"loss={loss.item():.4f} "
                  f"elapsed={format_duration(elapsed)} "
                  f"eta={format_duration(eta)}")

    avg_loss = total_loss / n_batches
    metrics  = compute_metrics(
        all_labels, all_preds,
        all_trusts, all_trust_preds
    )
    metrics["loss"] = round(avg_loss, 4)
    return metrics


# ─────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────
def evaluate(model, loader, ce_loss_fn, mse_loss_fn):

    model.eval()
    total_loss      = 0
    all_labels      = []
    all_preds       = []
    all_trusts      = []
    all_trust_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids   = batch["input_ids"].to(DEVICE)
            attn_mask   = batch["attention_mask"].to(DEVICE)
            labels      = batch["label"].to(DEVICE)
            trust_scores = batch["trust_score"].to(DEVICE)

            logits, trust_pred = model(input_ids, attn_mask)

            ce_loss  = ce_loss_fn(logits, labels)
            mse_loss = mse_loss_fn(trust_pred, trust_scores)
            loss     = (ALPHA * ce_loss +
                        (1 - ALPHA) * mse_loss)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_trusts.extend(trust_scores.cpu().numpy())
            all_trust_preds.extend(
                trust_pred.cpu().numpy()
            )

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(
        all_labels, all_preds,
        all_trusts, all_trust_preds
    )
    metrics["loss"] = round(avg_loss, 4)
    return metrics


# ─────────────────────────────────────────────────────────────
# SAVE CHECKPOINT
# ─────────────────────────────────────────────────────────────
def save_checkpoint(model, tokenizer, epoch,
                    val_metrics, is_best=False):
    checkpoint_meta = {
        "model_name"   : MODEL_NAME,
        "model_label"  : MODEL_LABEL,
        "max_len"      : MAX_LEN,
        "batch_size"   : BATCH_SIZE,
        "epochs"       : EPOCHS,
        "lr"           : LR,
        "dropout"      : DROPOUT,
        "alpha"        : ALPHA,
        "patient_chars": PATIENT_CHARS,
    }
    checkpoint_meta = to_python_types(checkpoint_meta)

    # Always save latest
    latest_path = os.path.join(OUTPUTS_DIR, "latest_model")
    os.makedirs(latest_path, exist_ok=True)
    model.encoder.save_pretrained(latest_path)
    tokenizer.save_pretrained(latest_path)

    # Save model state dict
    torch.save(
        model.state_dict(),
        os.path.join(latest_path, "model_state.pt")
    )
    with open(
        os.path.join(latest_path, "train_config.json"),
        "w"
    ) as f:
        json.dump(checkpoint_meta, f, indent=2)

    # Save best separately
    if is_best:
        best_path = os.path.join(OUTPUTS_DIR, "best_model")
        os.makedirs(best_path, exist_ok=True)
        model.encoder.save_pretrained(best_path)
        tokenizer.save_pretrained(best_path)
        torch.save(
            model.state_dict(),
            os.path.join(best_path, "model_state.pt")
        )
        with open(
            os.path.join(best_path, "train_config.json"),
            "w"
        ) as f:
            json.dump(checkpoint_meta, f, indent=2)

        # Save best metrics
        with open(
            os.path.join(best_path, "best_metrics.json"),
            "w"
        ) as f:
            json.dump(to_python_types({
                "epoch"      : epoch,
                "val_metrics": val_metrics,
                "saved_at"   : datetime.now().isoformat()
            }), f, indent=2)

        print(f"  Best model saved → outputs/best_model/")

    return best_path if is_best else latest_path


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train():
    log(f"Patient-Aware Hallucination Detection — {MODEL_LABEL}")
    print(f"  Device  : {DEVICE}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  LR      : {LR}")
    print(f"  Alpha   : {ALPHA} (CE) / "
          f"{1-ALPHA} (MSE)")

    # ── Load data ─────────────────────────────────────────────
    train_df, val_df = load_data()

    # ── Tokeniser ─────────────────────────────────────────────
    log("Loading tokeniser...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Vocab size : {tokenizer.vocab_size:,}")

    # ── Datasets + DataLoaders ────────────────────────────────
    train_dataset = HallucinationDataset(
        train_df, tokenizer
    )
    val_dataset   = HallucinationDataset(
        val_df,   tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,       # 0 = safe on Windows
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"\n  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")

    # ── Model ─────────────────────────────────────────────────
    log("Building model...")
    model = PatientAwareHallucinationDetector(MODEL_NAME)
    model = model.to(DEVICE)

    total_params = sum(
        p.numel() for p in model.parameters()
    )
    trainable    = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable:,}")

    # ── Optimiser + scheduler ─────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"  Total steps  : {total_steps:,}")
    print(f"  Warmup steps : {warmup_steps:,}")

    # ── Loss functions ────────────────────────────────────────
    ce_loss_fn  = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────
    log("Starting training...")

    best_val_f1  = 0.0
    training_log = []
    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n  ── Epoch {epoch}/{EPOCHS} ─────────────────────")

        # Train
        train_metrics = train_epoch(
            model, train_loader,
            optimizer, scheduler,
            ce_loss_fn, mse_loss_fn,
            epoch, EPOCHS
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader,
            ce_loss_fn, mse_loss_fn
        )

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start
        avg_epoch_time = total_elapsed / epoch
        remaining_epochs = EPOCHS - epoch
        total_eta = remaining_epochs * avg_epoch_time

        # Print epoch summary
        print(f"\n  Epoch {epoch} summary:")
        print(f"    Train loss  : {train_metrics['loss']}")
        print(f"    Train acc   : {train_metrics['accuracy']}")
        print(f"    Train F1    : {train_metrics['f1']}")
        print(f"    Val   loss  : {val_metrics['loss']}")
        print(f"    Val   acc   : {val_metrics['accuracy']}")
        print(f"    Val   F1    : {val_metrics['f1']}")
        print(f"    Val   prec  : {val_metrics['precision']}")
        print(f"    Val   rec   : {val_metrics['recall']}")
        print(f"    Val trust MAE: {val_metrics['trust_mae']}")
        print(f"    Time        : {format_duration(epoch_time)}")
        if remaining_epochs > 0:
            print(f"    ETA finish  : {format_duration(total_eta)}")

        # Check if best model
        is_best = val_metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["f1"]
            print(f"    New best F1 : {best_val_f1} ← saved")

        # Save checkpoint
        save_checkpoint(
            model, tokenizer, epoch,
            val_metrics, is_best
        )

        # Log epoch
        training_log.append({
            "epoch"        : epoch,
            "train"        : train_metrics,
            "val"          : val_metrics,
            "epoch_time_s" : round(epoch_time, 1),
            "is_best"      : is_best,
        })

        # Save running log after every epoch
        log_path = os.path.join(
            LOGS_DIR, "training_log.json"
        )
        with open(log_path, "w") as f:
            json.dump(to_python_types(training_log), f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    log("TRAINING COMPLETE")
    print(f"  Best val F1     : {best_val_f1:.4f}")
    print(f"  Best model path : outputs/best_model/")
    print(f"  Training log    : logs/training_log.json")

    return training_log


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
