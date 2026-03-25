"""
evaluate.py
===========
Loads the saved best MiniLM checkpoint and evaluates it on the
held-out test split for final paper-ready metrics.

Outputs:
  logs/test_metrics.json
  logs/test_predictions.jsonl
"""

import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from train import (
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    LOGS_DIR,
    OUTPUTS_DIR,
    SPLITS_DIR,
    HallucinationDataset,
    PatientAwareHallucinationDetector,
    format_duration,
    log,
    to_python_types,
)


BEST_MODEL_DIR = os.path.join(OUTPUTS_DIR, "best_model")
TEST_PATH = os.path.join(SPLITS_DIR, "test.jsonl")


def load_test_split():
    log("Loading held-out test split...")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Test split not found at {TEST_PATH}."
        )

    test_df = pd.read_json(TEST_PATH, lines=True)
    required = [
        "patient_context", "question", "answer",
        "is_hallucinated", "trust_score"
    ]
    for col in required:
        if col not in test_df.columns:
            raise ValueError(
                f"Missing column '{col}' in test.jsonl."
            )

    print(f"  Test rows : {len(test_df):,}")
    return test_df


def compute_test_metrics(labels, preds, trusts, trust_preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    prec = precision_score(
        labels, preds, average="binary", zero_division=0
    )
    rec = recall_score(
        labels, preds, average="binary", zero_division=0
    )
    mse = mean_squared_error(trusts, trust_preds)
    mae = mean_absolute_error(trusts, trust_preds)
    rmse = float(np.sqrt(mse))

    tn, fp, fn, tp = confusion_matrix(
        labels, preds, labels=[0, 1]
    ).ravel()
    specificity = tn / max(tn + fp, 1)

    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "trust_mse": round(mse, 4),
        "trust_mae": round(mae, 4),
        "trust_rmse": round(rmse, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate_on_test(model, loader, test_df):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    total_loss = 0.0
    labels = []
    preds = []
    trusts = []
    trust_preds = []
    prediction_rows = []

    start_time = torch.cuda.Event(enable_timing=True) \
        if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) \
        if torch.cuda.is_available() else None

    if start_time is not None:
        start_time.record()

    with torch.no_grad():
        sample_offset = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            batch_labels = batch["label"].to(DEVICE)
            batch_trusts = batch["trust_score"].to(DEVICE)

            logits, batch_trust_preds = model(input_ids, attn_mask)

            ce_loss = ce_loss_fn(logits, batch_labels)
            mse_loss = mse_loss_fn(batch_trust_preds, batch_trusts)
            loss = 0.7 * ce_loss + 0.3 * mse_loss
            total_loss += loss.item()

            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            batch_preds = torch.argmax(logits, dim=1)

            labels_np = batch_labels.cpu().numpy()
            preds_np = batch_preds.cpu().numpy()
            probs_np = batch_probs.cpu().numpy()
            trusts_np = batch_trusts.cpu().numpy()
            trust_preds_np = batch_trust_preds.cpu().numpy()

            labels.extend(labels_np)
            preds.extend(preds_np)
            trusts.extend(trusts_np)
            trust_preds.extend(trust_preds_np)

            batch_size = len(labels_np)
            batch_df = test_df.iloc[sample_offset: sample_offset + batch_size]
            for i, (_, row) in enumerate(batch_df.iterrows()):
                prediction_rows.append({
                    "question": row["question"],
                    "answer": row["answer"],
                    "difficulty": row.get("difficulty", "unknown"),
                    "hallucination_type": row.get(
                        "hallucination_type", "unknown"
                    ),
                    "label": int(labels_np[i]),
                    "pred": int(preds_np[i]),
                    "prob_hallucinated": float(probs_np[i]),
                    "trust_score": float(trusts_np[i]),
                    "predicted_trust_score": float(trust_preds_np[i]),
                    "correct": bool(labels_np[i] == preds_np[i]),
                })
            sample_offset += batch_size

    elapsed_seconds = None
    if start_time is not None:
        end_time.record()
        torch.cuda.synchronize()
        elapsed_seconds = start_time.elapsed_time(end_time) / 1000.0

    metrics = compute_test_metrics(labels, preds, trusts, trust_preds)
    metrics["loss"] = round(total_loss / len(loader), 4)
    if elapsed_seconds is not None:
        metrics["eval_time_s"] = round(elapsed_seconds, 2)

    return metrics, prediction_rows


def save_results(metrics, predictions):
    metrics_path = os.path.join(LOGS_DIR, "test_metrics.json")
    preds_path = os.path.join(LOGS_DIR, "test_predictions.jsonl")

    with open(metrics_path, "w") as f:
        json.dump(to_python_types(metrics), f, indent=2)

    pd.DataFrame(predictions).to_json(
        preds_path, orient="records", lines=True
    )

    print(f"  Saved metrics     : {metrics_path}")
    print(f"  Saved predictions : {preds_path}")


def main():
    log("Final Test Evaluation")
    print(f"  Device      : {DEVICE}")
    print(f"  Checkpoint  : {BEST_MODEL_DIR}")

    if not os.path.exists(BEST_MODEL_DIR):
        raise FileNotFoundError(
            f"Best model not found at {BEST_MODEL_DIR}."
        )

    test_df = load_test_split()

    log("Loading tokenizer + checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
    model = PatientAwareHallucinationDetector(
        BEST_MODEL_DIR, dropout=DROPOUT
    )

    state_path = os.path.join(BEST_MODEL_DIR, "model_state.pt")
    state_dict = torch.load(state_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)

    test_dataset = HallucinationDataset(test_df, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    log("Evaluating on test split...")
    wall_start = pd.Timestamp.now()
    metrics, predictions = evaluate_on_test(
        model, test_loader, test_df
    )
    wall_end = pd.Timestamp.now()

    metrics["wall_time"] = format_duration(
        (wall_end - wall_start).total_seconds()
    )

    print("\n  Test summary:")
    print(f"    Loss        : {metrics['loss']}")
    print(f"    Accuracy    : {metrics['accuracy']}")
    print(f"    F1          : {metrics['f1']}")
    print(f"    Precision   : {metrics['precision']}")
    print(f"    Recall      : {metrics['recall']}")
    print(f"    Specificity : {metrics['specificity']}")
    print(f"    Trust MAE   : {metrics['trust_mae']}")
    print(f"    Trust RMSE  : {metrics['trust_rmse']}")
    print(f"    Confusion   : {metrics['confusion_matrix']}")
    print(f"    Time        : {metrics['wall_time']}")

    save_results(metrics, predictions)


if __name__ == "__main__":
    main()
