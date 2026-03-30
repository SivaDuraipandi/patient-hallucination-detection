"""
evaluate.py
===========
Builds confidence-based trust artifacts from train/validation splits
and evaluates the saved best checkpoint on the held-out test split.

Outputs:
  logs/test_metrics.json
  logs/test_predictions.jsonl
  outputs/best_model/trust_config.json
  outputs/best_model/embedding_index.npz
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train import (
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    LOGS_DIR,
    OUTPUTS_DIR,
    SPLITS_DIR,
    HallucinationDataset,
    DEFAULT_INPUT_VARIANT,
    PatientAwareHallucinationDetector,
    format_duration,
    log,
    to_python_types,
)
from trust_utils import (
    compose_trust_score,
    compute_selective_accuracy,
    expected_calibration_error,
    find_abstention_threshold,
    fit_temperature,
    multiclass_brier_score,
    neighbor_trust_scores,
    predictive_entropy,
    save_embedding_index,
    save_trust_config,
    softmax_probs,
)


BEST_MODEL_DIR = os.path.join(OUTPUTS_DIR, "best_model")
TRAIN_PATH = os.path.join(SPLITS_DIR, "train.jsonl")
VAL_PATH = os.path.join(SPLITS_DIR, "val.jsonl")
TEST_PATH = os.path.join(SPLITS_DIR, "test.jsonl")
TRUST_CONFIG_PATH = os.path.join(BEST_MODEL_DIR, "trust_config.json")
EMBED_INDEX_PATH = os.path.join(BEST_MODEL_DIR, "embedding_index.npz")
NEIGHBOR_K = 15
MIN_COVERAGE = 0.85
TRUST_WEIGHTS = {
    "calibrated_probability": 0.55,
    "neighbor_trust": 0.25,
    "uncertainty_penalty": 0.20,
}


def load_split(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} split not found at {path}.")

    df = pd.read_json(path, lines=True)
    required = [
        "matched_patient_uid",
        "patient_context",
        "question",
        "answer",
        "is_hallucinated",
        "trust_score",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {label}.")

    return df


def run_model(model, loader):
    logits_rows = []
    trust_rows = []
    feature_rows = []
    label_rows = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits, trust_pred, features = model(
                input_ids, attn_mask, return_features=True
            )

            logits_rows.append(logits.cpu().numpy())
            trust_rows.append(trust_pred.cpu().numpy())
            feature_rows.append(features.cpu().numpy())
            label_rows.append(labels.cpu().numpy())

    return {
        "logits": np.concatenate(logits_rows, axis=0),
        "trust_pred": np.concatenate(trust_rows, axis=0),
        "features": np.concatenate(feature_rows, axis=0),
        "labels": np.concatenate(label_rows, axis=0),
    }


def prepare_trust_artifacts(model, tokenizer, input_variant):
    log("Preparing trust artifacts from train/validation splits...")

    train_df = load_split(TRAIN_PATH, "train.jsonl")
    val_df = load_split(VAL_PATH, "val.jsonl")

    train_loader = DataLoader(
        HallucinationDataset(
            train_df, tokenizer, input_variant=input_variant
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        HallucinationDataset(
            val_df, tokenizer, input_variant=input_variant
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    train_outputs = run_model(model, train_loader)
    val_outputs = run_model(model, val_loader)

    temperature = fit_temperature(
        val_outputs["logits"], val_outputs["labels"]
    )
    val_probs = softmax_probs(val_outputs["logits"], temperature)
    val_preds = val_probs.argmax(axis=1)
    val_uncertainty = predictive_entropy(val_probs)
    val_neighbor = neighbor_trust_scores(
        val_outputs["features"],
        train_outputs["features"],
        train_outputs["labels"],
        k=NEIGHBOR_K,
    )
    abstain = find_abstention_threshold(
        val_probs,
        val_outputs["labels"],
        min_coverage=MIN_COVERAGE,
    )

    save_embedding_index(
        EMBED_INDEX_PATH,
        train_outputs["features"],
        train_outputs["labels"],
    )

    trust_config = {
        "temperature": round(float(temperature), 6),
        "neighbor_k": NEIGHBOR_K,
        "min_coverage": MIN_COVERAGE,
        "abstention": abstain,
        "weights": TRUST_WEIGHTS,
        "validation_metrics": {
            "ece": round(
                expected_calibration_error(
                    val_probs, val_outputs["labels"]
                ),
                4,
            ),
            "brier": round(
                multiclass_brier_score(
                    val_probs, val_outputs["labels"]
                ),
                4,
            ),
            "mean_uncertainty": round(
                float(val_uncertainty.mean()), 4
            ),
            "mean_neighbor_trust": round(
                float(val_neighbor.mean()), 4
            ),
            "accuracy": round(
                float(
                    accuracy_score(
                        val_outputs["labels"], val_preds
                    )
                ),
                4,
            ),
        },
    }
    save_trust_config(TRUST_CONFIG_PATH, trust_config)
    print(f"  Saved trust config : {TRUST_CONFIG_PATH}")
    print(f"  Saved embed index  : {EMBED_INDEX_PATH}")
    return train_outputs, trust_config


def compute_test_metrics(labels, preds, trusts, trust_preds, probs):
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
    roc_auc = roc_auc_score(labels, probs[:, 1])

    tn, fp, fn, tp = confusion_matrix(
        labels, preds, labels=[0, 1]
    ).ravel()
    specificity = tn / max(tn + fp, 1)

    return {
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(specificity), 4),
        "trust_mse": round(float(mse), 4),
        "trust_mae": round(float(mae), 4),
        "trust_rmse": round(float(rmse), 4),
        "roc_auc": round(float(roc_auc), 4),
        "ece": round(
            expected_calibration_error(probs, labels), 4
        ),
        "brier": round(
            multiclass_brier_score(probs, labels), 4
        ),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate_on_test(
    model,
    tokenizer,
    train_outputs,
    trust_config,
    input_variant,
):
    test_df = load_split(TEST_PATH, "test.jsonl")
    test_loader = DataLoader(
        HallucinationDataset(
            test_df, tokenizer, input_variant=input_variant
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    log("Running trust-aware test evaluation...")
    outputs = run_model(model, test_loader)

    probs = softmax_probs(
        outputs["logits"], trust_config["temperature"]
    )
    preds = probs.argmax(axis=1)
    uncertainties = predictive_entropy(probs)
    neighbor_trust = neighbor_trust_scores(
        outputs["features"],
        train_outputs["features"],
        train_outputs["labels"],
        k=trust_config["neighbor_k"],
    )
    confidences = probs.max(axis=1)
    abstain_mask = (
        confidences <
        trust_config["abstention"]["confidence_threshold"]
    )
    final_trust = compose_trust_score(
        prob_not_hallucinated=probs[:, 0],
        uncertainty=uncertainties,
        neighbor_trust=neighbor_trust,
        abstain_mask=abstain_mask,
        weights=trust_config["weights"],
    )

    metrics = compute_test_metrics(
        outputs["labels"],
        preds,
        test_df["trust_score"].to_numpy(dtype=np.float32),
        final_trust,
        probs,
    )
    metrics["mean_uncertainty"] = round(
        float(uncertainties.mean()), 4
    )
    metrics["mean_neighbor_trust"] = round(
        float(neighbor_trust.mean()), 4
    )
    metrics["mean_final_trust"] = round(
        float(final_trust.mean()), 4
    )
    metrics["abstention_rate"] = round(
        float(abstain_mask.mean()), 4
    )
    metrics["selective"] = compute_selective_accuracy(
        outputs["labels"], preds, abstain_mask
    )

    prediction_rows = []
    for idx, (_, row) in enumerate(test_df.iterrows()):
        prediction_rows.append({
            "question": row["question"],
            "answer": row["answer"],
            "matched_patient_uid": row["matched_patient_uid"],
            "difficulty": row.get("difficulty", "unknown"),
            "hallucination_type": row.get(
                "hallucination_type", "unknown"
            ),
            "label": int(outputs["labels"][idx]),
            "pred": int(preds[idx]),
            "prob_hallucinated": float(probs[idx, 1]),
            "prob_not_hallucinated": float(probs[idx, 0]),
            "uncertainty": float(uncertainties[idx]),
            "neighbor_trust": float(neighbor_trust[idx]),
            "predicted_trust_score": float(final_trust[idx]),
            "abstain_for_review": bool(abstain_mask[idx]),
            "correct": bool(outputs["labels"][idx] == preds[idx]),
        })

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

    log("Loading tokenizer + checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
    model = PatientAwareHallucinationDetector(
        BEST_MODEL_DIR, dropout=DROPOUT
    )
    config_path = os.path.join(BEST_MODEL_DIR, "train_config.json")
    input_variant = DEFAULT_INPUT_VARIANT
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            train_config = json.load(f)
        input_variant = train_config.get(
            "input_variant", DEFAULT_INPUT_VARIANT
        )
    print(f"  Input mode  : {input_variant}")

    state_path = os.path.join(BEST_MODEL_DIR, "model_state.pt")
    state_dict = torch.load(state_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)

    wall_start = pd.Timestamp.now()
    train_outputs, trust_config = prepare_trust_artifacts(
        model, tokenizer, input_variant
    )
    metrics, predictions = evaluate_on_test(
        model,
        tokenizer,
        train_outputs,
        trust_config,
        input_variant,
    )
    wall_end = pd.Timestamp.now()

    metrics["wall_time"] = format_duration(
        (wall_end - wall_start).total_seconds()
    )

    print("\n  Test summary:")
    print(f"    Accuracy          : {metrics['accuracy']}")
    print(f"    F1                : {metrics['f1']}")
    print(f"    ROC AUC           : {metrics['roc_auc']}")
    print(f"    ECE               : {metrics['ece']}")
    print(f"    Brier             : {metrics['brier']}")
    print(f"    Trust MAE         : {metrics['trust_mae']}")
    print(f"    Mean uncertainty  : {metrics['mean_uncertainty']}")
    print(f"    Mean neighbor     : {metrics['mean_neighbor_trust']}")
    print(f"    Abstention rate   : {metrics['abstention_rate']}")
    print(f"    Selective results : {metrics['selective']}")
    print(f"    Time              : {metrics['wall_time']}")

    save_results(metrics, predictions)


if __name__ == "__main__":
    main()
