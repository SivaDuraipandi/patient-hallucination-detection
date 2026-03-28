import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score


def _to_tensor(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().float()
    return torch.tensor(array_like, dtype=torch.float32)


def apply_temperature(logits, temperature: float):
    logits = _to_tensor(logits)
    temperature = max(float(temperature), 1e-3)
    return logits / temperature


def softmax_probs(logits, temperature: float = 1.0) -> np.ndarray:
    scaled = apply_temperature(logits, temperature)
    return torch.softmax(scaled, dim=1).cpu().numpy()


def fit_temperature(logits, labels) -> float:
    logits_t = _to_tensor(logits)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=50
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).item())


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-8, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(probs.shape[1])
    return entropy / max(max_entropy, 1e-8)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for idx in range(num_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        mask = (
            (confidences >= left) &
            (confidences < right if idx < num_bins - 1 else confidences <= right)
        )
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.mean()

    return float(ece)


def multiclass_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-8, None)


def neighbor_trust_scores(
    query_embeddings: np.ndarray,
    index_embeddings: np.ndarray,
    index_labels: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    query_norm = normalize_embeddings(query_embeddings.astype(np.float32))
    index_norm = normalize_embeddings(index_embeddings.astype(np.float32))
    similarities = np.matmul(query_norm, index_norm.T)
    k = max(1, min(int(k), len(index_labels)))
    topk_idx = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]

    trust_scores = []
    for row_idx in range(len(query_embeddings)):
        row_neighbors = topk_idx[row_idx]
        row_sims = similarities[row_idx, row_neighbors]
        row_weights = np.clip((row_sims + 1.0) / 2.0, 1e-6, None)
        row_labels = index_labels[row_neighbors]
        support_correct = np.sum(row_weights[row_labels == 0])
        trust = support_correct / np.sum(row_weights)
        trust_scores.append(float(trust))

    return np.asarray(trust_scores, dtype=np.float32)


def find_abstention_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    min_coverage: float = 0.85,
) -> Dict[str, float]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float32)

    unique_thresholds = np.unique(np.round(confidences, 6))
    best_threshold = float(unique_thresholds.min())
    best_accuracy = -1.0
    best_coverage = 1.0

    for threshold in unique_thresholds:
        keep_mask = confidences >= threshold
        coverage = float(np.mean(keep_mask))
        if coverage < min_coverage or not np.any(keep_mask):
            continue
        selective_acc = float(correctness[keep_mask].mean())
        if selective_acc > best_accuracy:
            best_accuracy = selective_acc
            best_threshold = float(threshold)
            best_coverage = coverage

    return {
        "confidence_threshold": best_threshold,
        "coverage": round(best_coverage, 4),
        "selective_accuracy": round(best_accuracy, 4),
    }


def compose_trust_score(
    prob_not_hallucinated: np.ndarray,
    uncertainty: np.ndarray,
    neighbor_trust: np.ndarray,
    abstain_mask: np.ndarray,
    weights: Dict[str, float] | None = None,
) -> np.ndarray:
    weights = weights or {
        "calibrated_probability": 0.55,
        "neighbor_trust": 0.25,
        "uncertainty_penalty": 0.20,
    }
    trust = (
        weights["calibrated_probability"] * prob_not_hallucinated +
        weights["neighbor_trust"] * neighbor_trust +
        weights["uncertainty_penalty"] * (1.0 - uncertainty)
    )
    trust = np.clip(trust, 0.0, 1.0)
    trust = np.where(abstain_mask, np.minimum(trust, 0.35), trust)
    return trust.astype(np.float32)


def save_embedding_index(
    output_path: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> None:
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
    )


def load_embedding_index(index_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(index_path)
    return data["embeddings"], data["labels"]


def save_trust_config(config_path: str, config: Dict) -> None:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def compute_selective_accuracy(
    labels: np.ndarray,
    preds: np.ndarray,
    abstain_mask: np.ndarray,
) -> Dict[str, float]:
    covered_mask = ~abstain_mask
    coverage = float(np.mean(covered_mask))
    if not np.any(covered_mask):
        return {"coverage": round(coverage, 4), "selective_accuracy": 0.0}

    selective_acc = accuracy_score(labels[covered_mask], preds[covered_mask])
    return {
        "coverage": round(coverage, 4),
        "selective_accuracy": round(float(selective_acc), 4),
    }
