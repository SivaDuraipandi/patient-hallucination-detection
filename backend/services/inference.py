import json
import os
import sys
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoTokenizer


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MODEL_SRC_DIR = os.path.join(PROJECT_ROOT, "src", "model")
if MODEL_SRC_DIR not in sys.path:
    sys.path.insert(0, MODEL_SRC_DIR)

from train import (  # noqa: E402
    DEVICE,
    DROPOUT,
    MAX_LEN,
    OUTPUTS_DIR,
    PATIENT_CHARS,
    PatientAwareHallucinationDetector,
)
from trust_utils import (  # noqa: E402
    compose_trust_score,
    load_embedding_index,
    neighbor_trust_scores,
    predictive_entropy,
    softmax_probs,
)


BEST_MODEL_DIR = os.path.join(OUTPUTS_DIR, "best_model")
TRUST_CONFIG_PATH = os.path.join(BEST_MODEL_DIR, "trust_config.json")
EMBED_INDEX_PATH = os.path.join(BEST_MODEL_DIR, "embedding_index.npz")


class InferenceService:
    def __init__(self) -> None:
        self.device = DEVICE
        self.model_dir = BEST_MODEL_DIR
        self.tokenizer = None
        self.model = None
        self.max_len = MAX_LEN
        self.patient_chars = PATIENT_CHARS
        self.trust_config = {
            "temperature": 1.0,
            "neighbor_k": 15,
            "abstention": {"confidence_threshold": 0.55},
            "weights": {
                "calibrated_probability": 0.55,
                "neighbor_trust": 0.25,
                "uncertainty_penalty": 0.20,
            },
        }
        self.index_embeddings = None
        self.index_labels = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Trained model not found at {self.model_dir}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = PatientAwareHallucinationDetector(
            self.model_dir, dropout=DROPOUT
        )

        config_path = os.path.join(self.model_dir, "train_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                train_config = json.load(f)
            self.max_len = int(
                train_config.get("max_len", self.max_len)
            )
            self.patient_chars = int(
                train_config.get("patient_chars", self.patient_chars)
            )

        if os.path.exists(TRUST_CONFIG_PATH):
            with open(TRUST_CONFIG_PATH, "r") as f:
                self.trust_config = json.load(f)

        if os.path.exists(EMBED_INDEX_PATH):
            self.index_embeddings, self.index_labels = (
                load_embedding_index(EMBED_INDEX_PATH)
            )

        state_path = os.path.join(self.model_dir, "model_state.pt")
        state_dict = torch.load(state_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        patient_context: str,
        question: str,
        answer: str,
    ) -> Dict[str, Any]:
        self.load()

        patient = str(patient_context or "")[:self.patient_chars]
        question = str(question or "")
        answer = str(answer or "")
        text = (
            f"patient: {patient} "
            f"question: {question} "
            f"answer: {answer}"
        )

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits, _, features = self.model(
                input_ids,
                attention_mask,
                return_features=True,
            )

        probs = softmax_probs(
            logits.cpu(),
            self.trust_config.get("temperature", 1.0),
        )[0]
        pred_label = int(np.argmax(probs))
        uncertainty = float(predictive_entropy(probs[None, :])[0])

        neighbor_trust = 0.5
        if self.index_embeddings is not None and self.index_labels is not None:
            neighbor_trust = float(
                neighbor_trust_scores(
                    features.cpu().numpy(),
                    self.index_embeddings,
                    self.index_labels,
                    k=self.trust_config.get("neighbor_k", 15),
                )[0]
            )

        confidence = float(np.max(probs))
        abstain_for_review = bool(
            confidence <
            self.trust_config.get(
                "abstention", {}
            ).get("confidence_threshold", 0.55)
        )
        trust_score = float(
            compose_trust_score(
                prob_not_hallucinated=np.asarray([probs[0]]),
                uncertainty=np.asarray([uncertainty]),
                neighbor_trust=np.asarray([neighbor_trust]),
                abstain_mask=np.asarray([abstain_for_review]),
                weights=self.trust_config.get("weights"),
            )[0]
        )

        explanation_tags = []
        if abstain_for_review:
            explanation_tags.append("review_required")
        if uncertainty >= 0.45:
            explanation_tags.append("high_uncertainty")
        if neighbor_trust <= 0.4:
            explanation_tags.append("weak_neighbor_support")
        if probs[1] >= 0.5:
            explanation_tags.append("hallucination_risk")
        if not explanation_tags:
            explanation_tags.append("stable_prediction")

        return {
            "label_id": pred_label,
            "label": (
                "hallucinated" if pred_label == 1 else "not_hallucinated"
            ),
            "is_hallucinated": bool(pred_label == 1),
            "hallucination_probability": round(float(probs[1]), 4),
            "confidence": round(confidence, 4),
            "trust_score": round(trust_score, 4),
            "uncertainty": round(uncertainty, 4),
            "neighbor_trust": round(neighbor_trust, 4),
            "calibrated_probability": round(float(probs[pred_label]), 4),
            "abstain_for_review": abstain_for_review,
            "explanation_tags": explanation_tags,
            "device": str(self.device),
            "model_dir": self.model_dir,
        }


inference_service = InferenceService()
