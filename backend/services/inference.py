import os
import sys
import json
from typing import Any, Dict

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


BEST_MODEL_DIR = os.path.join(OUTPUTS_DIR, "best_model")


class InferenceService:
    def __init__(self) -> None:
        self.device = DEVICE
        self.model_dir = BEST_MODEL_DIR
        self.tokenizer = None
        self.model = None
        self.max_len = MAX_LEN
        self.patient_chars = PATIENT_CHARS

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
            logits, trust_pred = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[0]

        pred_label = int(torch.argmax(probs).item())
        hallucination_probability = float(probs[1].item())
        trust_score = float(trust_pred.squeeze().item())

        return {
            "label_id": pred_label,
            "label": (
                "hallucinated" if pred_label == 1 else "not_hallucinated"
            ),
            "is_hallucinated": bool(pred_label == 1),
            "hallucination_probability": round(
                hallucination_probability, 4
            ),
            "confidence": round(max(probs).item(), 4),
            "trust_score": round(trust_score, 4),
            "device": str(self.device),
            "model_dir": self.model_dir,
        }


inference_service = InferenceService()
