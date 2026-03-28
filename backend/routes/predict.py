from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.inference import inference_service


router = APIRouter(prefix="/predict", tags=["predict"])


class PredictionRequest(BaseModel):
    patient_context: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    label_id: int
    label: str
    is_hallucinated: bool
    hallucination_probability: float
    confidence: float
    trust_score: float
    uncertainty: float
    neighbor_trust: float
    calibrated_probability: float
    abstain_for_review: bool
    explanation_tags: list[str]
    device: str
    model_dir: str


@router.post("", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        result = inference_service.predict(
            patient_context=payload.patient_context,
            question=payload.question,
            answer=payload.answer,
        )
        return PredictionResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc
