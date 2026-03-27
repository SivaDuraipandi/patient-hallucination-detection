from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.predict import router as predict_router
from backend.services.inference import inference_service


app = FastAPI(
    title="Patient Hallucination Detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.on_event("startup")
def load_model() -> None:
    inference_service.load()


@app.get("/")
def root() -> dict:
    return {
        "message": "Patient hallucination detection API is running",
        "endpoints": ["/predict"],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": inference_service.model is not None,
    }
