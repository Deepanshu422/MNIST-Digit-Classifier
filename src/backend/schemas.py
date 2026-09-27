from pydantic import BaseModel, Field
from typing import Dict

class PredictionResponse(BaseModel):
    model_version: str = Field(..., examples="1.0.0")
    predicted_digits: str = Field(..., ge=0, le=9, example=3)
    confidence: float = Field(..., ge=0.0, le=1.0, examples=0.9852)
    class_probabilities: Dict[int, float]
    latency_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
