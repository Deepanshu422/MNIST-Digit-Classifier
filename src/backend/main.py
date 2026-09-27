from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.backend.predictor import predictor
from src.backend.schemas import PredictionResponse, HealthCheckResponse
from src.backend.preprocessor import preprocess_image

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading model
    try:
        predictor.load()
    except Exception as e:
        print(f"[!] Warning: Model could not be loaded: {e}")
    yield

app = FastAPI(
    title="MNIST Production API",
    description="Digit Classification Model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    return {
        "status": "healthy" if predictor.model is not None else "degraded",
        "model_loaded": predictor.model is not None,
        "model_version": predictor.version
    }

@app.post("/v1/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    image_bytes = await file.read()
    tensor = preprocess_image(image_bytes)
    result = predictor.predict(tensor)
    return result