import time
import numpy as np
import keras
from src.config import MODEL_PATH, MODEL_VERSION

class ModelPredictor:
    def __init__(self):
        self.model = None
        self.version = MODEL_VERSION

    def load(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run training first.")
        self.model = keras.models.load_model(MODEL_PATH)
        print(f"[+] Loaded Model v{self.version} into memory.")

    def predict(self, input_tensor: np.ndarray):
        start = time.perf_counter()

        raw_probs = self.model.predict(input_tensor, verbose=0)[0]

        latency = (time.perf_counter() - start) * 1000
        predicted_digit = int(np.argmax(raw_probs))
        confidence = float(raw_probs[predicted_digit])

        return{
            "model_version": self.version,
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 4),
            "class_probabilities": {i: round(float(raw_probs[i]), 4) for i in range(10)},
            "latency_ms": round(latency, 2)
        }

predictor = ModelPredictor()