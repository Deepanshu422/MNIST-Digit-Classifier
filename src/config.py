import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Versioning
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MODEL_FILENAME = f"mnist_model_v{MODEL_VERSION}.keras"
MODEL_PATH = ARTIFACTS_DIR / MODEL_FILENAME
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
VALIDATION_SPLIT = 0.1
INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10
