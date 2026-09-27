import json
import time
from src.config import MODEL_PATH, METADATA_PATH, MODEL_VERSION, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT
from src.training.model import build_neural_network
from src.training.data import load_and_preprocess_mnist

def run_training_pipeine():

    print(f"[*] Starting offline training for MNIST Model v{MODEL_VERSION}...")

    # load data
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_mnist()

    # instantiate model
    model = build_neural_network()

    # training loop
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )

    # testing on unseen data set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[+] Evaluation Complete. Test Accuracy: {test_acc*100:.2f}%, Test Loss: {test_loss:.4f}")

    # Save model binary
    model.save(MODEL_PATH)
    print(f"[+] Model artifact exported to: {MODEL_PATH}")

    # Save metadata
    metadata = {
        "model_version": MODEL_VERSION,
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[+] Metadata written to: {METADATA_PATH}")

if __name__ == "__main__":
    run_training_pipeine()