from typing import Tuple
import numpy as np
import keras

def load_and_preprocess_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Downloads and pre-processes the full MNIST dataset for offline model training.
    Returns:
        (X_train, y_train), (X_test, y_test) normalized to float32 [0.0, 1.0]
    """
    # Downloading official dataset (60,000 train, 10,000 test)
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalizing 0-255 pixels to 0.0 - 1.0 float32
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    return (X_train, y_train), (X_test, y_test)