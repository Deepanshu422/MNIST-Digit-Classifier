import keras
from src.config import INPUT_SHAPE, NUM_CLASSES

def build_neural_network() -> keras.Model:
    """
    Building a Feedforward Neural Network (Multi-Layer Perceptron).
    """
    model = keras.models.Sequential([
        keras.layers.Input(shape=INPUT_SHAPE),
        keras.layers.Flatten(name="flatten_input"),
        keras.layers.Dense(128, activation='relu', name="dense_hidden_1"),
        keras.layers.Dense(64, activation='relu', name="dense_hidden_2"),
        keras.layers.Dense(NUM_CLASSES, activation='softmax', name="softmax_probabilities")
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model