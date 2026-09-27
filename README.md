Here is the updated, accurate `README.md` reflecting your real files (`train.py`, `data.py`), your live Hugging Face deployment link, and the exact commands to run and retrain the project.

```markdown
# 🖊️ Handwritten Digit Recognition (MNIST)

An interactive, deep learning-powered handwritten digit classifier built with **TensorFlow / Keras** and an intuitive **Gradio** web interface.

Users can draw any single digit ($0$–$9$) on a canvas and receive real-time classification predictions and confidence breakdowns.

---

## 🌐 Live Demo

Try the interactive model directly in your browser without installing anything locally:

👉 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/deepanshu422/mnist-digit-classifier)**

---

## 📁 Repository Structure

```text
mnist-digit-classifier/
├── artifacts/
│   ├── metadata.json           # Model metadata, training parameters & test scores
│   └── mnist_model.keras       # Serialized Keras neural network weights
├── src/
│   ├── backend/
│   │   ├── predictor.py        # ModelPredictor wrapper for inference & timing
│   │   └── preprocessor.py     # Image normalization & MNIST-standard alignment
│   ├── config.py               # Central configuration (paths, hyperparameters)
│   └── training/
│       ├── data.py             # Dataset loading & [0.0, 1.0] scaling logic
│       ├── model.py            # Neural network architecture definition (MLP)
│       └── train.py            # Model training & artifact export loop
├── app.py                      # Interactive Gradio canvas application
├── requirements.txt            # Project dependencies
└── README.md

```

---

## 🧠 Model Architecture & Pipeline

### 1. Neural Network Architecture

The classifier uses a Feedforward Multi-Layer Perceptron (MLP) built in Keras:

* **Flatten Layer:** Reshapes the $28 \times 28$ image into a 784-dimensional vector.
* **Dense Layer 1:** 128 neurons, ReLU activation.
* **Dense Layer 2:** 64 neurons, ReLU activation.
* **Output Layer:** 10 neurons, Softmax activation (outputs class probabilities for digits 0–9).
* **Optimization:** Adam optimizer with `sparse_categorical_crossentropy` loss.

### 2. Preprocessing & Alignment Engine

Hand-drawn canvas inputs vary in scale and orientation. Before passing sketches to the neural network, `src/backend/preprocessor.py` applies the standard MNIST pipeline:

1. **Luminance Conversion:** Flattens RGBA canvas drawings to a single grayscale channel.
2. **Polarity Correction:** Inverts light canvas drawings to match MNIST's standard (white strokes on a solid black background).
3. **Aspect-Ratio Resizing:** Resizes the drawn character to fit inside a $20 \times 20$ bounding box without distortion.
4. **Padding & Centering:** Pastes the $20 \times 20$ digit into the center of a blank $28 \times 28$ matrix (creating the required 4-pixel border margin).
5. **Normalization:** Scales pixel values from integers $[0, 255]$ to `float32` values in $[0.0, 1.0]$.

---

## 🚀 Local Setup & Running

### Prerequisites

* Python 3.10+
* Virtual environment (`venv` recommended)

### 1. Setup & Installation

```bash
# Clone the repository
git clone [https://github.com/deepanshu422/mnist-digit-classifier.git](https://github.com/deepanshu422/mnist-digit-classifier.git)
cd mnist-digit-classifier

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Launch the Application

Start the interactive Gradio interface locally:

```bash
python app.py

```

Open your browser and navigate to:

```text
http://localhost:7860

```

Draw any digit from **0 to 9** on the canvas to see the live classification and confidence distribution.

---

## 🔁 Retraining the Model (Optional)

A pre-trained model artifact is already included in `artifacts/mnist_model.keras`. If you wish to retrain the network from scratch:

```bash
python -m src.training.train

```

This command will:

1. Load and scale the official MNIST dataset using `src/training/data.py`.
2. Build the model architecture from `src/training/model.py`.
3. Train the network across configured epochs and batch sizes.
4. Evaluate accuracy against the 10,000-image unseen test set.
5. Save the updated model weights and metrics into `artifacts/`.

```

```