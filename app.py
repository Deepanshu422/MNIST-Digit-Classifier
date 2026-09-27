import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

# 1. ZeroGPU compatibility (Cloud par GPU, local par fallback)
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

import gradio as gr
import numpy as np
from src.backend.preprocessor import preprocess_image
from src.backend.predictor import predictor

# Startup check satisfy karne ke liye decorator
@spaces.GPU
def classify_drawing(sketch):
    if sketch is None:
        return {}, None

    # Lazy-load model inside execution context
    if predictor.model is None:
        predictor.load()

    # Safe extraction without numpy boolean 'or'
    img_data = None
    if isinstance(sketch, dict):
        if "composite" in sketch and sketch["composite"] is not None:
            img_data = sketch["composite"]
        elif "image" in sketch and sketch["image"] is not None:
            img_data = sketch["image"]
        elif "layers" in sketch and len(sketch["layers"]) > 0:
            img_data = sketch["layers"][0]
    else:
        img_data = sketch

    if img_data is None:
        return {"Draw a digit first": 1.0}, None

    try:
        # Preprocess into (1, 28, 28)
        tensor = preprocess_image(img_data)

        # Check for blank canvas
        if tensor.sum() == 0:
            return {"Draw a digit first": 1.0}, None

        # Predict
        result = predictor.predict(tensor)
        confidences = {
            str(digit): float(prob)
            for digit, prob in result["class_probabilities"].items()
        }

        # Grayscale preview (28x28)
        preview = (tensor[0] * 255).astype(np.uint8)
        return confidences, preview

    except Exception as e:
        print(f"[!] Inference Exception: {e}")
        return {f"Error: {str(e)}": 1.0}, None


with gr.Blocks(title="MNIST Digit Classifier") as demo:
    gr.Markdown("# Production-Grade Handwritten Digit Recognizer")
    gr.Markdown("Draw any digit from **0 to 9** inside the canvas and click **Submit**.")

    with gr.Row():
        with gr.Column():
            sketch = gr.Sketchpad(label="Draw Digit Here")
            btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.ClearButton([sketch])
        with gr.Column():
            label_output = gr.Label(num_top_classes=10, label="Predictions (0-9)")
            preview_img = gr.Image(
                label="What Model Sees (28x28 MNIST format)",
                image_mode="L",
                width=140,
                height=140,
            )

    btn.click(
        fn=classify_drawing,
        inputs=[sketch],
        outputs=[label_output, preview_img],
    )

# ssr_mode=False taaki node server shutdown issue na aaye
demo.launch(ssr_mode=False)