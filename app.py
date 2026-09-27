import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

# 1. Import spaces for Hugging Face ZeroGPU compatibility
try:
    import spaces
except ImportError:
    # Dummy fallback when running locally without spaces installed
    class spaces:
        @staticmethod
        def GPU(func):
            return func

import gradio as gr
import numpy as np
from src.backend.preprocessor import preprocess_image
from src.backend.predictor import predictor

# Loading model
if predictor.model is None:
    predictor.load()

# 2. Add @spaces.GPU decorator here!
@spaces.GPU
def classify_drawing(sketch):
    if sketch is None:
        return {}, None

    # Fix: Extract the composite image from the Gradio Sketchpad dict
    if isinstance(sketch, dict):
        img_data = sketch.get("composite") or sketch.get("image")
    else:
        img_data = sketch

    if img_data is None:
        return {}, None

    # Preprocess into (1, 28, 28)
    tensor = preprocess_image(img_data)

    # If canvas is completely blank
    if tensor.sum() == 0:
        return {"Draw a digit first": 1.0}, None

    # Run inference
    result = predictor.predict(tensor)
    confidences = {str(digit): float(prob) for digit, prob in result["class_probabilities"].items()}

    # Show the 28x28 processed image the model actually saw
    preview = (tensor[0] * 255).astype(np.uint8)

    return confidences, preview

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
            preview_img = gr.Image(label="What Model Sees (28x28 MNIST format)", image_mode="L", width=140, height=140)

    btn.click(
        fn=classify_drawing,
        inputs=[sketch],
        outputs=[label_output, preview_img]
    )

demo.launch(ssr_mode=False)