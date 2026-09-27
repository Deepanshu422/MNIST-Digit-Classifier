import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import gradio as gr
import numpy as np
from src.backend.preprocessor import preprocess_image
from src.backend.predictor import predictor

# Loading model
if predictor.model is None:
    predictor.load()

def classify_drawing(sketch):
    if sketch is None:
        return {}, None

    # Preprocess into (1, 28, 28)
    tensor = preprocess_image(sketch)

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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)