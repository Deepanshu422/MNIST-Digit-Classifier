import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import gradio as gr
import numpy as np
from src.backend.preprocessor import preprocess_image
from src.backend.predictor import predictor

def classify_drawing(sketch):
    if sketch is None:
        return {}, None

    # 1. Ensure model is loaded safely
    if predictor.model is None:
        predictor.load()

    # 2. Extract image from Gradio Sketchpad dictionary
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
        # 3. Preprocess into (1, 28, 28)
        tensor = preprocess_image(img_data)

        # 4. Check if canvas is completely blank
        if tensor.sum() == 0:
            return {"Draw a digit first": 1.0}, None

        # 5. Run inference
        result = predictor.predict(tensor)
        confidences = {
            str(digit): float(prob)
            for digit, prob in result["class_probabilities"].items()
        }

        # 6. Show the 28x28 processed image the model actually saw
        preview = (tensor[0] * 255).astype(np.uint8)

        return confidences, preview

    except Exception as e:
        # Fallback to display actual error on UI instead of generic red badge
        print(f"[!] Inference Error: {e}")
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

if __name__ == "__main__":
    demo.launch(ssr_mode=False)