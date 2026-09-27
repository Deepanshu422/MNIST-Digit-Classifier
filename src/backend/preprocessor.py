import io
import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image_input) -> np.ndarray:
    """
    Production-grade preprocessor for Gradio Sketchpad:
    1. Extracts the drawn stroke directly.
    2. Crops to bounding box.
    3. Centers and fits inside a 20x20 area inside a standard 28x28 MNIST canvas.
    """

    raw_img = image_input
    if isinstance(image_input, dict):
        if "composite" in image_input and image_input["composite"] is not None:
            raw_img = image_input["composite"]
        elif "layers" in image_input and len(image_input["layers"]) > 0:
            raw_img = image_input["layers"][0]
        else:
            raw_img = image_input.get("background")

    # converting to PIL Image
    if isinstance(raw_img, bytes):
        pil_img = Image.open(io.BytesIO(raw_img))
    elif isinstance(raw_img, np.ndarray):
        pil_img = Image.fromarray(raw_img.astype("uint8"))
    elif isinstance(raw_img, Image.Image):
        pil_img = raw_img
    else:
        raise ValueError(f"Unsupported image type: {type(raw_img)}")

    # converting to RGBA to inspect transparent or white canvases
    pil_img = pil_img.convert("RGBA")
    r, g, b, a = pil_img.split()

    # In Gradio, drawing is typically black strokes on white canvas,
    # or black strokes on transparent canvas.
    # We want: 0 = Background, 255 = Digit Stroke.
    gray = pil_img.convert("L")
    gray_arr = np.array(gray)
    alpha_arr = np.array(a)

    # If alpha channel is active (transparent canvas with strokes)
    if np.any(alpha_arr > 0) and not np.all(alpha_arr == 255):
        # Stroke pixels are where alpha > 0
        stroke_mask = (alpha_arr > 50).astype("uint8") * 255
    else:
        # Standard white canvas with dark ink:
        # Stroke pixels are where lightness is low (dark)
        stroke_mask = (gray_arr < 180).astype("uint8") * 255

    stroke_img = Image.fromarray(stroke_mask, mode="L")

    # 4. Find bounding box of the drawn digit
    bbox = stroke_img.getbbox()
    if not bbox:
        # Empty canvas - return blank 28x28
        return np.zeros((1, 28, 28), dtype="float32")

    # crop to stroke
    digit = stroke_img.crop(bbox)

    # fitting inside 20x20 keeping aspect ratio (MNIST specification)
    w, h = digit.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))

    digit = digit.resize((new_w, new_h), Image.Resampling.BICUBIC)

    # paste centered into a 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(digit, (paste_x, paste_y))

    # normalizing to [0.0, 1.0] and add batch dimension (1, 28, 28)
    final_arr = np.array(canvas, dtype="float32") / 255.0
    return np.expand_dims(final_arr, axis=0)