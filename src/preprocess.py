import cv2
import numpy as np
from PIL import Image

def load_image_opencv(image_path: str) -> np.ndarray:
    """Load image using OpenCV and convert BGR to RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def resize_image(img: np.ndarray, width: int = 384, height: int = 384) -> np.ndarray:
    """Resize image to target dimensions expected by BLIP."""
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert a NumPy array (OpenCV format) to a PIL Image for Hugging Face."""
    return Image.fromarray(img)

def preprocess_image(image_path: str) -> Image.Image:
    """Full pipeline: load → resize → return PIL image ready for BLIP."""
    img = load_image_opencv(image_path)
    img = resize_image(img)
    pil_image = numpy_to_pil(img)
    return pil_image