from src.preprocess import preprocess_image
from src.model import VLMModel

def caption_image(image_path: str, model: VLMModel) -> dict:
    """
    Full captioning pipeline:
    Takes an image path → preprocesses → generates caption.
    Returns a result dictionary.
    """
    print(f"\nProcessing image: {image_path}")
    pil_image = preprocess_image(image_path)
    caption = model.generate_caption(pil_image)

    result = {
        "task": "captioning",
        "image_path": image_path,
        "caption": caption
    }

    print(f"Generated Caption: {caption}")
    return result