from src.preprocess import preprocess_image
from src.model import VLMModel

def answer_visual_question(image_path: str, question: str, model: VLMModel) -> dict:
    """
    Full VQA pipeline:
    Takes an image path + question → preprocesses → generates answer.
    Returns a result dictionary.
    """
    print(f"\nProcessing image: {image_path}")
    print(f"Question: {question}")
    pil_image = preprocess_image(image_path)
    answer = model.answer_question(pil_image, question)

    result = {
        "task": "vqa",
        "image_path": image_path,
        "question": question,
        "answer": answer
    }

    print(f"Answer: {answer}")
    return result