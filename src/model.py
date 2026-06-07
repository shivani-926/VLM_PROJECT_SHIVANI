import yaml
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class VLMModel:
    """
    Wrapper class that loads and manages BLIP models for
    both Image Captioning and Visual Question Answering.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.device = self.config["inference"]["device"]
        self.max_new_tokens = self.config["inference"]["max_new_tokens"]
        self.cache_dir = self.config["model"]["cache_dir"]

        # These are loaded lazily (only when first needed)
        self._caption_model = None
        self._caption_processor = None
        self._vqa_model = None
        self._vqa_processor = None

    def _load_captioning_model(self):
        """Download and load BLIP captioning model (runs once)."""
        if self._caption_model is None:
            print("Loading captioning model... (first run downloads ~990MB)")
            model_name = self.config["model"]["captioning"]
            self._caption_processor = BlipProcessor.from_pretrained(
                model_name, cache_dir=self.cache_dir
            )
            self._caption_model = BlipForConditionalGeneration.from_pretrained(
                model_name, cache_dir=self.cache_dir
            ).to(self.device)
            self._caption_model.eval()
            print("Captioning model loaded successfully.")

    def _load_vqa_model(self):
        """Download and load BLIP VQA model (runs once)."""
        if self._vqa_model is None:
            print("Loading VQA model... (first run downloads ~990MB)")
            model_name = self.config["model"]["vqa"]
            self._vqa_processor = BlipProcessor.from_pretrained(
                model_name, cache_dir=self.cache_dir
            )
            self._vqa_model = BlipForQuestionAnswering.from_pretrained(
                model_name, cache_dir=self.cache_dir
            ).to(self.device)
            self._vqa_model.eval()
            print("VQA model loaded successfully.")

    def generate_caption(self, pil_image) -> str:
        """Generate a natural language caption for the given PIL image."""
        self._load_captioning_model()
        inputs = self._caption_processor(
            images=pil_image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self._caption_model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        caption = self._caption_processor.decode(
            output[0], skip_special_tokens=True
        )
        return caption

    def answer_question(self, pil_image, question: str) -> str:
        """Answer a natural language question about the given PIL image."""
        self._load_vqa_model()
        inputs = self._vqa_processor(
            images=pil_image, text=question, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self._vqa_model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        answer = self._vqa_processor.decode(
            output[0], skip_special_tokens=True
        )
        return answer