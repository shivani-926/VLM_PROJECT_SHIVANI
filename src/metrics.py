import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Ensure necessary tokenizers are silently downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class VLMEvaluator:
    """
    A unified evaluation class for calculating standard NLP metrics (BLEU and ROUGE)
    to assess the quality of generated image captions and VQA answers.
    """
    def __init__(self):
        # Initialize ROUGE scorer for Unigram (rouge1), Bigram (rouge2), and Longest Common Subsequence (rougeL)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize a smoothing function for BLEU to prevent scores of 0 when higher-order n-grams are missing
        self.smoothie = SmoothingFunction().method1

    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """
        Calculates the BLEU-4 score between a reference string and the model's output.
        
        Parameters:
        - reference (str): The ground-truth human-annotated text.
        - candidate (str): The generated text from the VLM.
        
        Returns:
        - float: The computed BLEU score ranging from 0.0 to 1.0.
        """
        # BLEU expects a list of reference token lists, and a single candidate token list
        ref_tokens = [reference.lower().split()]
        cand_tokens = candidate.lower().split()
        
        return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=self.smoothie)

    def calculate_rouge(self, reference: str, candidate: str) -> dict:
        """
        Calculates the ROUGE metrics between a reference string and the model's output.
        
        Parameters:
        - reference (str): The ground-truth human-annotated text.
        - candidate (str): The generated text from the VLM.
        
        Returns:
        - dict: A dictionary containing precision, recall, and fmeasure for ROUGE-1, ROUGE-2, and ROUGE-L.
        """
        return self.scorer.score(reference, candidate)

if __name__ == "__main__":
    # --- DUMMY EVALUATION PIPELINE ---
    print("Initializing VLM Evaluator...\n")
    evaluator = VLMEvaluator()

    # Mock Data: Simulating a scenario where the model describes an image
    ground_truth_caption = "A city skyline beautifully lit up at night with reflections on the water."
    model_generated_caption = "A city lit up at night reflecting in the river."

    print(f"Reference: '{ground_truth_caption}'")
    print(f"Generated: '{model_generated_caption}'\n")

    # 1. Calculate BLEU
    bleu_score = evaluator.calculate_bleu(ground_truth_caption, model_generated_caption)
    print(f"BLEU Score: {bleu_score:.4f}")

    # 2. Calculate ROUGE
    rouge_scores = evaluator.calculate_rouge(ground_truth_caption, model_generated_caption)
    print("\nROUGE Scores:")
    print(f"ROUGE-1 F1: {rouge_scores['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2 F1: {rouge_scores['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L F1: {rouge_scores['rougeL'].fmeasure:.4f}")