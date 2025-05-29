"""
evaluate_domain.py
==================
This script demonstrates how to evaluate your domain-adapted model.
It loads the fine-tuned T5-small model (with the LoRA adapter) from disk,
then uses the model to perform inference on sample text prompts.
The goal is to qualitatively assess how well the model adapts to your domain.
"""

import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Configure logging.
logging.basicConfig(level=logging.INFO)


def load_finetuned_model(adapter_dir: str = "models/domain_adaptation_adapter", model_name: str = "t5-small"):
    """
    Loads the base model and attaches the fine-tuned LoRA adapter from disk.

    Args:
        adapter_dir (str): Directory containing the saved LoRA adapter.
        model_name (str): The base model name (default "t5-small").

    Returns:
        tuple: (tokenizer, model)
    """
    logging.info("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load the base model.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Load the adapter using the same API as 'from_pretrained'
    logging.info(f"Loading the LoRA adapter from '{adapter_dir}'...")
    model = model.from_pretrained(adapter_dir)
    logging.info("Model with LoRA adapter loaded successfully.")
    return tokenizer, model


def evaluate_sample_prompts():
    """
    Evaluates the finetuned model on a list of sample domain-specific inputs.
    Prints the input and the model's generated output.
    """
    tokenizer, model = load_finetuned_model()
    # Set up a text-to-text generation pipeline.
    gen_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    sample_prompts = [
        "Discuss the document management strategy in Minas Gerais.",
        "What are the key points of government financial facts?",
        "Explain the procedure from the SEPLAG instruction.",
    ]

    for prompt in sample_prompts:
        output = gen_pipeline(prompt, max_length=128)
        print("\nInput Prompt:", prompt)
        print("Model Output:", output[0]['generated_text'])


if __name__ == "__main__":
    evaluate_sample_prompts()
