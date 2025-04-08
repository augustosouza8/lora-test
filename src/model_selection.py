"""
model_selection.py
==================
This script demonstrates the selection and loading of a pre-trained model from Hugging Face Hub.
We are using "t5-small" as the base model, which is lightweight and versatile enough for further
fine-tuning with LoRA.

Implementation Decisions:
- Chose T5-small for resource efficiency and its text-to-text framework.
- Used AutoTokenizer and AutoModelForSeq2SeqLM for flexibility and ease-of-use.
- The script prints a simple confirmation message upon successfully loading the model and tokenizer.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_name: str = "t5-small"):
    """
    Loads the specified pre-trained model and tokenizer.

    Args:
        model_name (str): The Hugging Face model identifier.
                          Default is "t5-small".

    Returns:
        tuple: (tokenizer, model) loaded from the Hugging Face Hub.
    """
    try:
        print(f"Loading the tokenizer and model from '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Model and tokenizer loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading the model: {e}")
        raise


if __name__ == "__main__":
    load_model()
