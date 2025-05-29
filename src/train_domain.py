"""
train_domain.py
===============
This script fine-tunes the pre-trained T5-small model (with a LoRA adapter) on your
domain-specific dataset (prepared from your .txt and .pdf files). The training objective
is a simple text reconstruction task – the model is given a text chunk as input and is
trained to reproduce the same text as output. This helps the model adapt to the vocabulary,
stylistic features, and specific content of your documents.

Steps:
    1. Load the pre-trained T5-small model and its tokenizer.
    2. Integrate a LoRA adapter into the model.
    3. Load your domain-specific dataset (from Phase 4).
    4. Define a tokenization function that uses the "text" field from your dataset and sets
       both the input and the label to this text.
    5. Tokenize the dataset (using batched mapping) and remove original columns.
    6. Create a data collator for uniform padding.
    7. Configure the training parameters.
    8. Initialize a custom Trainer (subclassing the HF Trainer to allow passing label_names).
    9. Run the training loop.
   10. Save the trained LoRA adapter.
"""

import os
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from transformers.trainer import Trainer as HFTrainer
from peft import LoraConfig, get_peft_model
from data_preparation import prepare_document_dataset  # From Phase 4

# Configure logging.
logging.basicConfig(level=logging.INFO)


# -------------------------------
# Custom Trainer Subclass
# -------------------------------
class CustomTrainer(HFTrainer):
    """
    A subclass of Hugging Face's Trainer that accepts a `label_names` argument.
    This is necessary when working with PeftModel (which hides the base model's inputs)
    so that we can explicitly set the label names.
    """

    def __init__(self, *args, label_names=None, **kwargs):
        self.label_names = label_names if label_names is not None else []
        super().__init__(*args, **kwargs)


# -------------------------------
# Model and Adapter Setup
# -------------------------------
def load_base_model(model_name: str = "t5-small"):
    """
    Loads the pre-trained T5-small model and its tokenizer.

    Args:
        model_name (str): Hugging Face model identifier (default "t5-small").

    Returns:
        tuple: (tokenizer, model)
    """
    logging.info(f"Loading model and tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logging.info("Model and tokenizer loaded successfully.")
    return tokenizer, model


def add_lora_adapter(model, adapter_name: str, lora_r: int = 8, lora_alpha: int = 32,
                     lora_dropout: float = 0.1, target_modules: list = None):
    """
    Integrates a LoRA adapter into the model.

    Args:
        model: Pre-trained model.
        adapter_name (str): Identifier for the adapter.
        lora_r (int): Rank for LoRA.
        lora_alpha (int): Scaling factor.
        lora_dropout (float): Dropout rate for the LoRA layers.
        target_modules (list): List of module names to adapt (default: ["q", "v"]).

    Returns:
        Model with the integrated LoRA adapter.
    """
    if target_modules is None:
        target_modules = ["q", "v"]

    logging.info(f"Integrating LoRA adapter '{adapter_name}' with target modules: {target_modules}...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.config.lora_adapter = adapter_name
    logging.info("LoRA adapter integrated successfully.")
    return model


# -------------------------------
# Tokenization Function for Domain Data
# -------------------------------
def tokenize_domain_examples(examples, tokenizer):
    """
    Tokenizes a batch of domain examples. Since each sample consists of a single "text" field,
    we tokenize it and then set both the input and the label to that tokenized result.

    Args:
        examples (dict): Batch of examples with key "text".
        tokenizer: The tokenizer instance.

    Returns:
        dict: Dictionary with tokenized "input_ids", "attention_mask", and "labels".
    """
    # Tokenize the text (both for input and as labels).
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    # For autoencoding, use the same tokens as labels.
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# -------------------------------
# Main Training Function for Domain Data
# -------------------------------
def main():
    """
    Executes the domain-specific training process:
      1. Load model and tokenizer.
      2. Integrate the LoRA adapter.
      3. Load your custom domain dataset.
      4. Tokenize the dataset.
      5. Create a data collator.
      6. Set up training arguments.
      7. Initialize the CustomTrainer.
      8. Run training.
      9. Save the fine-tuned adapter.
    """
    # 1. Load the model and tokenizer.
    tokenizer, model = load_base_model(model_name="t5-small")

    # 2. Add the LoRA adapter.
    adapter_name = "domain_adaptation_adapter"
    model = add_lora_adapter(model, adapter_name=adapter_name)

    # 3. Load your domain-specific dataset (prepared from your .txt and .pdf files).
    dataset = prepare_document_dataset(max_chars=500)
    logging.info(f"Loaded domain dataset with {len(dataset)} samples.")

    # 4. Tokenize the dataset.
    logging.info("Tokenizing the domain dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_domain_examples(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    # 5. Create a data collator for uniform padding.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 6. Set up training arguments.
    training_args = TrainingArguments(
        output_dir="./results_domain",  # Directory to store domain training checkpoints.
        num_train_epochs=3,  # Adjust epochs based on your needs.
        per_device_train_batch_size=4,
        eval_strategy="no",  # No evaluation during training for this demo.
        logging_steps=10,
        save_steps=50,
        fp16=False,
    )

    # 7. Initialize the CustomTrainer with explicit label_names.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        label_names=["labels"],
    )

    # 8. Start training.
    logging.info("Starting domain-specific training with LoRA...")
    trainer.train()
    logging.info("Domain training completed successfully!")

    # 9. Save the fine-tuned LoRA adapter.
    output_adapter_dir = os.path.join("models", adapter_name)
    os.makedirs(output_adapter_dir, exist_ok=True)
    model.save_pretrained(output_adapter_dir)
    logging.info(f"Domain LoRA adapter saved to '{output_adapter_dir}'.")


if __name__ == "__main__":
    main()
