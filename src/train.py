"""
train.py
========
This script demonstrates how to configure a LoRA adapter for a pre-trained model
and set up a basic training pipeline using Hugging Face's Transformers and a custom Trainer.
We subclass the Trainer to accept a `label_names` argument in order to suppress warnings
from PeftModel regarding hidden base model input arguments.

Key Steps:
    - Load a lightweight base model (T5-small) and its tokenizer.
    - Integrate a LoRA adapter into the model.
    - Prepare and tokenize a sample dataset (subset of GLUE SST2).
      The tokenization function converts each example’s "sentence" into input tokens and converts
      its numeric label (0 or 1) into target text ("negative" or "positive") which is also tokenized.
    - Use a DataCollatorForSeq2Seq to pad sequences uniformly.
    - Use a custom Trainer that accepts `label_names` to set up training.
    - Run training and save the LoRA adapter.
"""

import os
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from transformers.trainer import Trainer as HFTrainer  # Base Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Configure logging for clear console output.
logging.basicConfig(level=logging.INFO)


# -------------------------------
# Custom Trainer Subclass
# -------------------------------
class CustomTrainer(HFTrainer):
    """
    A subclass of Hugging Face Trainer that accepts a 'label_names' keyword argument.
    This is used to assign label names to the Trainer (needed by PeftModel)
    without causing an unexpected keyword argument error.
    """

    def __init__(self, *args, label_names=None, **kwargs):
        # Save label_names if provided; default to empty list.
        self.label_names = label_names if label_names is not None else []
        super().__init__(*args, **kwargs)


# -------------------------------
# Model Loading and Adapter Setup
# -------------------------------
def load_base_model(model_name: str = "t5-small"):
    """
    Loads the pre-trained model and tokenizer.

    Args:
        model_name (str): Hugging Face model identifier; default "t5-small".

    Returns:
        tuple: (tokenizer, model) loaded from the Hugging Face Hub.
    """
    logging.info(f"Loading model and tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logging.info("Model and tokenizer loaded successfully.")
    return tokenizer, model


def add_lora_adapter(model, adapter_name: str, lora_r: int = 8, lora_alpha: int = 32,
                     lora_dropout: float = 0.1, target_modules: list = None):
    """
    Integrates a LoRA adapter into the model using PEFT.

    Args:
        model: Pre-trained model.
        adapter_name (str): Domain-specific identifier.
        lora_r (int): LoRA rank.
        lora_alpha (int): Scaling factor.
        lora_dropout (float): Dropout probability.
        target_modules (list): Names of model submodules to adapt (default: ["q", "v"]).

    Returns:
        Model with integrated LoRA adapter.
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
    # Optionally store the adapter name in the model's configuration.
    model.config.lora_adapter = adapter_name
    logging.info("LoRA adapter integrated successfully.")
    return model


# -------------------------------
# Tokenization and Dataset Preparation
# -------------------------------
def tokenize_function(examples, tokenizer):
    """
    Tokenizes a batch of examples.
    - Tokenizes the 'sentence' field.
    - Converts the numeric 'label' to text ("positive" if 1, else "negative")
      and tokenizes it as the target.
    Both inputs and targets are padded to a maximum length (128 tokens).

    Args:
        examples (dict): Batch of examples with keys "sentence" and "label".
        tokenizer: Tokenizer for the base model.

    Returns:
        dict: Tokenized inputs including "labels".
    """
    inputs = tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    # Convert numeric labels to text.
    label_texts = ["positive" if label == 1 else "negative" for label in examples["label"]]
    targets = tokenizer(
        label_texts,
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


def prepare_dummy_dataset():
    """
    Loads a small subset of the GLUE SST2 dataset (100 examples) for demonstration.

    Returns:
        Dataset: Loaded Hugging Face dataset.
    """
    logging.info("Loading a sample dataset for demonstration...")
    dataset = load_dataset("glue", "sst2", split="train[:100]")
    logging.info(f"Sample dataset loaded with {len(dataset)} examples.")
    return dataset


# -------------------------------
# Main Training Function
# -------------------------------
def main():
    """
    Executes the training process:
      1. Load model and tokenizer.
      2. Add LoRA adapter.
      3. Load and tokenize dataset.
      4. Set up data collator.
      5. Configure training arguments.
      6. Initialize CustomTrainer with label_names.
      7. Run training and save adapter.
    """
    # 1. Load the model and tokenizer.
    tokenizer, model = load_base_model(model_name="t5-small")

    # 2. Integrate the LoRA adapter.
    adapter_name = "domain_specific_adapter"
    model = add_lora_adapter(model, adapter_name=adapter_name)

    # 3. Load a dummy dataset.
    dataset = prepare_dummy_dataset()

    # 4. Tokenize the dataset using batched mapping and remove original columns.
    logging.info("Tokenizing the dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    # 5. Create a data collator for uniform padding.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 6. Configure training parameters.
    training_args = TrainingArguments(
        output_dir="./results",  # Directory for checkpoints.
        num_train_epochs=1,  # Single epoch for demo.
        per_device_train_batch_size=4,  # Batch size.
        eval_strategy="no",  # Disable evaluation.
        logging_steps=10,
        save_steps=50,
        fp16=False,
    )

    # 7. Initialize the CustomTrainer with label_names.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        label_names=["labels"],
    )

    # 8. Start training.
    logging.info("Starting training process with LoRA...")
    trainer.train()
    logging.info("Training completed successfully!")

    # 9. Save the adapter.
    output_adapter_dir = os.path.join("models", adapter_name)
    os.makedirs(output_adapter_dir, exist_ok=True)
    model.save_pretrained(output_adapter_dir)
    logging.info(f"LoRA adapter saved to '{output_adapter_dir}'.")


if __name__ == "__main__":
    main()
