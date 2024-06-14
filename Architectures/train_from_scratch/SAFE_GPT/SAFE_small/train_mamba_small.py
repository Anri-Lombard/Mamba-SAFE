# train_mamba_small.py

import argparse
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from mamba_model import MambaLMHeadModel, MambaConfig

def main(args):
    # Load the MOSES dataset
    dataset = load_dataset("json", data_files=args.dataset_path, field="data")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Load the Mamba configuration
    with open(args.config_path, "r") as f:
        config_data = json.load(f)
    config = MambaConfig(**config_data)

    # Initialize the Mamba model
    model = MambaLMHeadModel(config)

    # Prepare the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.max_checkpoints,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
    )

    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the MOSES dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the trained model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the Mamba configuration file")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Name of the tokenizer to use")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for training")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save model checkpoints every N steps")
    parser.add_argument("--max_checkpoints", type=int, default=2, help="Maximum number of checkpoints to keep")
    args = parser.parse_args()

    main(args)