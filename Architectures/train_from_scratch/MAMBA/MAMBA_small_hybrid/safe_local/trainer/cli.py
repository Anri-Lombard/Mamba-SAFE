import math
import os

import sys
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional
import json

import datasets
import evaluate
import torch
import transformers
from loguru import logger
from transformers import AutoTokenizer, TrainingArguments, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import log_levels as LOG_LEVELS

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # This should be the directory containing safe_local
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

try:
    import safe_local
    print("Successfully imported safe_local")
except ImportError as e:
    print(f"Failed to import safe_local: {e}")
    print("Contents of parent directory:")
    print(os.listdir(parent_dir))

from safe_local.tokenizer import SAFETokenizer
from safe_local.trainer.collator import SAFECollator
from safe_local.trainer.data_utils import get_dataset
from safe_local.trainer.trainer_utils import SAFETrainer

from mamba_model import MAMBAConfig, MAMBAModel


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_path: str = field(
        default=None, 
        metadata={"help": "Path to the Mamba config file"}
    )
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "Path to the tokenizer"},
    )
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length"},
    )
    wandb_project: Optional[str] = field(
        default="MAMBA_small",
        metadata={"help": "Name of the wandb project to use for logging"},
    )
    wandb_watch: Optional[Literal["gradients", "all"]] = field(
        default=None, 
        metadata={"help": "Whether to watch the model's parameters or gradients"}
    )


@dataclass
class DataArguments:
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset"},
    )
    text_column: str = field(
        default="text",
        metadata={"help": "Column in the dataset containing input text"},
    )
    max_train_samples: Optional[int] = field(
        default=None, 
        metadata={"help": "Maximum number of training samples to use"}
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use streaming mode for dataset"},
    )


def train(model_args, data_args, training_args):
    # Set up logging
    transformers.utils.logging.set_verbosity_info()

    # Load tokenizer
    tokenizer = SAFETokenizer.from_pretrained(model_args.tokenizer_path)

    # Initialize wandb if it's being used
    if "wandb" in training_args.report_to:
        import wandb
        wandb.init(project=model_args.wandb_project, name=training_args.run_name)

    # Load dataset
    dataset = get_dataset(
        data_args.dataset_path,
        tokenizer=tokenizer,
        streaming=data_args.streaming,
        tokenize_column=data_args.text_column,
        max_length=model_args.model_max_length,
    )

    if data_args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(data_args.max_train_samples))

    # Prepare data collator
    data_collator = SAFECollator(
        tokenizer=tokenizer,
        max_length=model_args.model_max_length,
        input_key=data_args.text_column,
        model_type="mamba",
    )

    # Load Mamba config and model
    with open(model_args.config_path, 'r') as f:
        config_dict = json.load(f)
    config = MAMBAConfig(**config_dict)
    
    if model_args.model_path:
        model = MAMBAModel.from_pretrained(model_args.model_path, config=config)
    else:
        model = MAMBAModel(config)

    # Set up trainer
    trainer = SAFETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if "wandb" in training_args.report_to:
            wandb.log({"eval/perplexity": perplexity})


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns=False

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(f"Checkpoint detected, resuming training at {last_checkpoint}")

    # Set up wandb logging
    if model_args.wandb_project:
        os.environ["WANDB_PROJECT"] = model_args.wandb_project
        training_args.report_to = ["wandb"]
    if model_args.wandb_watch:
        os.environ["WANDB_WATCH"] = model_args.wandb_watch
        if model_args.wandb_watch == "all":
            os.environ["WANDB_LOG_MODEL"] = "end"

    wandb_run_name = f"MAMBA_small_{uuid.uuid4().hex[:8]}"
    training_args.run_name = wandb_run_name

    set_seed(training_args.seed)

    train(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
