import argparse
import os
import sys

from config import MambaModelConfig
from mamba import MambaForTraining
from trainer import MambaTrainer
from nanotron.data.nanoset import Nanoset

from nanotron import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from run_train import get_dataloader  # noqa

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type="str", required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = MambaTrainer(config_file, model_config_class=MambaModelConfig, model_class=MambaForTraining)

    # Create and assign the nanoset to the trainer
    token_dtype = np.int32 if len(trainer.tokenizer) > np.iinfo(np.uint16).max + 1 else np.uint16
    trainer.nanoset = Nanoset(
        dataset_paths=trainer.config.data.dataset.dataset_path,
        dataset_weights=trainer.config.data.dataset.dataset_weights,
        sequence_length=trainer.sequence_length,
        token_dtype=token_dtype,
        train_split_num_samples=trainer.config.tokens.train_steps * trainer.global_batch_size,
        random_seed=trainer.config.data.seed,
    )

    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)