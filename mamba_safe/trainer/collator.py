import copy
import functools
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import torch
from tokenizers import Tokenizer
from transformers.data.data_collator import _torch_collate_batch

from safe_local.tokenizer import SAFETokenizer


class SAFECollator:

    def __init__(
        self,
        tokenizer,
        max_length: Optional[int] = None,
        input_key: str = "inputs",
        model_type: str = "mamba",
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_key = input_key
        self.model_type = model_type

    @functools.lru_cache()
    def get_tokenizer(self):
        """Get underlying tokenizer"""
        if isinstance(self.tokenizer, SAFETokenizer):
            return self.tokenizer.get_pretrained()
        return self.tokenizer

    def __call__(self, samples: List[Dict[str, Any]]):
        inputs = [example[self.input_key] for example in samples]
        tokenizer = self.get_tokenizer()

        # Use the tokenizer's encode_plus method instead of calling it directly
        batch = tokenizer.batch_encode_plus(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        if self.model_type == "mamba":
            batch.pop("attention_mask", None)
            batch.pop("token_type_ids", None)

        labels = batch["input_ids"].clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
