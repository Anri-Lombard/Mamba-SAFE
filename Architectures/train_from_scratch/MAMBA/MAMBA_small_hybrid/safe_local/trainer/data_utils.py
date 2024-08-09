import itertools
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import datasets
import upath
from tqdm.auto import tqdm

from safe_local.tokenizer import SAFETokenizer


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))


def get_dataset_column_names(dataset: Union[datasets.Dataset, datasets.IterableDataset, Mapping]):
    """Get the column names in a dataset

    Args:
        dataset: dataset to get the column names from

    """
    if isinstance(dataset, (datasets.IterableDatasetDict, Mapping)):
        column_names = {split: dataset[split].column_names for split in dataset}
    else:
        column_names = dataset.column_names
    if isinstance(column_names, dict):
        column_names = list(column_names.values())[0]
    return column_names


def tokenize_fn(
    row: Dict[str, Any],
    tokenizer: Callable,
    tokenize_column: str = "inputs",
    max_length: Optional[int] = None,
    padding: bool = False,
):
    """Perform the tokenization of a row
    Args:
        row: row to tokenize
        tokenizer: tokenizer to use
        tokenize_column: column to tokenize
        max_length: maximum size of the tokenized sequence
        padding: whether to pad the sequence
    """
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast

    fast_tokenizer = (
        tokenizer.get_pretrained() if isinstance(tokenizer, SAFETokenizer) else tokenizer
    )

    return fast_tokenizer(
        row[tokenize_column],
        truncation=(max_length is not None),
        max_length=max_length,
        padding=padding,
        return_tensors=None,
    )


def batch_iterator(datasets, batch_size=100, n_examples=None, column="inputs"):
    if isinstance(datasets, Mapping):
        datasets = list(datasets.values())

    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]

    for dataset in datasets:
        iter_dataset = iter(dataset)
        if n_examples is not None and n_examples > 0:
            for _ in tqdm(range(0, n_examples, batch_size)):
                out = [next(iter_dataset)[column] for _ in range(batch_size)]
                yield out
        else:
            for out in tqdm(iter(partial(take, batch_size, iter_dataset), [])):
                yield [x[column] for x in out]


def get_dataset(
    data_path,
    name: Optional[str] = None,
    tokenizer: Optional[Callable] = None,
    cache_dir: Optional[str] = None,
    streaming: bool = True,
    tokenize_column: str = "text",
    max_length: Optional[int] = None,
    num_shards: int = 1024,
):
    """Get the dataset for Mamba model training"""
    if data_path is None:
        raise ValueError("data_path must be provided")

    data_path = upath.UPath(str(data_path))

    if data_path.exists():
        raw_datasets = datasets.load_from_disk(str(data_path))

        if streaming:
            if isinstance(raw_datasets, datasets.DatasetDict):
                raw_datasets = datasets.IterableDatasetDict({
                    k: dt.to_iterable_dataset(num_shards=num_shards)
                    for k, dt in raw_datasets.items()
                })
            else:
                raw_datasets = raw_datasets.to_iterable_dataset(num_shards=num_shards)
    else:
        raw_datasets = datasets.load_dataset(
            data_path,
            name=name,
            cache_dir=cache_dir,
            streaming=streaming,
        )

    raw_datasets = raw_datasets.map(
        partial(
            tokenize_fn,
            tokenizer=tokenizer,
            tokenize_column=tokenize_column,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=[col for col in raw_datasets["train"].column_names if col != tokenize_column]
    )

    return raw_datasets
