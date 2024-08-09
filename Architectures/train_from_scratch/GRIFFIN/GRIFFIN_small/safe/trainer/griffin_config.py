# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Griffin configuration."""

import enum
import itertools
from typing import Any, Optional

from transformers import PretrainedConfig

@enum.unique
class TemporalBlockType(enum.Enum):
    """Type of temporal mixing to use in a residual block."""
    ATTENTION = enum.auto()
    RECURRENT = enum.auto()

@enum.unique
class ScanType(enum.Enum):
    """Which implementation to use for the scan in the RG-LRU."""
    AUTO = enum.auto()
    LINEAR_NATIVE = enum.auto()
    ASSOCIATIVE_NATIVE = enum.auto()
    LINEAR_PALLAS = enum.auto()

@enum.unique
class Preset(enum.Enum):
    """All default preset variants."""
    GRIFFIN_PAPER_7B = enum.auto()
    HAWK_PAPER_7B = enum.auto()
    RECURRENT_GEMMA_2B_V1 = enum.auto()
    RECURRENT_GEMMA_9B_V1 = enum.auto()

    @property
    def config_dict(self) -> dict[str, Any]:
        griffin_pattern = itertools.cycle([
            TemporalBlockType.RECURRENT,
            TemporalBlockType.RECURRENT,
            TemporalBlockType.ATTENTION,
        ])

        match self:
            case Preset.GRIFFIN_PAPER_7B:
                return dict(
                    width=4096,
                    mlp_expanded_width=3 * 4096,
                    num_heads=32,
                    lru_width=5632,
                    block_types=tuple(itertools.islice(griffin_pattern, 32)),
                    embeddings_scale_by_sqrt_dim=False,
                    attention_window_size=1024,
                    logits_soft_cap=0.0,
                    scan_type=ScanType.AUTO,
                )
            case Preset.HAWK_PAPER_7B:
                return dict(
                    width=4096,
                    mlp_expanded_width=3 * 4096,
                    num_heads=32,
                    lru_width=5632,
                    block_types=(TemporalBlockType.RECURRENT,) * 32,
                    embeddings_scale_by_sqrt_dim=False,
                    attention_window_size=1024,
                    logits_soft_cap=0.0,
                    scan_type=ScanType.AUTO,
                )
            case Preset.RECURRENT_GEMMA_2B_V1:
                return dict(
                    width=2560,
                    mlp_expanded_width=3 * 2560,
                    num_heads=10,
                    lru_width=2560,
                    block_types=tuple(itertools.islice(griffin_pattern, 26)),
                    embeddings_scale_by_sqrt_dim=True,
                    attention_window_size=2048,
                    logits_soft_cap=30.0,
                    scan_type=ScanType.AUTO,
                )
            case Preset.RECURRENT_GEMMA_9B_V1:
                return dict(
                    width=4096,
                    mlp_expanded_width=3 * 4096,
                    num_heads=16,
                    lru_width=4096,
                    block_types=tuple(itertools.islice(griffin_pattern, 38)),
                    embeddings_scale_by_sqrt_dim=True,
                    attention_window_size=2048,
                    logits_soft_cap=30.0,
                    scan_type=ScanType.AUTO,
                )

class GriffinConfig(PretrainedConfig):
    """Griffin config - https://arxiv.org/abs/2402.19427."""

    model_type = "griffin"

    def __init__(
        self,
        vocab_size: int = 32000,
        width: int = 4096,
        mlp_expanded_width: int = 12288,
        num_heads: int = 32,
        block_types: tuple[TemporalBlockType, ...] = (
            TemporalBlockType.RECURRENT,
            TemporalBlockType.RECURRENT,
            TemporalBlockType.ATTENTION,
        ) * 10,
        embeddings_scale_by_sqrt_dim: bool = False,
        attention_window_size: int = 1024,
        logits_soft_cap: float = 0.0,
        lru_width: Optional[int] = None,
        scan_type: ScanType = ScanType.AUTO,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.width = width
        self.mlp_expanded_width = mlp_expanded_width
        self.num_heads = num_heads
        self.block_types = block_types
        self.embeddings_scale_by_sqrt_dim = embeddings_scale_by_sqrt_dim
        self.attention_window_size = attention_window_size
        self.logits_soft_cap = logits_soft_cap
        self.lru_width = lru_width or width
        self.scan_type = scan_type

        super().__init__(**kwargs)

    @property
    def max_cache_length(self) -> int:
        """The maximum length of the cache used for the model."""
        return self.attention_window_size

    @property
    def num_layers(self) -> int:
        """The number of layers of the model."""
        return len(self.block_types)

    @classmethod
    def from_preset(
        cls,
        preset: Preset,
        vocab_size: int = 32000,
        max_sequence_length: Optional[int] = None,
    ) -> "GriffinConfig":
        """Creates a `GriffinConfig` for a given preset."""
        cls_kwargs = preset.config_dict
        if max_sequence_length is not None:
            w = min(cls_kwargs["attention_window_size"], max_sequence_length)
            cls_kwargs["attention_window_size"] = w

        return cls(vocab_size=vocab_size, **cls_kwargs)

def update_safe_config(safe_config, griffin_config: GriffinConfig):
    """
    Update the SAFE configuration with Griffin-specific parameters.
    
    Args:
        safe_config: The existing SAFE configuration object.
        griffin_config: The GriffinConfig object.
    
    Returns:
        Updated SAFE configuration object.
    """
    # Update existing parameters
    safe_config.vocab_size = griffin_config.vocab_size
    safe_config.n_embd = griffin_config.width
    safe_config.n_head = griffin_config.num_heads
    safe_config.n_layer = griffin_config.num_layers

    # Add Griffin-specific parameters
    safe_config.griffin_config = {
        "mlp_expanded_width": griffin_config.mlp_expanded_width,
        "block_types": [bt.value for bt in griffin_config.block_types],
        "embeddings_scale_by_sqrt_dim": griffin_config.embeddings_scale_by_sqrt_dim,
        "attention_window_size": griffin_config.attention_window_size,
        "logits_soft_cap": griffin_config.logits_soft_cap,
        "lru_width": griffin_config.lru_width,
        "scan_type": griffin_config.scan_type.value,
    }

    return safe_config