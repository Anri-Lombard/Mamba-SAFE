from dataclasses import dataclass
from typing import Dict

@dataclass
class MambaConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    ssm_cfg: Dict = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load the configuration from a pretrained model
        # You can implement this method based on how you store and load pretrained models
        raise NotImplementedError("Loading pretrained models is not implemented yet.")

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "norm_epsilon": self.norm_epsilon,
            "initializer_range": self.initializer_range,
            "ssm_cfg": self.ssm_cfg,
        }
