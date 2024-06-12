import math
from functools import partial
import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba, Block

class MambaLMHeadModel(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            Block(config.d_model, partial(Mamba, **config.ssm_cfg), layer_idx=i)
            for i in range(config.n_layer)
        ])
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids):
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states)
        hidden_states = self.norm_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits
