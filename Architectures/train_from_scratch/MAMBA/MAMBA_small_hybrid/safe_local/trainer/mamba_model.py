# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import namedtuple
# from typing import Optional, Tuple, Union, Any

# from safe_local.trainer.mixer_seq_simple import MambaLMHeadModel
# from safe_local.trainer.config_mamba import MambaConfig
# from mamba_ssm.utils.generation import GenerationMixin, decode
# from safe_local.trainer.model import PropertyHead
# from torch.nn import CrossEntropyLoss, MSELoss

import torch
import torch.nn as nn
from safe_local.trainer.mixer_seq_simple import MambaLMHeadModel, MixerModel
from mamba_ssm.utils.generation import GenerationMixin, decode
import os
import json


# from transformers import GenerationConfig

import os
import json
import copy

from collections import namedtuple

class MAMBAConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get('d_model', 2560)
        self.d_intermediate = kwargs.get('d_intermediate', 0)
        self.n_layer = kwargs.get('n_layer', 64)
        self.vocab_size = kwargs.get('vocab_size', 50277)
        self.ssm_cfg = kwargs.get('ssm_cfg', {})
        self.attn_layer_idx = kwargs.get('attn_layer_idx', [])
        self.attn_cfg = kwargs.get('attn_cfg', {})
        self.rms_norm = kwargs.get('rms_norm', True)
        self.residual_in_fp32 = kwargs.get('residual_in_fp32', True)
        self.fused_add_norm = kwargs.get('fused_add_norm', True)
        self.pad_vocab_size_multiple = kwargs.get('pad_vocab_size_multiple', 8)
        self.tie_embeddings = kwargs.get('tie_embeddings', True)
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)
        
        # Add any additional attributes that MambaLMHeadModel might expect
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        with open(os.path.join(pretrained_model_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.__dict__, f, indent=2)

    def to_dict(self):
        return self.__dict__


class MAMBAModel(nn.Module, GenerationMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mamba = MambaLMHeadModel(config)
        self._gradient_checkpointing = False

        self.backbone = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            vocab_size=config.vocab_size,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=None,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            dropout_rate=config.dropout_rate,
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._keys_to_ignore_on_save = []

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, position_ids=None, inference_params=None, labels=None, num_last_tokens=0, **mixer_kwargs):
        # Ignoring labels
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    # def generate(self, input_ids, max_length, **kwargs):
    #     input_ids = input_ids.to(self.device)
    #     return decode(
    #         input_ids,
    #         self,
    #         max_length,
    #         eos_token_id=self.config.eos_token_id,
    #         **kwargs
    #     )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._gradient_checkpointing = True
        if hasattr(self.mamba, 'gradient_checkpointing_enable'):
            self.mamba.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False
        if hasattr(self.mamba, 'gradient_checkpointing_disable'):
            self.mamba.gradient_checkpointing_disable()

    @classmethod
    def from_pretrained(cls, pretrained_model_path, device=None, dtype=None):
        config = MAMBAConfig.from_pretrained(pretrained_model_path)
        model = cls(config)
        state_dict = torch.load(os.path.join(pretrained_model_path, 'pytorch_model.bin'), map_location='cpu')
        
        # Remove 'mamba.' prefix from state_dict keys if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[6:] if k.startswith('mamba.') else k
            new_state_dict[new_k] = v
        
        # Handle potential key mismatches
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        
        model.load_state_dict(model_dict, strict=False)
        
        if device:
            model = model.to(device)
        if dtype:
            model = model.to(dtype=dtype)
        
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model's state dict
        state_dict = self.state_dict()
        if self.config.tie_embeddings:
            state_dict.pop('lm_head.weight', None)
        
        torch.save(state_dict, os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save the configuration
        self.config.save_pretrained(save_directory)

    def generate(self, input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        outputs = []
        for i in range(batch_size):
            output = decode(
                input_ids[i:i+1], 
                self, 
                kwargs.get('max_length', 100),
                top_k=kwargs.get('top_k', 0),
                top_p=kwargs.get('top_p', 1.0),
                temperature=kwargs.get('temperature', 1.0),
                repetition_penalty=kwargs.get('repetition_penalty', 1.0),
            )
            outputs.append(output.sequences)
        return torch.cat(outputs, dim=0)
