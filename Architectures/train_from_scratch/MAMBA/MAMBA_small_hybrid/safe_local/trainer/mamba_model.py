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

    def generate(self, input_ids, max_length, **kwargs):
        input_ids = input_ids.to(self.device)
        return decode(
            input_ids,
            self,
            max_length,
            eos_token_id=self.config.eos_token_id,
            **kwargs
        )

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
        model.load_state_dict(state_dict)
        if device:
            model = model.to(device)
        if dtype:
            model = model.to(dtype=dtype)
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        self.config.save_pretrained(save_directory)

# class MAMBADoubleHeadsModel(nn.Module, GenerationMixin):
#     def __init__(self, config: MAMBAConfig, tokenizer=None):
#         super().__init__()
#         self.config = config
#         self.tokenizer = tokenizer

#         if tokenizer is not None:
#             config.vocab_size = len(tokenizer)

#         self.mamba = MambaLMHeadModel(config)
#         self.multiple_choice_head = PropertyHead(config)

#         # Determine the hidden size
#         if hasattr(self.mamba.backbone, 'd_model'):
#             hidden_size = self.mamba.backbone.d_model
#         elif hasattr(self.mamba.backbone, 'layers') and self.mamba.backbone.layers:
#             hidden_size = self.mamba.backbone.layers[0].mixer.d_model
#         else:
#             hidden_size = config.d_model  # Fallback to config

#         # Ensure config.hidden_size is set correctly
#         config.hidden_size = hidden_size

#         # Initialize hidden_projection only if needed
#         if hidden_size != config.hidden_size:
#             self.hidden_projection = nn.Linear(hidden_size, config.hidden_size)
#         else:
#             self.hidden_projection = nn.Identity()

#         self._gradient_checkpointing = False

#         print(f"Initialized MAMBA model with dropout rate {config.dropout_rate}")

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     def generate(
#         self,
#         input_ids,
#         max_length,
#         num_return_sequences=1,
#         return_dict_in_generate=False,
#         output_scores=True,
#         **kwargs
#     ):
#         # If no input_ids are provided, create a default input (start token)
#         if input_ids is None:
#             input_ids = torch.tensor([[self.config.bos_token_id]], device=self.device)
        
#         # Convert input_ids to the correct format if necessary
#         if isinstance(input_ids, dict):
#             input_ids = input_ids['input_ids']
        
#         # Ensure input_ids is on the correct device
#         input_ids = input_ids.to(self.device)

#         # If num_return_sequences > 1, we need to repeat the input_ids
#         if num_return_sequences > 1:
#             input_ids = input_ids.repeat(num_return_sequences, 1)

#         # print kwargs
#         print(kwargs)

#         # Remove num_return_sequences from kwargs to avoid passing it to decode()
#         kwargs.pop('num_return_sequences', None)

#         # Call the decode function from mamba_ssm.utils.generation
#         output = decode(
#             input_ids,
#             self,
#             max_length,
#             eos_token_id=self.config.eos_token_id,
#             **kwargs
#         )

#         if not output_scores:
#             output.scores = None

#         return output

#     def allocate_inference_cache(self, batch_size, max_seqlen=100, dtype=None, **kwargs):
#         return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

#     # A lot of these arguments are not used, but are kept for compatibility with the Trainer
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         input_embeds: Optional[torch.FloatTensor] = None,
#         mc_token_ids: Optional[torch.LongTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         mc_labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         inputs: Optional[Any] = None, # the trainer might need these
#         # encoder_hidden_states: Optional[torch.Tensor] = None, # TODO: needed?
#         **kwargs
#     ):
#         # input_ids = input_ids.to(self.device)

#         # TODO: might use later
#         # is_generating = kwargs.get('is_generating', False)

#         mamba_output = self.mamba(
#             input_ids=input_ids,
#             position_ids=position_ids, # Just to be compatible with Transformer generation
#             inference_params=None, # for generation later
#         )

#         lm_logits = mamba_output.logits
#         hidden_states = self.mamba.backbone(input_ids)

#         if mc_token_ids is None and self.config.pad_token_id is not None and input_ids is not None:
#             mc_token_ids = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(
#                 lm_logits.device
#             )

#         mc_loss = None
#         mc_logits = None
#         if mc_labels is not None and getattr(self.config, "num_labels", 0) > 0:
#             mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
#             mc_labels = mc_labels.to(mc_logits.device)
#             loss_fct = MSELoss()
#             mc_loss = loss_fct(
#                 mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1, mc_logits.size(-1))
#             )

#         lm_loss = None
#         if labels is not None:
#             # Shift so tokens < n predict n
#             labels = labels.to(lm_logits.device)
#             shift_logits = lm_logits[..., :-1, :].contiguous() # similar to safe-gpt
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = nn.CrossEntropyLoss()
#             lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


#         # TODO: change this later to mamba specific
#         GPT2DoubleHeadsModelOutput = namedtuple('GPT2DoubleHeadsModelOutput',
#             ['loss', 'mc_loss', 'logits', 'mc_logits', 'past_key_values', 'hidden_states', 'attentions']
#         )

#         return GPT2DoubleHeadsModelOutput(
#             loss=lm_loss,
#             mc_loss=mc_loss,
#             logits=lm_logits,
#             mc_logits=mc_logits,
#             past_key_values=None, # MAMBA does not use attention, just for compatibility
#             hidden_states=hidden_states,
#             attentions=None, # MAMBA does not use attention, just for compatibility
#         )
    
#     def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
#         self._gradient_checkpointing = True
#         if hasattr(self.mamba.backbone, 'gradient_checkpointing_enable'):
#             self.mamba.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

#     def gradient_checkpointing_disable(self):
#         self._gradient_checkpointing = False
#         if hasattr(self.mamba.backbone, 'gradient_checkpointing_disable'):
#             self.mamba.backbone.gradient_checkpointing_disable()

#     def get_input_embeddings(self):
#         return self.mamba.backbone.embedding

#     def set_input_embeddings(self, value):
#         self.mamba.backbone.embedding = value

#     def get_output_embeddings(self):
#         return self.mamba.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.mamba.lm_head = new_embeddings

#     def resize_token_embeddings(self, new_num_tokens):
#         old_embeddings = self.get_input_embeddings()
#         new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
#         self.set_input_embeddings(new_embeddings)

#         # Update the config
#         self.config.vocab_size = new_num_tokens

#         # If we're using tied embeddings, we need to resize the output embeddings as well
#         if self.get_output_embeddings() is not None:
#             old_lm_head = self.get_output_embeddings()
#             new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
#             self.set_output_embeddings(new_lm_head)

#         return self.get_input_embeddings()

#     def _get_resized_embeddings(self, old_embeddings, new_num_tokens):
#         if new_num_tokens == old_embeddings.num_embeddings:
#             return old_embeddings

#         # Build new embeddings
#         new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
#         new_embeddings.to(old_embeddings.weight.device)

#         # Copy token embeddings from the previous weights
#         num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
#         new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

#         return new_embeddings

#     def _get_resized_lm_head(self, old_lm_head, new_num_tokens):
#         if new_num_tokens == old_lm_head.out_features:
#             return old_lm_head

#         # Build new lm head
#         new_lm_head = nn.Linear(old_lm_head.in_features, new_num_tokens)
#         new_lm_head.to(old_lm_head.weight.device)

#         # Copy weights from the previous lm head
#         num_tokens_to_copy = min(old_lm_head.out_features, new_num_tokens)
#         new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
#         if old_lm_head.bias is not None:
#             new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

#         return new_lm_head

#     # From mamba_ssm mixer_seq_simple.py
#     @classmethod
#     def from_pretrained(cls, pretrained_model_path, device=None, dtype=None):
#         # Load the configuration
#         with open(os.path.join(pretrained_model_path, 'config.json'), 'r') as f:
#             config_dict = json.load(f)
#         config = MAMBAConfig(**config_dict)

#         # Create the model
#         model = cls(config)

#         # Load the model weights
#         state_dict = torch.load(os.path.join(pretrained_model_path, 'pytorch_model.bin'), map_location='cpu')

#         # Handle shared weights
#         if 'mamba.lm_head.weight' not in state_dict and 'mamba.backbone.embedding.weight' in state_dict:
#             state_dict['mamba.lm_head.weight'] = state_dict['mamba.backbone.embedding.weight']

#         model.load_state_dict(state_dict)

#         if device is not None:
#             model = model.to(device)

#         if dtype is not None:
#             model = model.to(dtype=dtype)

#         return model

#     # From mamba_ssm mixer_seq_simple.py
#     def save_pretrained(self, save_directory):
#         os.makedirs(save_directory, exist_ok=True)

#         # Save the model's state dict, handling shared tensors
#         state_dict = self.state_dict()
#         if torch.equal(state_dict['mamba.lm_head.weight'], state_dict['mamba.backbone.embedding.weight']):
#             del state_dict['mamba.lm_head.weight']

#         torch.save(state_dict, os.path.join(save_directory, 'pytorch_model.bin'))

#         # Save the configuration
#         config_dict = self.config.to_dict()
#         with open(os.path.join(save_directory, 'config.json'), 'w') as f:
#             json.dump(config_dict, f, indent=2)
