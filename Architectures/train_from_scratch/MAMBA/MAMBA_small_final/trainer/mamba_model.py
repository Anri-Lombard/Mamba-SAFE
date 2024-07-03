import torch
import torch.nn as nn
from collections import namedtuple
from typing import Optional, Tuple, Union, Any

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from safe.trainer.model import PropertyHead

import os
import json

class MAMBAConfig(MambaConfig):
    def __init__(
        self,
        num_labels: int = None,
        vocab_size: int = None,
        pad_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

class MAMBADoubleHeadsModel(nn.Module):
    def __init__(self, config: MAMBAConfig, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        if tokenizer is not None:
            config.vocab_size = len(tokenizer)

        self.mamba = MambaLMHeadModel(config)
        self.property_head = PropertyHead(config)

        # Not sure if this is needed, but trying it
        if self.mamba.backbone.d_model != config.hidden_size:
            self.hidden_projection = nn.Linear(self.mamba.backbone.d_model, config.hidden_size)
        else:
            self.hidden_projection = nn.Identity()

    # A lot of these arguments are not used, but are kept for compatibility with the Trainer
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mc_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inputs: Optional[Any] = None, # the trainer might need these
        # encoder_hidden_states: Optional[torch.Tensor] = None, # TODO: needed?
    ):
        mamba_output = self.mamba(
            input_ids=input_ids,
            position_ids=position_ids, # Just to be compatible with Transformer generation
            inference_params=None, # for generation later
        )

        lm_logits = mamba_output.logits
        hidden_states = self.mamba.backbone(input_ids)
        # Not sure if this is right, not using lm_head from mamba_ssm
        # and not similar to safe-gpt
        hidden_states = self.hidden_projection(hidden_states) # Not sure if this is needed
        mc_logits = self.property_head(hidden_states)

        loss = None
        mc_loss = None
        if labels is not None:
            # Shift so tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous() # similar to safe-gpt
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Similar to safe-gpt
        if mc_labels is not None and getattr(self.config, "num_labels", 0) > 0:
            loss_fct = nn.MSELoss()
            mc_loss = loss_fct(
                mc_logits.view(-1, mc_logits.size(-1)),
                mc_labels.view(-1, mc_logits.size(-1)),
            )

        GPT2DoubleHeadsModelOutput = namedtuple('GPT2DoubleHeadsModelOutput',
            ['loss', 'mc_loss', 'logits', 'mc_logits', 'past_key_values', 'hidden_states', 'attentions']
        )

        return GPT2DoubleHeadsModelOutput(
            loss=loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=None, # MAMBA does not use attention, just for compatibility
            hidden_states=hidden_states,
            attentions=None, # MAMBA does not use attention, just for compatibility
        )

    # From mamba_ssm mixer_seq_simple.py
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config = MAMBAConfig.from_pretrained(pretrained_model_name, **kwargs)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(torch.load(pretrained_model_name, device=device, dtype=dtype))
        return model

    # From mamba_ssm mixer_seq_simple.py
    def save_pretrained(self, save_directory):
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)
