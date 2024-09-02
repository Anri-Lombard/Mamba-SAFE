from transformers import Trainer

import torch
import os

from typing import Optional

class SAFETrainer(Trainer):
    """
    Custom trainer for training SAFE model.

    This custom trainer changes the loss function to support the property head

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        # Remove 'labels' from inputs if present
        # This is to be compatible with new transformers versions
        if 'labels' in inputs:
            inputs.pop('labels')

        input_ids = inputs["input_ids"]
        outputs = model(input_ids)

        if isinstance(outputs, dict):
            lm_logits = outputs.get('logits', outputs.get('lm_logits'))
        elif isinstance(outputs, tuple):
            lm_logits = outputs[0]
        else:
            lm_logits = outputs

        if lm_logits is None:
            raise ValueError("Model output does not contain logits")

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()

        # Handle shared weights if necessary
        if 'mamba.lm_head.weight' in state_dict and 'mamba.backbone.embedding.weight' in state_dict:
            if torch.equal(state_dict['mamba.lm_head.weight'], state_dict['mamba.backbone.embedding.weight']):
                del state_dict['mamba.lm_head.weight']

        torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))

        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(output_dir)
