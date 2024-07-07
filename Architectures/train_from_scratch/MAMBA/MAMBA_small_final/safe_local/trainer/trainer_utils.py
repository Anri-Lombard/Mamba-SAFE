from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

import torch
import os

from typing import Optional

class SAFETrainer(Trainer):
    """
    Custom trainer for training SAFE model.

    This custom trainer changes the loss function to support the property head

    """

    def __init__(self, *args, prop_loss_coeff: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.prop_loss_coeff = prop_loss_coeff

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        labels = (
            inputs.pop("labels") if self.label_smoother is not None and "labels" in inputs else None
        )

        outputs = model(**inputs)

        if len(outputs) == 4:  # For MAMBADoubleHeadsModel
            lm_loss, mc_loss, lm_logits, mc_logits = outputs
        else:  # For SAFEDoubleHeadsModel
            lm_loss, mc_loss = outputs[:2]
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        mc_loss = outputs.get("mc_loss", None) if isinstance(outputs, dict) else outputs[1]
        if mc_loss is not None:
            loss = loss + self.prop_loss_coeff * mc_loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Handle shared weights
        if state_dict is None:
            state_dict = self.model.state_dict()

        # Remove duplicate weights
        if 'mamba.lm_head.weight' in state_dict and 'mamba.backbone.embedding.weight' in state_dict:
            if torch.equal(state_dict['mamba.lm_head.weight'], state_dict['mamba.backbone.embedding.weight']):
                del state_dict['mamba.lm_head.weight']

        # Save the state dict
        torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))

        # Save the config
        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(output_dir)
