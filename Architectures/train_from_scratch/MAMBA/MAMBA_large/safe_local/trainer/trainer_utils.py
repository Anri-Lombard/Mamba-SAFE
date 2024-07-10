from transformers import Trainer, get_scheduler
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

import torch
import os

from typing import Optional, Dict, Any, Union

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

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            self.clip_gradients(model, self.args.max_grad_norm)

        return loss.detach()

    def clip_gradients(self, model, max_grad_norm):
        if hasattr(self.optimizer, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.optimizer.clip_grad_norm(max_grad_norm)
        elif hasattr(model, "clip_grad_norm_"):
            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
            model.clip_grad_norm_(max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling both nn.DataParallel and non-parallel
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

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

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(torch.utils.data.DataLoader(train_dataset, **dataloader_params))

    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._save(self.args.output_dir)
        return control
