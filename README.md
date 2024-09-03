# Mamba-SAFE: Molecular Generation with Mamba and SAFE

Mamba-SAFE is a framework for generating molecules using the Mamba architecture and the SAFE (Structure-Agnostic Few-shot Encoding) representation (although any other representation could be used if needed). This library combines the power of the Mamba sequence modeling architecture with the versatility of the SAFE molecular representation.

## Features

- Generate molecules using the Mamba architecture
- Utilize the SAFE representation for molecular encoding

## Installation

### From PyPI

To install the latest stable version from PyPI:

```bash
pip install mamba-safe
```

### From Source

To install the latest development version from source:

```bash
git clone https://github.com/Anri-Lombard/Mamba-SAFE.git
cd Mamba-SAFE/mamba_safe
pip install -e .
```

**Note:** Make sure you have CUDA installed, as `mamba_ssm` requires it (https://github.com/state-spaces/mamba).

## Usage

### Generating Molecules

Here's a simple example of how to generate molecules using a trained Mamba-SAFE model:

```python
import torch
from mamba_safe import MAMBAModel, SAFETokenizer, SAFEDesign

# Set up your model and parameters
model_dir = "path/to/your/model"
tokenizer_path = "path/to/your/tokenizer"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer
mamba_model = MAMBAModel.from_pretrained(model_dir, device=device)
safe_tokenizer = SAFETokenizer.from_pretrained(tokenizer_path)

# Create designer
designer = SAFEDesign(model=mamba_model, tokenizer=safe_tokenizer, verbose=True)

# Generate molecules
generated_smiles = designer.de_novo_generation(
    n_samples_per_trial=100,
    max_length=50,
    sanitize=True,
    top_k=15,
    top_p=0.9,
    temperature=0.7,
    n_trials=10,
    repetition_penalty=1.0
)

# Print the first 10 generated SMILES
for smi in generated_smiles[:10]:
    print(smi)
```

### Training a Model from Scratch

To train a Mamba-SAFE model from scratch, you can use the `safe-train` CLI, adapted from https://safe-docs.datamol.io/stable/. Here's an example script:

```bash
#!/bin/bash

# Set up environment variables
export WANDB_API_KEY="your_wandb_api_key"

# Set up paths
config_path="example_config.json"
tokenizer_path="tokenizer.json"
dataset_path="/path/to/safe_zinc_dataset"
output_dir="/path/to/output"

# Run the training script
safe-train \
    --config_path $config_path \
    --tokenizer_path $tokenizer_path \
    --dataset_path $dataset_path \
    --text_column "safe" \
    --optim "adamw_torch" \
    --report_to "wandb" \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 100 \
    --per_device_eval_batch_size 100 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 10000 \
    --logging_first_step True \
    --save_steps 10000 \
    --eval_steps 10000 \
    --eval_accumulation_steps 1000 \
    --eval_strategy "steps" \
    --wandb_project "MAMBA_large" \
    --logging_steps 100 \
    --save_total_limit 1 \
    --output_dir $output_dir \
    --overwrite_output_dir True \
    --do_train True \
    --do_eval True \
    --save_safetensors True \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --max_steps 250000
```

Make sure to adjust the paths and parameters according to your specific setup and requirements.

## Important Notes

1. Do not install both `safe-mol` and `mamba-safe` in the same environment to avoid conflicts. Use `safe-mol` for transformer architectures and `mamba-safe` for Mamba-based models.

2. CUDA is required to run this package efficiently, as `mamba_ssm` relies on CUDA for optimal performance.

## Citation

If you use Mamba-SAFE in your research, please cite the following papers:

```bibtex
@article{noutahi2024gotta,
  title={Gotta be SAFE: a new framework for molecular design},
  author={Noutahi, Emmanuel and Gabellini, Cristian and Craig, Michael and Lim, Jonathan SC and Tossou, Prudencio},
  journal={Digital Discovery},
  volume={3},
  number={4},
  pages={796--804},
  year={2024},
  publisher={Royal Society of Chemistry}
}

@article{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

## Acknowledgments

We would like to express our sincere gratitude to:

- The SAFE authors for their pivotal work in sequence representation and molecular generation. Their contributions have been instrumental in the development of this library.
- The Mamba authors for their groundbreaking work in language model architectures. Their innovations have made this work possible.
- [SAFE](https://github.com/datamol-io/safe) for providing the molecular representation framework that forms the backbone of our approach.
- [Mamba](https://github.com/state-spaces/mamba) for developing the sequence modeling architecture that powers our models.

This library and the work it enables would not have been possible without their significant contributions to the field.
