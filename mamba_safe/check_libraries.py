# test_torch.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# test_transformers.py
import transformers
print(f"Transformers version: {transformers.__version__}")

# test_mamba.py
import mamba_ssm
print(f"Mamba SSM version: {mamba_ssm.__version__}")

# test_mamba_model.py
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
config = MambaConfig(d_model=32, n_layer=2)
model = MambaLMHeadModel(config)
print("Successfully initialized Mamba model")