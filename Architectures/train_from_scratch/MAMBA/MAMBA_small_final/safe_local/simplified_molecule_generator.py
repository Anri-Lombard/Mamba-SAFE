import sys
import os
import torch
import argparse
import logging
from rdkit import Chem

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

trainer_dir = os.path.join(parent_dir, 'trainer')
sys.path.insert(0, trainer_dir)

from safe_local.tokenizer import SAFETokenizer
from safe_local.trainer.mamba_model import MAMBAModel
import safe_local as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_model(model_dir, tokenizer_path, device):
    mamba_model = MAMBAModel.from_pretrained(model_dir, device=device)
    safe_tokenizer = SAFETokenizer.from_pretrained(tokenizer_path)
    mamba_model.to(device)
    return mamba_model, safe_tokenizer

def generate_molecules(designer, n_samples, max_length, top_k, top_p, temperature):
    return designer.de_novo_generation(
        n_samples_per_trial=n_samples,
        max_length=max_length,
        sanitize=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        n_trials=1,
        repetition_penalty=1.0,
        max_retries=0
    )

def evaluate_model(model_dir, tokenizer_path, n_samples, max_length, top_k, top_p, temperature):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    mamba_model, safe_tokenizer = setup_model(model_dir, tokenizer_path, device)
    designer = sf.SAFEDesign(model=mamba_model, tokenizer=safe_tokenizer, verbose=True)

    generated_smiles = generate_molecules(designer, n_samples, max_length, top_k, top_p, temperature)

    print(f"Number of samples generated: {len(generated_smiles)}")
    print("\nFirst 10 generated SMILES:")
    for smi in generated_smiles[:10]:
        print(smi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate molecules using Mamba model")
    parser.add_argument("--model_dir", required=True, help="Directory containing the model")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated sequences")
    parser.add_argument("--top_k", type=int, default=15, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")

    args = parser.parse_args()

    evaluate_model(args.model_dir, args.tokenizer_path, args.num_samples, args.max_length, 
                   args.top_k, args.top_p, args.temperature)
