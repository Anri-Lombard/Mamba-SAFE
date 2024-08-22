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
from safe_local._exception import SAFEDecodeError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_model(model_dir, tokenizer_path, device, dtype):
    mamba_model = MAMBAModel.from_pretrained(model_dir, device=device)
    mamba_model = mamba_model.to(dtype=dtype)
    safe_tokenizer = SAFETokenizer.from_pretrained(tokenizer_path)
    mamba_model.to(device)
    return mamba_model, safe_tokenizer

def generate_molecules(designer, n_samples, max_length, top_k, top_p, temperature, n_trials, device, dtype, max_retries=10):
    all_generated_smiles = []
    for trial in range(n_trials):
        batch_smiles = []
        retry_count = 0
        while retry_count < max_retries:
            try:
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast(dtype=dtype):
                        batch_smiles = designer.de_novo_generation(
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
                else:
                    batch_smiles = designer.de_novo_generation(
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
                break  # If successful, break out of the retry loop
            except SAFEDecodeError as e:
                retry_count += 1
                logging.warning(f"SAFEDecodeError occurred (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count == max_retries:
                    logging.error(f"Failed to generate batch after {max_retries} attempts. Moving to next trial.")
                else:
                    logging.info("Retrying batch...")
        
        all_generated_smiles.extend(batch_smiles)
        logging.info(f"Completed trial {trial + 1}/{n_trials}. Total molecules generated: {len(all_generated_smiles)}")
    
    return all_generated_smiles

def save_molecules_to_file(molecules, output_file, append=False):
    mode = 'a' if append else 'w'
    with open(output_file, mode) as f:
        for molecule in molecules:
            f.write(f"{molecule}\n")
    logging.info(f"Generated molecules {'appended to' if append else 'saved to'} {output_file}")

def evaluate_model(model_dir, tokenizer_path, n_samples, max_length, top_k, top_p, temperature, n_trials, output_file, append, max_retries, device_type, dtype_str):
    device = torch.device(device_type)
    dtype = getattr(torch, dtype_str)
    logging.info(f"Using device: {device}, dtype: {dtype}")

    mamba_model, safe_tokenizer = setup_model(model_dir, tokenizer_path, device, dtype)
    designer = sf.SAFEDesign(model=mamba_model, tokenizer=safe_tokenizer, verbose=True)

    generated_smiles = generate_molecules(designer, n_samples, max_length, top_k, top_p, temperature, n_trials, device, dtype, max_retries)

    save_molecules_to_file(generated_smiles, output_file, append)

    print(f"Number of samples generated: {len(generated_smiles)}")
    print(f"\nFirst 10 generated SMILES ({'appended to' if append else 'saved to'} {output_file}):")
    for smi in generated_smiles[:10]:
        print(smi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate molecules using Mamba model")
    parser.add_argument("--model_dir", required=True, help="Directory containing the model")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate per trial")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated sequences")
    parser.add_argument("--top_k", type=int, default=15, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials for generation")
    parser.add_argument("--output_file", required=True, help="Path to save generated molecules")
    parser.add_argument("--append", action="store_true", help="Append to output file instead of overwriting")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries per batch")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"], help="Data type to use")

    args = parser.parse_args()

    evaluate_model(args.model_dir, args.tokenizer_path, args.num_samples, args.max_length, 
                   args.top_k, args.top_p, args.temperature, args.n_trials, args.output_file, 
                   args.append, args.max_retries, args.device, args.dtype)