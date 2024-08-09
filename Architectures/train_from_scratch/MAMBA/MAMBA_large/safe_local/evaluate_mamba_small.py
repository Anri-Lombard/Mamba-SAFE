import sys
import os
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, Lipinski

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

trainer_dir = os.path.join(parent_dir, 'trainer')
sys.path.insert(0, trainer_dir)

from safe_local.tokenizer import SAFETokenizer
from safe_local.trainer.mamba_model import MAMBADoubleHeadsModel
import safe_local as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_model(model_dir, tokenizer_path, device):
    mamba_model = MAMBADoubleHeadsModel.from_pretrained(model_dir, device=device)
    safe_tokenizer = SAFETokenizer.from_pretrained(tokenizer_path)
    mamba_model.to(device)
    return mamba_model, safe_tokenizer

def generate_molecules(designer, n_samples=1000, max_length=100):
    return designer.de_novo_generation(
        n_samples_per_trial=n_samples,
        max_length=max_length,
        sanitize=True,
        top_k=20, # No filtering based on top-k
        top_p=0.9, # No filtering based on top-p
        temperature=0.7, # Just temperature sampling
        n_trials=1,
        repetition_penalty=1.0,
        max_retries=30
    )

def process_molecules(smiles_list):
    return [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]

def calculate_properties(mol_list):
    properties = {
        'MW': [], 'LogP': [], 'HBD': [], 'HBA': [], 'TPSA': [], 'RotBonds': [], 'QED': []
    }
    for mol in mol_list:
        properties['MW'].append(Descriptors.ExactMolWt(mol))
        properties['LogP'].append(Crippen.MolLogP(mol))
        properties['HBD'].append(Lipinski.NumHDonors(mol))
        properties['HBA'].append(Lipinski.NumHAcceptors(mol))
        properties['TPSA'].append(Descriptors.TPSA(mol))
        properties['RotBonds'].append(Lipinski.NumRotatableBonds(mol))
        properties['QED'].append(QED.qed(mol))
    return properties

def get_top_k_tokens(model, tokenizer, k=100):
    # Create input with just the BOS token
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        # Get probabilities for the next token after BOS
        probs = torch.softmax(logits[0, -1], dim=-1)
    
    top_k_probs, top_k_indices = torch.topk(probs, k)
    
    results = []
    for prob, index in zip(top_k_probs, top_k_indices):
        token = tokenizer.decode([index.item()])
        token_id = index.item()
        token_repr = token if token else '<empty>'  # Handle empty string tokens
        results.append((token, token_repr, token_id, prob.item()))
    
    return results

def evaluate_model(model_dir, tokenizer_path, n_samples=1000, max_length=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    mamba_model, safe_tokenizer = setup_model(model_dir, tokenizer_path, device)
    designer = sf.SAFEDesign(model=mamba_model, tokenizer=safe_tokenizer, verbose=True)

    generated_smiles = generate_molecules(designer, n_samples, max_length=max_length)
    generated_mols = process_molecules(generated_smiles)

    validity = len(generated_mols) / len(generated_smiles) if generated_smiles else 0
    uniqueness = len(set(Chem.MolToSmiles(mol) for mol in generated_mols)) / len(generated_smiles) if generated_smiles else 0

    properties = calculate_properties(generated_mols)

    print("\nFirst 10 generated SMILES:")
    for smi in generated_smiles:
        print(smi)

    print(f"Number of samples generated: {len(generated_smiles)}")
    print(f"Validity: {validity:.4f}")
    print(f"Uniqueness: {uniqueness:.4f}")
    print("\nProperty Summary Statistics:")
    for prop, values in properties.items():
        print(f"{prop}: {np.mean(values):.2f} ± {np.std(values):.2f}")

    # # Examine top-k tokens after 'C'
    # print("\nTop 100 tokens after 'C':")
    # top_tokens = get_top_k_tokens(mamba_model, safe_tokenizer, safe_tokenizer.bos_token_id, k=100)
    # top_tokens = get_top_k_tokens(mamba_model, safe_tokenizer, k=100)
    # print(f"The eos token id is: {safe_tokenizer.eos_token_id} and the token is: {safe_tokenizer.decode([safe_tokenizer.eos_token_id])}")
    # eos_found = False
    # for i, (token, token_repr, token_id, prob) in enumerate(top_tokens, 1):
    #     print(f"{i}. Token: '{token_repr}', ID: {token_id}, Probability: {prob:.4f}")
    #     if token_id == designer.model.config.eos_token_id:
    #         print(f"EOS token (ID: {token_id}) found at position {i}")
    #         eos_found = True
    # if not eos_found:
    #     print("EOS token not found in top 100 tokens")
    # else:
    #     print("EOS token found in top 100 tokens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mamba model for molecule generation")
    parser.add_argument("--model_dir", required=True, help="Directory containing the model")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated sequences")

    args = parser.parse_args()

    evaluate_model(args.model_dir, args.tokenizer_path, n_samples=args.num_samples, max_length=args.max_length)