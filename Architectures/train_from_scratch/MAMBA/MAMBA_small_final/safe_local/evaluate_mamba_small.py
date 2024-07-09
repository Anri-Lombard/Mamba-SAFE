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
        top_k=15,
        top_p=0.9,
        temperature=0.8,
        n_trials=1,
        repetition_penalty=1.0,
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

def get_top_k_tokens(model, tokenizer, input_text, k=50):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        probs = torch.softmax(logits[0, -1], dim=-1)
    
    top_k_probs, top_k_indices = torch.topk(probs, k)
    
    results = []
    for prob, index in zip(top_k_probs, top_k_indices):
        token = tokenizer.decode([index])
        results.append((token, prob.item()))
    
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

    print(f"Number of samples generated: {len(generated_smiles)}")
    print(f"Validity: {validity:.4f}")
    print(f"Uniqueness: {uniqueness:.4f}")
    print("\nProperty Summary Statistics:")
    for prop, values in properties.items():
        print(f"{prop}: {np.mean(values):.2f} ± {np.std(values):.2f}")

    print("\nFirst 10 generated SMILES:")
    for smi in generated_smiles[:10]:
        print(smi)

    # Examine top-k tokens after 'C'
    print("\nTop 50 tokens after 'C':")
    top_tokens = get_top_k_tokens(mamba_model, safe_tokenizer, 'C', k=50)
    for token, prob in top_tokens:
        print(f"Token: '{token}', Probability: {prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mamba model for molecule generation")
    parser.add_argument("--model_dir", required=True, help="Directory containing the model")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated sequences")

    args = parser.parse_args()

    evaluate_model(args.model_dir, args.tokenizer_path, n_samples=args.num_samples, max_length=args.max_length)