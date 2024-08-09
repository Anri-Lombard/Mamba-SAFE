import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safe_local.tokenizer import SAFETokenizer
from safe_local.trainer.mamba_model import MAMBADoubleHeadsModel
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, QED, Crippen
import os
import safe_local as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from rdkit import DataStructs
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate molecules using a trained SAFE model")
    parser.add_argument("--model_dir", required=True, help="Directory containing the model")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated sequences")
    return parser.parse_args()

def load_model_and_tokenizer(model_dir, tokenizer_path):
    safe_model = MAMBADoubleHeadsModel.from_pretrained(model_dir)
    safe_tokenizer = SAFETokenizer.from_pretrained(tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_model.to(device)
    return safe_model, safe_tokenizer

def generate_molecules(model, tokenizer, num_samples, max_length):
    designer = sf.SAFEDesign(
        model=model,
        tokenizer=tokenizer,
        verbose=True,
    )
    generated_smiles = []
    with tqdm(total=num_samples, desc="Generating molecules", unit="molecule") as pbar:
        while len(generated_smiles) < num_samples:
            batch = designer.de_novo_generation(
                sanitize=True,
                n_samples_per_trial=min(1000, num_samples - len(generated_smiles)),
                early_stopping=False,
                max_length=max_length
            )
            generated_smiles.extend(batch)
            pbar.update(len(batch))
    return generated_smiles[:num_samples]

def process_molecules(smiles_list, desc):
    mols = []
    for smi in tqdm(smiles_list, desc=desc, unit="molecule"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
    return mols

def calculate_fingerprints(mols, radius=2, nBits=2048):
    fingerprints = []
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits)
    
    for mol in tqdm(mols, desc="Calculating fingerprints", unit="molecule"):
        fp = morgan_gen.GetFingerprint(mol)
        fingerprints.append(fp)
    return fingerprints

def calculate_diversity(fingerprints):
    print("Calculating pairwise diversities...")
    n = len(fingerprints)
    diversity = 0
    total_pairs = (n * (n - 1)) // 2
    
    with tqdm(total=total_pairs, desc="Calculating diversity", unit="pair") as pbar:
        for i in range(n):
            diversity += sum(1 - DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j]) 
                             for j in range(i + 1, n))
            pbar.update(n - i - 1)
    
    return diversity / total_pairs if n > 1 else 0

def calculate_properties(mol_list):
    properties = {
        'MW': [], 'LogP': [], 'HBD': [], 'HBA': [], 'TPSA': [], 'RotBonds': [], 'QED': []
    }
    
    for mol in tqdm(mol_list, desc="Calculating properties", unit="molecule"):
        properties['MW'].append(Descriptors.ExactMolWt(mol))
        properties['LogP'].append(Crippen.MolLogP(mol))
        properties['HBD'].append(Lipinski.NumHDonors(mol))
        properties['HBA'].append(Lipinski.NumHAcceptors(mol))
        properties['TPSA'].append(Descriptors.TPSA(mol))
        properties['RotBonds'].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        properties['QED'].append(Descriptors.qed(mol))
    
    return properties

def plot_property_distributions(gen_props, property_name, xlabel, ylabel="Density", plot_type='line', output_dir=''):
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'line':
        sns.kdeplot(gen_props, label='Generated', color='blue', fill=True)
    elif plot_type == 'bar':
        gen_counts, gen_bins = np.histogram(gen_props, bins=range(min(gen_props), max(gen_props) + 2))
        gen_counts = gen_counts / len(gen_props)
        width = 0.35
        plt.bar(gen_bins[:-1], gen_counts, width, label='Generated', alpha=0.7, color='blue')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Distribution of {property_name}')
    plt.legend()
    filename = f'{property_name.lower().replace(" ", "_")}_distribution.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def save_molecule_images(mols, output_dir, num_to_save=10):
    print(f"Saving {num_to_save} molecule images...")
    img_dir = os.path.join(output_dir, "molecule_images")
    os.makedirs(img_dir, exist_ok=True)
    
    for i, mol in enumerate(tqdm(mols[:num_to_save], desc="Saving molecule images", unit="image")):
        img = Draw.MolToImage(mol)
        img.save(os.path.join(img_dir, f"molecule_{i+1}.png"))

def analyze_scaffolds(mol_list):
    scaffolds = {}
    for mol in tqdm(mol_list, desc="Analyzing scaffolds", unit="molecule"):
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        scaffolds[scaffold_smiles] = scaffolds.get(scaffold_smiles, 0) + 1
    return scaffolds

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    safe_model, safe_tokenizer = load_model_and_tokenizer(args.model_dir, args.tokenizer_path)
    
    # Generate molecules
    generated_smiles = generate_molecules(safe_model, safe_tokenizer, args.num_samples, args.max_length)
    
    # Process molecules
    valid_mols = process_molecules(generated_smiles, "Processing generated molecules")
    
    # Save molecule images
    save_molecule_images(valid_mols, args.output_dir)
    
    # Calculate fingerprints
    gen_fps = calculate_fingerprints(valid_mols)
    
    # Calculate basic properties
    print("Calculating basic properties...")
    unique_smiles = set(Chem.MolToSmiles(mol) for mol in valid_mols)
    qed_scores = [QED.qed(mol) for mol in tqdm(valid_mols, desc="Calculating QED scores", unit="molecule")]
    sas_scores = [sascorer.calculateScore(mol) for mol in tqdm(valid_mols, desc="Calculating SAS scores", unit="molecule")]
    
    validity = len(valid_mols) / len(generated_smiles) if generated_smiles else 0
    uniqueness = len(unique_smiles) / len(generated_smiles) if generated_smiles else 0
    diversity = calculate_diversity(gen_fps)
    
    qed_mean = sum(qed_scores) / len(qed_scores) if qed_scores else 0
    sas_mean = sum(sas_scores) / len(sas_scores) if sas_scores else 0
    
    # Print and save results
    results = f"""
    Validity: {validity:.3f}
    Uniqueness: {uniqueness:.3f}
    Diversity: {diversity:.3f}
    QED mean: {qed_mean:.3f}
    SAS mean: {sas_mean:.3f}
    """
    print(results)
    
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(results)
    
    # Calculate and plot property distributions
    properties = calculate_properties(valid_mols)
    properties_to_plot = [
        ('MW', 'Molecular Weight', 'Molecular Weight (Da)', 'line'),
        ('LogP', 'LogP', 'LogP', 'line'),
        ('HBD', 'H-Bond Donors', 'Number of H-Bond Donors', 'bar'),
        ('HBA', 'H-Bond Acceptors', 'Number of H-Bond Acceptors', 'bar'),
        ('TPSA', 'Topological Polar Surface Area', 'TPSA (Å²)', 'line'),
        ('RotBonds', 'Rotatable Bonds', 'Number of Rotatable Bonds', 'bar'),
        ('QED', 'QED', 'Quantitative Estimate of Drug-likeness', 'line')
    ]
    
    print("Plotting property distributions...")
    for prop, name, xlabel, plot_type in tqdm(properties_to_plot, desc="Plotting distributions", unit="plot"):
        plot_property_distributions(properties[prop], name, xlabel, plot_type=plot_type, output_dir=args.output_dir)
    
    # Scaffold analysis
    scaffolds = analyze_scaffolds(valid_mols)
    scaffold_results = f"""
    Unique scaffolds in generated set: {len(scaffolds)}
    Scaffold diversity: {len(scaffolds) / len(valid_mols):.4f}
    """
    print(scaffold_results)
    
    with open(os.path.join(args.output_dir, "scaffold_results.txt"), "w") as f:
        f.write(scaffold_results)
    
    # Save generated SMILES
    print("Saving generated SMILES...")
    with open(os.path.join(args.output_dir, "generated_smiles.txt"), "w") as f:
        for smiles in tqdm(generated_smiles, desc="Saving SMILES", unit="molecule"):
            f.write(smiles + "\n")

    print("All operations completed successfully.")

if __name__ == "__main__":
    main()