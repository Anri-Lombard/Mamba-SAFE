import sys
import os
import torch
import argparse
import traceback
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import pickle
import hashlib
from tabulate import tabulate

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

trainer_dir = os.path.join(parent_dir, 'trainer')
sys.path.insert(0, trainer_dir)

from safe_local.tokenizer import SAFETokenizer
from safe_local.trainer.mamba_model import MAMBADoubleHeadsModel
import safe_local as sf
import datamol as dm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, rdMolDescriptors, Lipinski, AllChem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_model(model_dir, tokenizer_path, device):
    logging.info(f"Loading model from {model_dir}")
    mamba_model = MAMBADoubleHeadsModel.from_pretrained(model_dir, device=device)
    logging.info(f"Loading tokenizer from {tokenizer_path}")
    safe_tokenizer = SAFETokenizer.from_pretrained(tokenizer_path)
    mamba_model.to(device)
    return mamba_model, safe_tokenizer

def generate_molecules(designer, n_samples=10000, max_length=100):
    logging.info(f"Generating {n_samples} molecules")
    generated_smiles = designer.de_novo_generation(
        n_samples_per_trial=n_samples,
        max_length=max_length,
        sanitize=True,
        top_k=15,
        top_p=0.9,
        temperature=0.8,
        n_trials=1,
        repetition_penalty=1.0,
    )
    return generated_smiles

def process_molecules(smiles_list, desc):
    mols = []
    for smi in tqdm(smiles_list, desc=desc, unit="molecule"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
    return mols

def calculate_fingerprints(mols, radius=2, nBits=2048):
    fingerprints = []
    for mol in tqdm(mols, desc="Calculating fingerprints", unit="molecule"):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        fingerprints.append(fp)
    return fingerprints

def calculate_diversity(fingerprints):
    logging.info("Calculating pairwise diversities...")
    n = len(fingerprints)
    diversity = 0
    total_pairs = (n * (n - 1)) // 2

    with tqdm(total=total_pairs, desc="Calculating diversity", unit="pair") as pbar:
        for i in range(n):
            diversity += sum(1 - DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j]) 
                             for j in range(i + 1, n))
            pbar.update(n - i - 1)

    return diversity / total_pairs if n > 1 else 0

def calculate_novelty(gen_fps, ref_fps, similarity_threshold=0.9):
    logging.info("Calculating novelty...")
    novel_count = 0
    for gen_fp in tqdm(gen_fps, desc="Novelty calculation", unit="molecule"):
        similarities = DataStructs.BulkTanimotoSimilarity(gen_fp, ref_fps)
        if max(similarities) < similarity_threshold:
            novel_count += 1

    return novel_count / len(gen_fps) if gen_fps else 0

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
        properties['RotBonds'].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        properties['QED'].append(Descriptors.qed(mol))
    return properties

def plot_property_distributions(gen_props, ref_props, property_name, xlabel, ylabel="Density", plot_type='line', output_dir=''):
    plt.figure(figsize=(10, 6))

    if plot_type == 'line':
        sns.kdeplot(gen_props, label='Generated', color='blue', fill=True)
        sns.kdeplot(ref_props, label='MOSES', color='red', fill=True)
    elif plot_type == 'bar':
        gen_counts, gen_bins = np.histogram(gen_props, bins=range(min(min(gen_props), min(ref_props)), max(max(gen_props), max(ref_props)) + 2))
        ref_counts, ref_bins = np.histogram(ref_props, bins=gen_bins)

        gen_counts = gen_counts / len(gen_props)
        ref_counts = ref_counts / len(ref_props)

        width = 0.35
        plt.bar(gen_bins[:-1], gen_counts, width, label='Generated', alpha=0.7, color='blue')
        plt.bar(ref_bins[:-1] + width, ref_counts, width, label='MOSES', alpha=0.7, color='red')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Distribution of {property_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{property_name.lower().replace(" ", "_")}_distribution.png'))
    plt.close()

def plot_property_boxplots(gen_props, ref_props, properties_to_plot, output_dir=''):
    plt.figure(figsize=(15, 10))

    data = []
    for prop, name, _, _ in properties_to_plot:
        for value in gen_props[prop]:
            data.append({
                'Property': name,
                'Value': value,
                'Dataset': 'Generated'
            })
        for value in ref_props[prop]:
            data.append({
                'Property': name,
                'Value': value,
                'Dataset': 'MOSES'
            })

    df = pd.DataFrame(data)

    sns.boxplot(x='Property', y='Value', hue='Dataset', data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Box Plots of Molecular Properties')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'molecular_properties_boxplots.png'))
    plt.close()

def plot_individual_boxplots(gen_props, ref_props, properties_to_plot, output_dir=''):
    for prop, name, _, _ in properties_to_plot:
        plt.figure(figsize=(10, 6))
        data = [
            {'Property': name, 'Value': value, 'Dataset': 'Generated'} for value in gen_props[prop]
        ] + [
            {'Property': name, 'Value': value, 'Dataset': 'MOSES'} for value in ref_props[prop]
        ]
        df = pd.DataFrame(data)

        sns.boxplot(x='Dataset', y='Value', data=df)
        plt.title(f'Box Plot of {name}')
        plt.ylabel(name)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name.lower().replace(" ", "_")}_boxplot.png'))
        plt.close()

def analyze_scaffolds(mol_list):
    scaffolds = {}
    for mol in mol_list:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        scaffolds[scaffold_smiles] = scaffolds.get(scaffold_smiles, 0) + 1
    return scaffolds

def evaluate_model(model_dir, tokenizer_path, output_dir, n_samples=10000, max_length=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    try:
        mamba_model, safe_tokenizer = setup_model(model_dir, tokenizer_path, device)
        logging.info(f"Model loaded successfully. Type: {type(mamba_model)}")
        logging.info(f"Tokenizer loaded successfully. Type: {type(safe_tokenizer)}")
    except Exception as e:
        logging.error(f"Error setting up model: {str(e)}")
        logging.error(traceback.format_exc())
        return

    designer = sf.SAFEDesign(
        model=mamba_model,
        tokenizer=safe_tokenizer,
        verbose=True,
    )

    try:
        logging.info(f"Starting molecule generation with parameters: n_samples={n_samples}, max_length={max_length}")
        generated_smiles = generate_molecules(designer, n_samples, max_length=max_length)
        logging.info(f"Molecule generation completed. Generated {len(generated_smiles)} molecules.")
    except Exception as e:
        logging.error(f"Error generating molecules: {str(e)}")
        logging.error(traceback.format_exc())
        return

    # Load MOSES dataset
    train_set = pd.read_csv("../../../train_from_scratch/Datasets/MOSES/train.csv")
    test_set = pd.read_csv("../../../train_from_scratch/Datasets/MOSES/test.csv")
    all_smiles = pd.concat([train_set, test_set])["SMILES"].unique()
    moses_smiles = all_smiles.tolist()

    # Process molecules
    generated_mols = process_molecules(generated_smiles, "Processing generated molecules")
    moses_mols = process_molecules(moses_smiles, "Processing MOSES molecules")

    # Calculate fingerprints
    gen_fps = calculate_fingerprints(generated_mols)
    ref_fps = calculate_fingerprints(moses_mols)

    # Calculate basic properties
    unique_smiles = set(Chem.MolToSmiles(mol) for mol in generated_mols)
    
    validity = len(generated_mols) / len(generated_smiles) if generated_smiles else 0
    uniqueness = len(unique_smiles) / len(generated_smiles) if generated_smiles else 0
    diversity = calculate_diversity(gen_fps)
    novelty = calculate_novelty(gen_fps, ref_fps)

    # Calculate molecular properties
    generated_properties = calculate_properties(generated_mols)
    moses_properties = calculate_properties(moses_mols)

    # Plot property distributions
    properties_to_plot = [
        ('MW', 'Molecular Weight', 'Molecular Weight (Da)', 'line'),
        ('LogP', 'LogP', 'LogP', 'line'),
        ('HBD', 'H-Bond Donors', 'Number of H-Bond Donors', 'bar'),
        ('HBA', 'H-Bond Acceptors', 'Number of H-Bond Acceptors', 'bar'),
        ('TPSA', 'Topological Polar Surface Area', 'TPSA (Å²)', 'line'),
        ('RotBonds', 'Rotatable Bonds', 'Number of Rotatable Bonds', 'bar'),
        ('QED', 'QED', 'Quantitative Estimate of Drug-likeness', 'line')
    ]

    for prop, name, xlabel, plot_type in properties_to_plot:
        plot_property_distributions(generated_properties[prop], moses_properties[prop], name, xlabel, plot_type=plot_type, output_dir=output_dir)

    plot_property_boxplots(generated_properties, moses_properties, properties_to_plot, output_dir=output_dir)
    plot_individual_boxplots(generated_properties, moses_properties, properties_to_plot, output_dir=output_dir)

    # Scaffold analysis
    generated_scaffolds = analyze_scaffolds(generated_mols)
    moses_scaffolds = analyze_scaffolds(moses_mols)

    gen_scaffold_diversity = len(generated_scaffolds) / len(generated_mols)
    moses_scaffold_diversity = len(moses_scaffolds) / len(moses_mols)

    # Prepare results
    results = {
        "validity": validity,
        "uniqueness": uniqueness,
        "diversity": diversity,
        "novelty": novelty,
        "qed_mean": np.mean(generated_properties['QED']),
        "sas_mean": np.mean([sascorer.calculateScore(mol) for mol in generated_mols]),
        "scaffold_diversity": gen_scaffold_diversity,
    }

    # Write results to file
    with open(os.path.join(output_dir, "evaluation_results.txt"), 'w') as f:
        f.write("Mamba Model Evaluation Results\n")
        f.write("==============================\n\n")
        f.write(f"Number of samples generated: {len(generated_smiles)}\n")
        f.write(f"Validity: {results['validity']:.4f}\n")
        f.write(f"Uniqueness: {results['uniqueness']:.4f}\n")
        f.write(f"Diversity: {results['diversity']:.4f}\n")
        f.write(f"Novelty: {results['novelty']:.4f}\n")
        f.write(f"QED mean: {results['qed_mean']:.4f}\n")
        f.write(f"SAS mean: {results['sas_mean']:.4f}\n")
        f.write(f"Scaffold diversity: {results['scaffold_diversity']:.4f}\n\n")

        # Write property summary statistics
        f.write("Property Summary Statistics\n")
        f.write("===========================\n\n")
        headers = ["Property", "Generated (mean ± std)", "MOSES (mean ± std)"]
        table_rows = []
        for prop, name, _, _ in properties_to_plot:
            gen_mean, gen_std = np.mean(generated_properties[prop]), np.std(generated_properties[prop])
            ref_mean, ref_std = np.mean(moses_properties[prop]), np.std(moses_properties[prop])
            row = [name, f"{gen_mean:.2f} ± {gen_std:.2f}", f"{ref_mean:.2f} ± {ref_std:.2f}"]
            table_rows.append(row)
        f.write(tabulate(table_rows, headers=headers, tablefmt="grid"))

        # Write scaffold analysis results
        f.write("\n\nScaffold Analysis\n")
        f.write("=================\n\n")
        f.write(f"Unique scaffolds in generated set: {len(generated_scaffolds)}\n")
        f.write(f"Unique scaffolds in MOSES set: {len(moses_scaffolds)}\n")
        f.write(f"Scaffold diversity in generated set: {gen_scaffold_diversity:.4f}\n")
        f.write(f"Scaffold diversity in MOSES set: {moses_scaffold_diversity:.4f}\n")

    logging.info(f"Results written to {os.path.join(output_dir, 'evaluation_results.txt')}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mamba model for molecule generation")
    parser.add_argument("--model_dir", required=True, help="Directory containing the model")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated sequences")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Starting Mamba model evaluation")
    logging.info(f"Samples: {args.num_samples}, Max Length: {args.max_length}")

    try:
        results = evaluate_model(args.model_dir, args.tokenizer_path, args.output_dir, n_samples=args.num_samples, max_length=args.max_length)
        logging.info("Evaluation completed successfully")
        logging.info(f"Validity: {results['validity']:.4f}")
        logging.info(f"Uniqueness: {results['uniqueness']:.4f}")
        logging.info(f"Diversity: {results['diversity']:.4f}")
        logging.info(f"Novelty: {results['novelty']:.4f}")
        logging.info(f"QED mean: {results['qed_mean']:.4f}")
        logging.info(f"SAS mean: {results['sas_mean']:.4f}")
        logging.info(f"Scaffold diversity: {results['scaffold_diversity']:.4f}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        logging.error(traceback.format_exc())

# def check_predictions(model, tokenizer, input_text='C'):
#     # Encode the input text
#     input_ids = tokenizer.encode(input_text, ids_only=True)
    
#     # Convert to tensor and add batch dimension
#     input_ids = torch.tensor([input_ids]).to(model.device)
    
#     logging.info(f"Input ids: {input_ids}")
    
#     with torch.no_grad():
#         outputs = model(input_ids)
#         logits = outputs.logits[:, -1, :]
#         probs = torch.softmax(logits, dim=-1)
    
#     top_k = 10
#     top_probs, top_indices = torch.topk(probs, k=top_k)
    
#     logging.info(f"Top {top_k} predictions for input '{input_text}':")
#     for prob, idx in zip(top_probs[0], top_indices[0]):
#         token = tokenizer.decode([idx.item()])
#         logging.info(f"Token: {token}, Probability: {prob.item():.4f}")

# def generate_molecules(designer, n_samples=1000, max_length=1000):
#     logging.info(f"Generating {n_samples} molecules")
#     generated_smiles = designer.de_novo_generation(
#         n_samples_per_trial=n_samples,
#         max_length=max_length,
#         sanitize=True,
#         top_k=15,
#         top_p=0.9,
#         temperature=0.9,
#         n_trials=1,
#         repetition_penalty=1.0,
#     )
#     return generated_smiles

# def analyze_molecules(smiles_list):
#     logging.info("Analyzing generated molecules")
#     valid_mols = []
#     for smi in tqdm(smiles_list, desc="Processing molecules"):
#         mol = Chem.MolFromSmiles(smi)
#         if mol is not None:
#             valid_mols.append(mol)
    
#     n_valid = len(valid_mols)
#     n_total = len(smiles_list)
    
#     unique_smiles = set(Chem.MolToSmiles(mol) for mol in valid_mols)
#     n_unique = len(unique_smiles)
    
#     mol_weights = [Descriptors.ExactMolWt(mol) for mol in valid_mols]
#     logp_values = [Descriptors.MolLogP(mol) for mol in valid_mols]
#     qed_values = [QED.qed(mol) for mol in valid_mols]
    
#     return {
#         "n_total": n_total,
#         "n_valid": n_valid,
#         "n_unique": n_unique,
#         "validity": n_valid / n_total if n_total > 0 else 0,
#         "uniqueness": n_unique / n_valid if n_valid > 0 else 0,
#         "mol_weight_mean": np.mean(mol_weights),
#         "mol_weight_std": np.std(mol_weights),
#         "logp_mean": np.mean(logp_values),
#         "logp_std": np.std(logp_values),
#         "qed_mean": np.mean(qed_values),
#         "qed_std": np.std(qed_values),
#     }

# def evaluate_model(model_dir, tokenizer_path, output_file, n_samples=1000, max_length=100):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f"Using device: {device}")

#     try:
#         mamba_model, safe_tokenizer = setup_model(model_dir, tokenizer_path, device)
#         logging.info(f"Model loaded successfully. Type: {type(mamba_model)}")
#         logging.info(f"Tokenizer loaded successfully. Type: {type(safe_tokenizer)}")
#     except Exception as e:
#         logging.error(f"Error setting up model: {str(e)}")
#         logging.error(traceback.format_exc())
#         return

#     logging.info("Checking vocabulary:")
#     try:
#         check_vocab(safe_tokenizer)
#     except Exception as e:
#         logging.error(f"Error checking vocabulary: {str(e)}")
#         logging.error(traceback.format_exc())

#     logging.info("\nChecking predictions:")
#     try:
#         check_predictions(mamba_model, safe_tokenizer)
#     except Exception as e:
#         logging.error(f"Error checking predictions: {str(e)}")
#         logging.error(traceback.format_exc())

#     designer = sf.SAFEDesign(
#         model=mamba_model,
#         tokenizer=safe_tokenizer,
#         verbose=True,
#     )

#     try:
#         logging.info(f"Starting molecule generation with parameters: n_samples={n_samples}, max_length={max_length}")
#         generated_smiles = generate_molecules(designer, n_samples, max_length=max_length)
#         logging.info(f"Molecule generation completed. Generated {len(generated_smiles)} molecules.")
#     except Exception as e:
#         logging.error(f"Error generating molecules: {str(e)}")
#         logging.error(traceback.format_exc())
#         return

#     logging.info("\nSample of generated molecules:")
#     for i, smi in enumerate(generated_smiles[:10]):  # Print first 10 molecules
#         logging.info(f"{i+1}. {smi}")
#     logging.info("...")

#     logging.info("Moving on to molecule analysis")

#     try:
#         results = analyze_molecules(generated_smiles)
#     except Exception as e:
#         logging.error(f"Error analyzing molecules: {str(e)}")
#         logging.error(traceback.format_exc())
#         return

#     # Print metrics
#     logging.info("\nMolecule Generation Metrics:")
#     logging.info(f"Total molecules generated: {results['n_total']}")
#     logging.info(f"Valid molecules: {results['n_valid']} ({results['validity']:.2%})")
#     logging.info(f"Unique molecules: {results['n_unique']} ({results['uniqueness']:.2%})")
#     logging.info(f"Molecular weight: {results['mol_weight_mean']:.2f} ± {results['mol_weight_std']:.2f}")
#     logging.info(f"LogP: {results['logp_mean']:.2f} ± {results['logp_std']:.2f}")
#     logging.info(f"QED: {results['qed_mean']:.2f} ± {results['qed_std']:.2f}")


#     try:
#         with open(output_file, 'w') as f:
#             f.write("Mamba Model Evaluation Results\n")
#             f.write("==============================\n\n")
#             f.write(f"Number of samples requested: {n_samples}\n")
#             f.write(f"Total molecules generated: {results['n_total']}\n")
#             f.write(f"Valid molecules: {results['n_valid']} ({results['validity']:.2%})\n")
#             f.write(f"Unique molecules: {results['n_unique']} ({results['uniqueness']:.2%})\n")
#             f.write(f"Molecular weight: {results['mol_weight_mean']:.2f} ± {results['mol_weight_std']:.2f}\n")
#             f.write(f"LogP: {results['logp_mean']:.2f} ± {results['logp_std']:.2f}\n")
#             f.write(f"QED: {results['qed_mean']:.2f} ± {results['qed_std']:.2f}\n")

#             # Add sample of generated molecules to the output file
#             f.write("\nSample of generated molecules:\n")
#             for i, smi in enumerate(generated_smiles[:10]):
#                 f.write(f"{i+1}. {smi}\n")

#         logging.info(f"Results written to {output_file}")

#         # Debug: Save generated SMILES to a separate file
#         debug_file = output_file.replace('.txt', '_debug_smiles.txt')
#         with open(debug_file, 'w') as f:
#             for smi in generated_smiles:
#                 f.write(f"{smi}\n")
#         logging.info(f"Debug: Generated SMILES saved to {debug_file}")
#     except Exception as e:
#         logging.error(f"Error writing results: {str(e)}")
#         logging.error(traceback.format_exc())

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate Mamba model for molecule generation")
#     parser.add_argument("--model_dir", required=True, help="Directory containing the model")
#     parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
#     parser.add_argument("--output_file", required=True, help="Path to the output file")
#     parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
#     parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated sequences")

#     args = parser.parse_args()

#     logging.info("Starting Mamba model evaluation")
#     logging.info(f"Samples: {args.num_samples}, Max Length: {args.max_length}")

#     try:
#         evaluate_model(args.model_dir, args.tokenizer_path, args.output_file, n_samples=args.num_samples, max_length=args.max_length)
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {str(e)}")
#         logging.error(traceback.format_exc())