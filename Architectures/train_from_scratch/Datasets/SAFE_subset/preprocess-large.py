import argparse
import os
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

def main(fraction):
    dataset = load_dataset("datamol-io/safe-gpt", split="train", streaming=True)
    
    # Get the total number of examples in the dataset
    total_examples = 1182796783  # Use the known dataset size
    
    # Calculate the number of examples to keep
    num_examples_to_keep = int(fraction * total_examples)
    
    # Create a progress bar
    progress_bar = tqdm(total=num_examples_to_keep, unit='examples')
    
    # Check if a subset dataset already exists
    # subset_dir = "safe_gpt_subset"
    subset_dir = "/scratch/lmbanr001/safe_gpt_subset"
    if os.path.exists(os.path.join(subset_dir, "dataset_info.json")):
        subset_dataset = Dataset.load_from_disk(subset_dir)
        subset_examples = subset_dataset.to_list()
        progress_bar.update(len(subset_examples))
    else:
        subset_examples = []
    
    # Iterate over the dataset and collect the subset examples
    for item in dataset:
        if len(subset_examples) < num_examples_to_keep:
            subset_examples.append(item)
            progress_bar.update(1)
            
            if len(subset_examples) % 1000000 == 0:
                # Save the current subset incrementally
                subset_dataset = Dataset.from_list(subset_examples)
                subset_dataset.save_to_disk(subset_dir, max_shard_size="30GB")
        else:
            break
    
    # Close the progress bar
    progress_bar.close()
    
    # Save the final subset dataset
    subset_dataset = Dataset.from_list(subset_examples)
    subset_dataset.save_to_disk(subset_dir, max_shard_size="30GB")
    
    print(f"{len(subset_examples)} examples from the original {total_examples} examples in the safe dataset")
    
    subset_dataset_dict = DatasetDict({"train": subset_dataset})
    subset_dataset_dict.save_to_disk("safe_gpt_subset_dict", max_shard_size="30GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a subset of the datamol-io/safe-gpt dataset.')
    parser.add_argument('--fraction', type=float, default=0.1, help='Fraction of the dataset to include in the subset (default: 0.1)')
    args = parser.parse_args()
    
    main(args.fraction)
