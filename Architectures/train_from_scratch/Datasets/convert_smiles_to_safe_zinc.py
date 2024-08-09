import datasets
import safe
from tqdm.auto import tqdm
import os

def smiles_to_safe(example):
    try:
        safe_encoded = safe.encode(example['smiles'], ignore_stereo=True)
        return {'safe': safe_encoded}
    except (safe.SAFEEncodeError, safe.SAFEDecodeError, safe.SAFEFragmentationError) as e:
        print(f"Error processing SMILES: {example['smiles']}. Error: {str(e)}")
        return {'safe': None}

def process_dataset(dataset, split_name):
    total = dataset.info.splits[split_name].num_examples
    safe_encoded_split = dataset.map(smiles_to_safe)
    filtered_split = safe_encoded_split.filter(lambda x: x['safe'] is not None)

    # Process the streaming dataset in batches
    processed_data = []
    molecule_count = 0
    for batch in tqdm(filtered_split.iter(batch_size=1000), total=total//1000, desc=f"Processing {split_name}"):
        for item in batch['safe']:
            if item is not None:
                processed_data.append(item)
                molecule_count += 1

    print(f"Number of molecules successfully converted in {split_name} split: {molecule_count}")
    return datasets.Dataset.from_dict({'safe': processed_data}), molecule_count

# Load the dataset in streaming mode for each split separately
splits = ['train','validation']
dataset = {split: datasets.load_dataset("sagawa/ZINC-canonicalized", split=split, streaming=True) for split in splits}

# Process each split
safe_dataset = {}
total_molecules = 0
for split_name, split_data in dataset.items():
    print(f"Processing {split_name} split...")
    safe_dataset[split_name], molecule_count = process_dataset(split_data, split_name)
    total_molecules += molecule_count

# Create a DatasetDict
safe_dataset_dict = datasets.DatasetDict(safe_dataset)

zinc_folder = os.path.join(os.getcwd(), 'ZINC')
os.makedirs(zinc_folder, exist_ok=True)

# Save the SAFE-encoded dataset as a DatasetDict
save_path = os.path.join(zinc_folder, 'safe_zinc_dataset')
print(f"Saving the dataset to {save_path}...")
safe_dataset_dict.save_to_disk(save_path, num_proc=4)

print(f"Dataset saved as a DatasetDict in the ZINC folder. To load it later, use:")
print(f"loaded_dataset = datasets.load_from_disk('{save_path}')")
print(f"Total number of molecules successfully converted: {total_molecules}")