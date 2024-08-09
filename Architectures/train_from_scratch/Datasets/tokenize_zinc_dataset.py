import datasets
import safe
from tqdm.auto import tqdm

def tokenize_smiles(example):
    try:
        safe_encoded = safe.encode(example['smiles'], ignore_stereo=True)
        tokens = list(safe.split(safe_encoded))
        return {'tokenized': tokens}
    except (safe.SAFEEncodeError, safe.SAFEDecodeError, safe.SAFEFragmentationError) as e:
        print(f"Error processing SMILES: {example['smiles']}. Error: {str(e)}")
        return {'tokenized': None}

def process_dataset(dataset, split_name):
    total = dataset.info.splits[split_name].num_examples
    tokenized_split = dataset.map(tokenize_smiles)
    filtered_split = tokenized_split.filter(lambda x: x['tokenized'] is not None)
    
    # Process the streaming dataset in batches
    processed_data = []
    molecule_count = 0
    for batch in tqdm(filtered_split.iter(batch_size=1000), total=total//1000, desc=f"Processing {split_name}"):
        for item in batch['tokenized']:
            if item is not None:
                processed_data.append(item)
                molecule_count += 1
    
    print(f"Number of molecules left after processing {split_name} split: {molecule_count}")
    return datasets.Dataset.from_dict({'tokenized': processed_data}), molecule_count

# Load the dataset in streaming mode for each split separately
# splits = ['train', 'validation']
splits = ['validation']
dataset = {split: datasets.load_dataset("sagawa/ZINC-canonicalized", split=split, streaming=True) for split in splits}

# Process each split
tokenized_dataset = {}
total_molecules = 0
for split_name, split_data in dataset.items():
    print(f"Processing {split_name} split...")
    tokenized_dataset[split_name], molecule_count = process_dataset(split_data, split_name)
    total_molecules += molecule_count

# Create a DatasetDict
tokenized_dataset_dict = datasets.DatasetDict(tokenized_dataset)

# Save the tokenized dataset as a DatasetDict
print("Saving the dataset...")
tokenized_dataset_dict.save_to_disk("tokenized_zinc_dataset", num_proc=4)

print("Dataset saved as a DatasetDict. To load it later, use:")
print("loaded_dataset = datasets.load_from_disk('tokenized_zinc_dataset')")

print(f"Total number of molecules left after processing all splits: {total_molecules}")
