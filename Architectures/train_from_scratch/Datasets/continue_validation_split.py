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
    
    processed_data = []
    molecule_count = 0
    for batch in tqdm(filtered_split.iter(batch_size=1000), total=total//1000, desc=f"Processing {split_name}"):
        for item in batch['tokenized']:
            if item is not None:
                processed_data.append(item)
                molecule_count += 1
    
    print(f"Number of molecules left after processing {split_name} split: {molecule_count}")
    return datasets.Dataset.from_dict({'tokenized': processed_data}), molecule_count

# Load the existing dataset
print("Loading existing dataset...")
existing_dataset = datasets.load_from_disk("tokenized_zinc_dataset")

# Load and process the validation split
print("Processing validation split...")
validation_dataset = datasets.load_dataset("sagawa/ZINC-canonicalized", split="validation", streaming=True)
tokenized_validation, validation_count = process_dataset(validation_dataset, "validation")

# Append validation split to the existing dataset
print("Appending validation split to the existing dataset...")
updated_dataset = datasets.DatasetDict({
    'train': existing_dataset['train'],
    'validation': tokenized_validation
})

# Save the updated dataset
print("Saving the updated dataset...")
updated_dataset.save_to_disk("tokenized_zinc_dataset_with_validation", num_proc=4)

print("Updated dataset saved. To load it later, use:")
print("loaded_dataset = datasets.load_from_disk('tokenized_zinc_dataset_with_validation')")
print(f"Total number of molecules in train split: {len(existing_dataset['train'])}")
print(f"Total number of molecules in validation split: {validation_count}")
print(f"Total number of molecules in the updated dataset: {len(existing_dataset['train']) + validation_count}")