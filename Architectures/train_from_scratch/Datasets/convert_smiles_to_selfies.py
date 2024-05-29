import selfies as sf
import pandas as pd
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    print(f"Read in original dataset from {input_file}")
    print("Number of SMILES: ", len(df))

    molecule_nr = 0
    start_time = time.time()

    def convert_smiles_to_selfies(row):
        nonlocal molecule_nr
        nonlocal start_time
        molecule_nr += 1
        idx = row.name
        smile = row['SMILES']
        if molecule_nr % 100000 == 0:
            print(f"Molecule {molecule_nr}/{len(df)} in time {time.time() - start_time}")
        try:
            selfies = sf.encoder(smile)
            return selfies
        except sf.EncoderError:
            return None

    df['SELFIES'] = df.apply(convert_smiles_to_selfies, axis=1)
    df = df.dropna(subset=['SELFIES'])
    df.to_csv(output_file, index=False)
    print(f"Saved updated dataset to {output_file}")

# Convert train dataset
train_input_file = './Datasets/MOSES/train.csv'
train_output_file = './Datasets/MOSES/train_selfies.csv'
convert_dataset(train_input_file, train_output_file)

# Convert test dataset
test_input_file = './Datasets/MOSES/test.csv'
test_output_file = './Datasets/MOSES/test_selfies.csv'
convert_dataset(test_input_file, test_output_file)