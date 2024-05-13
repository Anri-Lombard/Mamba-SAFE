import safe as sf
import pandas as pd
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# df = pd.read_csv('./Datasets/MOSES/train.csv')
df = pd.read_csv('./Datasets/MOSES/test.csv')

print("Read in original dataset from ./Datasets/MOSES/train.csv")
print("Number of SMILES: ", len(df))

molecule_nr = 0
start_time = time.time()
def convert_smiles_to_safe(row):
    global molecule_nr
    global start_time
    molecule_nr += 1
    idx = row.name
    smile = row['SMILES']
    if molecule_nr % 100000 == 0:
        print(f"Molecule {molecule_nr}/{len(df)} in time {time.time() - start_time}")
    try:
        safe = sf.encode(smile)
        return safe
    except:
        return None

df['SAFE'] = df.apply(convert_smiles_to_safe, axis=1)
df = df.dropna(subset=['SAFE'])
# df.to_csv('./Datasets/MOSES/train_safe.csv', index=False)
df.to_csv('./Datasets/MOSES/test_safe.csv', index=False)
# print("Saved updated dataset to ./Datasets/MOSES/train_safe.csv")
print("Saved updated dataset to ./Datasets/MOSES/test_safe.csv")
