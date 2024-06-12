import torch
from torch.utils.data import Dataset

class MambaDataset(Dataset):
    def __init__(self, datadict, max_length):
        self.data = []
        for key in datadict:
            text = datadict[key]['text']
            tokens = text.split()
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i+max_length]
                self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tokens = self.data[index]
        input_ids = torch.tensor([YOUR_TOKENIZER.encode(token) for token in tokens])
        return {'input_ids': input_ids}