import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import defaultdict
import json

import shutil

from rdkit import Chem
from typing import List, Optional
from tqdm import tqdm

from time import perf_counter

import torch
import torch.nn.functional as F

from mambapy.mamba_lm import from_pretrained
from mambapy.mamba_lm import MambaLM, MambaLMConfig

from transformers import AutoTokenizer
import datasets

import numpy as np
import random

from safe.tokenizer import SAFETokenizer
from datasets import DatasetDict

# Automated device selection based on available backends
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available() and False
        else "cpu"
    )

print(f"> Using {device} device")

def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            files.append(f"{path}/{f}")
    return files

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_checkpoint(filepath, model, scheduler, optimizer):
    print(f"> Loading model from: {filepath}")
    try:
        loaded_checkpoint = torch.load(filepath, map_location=device)

        loaded_epoch = loaded_checkpoint['epoch']
        loaded_model = model
        loaded_scheduler = scheduler
        loaded_optimizer = optimizer

        loaded_model.load_state_dict(loaded_checkpoint['model_state'])
        if scheduler is not None:
            loaded_scheduler.load_state_dict(loaded_checkpoint['scheduler_state'])
        if optimizer is not None:
            loaded_optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])

        print("> Loaded model")
        return True, loaded_epoch, loaded_model, loaded_scheduler, loaded_optimizer
    except Exception as e:
        print("> Cannot load model")
        return False, 0, model, scheduler, optimizer

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def prepare_molecular_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return {"input_ids": tokenizer.encode(examples["SAFE"], ids_only=True)}
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Sort the dataset by length
    def get_length(example):
        return len(example['input_ids'])
    
    sorted_dataset = sorted(tokenized_dataset, key=get_length)
    
    return sorted_dataset

def create_batches(dataset, batch_size):
    batches = []
    current_batch = []
    current_length = 0
    
    for item in dataset:
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []
            current_length = 0
        
        current_batch.append(item)
        current_length = max(current_length, len(item['input_ids']))
        
    if current_batch:
        batches.append(current_batch)
    
    return batches

def pad_batch(batch, pad_token_id=0):
    max_length = max(len(item['input_ids']) for item in batch)
    padded_batch = []
    for item in batch:
        padded_input = item['input_ids'] + [pad_token_id] * (max_length - len(item['input_ids']))
        padded_batch.append(padded_input)
    return torch.tensor(padded_batch)


def train(pretrained=False):
    # Training parameters
    '''
    epochs - number of epochs during training
    batch_size - size of a single batch during training
    seq_length - number of tokens in model's context during training
    learning_rate - initial learning rate of the training
    model_path - path to the saved weights; if empty it'll save there new weights during training
    backup_path - path to the backup of a model. if None - no backup is created
    '''
    epochs = 1
    checkpoint_interval = 1000
    batch_size = 64 #32 for 24GB and 130m model
    seq_length = 128
    learning_rate = 1e-4
    model_path = f'saves/model.pth'
    max_checkpoints = 2  # Keep only the last 2 checkpoints


    # Usage of datasets' built in datasets
    # dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1')
    dataset = DatasetDict.load_from_disk('../../Datasets/MOSES/datasets')

    # https://www.kaggle.com/datasets/nltkdata/gutenberg
    #dataset = datasets.load_dataset('text', data_files={'train': listdir_nohidden("./gutenberg")}, encoding='utf-8',encoding_errors='ignore')
    
    # Usage of custom txt datasets
    '''
    In order to load custom training data add filepaths to the list
    For example to use one txt file change the name of the file in the command below:

    dataset = datasets.load_dataset('text', data_files={'train': ['austen-emma.txt']})

    For more files add them to the list after comma

    https://huggingface.co/docs/datasets/v1.2.1/loading_datasets.html
    '''

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer = SAFETokenizer.from_pretrained("./tokenizer.json")
    # Add eos tokens
    # tokenizer.eos_token = "<|endoftext|>"
    # tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    # def tokenize_function(examples):
    #     return {"input_ids": tokenizer.encode(examples["SAFE"], ids_only=True)}

    # Map tokenizer to the dataset
    # tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)
    # tokenize_data = lambda example, tokenizer: {'tokens': tokenizer.tokenize(example['text'], truncation=True)} 
    # tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'], 
    #     fn_kwargs={'tokenizer': tokenizer})


    
    # Prepare and load tokenizer's vocabulary for later use
    # vocab = tokenizer.vocab
    # print(f"Vocab size: {len(vocab)}")
    vocab_size = 1180 # safe tokenizer vocab size
    print(f"Vocab size: {vocab_size}")

    
    # Select the wanted model
    '''
    If pretrained==True - the script loads pretrained mamba weights specified by the string.
    If pretrained==False - the script creates a new MambaLM model with parameters specified in config variable
    '''
    if pretrained:
        model = from_pretrained('state-spaces/mamba-130m').to(device)
    else:
        # config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=len(tokenizer.vocab))
        config = MambaLMConfig(d_model=768, n_layers=6, vocab_size=vocab_size)
        model = MambaLM(config).to(device)

    print(f"Number of trainable parameters: {count_parameters(model):,}")

    # Create optimizer and pass the model
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            optim,
                                                            mode='min',
                                                            factor=0.1, #factor by which the lr is multiplied
                                                            patience=2,
                                                        )

    # Load previously trained weights
    ''' 
    If the model is the same it will load previous weights located in specified path
    If the model differs or the path is empty it'll skip loading and train from scratch
    '''
    _, epoch, model, scheduler, optim = load_checkpoint(model_path, model, scheduler, optim)
    
    # Create data loader functions
    # def get_data(dataset, batch_size, max_length):
    #     # data = []                                   
    #     # for example in dataset:
    #     #     if example['tokens']:
    #     #         tokens = [vocab[token] for token in example['tokens']]
    #     #         data.extend(tokens)
        
    #     # data = torch.LongTensor(data)              
    #     # num_batches = data.shape[0] // batch_size 
    #     # data = data[:num_batches * batch_size]                       
    #     # data = data.view(batch_size, num_batches)
    #     # return data     

    #     # Pad sequences to max_length
    #     padded_data = [seq + [0] * (max_length - len(seq)) for seq in dataset['input_ids']]
    #     data = torch.LongTensor(padded_data)
    #     num_batches = data.shape[0] // batch_size 
    #     data = data[:num_batches * batch_size]
    #     data = data.view(batch_size, -1)
    #     return data

    # def get_batch(data, seq_len, idx):
    #     src = data[:, idx:idx+seq_len]
    #     target = data[:, idx+1:idx+seq_len+1]
    #     return src, target


    # # Get data
    # train_data = get_data(tokenized_dataset['train'], batch_size, max_seq_length)
    # print(f"Train data shape: {train_data.shape}")

    prepared_dataset = prepare_molecular_dataset(dataset['train'], tokenizer)

    # Training loop
    losses = defaultdict(list)
    t0_start = perf_counter()
    for z in range(epoch, epochs):
        # idx = 0
        avg_loss = 0
        print(f"\n> Epoch {z+1}/{epochs}")

        t2_start = perf_counter()

        # Shuffle the dataset at the start of each epoch
        random.shuffle(prepared_dataset)

        # Create batches
        batches = create_batches(prepared_dataset, batch_size)
        
        # for i in range(train_data.shape[-1]):
        for i, batch in enumerate(batches):
            model.train()
            t1_start = perf_counter()

            # idx = i * seq_length
            # input, output = get_batch(train_data, seq_length, idx)
            # output = output.reshape(-1)
            # input = input.to(device)
            # output = output.to(device)

            input_ids = pad_batch(batch).to(device)
            attention_mask = (input_ids != 0).float().to(device)

            # Shift the input_ids to create targets
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

            # logits = model(input)
            logits = model(input_ids)

            # # If the batch is not complete - skip
            # if (logits.view(-1, logits.size(-1)).shape[0] != output.view(-1).shape[0]):
            #     print("skip")
            # else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)  # ignore padding
            avg_loss += loss.item()
            losses[z].append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

            t1_stop = perf_counter()

            # Print the progress during training and save the model
            if i%100==0:
                print(f"\r> Batch: {i}/{len(batches)} loss: {avg_loss/(i+1):.5f} time: {t1_stop-t1_start:.2f} sec ", end="")

                # checkpoint = {
                #     'epoch': z,
                #     'model_state': model.state_dict(),
                #     'optimizer_state': optim.state_dict(),
                #     'scheduler_state': scheduler.state_dict(),
                # }
                # # Create backup file
                # if backup_path is not None and os.path.isfile(model_path):
                #     shutil.copyfile(model_path, backup_path)
                # torch.save(checkpoint, model_path)

            if i % checkpoint_interval == 0:
                checkpoint = {
                    'epoch': z,
                    'model_state': model.state_dict(),
                    'optimizer_state': optim.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }
                checkpoint_path = f'saves/checkpoint_epoch{z}_step{i}.pth'
                torch.save(checkpoint, checkpoint_path)
                
                # Remove old checkpoints if there are more than max_checkpoints
                checkpoints = sorted([f for f in os.listdir('saves') if f.startswith('checkpoint') and f.endswith('.pth')])
                if len(checkpoints) > max_checkpoints:
                    os.remove(os.path.join('saves', checkpoints[0]))

            # Increment idx
            # idx += 1
            # if idx >= train_data.shape[-1] - max_seq_length:
            #     idx = 0
            #     break

        t2_stop = perf_counter()
        print(f"\n> Epoch time: {t2_stop - t2_start:.3f} seconds")
        # Update schedulers
        scheduler.step(avg_loss/len(batches))

    t0_stop = perf_counter()
    print(f"\n> Finished training in: {t0_stop-t0_start} seconds")

    # print("> Generating answer: ")
    # # Generate sample text after training
    # output = model.generate(tokenizer, "She was the youngest of the two daughters of a most affectionate "
    #                         , num_tokens=50
    #                         , temperature=1.0
    #                         , top_k=None)

    # print(f"Answer: {output}")

    # Save the final model
    final_checkpoint = {
        'epoch': epochs,
        'model_state': model.state_dict(),
        'optimizer_state': optim.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }
    torch.save(final_checkpoint, model_path)

    # Save the loss history
    with open('loss_history.json', 'w') as f:
        json.dump(losses, f)


def de_novo_generation(
    model: MambaLM,
    tokenizer: SAFETokenizer,
    n_samples: int = 10,
    sanitize: bool = False,
    n_trials: Optional[int] = None,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[str]:
    """
    Perform de novo generation using the trained Mamba model.

    Args:
        model: The trained Mamba model
        tokenizer: The SAFE tokenizer
        n_samples: Number of new molecules to generate per trial
        sanitize: Whether to perform sanitization to ensure validity
        n_trials: Number of trials to perform (default is 1)
        max_length: Maximum length of generated sequences
        temperature: Sampling temperature
        top_k: If set, only sample from the top k most likely next tokens
        device: Device to use for generation

    Returns:
        List of generated molecular sequences
    """
    model.eval()
    model.to(device)

    total_sequences = []
    n_trials = n_trials or 1

    for _ in tqdm(range(n_trials), desc="Generating molecules", leave=False):
        sequences = []
        for _ in range(n_samples):
            start_token = tokenizer.encode("[START]", ids_only=True)
            input_ids = torch.tensor(start_token).unsqueeze(0).to(device)
            
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = model(input_ids)
                    next_token_logits = outputs[:, -1, :] / temperature
                    
                    if top_k is not None:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits[0, :] = float('-inf')
                        next_token_logits[0, top_k_indices] = top_k_logits
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    if next_token.item() == tokenizer.encode("[END]", ids_only=True)[0]:
                        break
                    
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            generated_ids = input_ids[0].tolist()
            sequences.append(tokenizer.decode(generated_ids))
        
        total_sequences.extend(sequences)

    if sanitize:
        total_sequences = [seq for seq in total_sequences if is_valid_molecule(seq)]

    return total_sequences

def is_valid_molecule(smiles: str) -> bool:
    """Check if a SMILES string represents a valid molecule."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Usage example
def generate_molecules(model_path: str, tokenizer_path: str, n_samples: int = 10):
    tokenizer = SAFETokenizer.from_pretrained(tokenizer_path)
    
    config = MambaLMConfig(d_model=768, n_layers=6, vocab_size=tokenizer.vocab_size)
    model = MambaLM(config)

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])

    generated_molecules = de_novo_generation(
        model,
        tokenizer,
        n_samples=n_samples,
        sanitize=True,
        n_trials=1,
        max_length=100,
        temperature=0.8,
        top_k=50
    )

    print(f"Generated {len(generated_molecules)} valid molecules:")
    for i, mol in enumerate(generated_molecules, 1):
        print(f"Molecule {i}: {mol}")

    return generated_molecules

def prepare_folders():
    try:
        os.makedirs("./saves/")
    except:
        pass
seed_everything(534)
prepare_folders()
train(pretrained=False)
