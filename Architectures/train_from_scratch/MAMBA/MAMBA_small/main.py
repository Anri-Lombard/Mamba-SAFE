import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mamba_model import MambaLMHeadModel
from mamba_dataset import MambaDataset
from mamba_sampler import MambaSampler
from mamba_config import MambaConfig

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = input_ids.clone().to(device)
        outputs = model(input_ids)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, model.config.vocab_size), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = input_ids.clone().to(device)
            outputs = model(input_ids)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, model.config.vocab_size), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = MambaConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        ssm_cfg=args.ssm_cfg,
    )
    model = MambaLMHeadModel(config).to(device)
    dataset = MambaDataset(args.datadict, args.max_length)
    sampler = MambaSampler(dataset, args.batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.num_epochs):
        train_loss = train(model, dataloader, optimizer, device)
        eval_loss = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")
    
    torch.save(model.state_dict(), args.output_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadict', type=str, required=True, help='Path to the MOSES datadict')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of Mamba layers')
    parser.add_argument('--ssm_cfg', type=dict, default=None, help='State space model configuration')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--output_model_path', type=str, default='mamba_model.pt', help='Path to save the trained model')
    args = parser.parse_args()
    
    main(args)