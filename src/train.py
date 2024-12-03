import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import read_data, create_bpe_tokenizer, TranslateDataset
from decoder import Transformer

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        src = batch["source_ids"].to(device)
        trg = batch["target_ids"].to(device)
        
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        loss = criterion(output.contiguous().view(-1, output.shape[-1]), trg[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            src = batch["source_ids"].to(device)
            trg = batch["target_ids"].to(device)
            
            output = model(src, trg[:, :-1])
            loss = criterion(output.contiguous().view(-1, output.shape[-1]), trg[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'transformer.pt')
            print("Best model saved!")
        
        plot_loss(train_losses, val_losses)

if __name__ == "__main__":
    configs = {
        "train_source_data": "ted-talks-corpus/train.en",
        "train_target_data": "ted-talks-corpus/train.fr",
        "valid_source_data": "ted-talks-corpus/dev.en",
        "valid_target_data": "ted-talks-corpus/dev.fr",
        "source_max_seq_len": 300,
        "target_max_seq_len": 300,
        "batch_size": 96,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "embedding_dim": 256,
        "n_layers": 3,
        "n_heads": 4,
        "dropout": 0.1,
        "lr": 1e-3,
        "n_epochs": 1,
        "beam_size": 5,
        "vocab_size": 5000
    }

    # Load and preprocess data
    train_src, train_trg = read_data(configs["train_source_data"], configs["train_target_data"])
    val_src, val_trg = read_data(configs["valid_source_data"], configs["valid_target_data"])

    # Create BPE tokenizers
    source_tokenizer = create_bpe_tokenizer(train_src, configs["vocab_size"])
    target_tokenizer = create_bpe_tokenizer(train_trg, configs["vocab_size"])

    # Create datasets
    train_dataset = TranslateDataset(source_tokenizer, target_tokenizer, train_src, train_trg, configs["source_max_seq_len"], configs["target_max_seq_len"])
    val_dataset = TranslateDataset(source_tokenizer, target_tokenizer, val_src, val_trg, configs["source_max_seq_len"], configs["target_max_seq_len"])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=configs["batch_size"])

    # Initialize model
    model = Transformer(
        source_vocab_size=source_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
        embedding_dim=configs["embedding_dim"],
        num_heads=configs["n_heads"],
        num_layers=configs["n_layers"],
        dropout=configs["dropout"]
    ).to(configs["device"])

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=target_tokenizer.token_to_id("[PAD]"))

    # Train the model
    train(model, train_loader, val_loader, optimizer, criterion, configs["device"], configs["n_epochs"])