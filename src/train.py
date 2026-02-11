
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from src.data import TabularDataset, load_data
from src.modules import TabularDenoiseModel
from src.diffusion import DSDDPM

def train(
    data_path: str,
    output_dir: str = "checkpoints",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    load_checkpoint: str = None
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {data_path}...")
    df, num_cols, cat_cols = load_data(data_path)
    dataset = TabularDataset(df, num_cols, cat_cols)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Setup Model
    d_in_num, cat_dims = dataset.get_dims()
    print(f"Numerical dims: {d_in_num}, Categorical dims: {cat_dims}")
    
    denoise_model = TabularDenoiseModel(
        d_in_num=d_in_num,
        cat_dims=cat_dims,
        d_embed=16,
        d_hidden=256,
        n_layers=3
    ).to(device)
    
    diffusion = DSDDPM(denoise_model, num_timesteps=1000, device=device).to(device)
    
    optimizer = optim.AdamW(denoise_model.parameters(), lr=lr)
    
    start_epoch = 0
    if load_checkpoint:
        print(f"Loading checkpoint {load_checkpoint}...")
        ckpt = torch.load(load_checkpoint, map_location=device)
        denoise_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # 3. Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, epochs):
        diffusion.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        total_loss = 0
        total_num = 0
        total_cat = 0
        
        for batch in pbar:
            x_num = batch['x_num'].to(device)
            x_cat = batch['x_cat'].to(device)
            
            optimizer.zero_grad()
            loss, metrics = diffusion.get_loss(x_num, x_cat) # No need to call forward explicitly
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_num += metrics['loss_num']
            total_cat += metrics['loss_cat']
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'num': f"{metrics['loss_num']:.4f}", 
                'cat': f"{metrics['loss_cat']:.4f}",
                'wc': f"{metrics['weight_c']:.2f}"
            })
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': denoise_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            
    print("Training complete.")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': denoise_model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(output_dir, "model_final.pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to csv data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    train(args.data, epochs=args.epochs, batch_size=args.batch_size)
