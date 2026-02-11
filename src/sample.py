
import os
import torch
import pandas as pd
import argparse
from src.data import load_data, TabularDataset
from src.modules import TabularDenoiseModel
from src.diffusion import DSDDPM

def sample(
    model_path: str,
    data_path: str,
    output_path: str = "generated_data.csv",
    num_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # Load original data to get dimensions and encoders
    # (In a real app, save metadata separately)
    df, num_cols, cat_cols = load_data(data_path)
    dataset = TabularDataset(df, num_cols, cat_cols)
    d_in_num, cat_dims = dataset.get_dims()
    
    # Init Model
    model = TabularDenoiseModel(
        d_in_num=d_in_num,
        cat_dims=cat_dims,
        d_embed=16, 
        d_hidden=256,
        n_layers=3
    ).to(device)
    
    # Load Weights
    print(f"Loading weights from {model_path}...")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    diffusion = DSDDPM(model, device=device).to(device)
    
    # Sample
    print(f"Generating {num_samples} samples...")
    # Prototype for shape reference
    x_prototype = torch.zeros(1, d_in_num, device=device)
    
    x_num_gen, x_cat_gen = diffusion.sample(num_samples, x_prototype, cat_dims)
    
    # Inverse Transform
    df_gen = dataset.inverse_transform(x_num_gen, x_cat_gen)
    
    df_gen.to_csv(output_path, index=False)
    print(f"Saved generated data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="generated_data.csv")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    
    sample(args.model, args.data, output_path=args.output, num_samples=args.num_samples)
