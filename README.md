
# Tabular Data Generation with DSDDPM

This repository contains the implementation of the **Dual-Scale Diffusion Probabilistic Model (DSDDPM)** for generating synthetic tabular data. This model is designed to capture both global dependencies (coarse-scale) and local details (fine-scale) using a dual-branch diffusion process.

## Features
- **Dual-Scale Noise Handling**: Treats numerical and categorical features with specialized noise schedules.
- **Mixed-Type Data Support**: Handles both continuous and categorical columns.
- **PyTorch Implementation**: Modular and easy to extend.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MuniSurya18/Tabular-Data-Generation.git
   cd Tabular-Data-Generation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Data
Prepare your dataset as a CSV file. The script automatically detects numerical and categorical columns (simple heuristic), or you can modify `src/data.py` to specify them.

To generate dummy data for testing:
```bash
python scripts/generate_synthetic.py
```

### 2. Train the Model
Train the DSDDPM model on your dataset:
```bash
python -m src.train --data data/dummy.csv --epochs 100 --batch_size 64
```
Checkpoints will be saved in `checkpoints/`.

### 3. Generate Synthetic Data
Generate new samples using a trained checklist:
```bash
python -m src.sample --model checkpoints/model_final.pt --data data/dummy.csv --output generated.csv --num_samples 1000
```

## Structure
- `src/diffusion.py`: Core DSDDPM logic (Forward/Reverse processes).
- `src/modules.py`: Neural Network architecture (Dual-Branch MLP).
- `src/data.py`: Data loading and preprocessing (Quantile Transform, Label Encoding).
- `src/train.py`: Training loop.

## References
Based on the paper "Enhancing Tabular Data Generation With Dual-Scale Noise Modeling" by Zhang et al.
