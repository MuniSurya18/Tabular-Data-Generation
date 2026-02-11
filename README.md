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
   *Note: On Windows systems with strict path limits, you may encounter issues installing `torch`. We recommend using the provided Colab Notebook.*

## Usage

### Run on Google Colab (Recommended)
Open `DSDDPM_Colab.ipynb` in Google Colab (upload it or open from GitHub) to run the training and generation without local setup issues.

### Local Usage

1. **Prepare Data**
   ```bash
   python scripts/generate_synthetic.py
   ```

2. **Train the Model**
   ```bash
   python -m src.train --data data/dummy.csv --epochs 100 --batch_size 64
   ```
   Checkpoints will be saved in `checkpoints/`.

3. **Generate Synthetic Data**
   ```bash
   python -m src.sample --model checkpoints/model_final.pt --data data/dummy.csv --output generated.csv --num_samples 1000
   ```
