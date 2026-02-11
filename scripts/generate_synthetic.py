
import numpy as np
import pandas as pd
import random

def generate_dummy_data(n_samples=1000, path='data/dummy.csv'):
    # Numerical
    age = np.random.normal(30, 10, n_samples)
    income = np.random.lognormal(10, 1, n_samples)
    
    # Categorical
    gender = np.random.choice(['Male', 'Female', 'Other'], n_samples)
    city = np.random.choice(['New York', 'London', 'Paris', 'Tokyo'], n_samples)
    churn = np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8])
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Gender': gender,
        'City': city,
        'Churn': churn
    })
    
    df.to_csv(path, index=False)
    print(f"Generated {n_samples} samples to {path}")

if __name__ == "__main__":
    generate_dummy_data()
