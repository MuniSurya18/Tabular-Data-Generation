
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from typing import Tuple, List, Optional

class TabularDataset(Dataset):
    def __init__(self, 
                 data: pd.DataFrame, 
                 numerical_cols: List[str], 
                 categorical_cols: List[str]):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Preprocessing
        self.X_num = data[numerical_cols].values.astype(np.float32)
        self.X_cat = data[categorical_cols].values
        
        # Quantile Transform for Numerical
        if len(self.numerical_cols) > 0:
            self.qt = QuantileTransformer(output_distribution='normal', random_state=42)
            self.X_num = self.qt.fit_transform(self.X_num)
        else:
            self.qt = None
            
        # Label Encode Categorical
        self.label_encoders = {}
        self.cat_dims = []
        X_cat_encoded = np.zeros_like(self.X_cat, dtype=np.int64)
        
        for i, col in enumerate(categorical_cols):
            le = LabelEncoder()
            X_cat_encoded[:, i] = le.fit_transform(self.X_cat[:, i])
            self.label_encoders[col] = le
            self.cat_dims.append(len(le.classes_))
            
        self.X_cat = torch.tensor(X_cat_encoded, dtype=torch.long)
        self.X_num = torch.tensor(self.X_num, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X_num)
    
    def __getitem__(self, idx):
        return {
            "x_num": self.X_num[idx],
            "x_cat": self.X_cat[idx]
        }
    
    def get_dims(self):
        return self.X_num.shape[1], self.cat_dims

    def inverse_transform(self, x_num, x_cat):
        # x_num: (B, D_num), x_cat: (B, D_cat)
        
        if self.qt is not None:
            x_num_inv = self.qt.inverse_transform(x_num.cpu().numpy())
        else:
            x_num_inv = x_num.cpu().numpy()
            
        x_cat_inv_list = []
        x_cat_np = x_cat.cpu().numpy()
        for i, col in enumerate(self.categorical_cols):
            le = self.label_encoders[col]
            # Clip to valid range just in case
            vals = np.clip(x_cat_np[:, i], 0, len(le.classes_) - 1)
            x_cat_inv_list.append(le.inverse_transform(vals))
            
        df_num = pd.DataFrame(x_num_inv, columns=self.numerical_cols)
        df_cat = pd.DataFrame(np.stack(x_cat_inv_list, axis=1), columns=self.categorical_cols)
        
        return pd.concat([df_num, df_cat], axis=1)

def load_data(path: str, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Simple loader - assumes csv
    df = pd.read_csv(path)
    
    # Auto-detect types if not specified (simple heuristic)
    # In a real app, user should specify which is which
    # For now, we assume object/bool/category are categorical, others numerical
    
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # If target_col is specified and numerical, keep it in numerical, etc.
    # Usually target is what we want to generate too, so it's just another column.
    
    return df, numerical_cols, categorical_cols

