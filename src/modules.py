
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TabularDenoiseModel(nn.Module):
    def __init__(self, 
                 d_in_num: int, 
                 cat_dims: list, 
                 d_embed: int = 16, 
                 d_hidden: int = 256, 
                 n_layers: int = 4,
                 d_time_emb: int = 128):
        super().__init__()
        
        self.d_in_num = d_in_num
        self.cat_dims = cat_dims
        self.d_embed = d_embed
        
        # Embeddings for categorical features
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, d_embed) for num_classes in cat_dims
        ])
        
        # Input dimension: Numerical + Flattened Embeddings + Time Emb
        self.d_input = d_in_num + len(cat_dims) * d_embed + d_time_emb
        
        # Time Embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_time_emb),
            nn.Linear(d_time_emb, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        # -- Separate Branches for Numerical and Categorical Denoising --
        # Numerical Branch (Coarse Scale)
        self.num_mlp = self._make_mlp(self.d_input, d_hidden, n_layers, d_in_num)
        
        # Categorical Branch (Fine Scale)
        # Output dim is sum of all classes (for logits)
        total_cat_out = sum(cat_dims)
        self.cat_mlp = self._make_mlp(self.d_input, d_hidden, n_layers, total_cat_out)

    def _make_mlp(self, d_in, d_hidden, n_layers, d_out):
        layers = []
        layers.append(nn.Linear(d_in, d_hidden))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(0.1))
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.1))
            
        layers.append(nn.Linear(d_hidden, d_out))
        return nn.Sequential(*layers)

    def forward(self, x_num, x_cat, t):
        # x_num: (B, D_num)
        # x_cat: (B, D_cat_cols) - integer indices
        # t: (B,)
        
        # 1. Embed Time
        t_emb = self.time_emb(t)
        
        # 2. Embed Categorical
        cat_embs = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            cat_embs.append(emb_layer(x_cat[:, i]))
        
        if len(cat_embs) > 0:
            x_cat_embed = torch.cat(cat_embs, dim=1)
        else:
            x_cat_embed = torch.tensor([], device=x_num.device)
            
        # 3. Concatenate Inputs
        # All branches see everything to capture correlations
        inp = torch.cat([x_num, x_cat_embed, t_emb], dim=1)
        
        # 4. Forward Pass
        # Numerical Output (Noise Prediction)
        out_num = self.num_mlp(inp)
        
        # Categorical Output (Logits)
        out_cat_flat = self.cat_mlp(inp)
        
        # Reshape categorical output to list of logits per feature
        out_cat_logits = []
        start = 0
        for dim in self.cat_dims:
            out_cat_logits.append(out_cat_flat[:, start:start+dim])
            start += dim
            
        return out_num, out_cat_logits
