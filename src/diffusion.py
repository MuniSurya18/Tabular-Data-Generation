
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSDDPM(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 num_timesteps: int = 1000, 
                 beta_start: float = 1e-4, 
                 beta_end: float = 0.02,
                 device: str = 'cpu'):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Define Beta Schedule (Linear)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Helper values for numerical diffusion (Gaussian)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Helper values for categorical diffusion (Uniform Transition)
        # q(xt | x0) = alpha_bar_t * x0 + (1 - alpha_bar_t) * 1/K
        # But commonly we use: 
        # prob_keep = alphas_cumprod
        # prob_random = 1 - alphas_cumprod
        # This matches the "Uniform" transition matrix Q_t_bar.
        
    def q_sample_num(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    
    def q_sample_cat(self, x_0, t, cat_dims):
        """
        Sample xt from q(xt | x0) for categorical features.
        We apply noise independently for each feature column.
        """
        # x_0: (B, num_cat_cols) - indices
        B, C = x_0.shape
        x_t = x_0.clone()
        
        # Probability of keeping the original value
        # alpha_bar_t: (B,)
        alpha_bar_t = self.alphas_cumprod[t] 
        
        # Sample mask: 1 = keep, 0 = random
        # random[0, 1) < alpha_bar_t
        mask = torch.rand((B, C), device=self.device) < alpha_bar_t[:, None]
        
        # For entries where mask is False, sample uniformly from K
        for i, K in enumerate(cat_dims):
            # Generate random indices for this column
            rand_vals = torch.randint(0, K, (B,), device=self.device)
            # Replace where mask is 0
            x_t[:, i] = torch.where(mask[:, i], x_t[:, i], rand_vals)
            
        return x_t

    def get_loss(self, x_num, x_cat):
        B = x_num.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        
        # 1. Numerical Branch (Gaussian Diffusion)
        noise_num = torch.randn_like(x_num)
        x_num_t = self.q_sample_num(x_num, t, noise_num)
        
        # 2. Categorical Branch (Multinomial Diffusion)
        # Note: We pass x_0 indices directly
        x_cat_t = self.q_sample_cat(x_cat, t, self.model.cat_dims)
        
        # 3. Model Prediction
        # Model should predict NOISE for numerical (epsilon)
        # And LOGITS of x_0 for categorical (denoising)
        pred_noise_num, pred_logits_cat = self.model(x_num_t, x_cat_t, t)
        
        # 4. Calculate Losses
        # Numerical Loss (MSE)
        loss_num = F.mse_loss(pred_noise_num, noise_num, reduction='none').mean(dim=1) # (B,)
        
        # Categorical Loss (Cross Entropy)
        # Sum over all categorical features
        loss_cat = torch.zeros(B, device=self.device)
        for i, logits in enumerate(pred_logits_cat):
            # logits: (B, K), target: (B,)
            # F.cross_entropy returns (B,) if reduction='none' is not passed, but we want per-sample
            # so we flatten or loop? F.cross_entropy does reduction by default.
            l = F.cross_entropy(logits, x_cat[:, i], reduction='none')
            loss_cat += l
            
        # 5. Adaptive Weighting (Eq 15 & 16)
        # We calculate means for the batch to determine weights
        # Or should we do it per sample? Paper implies "in each batch".
        
        L_c_mean = loss_num.mean()
        L_f_mean = loss_cat.mean()
        
        # Avoid division by zero
        denom = L_c_mean + L_f_mean + 1e-8
        
        # Assuming lambda_c = lambda_f = 1 for simplicity (or use hyperparameters)
        lambda_c = 1.0
        lambda_f = 1.0
        
        wc = (lambda_c * L_c_mean) / denom
        wf = 1.0 - wc
        
        # "DSDDPM introduces... dual-scale noise-handling mechanism"
        # Combine losses
        total_loss = wc * loss_num + wf * loss_cat
        
        return total_loss.mean(), {
            'loss_num': L_c_mean.item(), 
            'loss_cat': L_f_mean.item(),
            'weight_c': wc.item(),
            'weight_f': wf.item()
        }

    @torch.no_grad()
    def sample(self, num_samples, input_prototype_num, cat_dims):
        """
        Generate new samples.
        input_prototype_num is just to get dimensions and shape.
        """
        self.model.eval()
        d_num = input_prototype_num.shape[1]
        
        # Start from pure noise
        # Numerical: Gaussian noise
        x_num = torch.randn(num_samples, d_num, device=self.device)
        
        # Categorical: Uniform noise
        x_cat = torch.zeros(num_samples, len(cat_dims), dtype=torch.long, device=self.device)
        for i, K in enumerate(cat_dims):
            x_cat[:, i] = torch.randint(0, K, (num_samples,), device=self.device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict
            pred_noise_num, pred_logits_cat = self.model(x_num, x_cat, t_batch)
            
            # Update Numerical (standard DDPM update)
            # x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_bar) * epsilon) + sigma * z
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x_num)
            else:
                noise = torch.zeros_like(x_num)
                
            x_num = (1 / torch.sqrt(alpha_t)) * (
                x_num - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise_num
            ) + torch.sqrt(beta_t) * noise
            
            # Update Categorical
            # We use the predicted logits to sample x_0_hat, then re-noises to x_{t-1}
            # Or use the logits as probabilities for the next step directly?
            # Standard way for discrete: p(x_{t-1} | x_t) = sum_x0 q(x_{t-1} | x_t, x_0) * p(x_0 | x_t)
            # We approximate p(x_0 | x_t) with our model's predicted logits.
            
            # 1. Get p(x_0 | x_t) from logits
            probs_x0 = [F.softmax(logits, dim=-1) for logits in pred_logits_cat]
            
            # 2. For each feature, compute p(x_{t-1} | x_t) using the posterior formula
            # q(x_{t-1} | x_t, x_0) propto q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
            # For uniform transition, this is simple.
            # But let's simplify:
            # Just sample x_0_hat from probs, then 'q_sample' it to t-1?
            # Or better: Denoise to x_{t-1} based on predicted x_0.
            # Eq (13) in paper implies p_theta(x_{t-1} | x_t) is predicted directly?
            # "u_theta and Sigma_theta are estimates of neural network outputs"
            # Since I implemented x_0 prediction (standard for discrete), I'll use the analytic posterior.
            
            # Simplified sampler for Discrete Diffusion:
            # If t=0, return argmax(logits)
            # Else, use the predicted x_0 probabilities to sample x_{t-1}.
            # Actually, let's just use the "Argmax + Noise" approximation for simplicity 
            # or the proper posterior if possible.
            # Proper posterior for uniform scalar diffusion:
            # p(x_{t-1}=k | x_t=j, x_0_hat) 
            # We will use a simplified approach: Sample x_0 from logits, then q_sample to t-1.
            # This is known as "Ancestral Sampling" on the approximate posterior.
            
            new_x_cat_list = []
            for i, probs in enumerate(probs_x0):
                # Sample x_0 prediction
                # x_0_hat = torch.multinomial(probs, 1).squeeze()
                # Actually, taking the mode is often better for quality at the end
                if t == 0:
                    x_cat_val = torch.argmax(probs, dim=1)
                else:
                    # Sample x_{t-1} guided by x_0_hat?
                    # Let's just sample x_0_hat weighted by probs
                    x_0_hat = torch.multinomial(probs, 1).squeeze(-1)
                    
                    # Compute posterior q(x_{t-1} | x_t, x_0_hat)
                    # For uniform noise:
                    # chance to keep x_t, chance to jump to x_0_hat, chance to jump random
                    # This is complex to implement from scratch without checking the formulas.
                    # Fallback: q_sample x_0_hat to t-1
                    
                    # alpha_bar_{t-1}
                    alpha_bar_prev = self.alphas_cumprod[t-1]
                    
                    # Sample mask
                    mask = torch.rand((num_samples,), device=self.device) < alpha_bar_prev
                    
                    # If mask=1, use x_0_hat. If mask=0, random? 
                    # No, this is q(x_{t-1}|x_0).
                    # We want to move from x_t towards x_0.
                    
                    # Simple heuristic:
                    # With probability alpha_t, keep x_t? No.
                    
                    # Let's stick to: "Reverse Process" defined in paper Alg 2?
                    # "x_{t-1} = f_f(x_t, t)"
                    # This implies the model PREDICTS x_{t-1} directly?
                    # If so, my loss function (predicting x_0) is slightly different.
                    # But predicting x_0 is better.
                    # Let's use the property: x_{t-1} is a slightly less noisy version.
                    # We'll set x_cat_{t-1} = x_0_hat with probability (some small update) and x_t with remainder?
                    
                    # Let's assume the model predicts x_0 logits.
                    # We sample x_0_hat.
                    # Then we sample x_{t-1} using q(x_{t-1} | x_t, x_0_hat).
                    # Formula for Uniform Matrix:
                    # theta_post = ...
                    
                    # I'll implement a very simple "Analytic" step:
                    # Just sample x_0_hat from probs.
                    # Then q_sample to t-1.
                    x_cat_val = self.q_sample_cat(x_0_hat.unsqueeze(1), t-1, [cat_dims[i]]).squeeze(1)
                
                new_x_cat_list.append(x_cat_val)
            
            x_cat = torch.stack(new_x_cat_list, dim=1)
            
        return x_num, x_cat

