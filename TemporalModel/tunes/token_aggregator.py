import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import MyLinear, MyLayerNorm, MyDropout1d, MyActivation


class TokenAggregator(nn.Module):
    """
    Aggregates multiple tokens per timestep into a single representation.
    Converts input from (N, T, num_tokens, D) to (N, T, D).
    """
    
    def __init__(self, input_dim, output_dim, num_tokens, aggregation_type="linear", 
                 hidden_dim=None, init_method='trunc_normal', dropout=0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.aggregation_type = aggregation_type
        
        if hidden_dim is None:
            hidden_dim = output_dim
            
        if aggregation_type == "linear":
            # Simple linear projection of flattened tokens
            self.aggregator = MyLinear(
                input_dim * num_tokens, 
                dim_out=output_dim, 
                init_method=init_method, 
                followed_by_relu=False
            )
            
        elif aggregation_type == "attention":
            # Attention-based pooling using simple query-key-value attention
            self.query = nn.Parameter(torch.randn(1, 1, input_dim))
            self.output_proj = MyLinear(
                input_dim, 
                dim_out=output_dim, 
                init_method=init_method, 
                followed_by_relu=False
            )
            
        elif aggregation_type == "mlp":
            # MLP-based aggregation
            self.aggregator = nn.Sequential(
                MyLinear(input_dim * num_tokens, dim_out=hidden_dim, 
                        init_method=init_method, followed_by_relu=True),
                MyActivation("gelu"),
                MyDropout1d(p=dropout),
                MyLinear(hidden_dim, dim_out=output_dim, 
                        init_method=init_method, followed_by_relu=False)
            )
            
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
            
        # Initialize query parameter for attention
        if aggregation_type == "attention":
            nn.init.normal_(self.query, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (N, T, num_tokens, D)
            mask: Optional mask of shape (N, T) where True means valid timestep
            
        Returns:
            aggregated: Tensor of shape (N, T, output_dim)
            mask: Same mask as input (N, T)
        """
        N, T, num_tokens, D = x.shape
        assert num_tokens == self.num_tokens, f"Expected {self.num_tokens} tokens, got {num_tokens}"
        assert D == self.input_dim, f"Expected input dim {self.input_dim}, got {D}"
        
        if self.aggregation_type == "linear" or self.aggregation_type == "mlp":
            # Flatten tokens and apply linear layer
            x_flat = x.view(N, T, -1)  # (N, T, num_tokens * D)
            aggregated = self.aggregator(x_flat)  # (N, T, output_dim)
            
        elif self.aggregation_type == "attention":
            # Reshape for attention computation
            x_reshaped = x.view(N * T, num_tokens, D)  # (N*T, num_tokens, D)
            
            # Create query for attention pooling
            query = self.query.expand(N * T, 1, D)  # (N*T, 1, D)
            
            # Create token mask (all tokens are valid)
            token_mask = torch.ones(N * T, num_tokens, device=x.device, dtype=torch.bool)
            
            # Apply cross-attention: query attends to all tokens
            # We need to implement a simple attention pooling here
            # Since we don't have access to the exact MyAttention interface, use a simple version
            
            # Compute attention weights
            scores = torch.bmm(query, x_reshaped.transpose(1, 2))  # (N*T, 1, num_tokens)
            attn_weights = torch.softmax(scores, dim=-1)  # (N*T, 1, num_tokens)
            
            # Apply attention weights to get pooled representation
            pooled = torch.bmm(attn_weights, x_reshaped).squeeze(1)  # (N*T, D)
            
            # Project to output dimension
            aggregated = self.output_proj(pooled)  # (N*T, output_dim)
            aggregated = aggregated.view(N, T, self.output_dim)  # (N, T, output_dim)
            
        return aggregated, mask
    
    def extra_repr(self):
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, ' \
               f'num_tokens={self.num_tokens}, aggregation_type={self.aggregation_type}'


class SimpleTokenAggregator(nn.Module):
    """
    Simple token aggregator with predefined strategies (no learnable parameters).
    """
    
    def __init__(self, aggregation_type="cls"):
        super().__init__()
        self.aggregation_type = aggregation_type
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (N, T, num_tokens, D)
            mask: Optional mask of shape (N, T)
            
        Returns:
            aggregated: Tensor of shape (N, T, D)
            mask: Same mask as input (N, T)
        """
        if self.aggregation_type == "cls":
            # Use first token (CLS token)
            aggregated = x[:, :, 0, :]  # (N, T, D)
            
        elif self.aggregation_type == "mean":
            # Average all tokens
            aggregated = x.mean(dim=2)  # (N, T, D)
            
        elif self.aggregation_type == "max":
            # Max pooling across tokens
            aggregated, _ = x.max(dim=2)  # (N, T, D)
            
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")
            
        return aggregated, mask
    
    def extra_repr(self):
        return f'aggregation_type={self.aggregation_type}'