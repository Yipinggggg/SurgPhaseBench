"""
Mamba-based Temporal Model for Surgical Phase Recognition

This model uses the Mamba architecture (Selective State Space Model) for temporal sequence modeling.
Mamba is efficient for long sequences and provides an alternative to Transformers and TCNs.

Reference: Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

Installation required:
    pip install mamba-ssm causal-conv1d>=1.2.0
    
Note: mamba-ssm requires CUDA for efficient implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Please install with: pip install mamba-ssm causal-conv1d>=1.2.0")

from ...utils import LOGITS


class TemporalDilation(nn.Module):
    """
    Temporal dimension dilation layer.
    Dilates the temporal dimension by a factor using max pooling followed by upsampling.
    
    Args:
        dilation_factor: Factor by which to dilate the temporal dimension (default: 2)
    """
    def __init__(self, dilation_factor: int = 2):
        super().__init__()
        self.dilation_factor = dilation_factor
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, D) where B=batch, L=length, D=dimension
        Returns:
            Dilated tensor of shape (B, L, D)
        """
        if self.dilation_factor <= 1:
            return x
            
        # x shape: (B, L, D)
        B, L, D = x.shape
        
        # Transpose to (B, D, L) for 1D operations
        x = x.transpose(1, 2)  # (B, D, L)
        
        # Apply max pooling for downsampling
        x_pooled = F.max_pool1d(x, kernel_size=self.dilation_factor, stride=self.dilation_factor)
        
        # Upsample back to original length
        x_upsampled = F.interpolate(x_pooled, size=L, mode='linear', align_corners=False)
        
        # Transpose back to (B, L, D)
        x = x_upsampled.transpose(1, 2)
        
        return x


class MambaBlock(nn.Module):
    """
    A single Mamba block with residual connection and normalization.
    
    Args:
        d_model: Model dimension
        d_state: State dimension (SSM state expansion factor)
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        dropout: Dropout rate
        temporal_dilation: Temporal dilation factor (default: 1, no dilation)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        temporal_dilation: int = 2,
    ):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm causal-conv1d>=1.2.0")
        
        # Temporal dilation layer (applied before mamba)
        self.temporal_dilation = TemporalDilation(dilation_factor=temporal_dilation)
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, D) where B=batch, L=length, D=d_model
        Returns:
            Output tensor of shape (B, L, D)
        """
        # Apply temporal dilation before mamba processing
        x = self.temporal_dilation(x)
        
        # Mamba expects (B, L, D) format
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaEncoder(nn.Module):
    """
    Stack of Mamba blocks for encoding temporal sequences.
    
    Args:
        num_layers: Number of Mamba blocks
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        dropout: Dropout rate
        temporal_dilation: Temporal dilation factor for all blocks
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        temporal_dilation: int = 1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                temporal_dilation=temporal_dilation,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, D)
        Returns:
            Output tensor of shape (B, L, D)
        """
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return x


class MambaTemporalModel(nn.Module):
    """
    Mamba-based model for surgical phase recognition.
    
    This model uses selective state space models (Mamba) for temporal modeling,
    providing an efficient alternative to Transformers with linear complexity in sequence length.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes (phases)
        d_model: Hidden dimension for Mamba layers
        num_layers: Number of Mamba blocks
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor for Mamba (default: 2)
        dropout: Dropout rate (default: 0.1)
        bidirectional: Whether to use bidirectional processing (default: True)
        temporal_dilation: Temporal dilation factor for all blocks (default: 2)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        temporal_dilation: int = 2,
    ):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for MambaTemporalModel. "
                "Install with: pip install mamba-ssm causal-conv1d>=1.2.0"
            )
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.bidirectional = bidirectional
        
        if input_dim != d_model:
            print(
                f"Warning: input_dim ({input_dim}) != d_model ({d_model}). "
                "An input projection layer will be used."
            )
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            print(
                f"Input dimension ({input_dim}) matches d_model ({d_model}). "
                "No input projection layer will be used."
            )
            self.input_projection = nn.Identity()
        
        # Forward Mamba encoder
        self.forward_encoder = MambaEncoder(
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            temporal_dilation=temporal_dilation,
        )
        
        # Backward Mamba encoder (for bidirectional processing)
        if bidirectional:
            self.backward_encoder = MambaEncoder(
                num_layers=num_layers,
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                temporal_dilation=temporal_dilation,
            )
            classifier_input_dim = d_model * 2
        else:
            self.backward_encoder = None
            classifier_input_dim = d_model
        
        # Output projection
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, num_classes)
        )
        
        # For compatibility with different input formats
        self.input_format = 'NCT'  # Default: (batch, channels, time)
        
    def forward(self, x, mask=None):
        """
        Forward pass of the Mamba temporal model.
        
        Args:
            x: Input tensor of shape (N, C, T) or (N, T, C)
               where N=batch, C=channels/features, T=time
            mask: Optional mask of shape (N, T) where 1=keep, 0=mask
        
        Returns:
            Dictionary containing:
                - LOGITS: Output logits of shape (N, num_classes, T)
        """
        # Handle input format: convert to (N, T, C)
        if x.dim() == 3:
            if x.size(1) == self.input_dim:  # (N, C, T) format
                x = x.transpose(1, 2)  # -> (N, T, C)
        
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimension
        x = self.input_projection(x)  # (N, T, d_model)
        
        # Forward pass through Mamba encoder
        forward_out = self.forward_encoder(x)  # (N, T, d_model)
        
        # Bidirectional processing
        if self.bidirectional:
            # Reverse sequence for backward pass
            x_reversed = torch.flip(x, dims=[1])
            backward_out = self.backward_encoder(x_reversed)
            # Reverse back to original order
            backward_out = torch.flip(backward_out, dims=[1])
            # Concatenate forward and backward
            combined = torch.cat([forward_out, backward_out], dim=-1)  # (N, T, 2*d_model)
        else:
            combined = forward_out
        
        # Classify each timestep
        logits = self.classifier(combined)  # (N, T, num_classes)
        
        # Convert to (N, num_classes, T) format for consistency with other models
        logits = logits.transpose(1, 2)  # (N, num_classes, T)
        
        # Apply mask if provided
        if mask is not None:
            logits = logits * mask.unsqueeze(1)
        
        return {
            LOGITS: logits,
        }


class MultiStageMambaModel(nn.Module):
    """
    Multi-stage refinement Mamba model similar to MS-TCN architecture.
    
    The first stage processes input features, and subsequent stages refine
    the predictions using softmax outputs from the previous stage.
    
    Args:
        num_stages: Number of refinement stages
        input_dim: Input feature dimension
        num_classes: Number of output classes
        d_model: Hidden dimension for Mamba layers
        num_layers: Number of Mamba blocks per stage
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional processing
        temporal_dilation: Temporal dilation factor for all blocks (default: 2)
    """
    def __init__(
        self,
        num_stages: int,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        temporal_dilation: int = 2,
    ):
        super().__init__()
        
        self.num_stages = num_stages
        
        # First stage: process input features
        self.stage1 = MambaTemporalModel(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bidirectional=bidirectional,
            temporal_dilation=temporal_dilation,
        )
        
        # Subsequent stages: refine predictions
        self.refinement_stages = nn.ModuleList([
            MambaTemporalModel(
                input_dim=num_classes,
                num_classes=num_classes,
                d_model=d_model,
                num_layers=num_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                bidirectional=bidirectional,
                temporal_dilation=temporal_dilation,
            )
            for _ in range(num_stages - 1)
        ])
        
        self.input_format = 'NCT'
        
    def forward(self, x, mask=None):
        """
        Forward pass through all stages.
        
        Args:
            x: Input features of shape (N, C, T)
            mask: Optional mask of shape (N, T)
        
        Returns:
            Dictionary containing:
                - LOGITS: Tuple of output logits from each stage
        """
        # First stage
        out = self.stage1(x, mask)
        outputs = (out[LOGITS],)
        
        # Refinement stages
        for stage in self.refinement_stages:
            # Use softmax of previous stage as input
            stage_input = F.softmax(outputs[-1], dim=1)
            out = stage(stage_input, mask)
            outputs = outputs + (out[LOGITS],)
        
        return {
            LOGITS: outputs,  # Tuple of tensors, each of shape (N, num_classes, T)
        }


# Utility function to create model instances
def create_mamba_model(
    model_type: str = 'single',
    input_dim: int = 2048,
    num_classes: int = 7,
    d_model: int = 256,
    num_layers: int = 4,
    num_stages: int = 2,
    **kwargs
):
    """
    Factory function to create Mamba models.
    
    Args:
        model_type: 'single' or 'multistage'
        input_dim: Input feature dimension
        num_classes: Number of output classes
        d_model: Hidden dimension
        num_layers: Number of Mamba blocks
        num_stages: Number of stages (for multistage model)
        **kwargs: Additional arguments for the model
    
    Returns:
        Mamba model instance
    """
    if model_type == 'single':
        return MambaTemporalModel(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=num_layers,
            **kwargs
        )
    elif model_type == 'multistage':
        return MultiStageMambaModel(
            num_stages=num_stages,
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=num_layers,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'single' or 'multistage'")


if __name__ == "__main__":
    """
    Test the Mamba models with sample data.
    """
    if not MAMBA_AVAILABLE:
        print("Mamba-ssm not available. Please install it to test the models.")
        exit(1)
    
    # Test configuration
    batch_size = 2
    seq_len = 100
    input_dim = 2048
    num_classes = 7
    d_model = 128
    
    print("Testing MambaTemporalModel...")
    print("=" * 50)
    
    # Create model
    model = MambaTemporalModel(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=d_model,
        num_layers=4,
        bidirectional=True,
    )
    
    # Create sample input (N, C, T) format
    x = torch.randn(batch_size, input_dim, seq_len)
    mask = torch.ones(batch_size, seq_len)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # everything to cuda
    model = model.cuda()
    x = x.cuda()
    mask = mask.cuda()

    # Forward pass
    output = model(x, mask)
    logits = output[LOGITS]
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {num_classes}, {seq_len})")
    assert logits.shape == (batch_size, num_classes, seq_len), "Output shape mismatch!"
    print("✓ Single-stage model test passed!")
    
    print("\n" + "=" * 50)
    print("Testing MultiStageMambaModel...")
    print("=" * 50)
    
    # Create multi-stage model
    num_stages = 2
    multistage_model = MultiStageMambaModel(
        num_stages=num_stages,
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=d_model,
        num_layers=4,
        bidirectional=True,
    )
    
    print(f"Number of stages: {num_stages}")
    print(f"Model parameters: {sum(p.numel() for p in multistage_model.parameters()):,}")
    
    # everything to cuda
    multistage_model = multistage_model.cuda()
    x = x.cuda()
    mask = mask.cuda()

    # Forward pass
    output = multistage_model(x, mask)
    logits_tuple = output[LOGITS]
    
    print(f"Number of stage outputs: {len(logits_tuple)}")
    for i, stage_logits in enumerate(logits_tuple):
        print(f"Stage {i+1} logits shape: {stage_logits.shape}")
        assert stage_logits.shape == (batch_size, num_classes, seq_len), f"Stage {i+1} shape mismatch!"
    
    print("✓ Multi-stage model test passed!")
    
    print("\n" + "=" * 50)
    print("All tests passed successfully! ✓")
    print("=" * 50)
