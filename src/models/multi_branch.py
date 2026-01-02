import torch
import torch.nn as nn
from torchvision import models

from src.models.img_encoder import CNN, ResNet

class SimplePoolingAttention(nn.Module):
    """
    Simple Pooling Attention (SPA) module.

    This module applies a simple pooling operation to the input features. 
    Args:
        emb_dim (int): Dimension of the input features
        hidden_dim (int): Dimension of the hidden layer
    """
    def __init__(self, emb_dim, hidden_dim=128):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_weights=False):
        scores = self.score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        
        if return_weights:
            return pooled, weights
        return pooled

class PoolingByMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=3, dropout=0.1):
        """
        Pooling-by-Multi-Head-Attention (PMA) module.
        
        Args:
            input_dim (int): Dimension of input features
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"
        
        # Multi-head attention layers
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        
        # Output projection
        self.out_proj = nn.Linear(input_dim, input_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Temperature parameter for softmax
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x, return_weights=False):
        """
        Forward pass of PMA attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_branches, features]
            return_weights (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Attended features of shape [batch_size, features]
            torch.Tensor (optional): Attention weights of shape [batch_size, num_heads, num_branches]
        """
        batch_size, num_branches, _ = x.size()
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, num_branches, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_branches, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_branches, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, num_branches, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, num_branches, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, num_branches, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores / self.temperature
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, num_branches, head_dim]
        
        # Reshape and combine heads
        context = context.transpose(1, 2).contiguous()  # [batch_size, num_branches, num_heads, head_dim]
        context = context.view(batch_size, num_branches, -1)  # [batch_size, num_branches, input_dim]
        
        # Project to output dimension
        output = self.out_proj(context)
        
        # Sum over branches
        output = output.sum(dim=1)  # [batch_size, input_dim]
        
        if return_weights:
            return output, attn_weights
        return output

class MultiBranchArchitecture(nn.Module):
    def __init__(self, num_branches=3, name="MBArch", branch_model='resnet18', custom_cnn_layers:list=None, pretrained=True, 
                 filters_per_layer:list=[16,32,64,128], dropout:float=0.5, branch_predicition_head:list=None, context_vector_size=17,
                 pooling_method='spa', num_attention_heads=4, branch_output_dim=256, final_prediction_head:list=None, output_size=9):
        super(MultiBranchArchitecture, self).__init__()
        self.name = name
        self.num_branches = num_branches

        # Validate and create branches based on branch_model type
        if 'cnn' in branch_model.lower():
            # CNN requires custom_cnn_layers
            if custom_cnn_layers is None:
                raise ValueError(f"custom_cnn_layers is required when using CNN (branch_model='{branch_model}')")
            if not isinstance(custom_cnn_layers, list) or len(custom_cnn_layers) == 0:
                raise ValueError("custom_cnn_layers must be a non-empty list of tuples")
            
            # Create CNN branches
            self.branches = nn.ModuleList([
                CNN(name=f"branch_{i}", 
                    dropout=dropout, 
                    custom_cnn_layers=custom_cnn_layers,
                    custom_prediction_head=branch_predicition_head,
                    context_vector_size=context_vector_size,
                    branch_output_dim=branch_output_dim
                    ) for i in range(num_branches)
                ])
        elif 'resnet' in branch_model.lower():
            # ResNet requires base_model, pretrained, and filters_per_layer
            # Create ResNet branches
            self.branches = nn.ModuleList([
                ResNet(name=f"branch_{i}", 
                       base_model=branch_model,
                       pretrained=pretrained,
                       dropout=dropout, 
                       custom_prediction_head=branch_predicition_head,
                       filters_per_layer=filters_per_layer,
                       context_vector_size=context_vector_size,
                       branch_output_dim=branch_output_dim
                       ) for i in range(num_branches)
                ])
        else:
            raise ValueError(f"Unsupported branch_model: {branch_model}. Use 'resnet<num_layers>' (e.g., 'resnet18', 'resnet34') or 'cnn'")
        
        # Use the provided branch_output_dim for attention (already set from parameter)
        
        # PMA attention module to combine features from different branches
        if pooling_method == 'spa':
            self.attention = SimplePoolingAttention(branch_output_dim)
        elif pooling_method == 'pma':
            self.attention = PoolingByMultiHeadAttention(branch_output_dim, num_heads=num_attention_heads, dropout=dropout)
        else:
            raise ValueError(f'Unsupported pooling method: {pooling_method}. Please use:\n - spa for Simple Pooling Attention\n - pma for Pooling by Multi-Head Attention')
        
        # Final prediction head
        if final_prediction_head is not None:
            layers = []
            in_features = branch_output_dim
            
            # Build layers based on the provided list
            for out_features in final_prediction_head[:-1]:  # All layers except the last one
                layers.extend([
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                in_features = out_features
            
            # Add final layer to match required output size
            layers.append(nn.Linear(in_features, output_size))
            
            self.prediction_head = nn.Sequential(*layers)
        else:
            # Default architecture if no custom_head is provided
            self.prediction_head = nn.Sequential(
                nn.Linear(branch_output_dim, 512), # Q: Should branch output be the same as the output size of the final prediction head after pooling block?
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, output_size)
            )

    def forward(self, x, context_vector, return_attention=False):
        """
        Forward pass of the multi-branch CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_branches, channels, height, width)
            context_vector (torch.Tensor): Context vector tensor of shape (batch_size, context_vector_size)
            return_attention (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
            torch.Tensor (optional): Attention weights if return_attention is True
        """
        batch_size = x.size(0)
        
        # Verify input shape matches number of branches
        if x.size(1) != self.num_branches:
            raise ValueError(f"Input tensor has {x.size(1)} branches but model expects {self.num_branches} branches")
                
        # Process each branch
        branch_outputs = []
        for i in range(self.num_branches):
            branch_input = x[:, i]  # Select the i-th image for this branch
            context_vector_input = context_vector[:, i]
            branch_output = self.branches[i](branch_input, context_vector_input)
            branch_outputs.append(branch_output)
        
        # Stack branch outputs
        # Shape: (batch_size, num_branches, features)
        stacked_outputs = torch.stack(branch_outputs, dim=1)
        
        # Apply PMA attention to combine branch outputs
        if return_attention:
            attended_features, attention_weights = self.attention(stacked_outputs, return_weights=True)
        else:
            attended_features = self.attention(stacked_outputs)
        
        # Final prediction
        output = self.prediction_head(attended_features)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def __str__(self):
        return self.name
