import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from torch.autograd import Variable
from tqdm.notebook import tqdm

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import torch
import torchvision as tv
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Overall this should works on pretrained models
# CNN models or Multi-branch CNNs models
# working on the 5 examples the best fitted: 100% 75% 50% 25% 1%
# TODO: 1st make Salicy mrthod
# TODO: Simple recreation of the prediction and target
# TODO: Attetion block visualization and branch contribution

class InterpModel:
    # TODO: Would like to create one class for all interpolation methods needed for the project
    def __init__(self, model, steps=50, n_samples=50, noise_level=0.1):
        """
        Initialize the interpreter model
        
        Args:
            model: The model to interpret
            steps: Number of steps for integrated gradients
            n_samples: Number of samples for SmoothGrad
            noise_level: Standard deviation of noise to add for SmoothGrad
        """
        self.model = model
        self.steps = steps
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.model.eval()  # Set model to evaluation mode
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture gradients and activations"""
        def save_gradient(grad):
            self.gradients = grad.detach()
            
        def save_activation(module, input, output):
            self.activations = output.detach()
        
        # Find the last convolutional layer
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.target_layer = module
                # Register forward hook for activations
                self.target_layer.register_forward_hook(save_activation)
                # Register backward hook for gradients
                self.target_layer.register_full_backward_hook(
                    lambda module, grad_input, grad_output: save_gradient(grad_output[0])
                )
                break
        
        if not hasattr(self, 'target_layer'):
            raise ValueError("No convolutional layer found in the model. Grad-CAM requires at least one convolutional layer.")
    
    def _get_gradcam_weights(self, target_class=None):
        """
        Compute Grad-CAM++ weights using higher-order derivatives and pixel-wise weighting
        
        Args:
            target_class: Target class index (for classification tasks)
            
        Returns:
            Pixel-wise weights for Grad-CAM++
        """
        if self.gradients is None or self.activations is None:
            raise ValueError("No gradients or activations captured. Run forward pass first.")
            
        # Get first-order gradients
        first_grad = self.gradients
        
        # Get second-order gradients
        second_grad = torch.pow(first_grad, 2)
        
        # Get third-order gradients
        third_grad = torch.pow(first_grad, 3)
        
        # Global average pooling
        global_sum = torch.sum(self.activations, dim=(2, 3), keepdim=True)
        
        # Compute pixel-wise weights
        alpha_num = second_grad
        alpha_denom = second_grad + torch.sum(second_grad * self.activations, dim=(2, 3), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-7)
        
        # Compute weights
        weights = torch.sum(alpha * torch.relu(first_grad), dim=(2, 3))
        
        return weights

    def compute_gradcam(self, inputs, target_class=None, target_dim=None):
        """
        Compute Grad-CAM++ visualization for the input (replaces original Grad-CAM)
        
        Args:
            inputs: List of input tensors or single input tensor
            target_class: Index of target class for classification tasks, or None for regression
            target_dim: For regression tasks, which output dimension to visualize. If None, uses mean of all outputs.
            
        Returns:
            Grad-CAM++ heatmap for each input
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        gradcam_maps = []
        
        for input_tensor in inputs:
            # Forward pass
            output = self.model(input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor)
            
            # For regression models
            if target_class is None:
                if target_dim is not None:
                    # Use specific output dimension
                    target = output[:, target_dim]
                else:
                    # Use mean of all outputs for visualization
                    target = output.mean()
            # For classification models
            else:
                target = output[:, target_class]
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass
            target.backward(retain_graph=True)
            
            # Get weights and activations using Grad-CAM++ method
            weights = self._get_gradcam_weights(target_class)
            activations = self.activations[0]  # Remove batch dimension
            
            # Weight the activations
            weighted_activations = torch.zeros_like(activations)
            for i, w in enumerate(weights[0]):
                weighted_activations[i] = w * activations[i]
            
            # Generate heatmap
            heatmap = torch.sum(weighted_activations, dim=0)
            heatmap = torch.relu(heatmap)  # Apply ReLU
            
            # Normalize heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Resize heatmap to match input size
            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=input_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            gradcam_maps.append(heatmap)
            
        return gradcam_maps[0] if len(gradcam_maps) == 1 else gradcam_maps

    def visualize_gradcam(self, inputs, gradcam_maps, save_path=None, show=True):
        """
        Visualize Grad-CAM++ heatmaps with consistent color scaling
        
        Args:
            inputs: List of input tensors or single input tensor
            gradcam_maps: Grad-CAM++ heatmaps from compute_gradcam
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(gradcam_maps, list):
            gradcam_maps = [gradcam_maps]
            
        n_inputs = len(inputs)
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(4*n_inputs + 1.5, 8))
        # Create a grid with extra space on the right for colorbar
        gs = plt.GridSpec(2, n_inputs, figure=fig, right=0.85)
        axes = np.empty((2, n_inputs), dtype=object)
        
        # Create subplots
        for i in range(n_inputs):
            axes[0, i] = fig.add_subplot(gs[0, i])
            axes[1, i] = fig.add_subplot(gs[1, i])
        
        # Find global min and max for consistent color scaling
        global_min = min(heatmap.min().item() for heatmap in gradcam_maps)
        global_max = max(heatmap.max().item() for heatmap in gradcam_maps)
        
        for i, (input_tensor, heatmap) in enumerate(zip(inputs, gradcam_maps)):
            # Original image
            img = input_tensor.cpu().detach()
            if img.dim() == 4:  # (B, C, H, W)
                img = img.squeeze(0)  # Remove batch dimension
            if img.dim() == 3:  # (C, H, W)
                if img.shape[0] == 1:  # Single channel
                    img = img.squeeze(0)  # Remove channel dimension
                else:  # Multiple channels
                    img = img.permute(1, 2, 0)  # Convert to (H, W, C)
            img = img.numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Heatmap
            heatmap = heatmap.cpu().numpy()
            
            # Plot original image
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            
            # Plot heatmap with consistent color scaling
            im = axes[1, i].imshow(heatmap, cmap='jet', vmin=global_min, vmax=global_max)
            axes[1, i].set_title(f"Grad-CAM++ {i+1}")
            axes[1, i].axis('off')
        
        # Add colorbar in a separate axes
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        if show:
            plt.show()
        plt.close()

    def compute_gradcam_plus_plus(self, inputs, target_class=None, target_dim=None):
        """
        Compute Grad-CAM++ visualization for the input
        
        Args:
            inputs: List of input tensors or single input tensor
            target_class: Index of target class for classification tasks, or None for regression
            target_dim: For regression tasks, which output dimension to visualize. If None, uses mean of all outputs.
            
        Returns:
            Grad-CAM++ heatmap for each input
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        gradcam_plus_plus_maps = []
        
        for input_tensor in inputs:
            # Forward pass
            output = self.model(input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor)
            
            # For regression models
            if target_class is None:
                if target_dim is not None:
                    # Use specific output dimension
                    target = output[:, target_dim]
                else:
                    # Use mean of all outputs for visualization
                    target = output.mean()
            # For classification models
            else:
                target = output[:, target_class]
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass
            target.backward(retain_graph=True)
            
            # Get weights and activations using Grad-CAM++ method
            weights = self._get_gradcam_weights(target_class)
            activations = self.activations[0]  # Remove batch dimension
            
            # Weight the activations
            weighted_activations = torch.zeros_like(activations)
            for i, w in enumerate(weights[0]):
                weighted_activations[i] = w * activations[i]
            
            # Generate heatmap
            heatmap = torch.sum(weighted_activations, dim=0)
            heatmap = torch.relu(heatmap)  # Apply ReLU
            
            # Normalize heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Resize heatmap to match input size
            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=input_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            gradcam_plus_plus_maps.append(heatmap)
            
        return gradcam_plus_plus_maps[0] if len(gradcam_plus_plus_maps) == 1 else gradcam_plus_plus_maps

    def compute_guided_gradcam(self, inputs, target_class=None, target_dim=None):
        """
        Compute Guided Grad-CAM visualization for the input
        
        Args:
            inputs: List of input tensors or single input tensor
            target_class: Index of target class for classification tasks, or None for regression
            target_dim: For regression tasks, which output dimension to visualize. If None, uses mean of all outputs.
            
        Returns:
            Guided Grad-CAM heatmap for each input
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        guided_gradcam_maps = []
        
        for input_tensor in inputs:
            # Get Grad-CAM heatmap
            gradcam = self.compute_gradcam(input_tensor, target_class, target_dim)
            
            # Get guided backpropagation
            guided_grad = self._compute_guided_backprop(input_tensor, target_class, target_dim)
            
            # Combine Grad-CAM with guided backpropagation
            guided_gradcam = guided_grad * gradcam.unsqueeze(0)
            
            # Normalize
            guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (guided_gradcam.max() - guided_gradcam.min() + 1e-8)
            
            guided_gradcam_maps.append(guided_gradcam)
            
        return guided_gradcam_maps[0] if len(guided_gradcam_maps) == 1 else guided_gradcam_maps
    
    def _compute_guided_backprop(self, input_tensor, target_class=None, target_dim=None):
        """
        Compute guided backpropagation gradients
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (for classification)
            target_dim: For regression tasks, which output dimension to visualize. If None, uses mean of all outputs.
            
        Returns:
            Guided backpropagation gradients
        """
        # Register hooks for guided backpropagation
        guided_gradients = []
        
        def guided_relu_hook(module, grad_in, grad_out):
            # Only backpropagate positive gradients
            guided_gradients.append(grad_in[0].clamp(min=0))
        
        # Register hooks for all ReLU layers
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                hooks.append(module.register_backward_hook(guided_relu_hook))
        
        # Forward pass
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        # For regression models
        if target_class is None:
            if target_dim is not None:
                # Use specific output dimension
                target = output[:, target_dim]
            else:
                # Use mean of all outputs for visualization
                target = output.mean()
        # For classification models
        else:
            target = output[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Get guided gradients
        guided_grad = input_tensor.grad.data[0]
        
        # Normalize
        guided_grad = (guided_grad - guided_grad.min()) / (guided_grad.max() - guided_grad.min() + 1e-8)
        
        return guided_grad
    
    def visualize_guided_gradcam(self, inputs, guided_gradcam_maps, save_path=None, show=True):
        """
        Visualize Guided Grad-CAM heatmaps
        
        Args:
            inputs: List of input tensors or single input tensor
            guided_gradcam_maps: Guided Grad-CAM heatmaps from compute_guided_gradcam
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(guided_gradcam_maps, list):
            guided_gradcam_maps = [guided_gradcam_maps]
            
        n_inputs = len(inputs)
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(4*n_inputs + 1.5, 8))
        # Create a grid with extra space on the right for colorbar
        gs = plt.GridSpec(2, n_inputs, figure=fig, right=0.85)
        axes = np.empty((2, n_inputs), dtype=object)
        
        # Create subplots
        for i in range(n_inputs):
            axes[0, i] = fig.add_subplot(gs[0, i])
            axes[1, i] = fig.add_subplot(gs[1, i])
        
        # Find global min and max for consistent color scaling
        global_min = min(heatmap.min().item() for heatmap in guided_gradcam_maps)
        global_max = max(heatmap.max().item() for heatmap in guided_gradcam_maps)
        
        for i, (input_tensor, heatmap) in enumerate(zip(inputs, guided_gradcam_maps)):
            # Original image
            img = input_tensor.cpu().detach()
            if img.dim() == 4:  # (B, C, H, W)
                img = img.squeeze(0)  # Remove batch dimension
            if img.dim() == 3:  # (C, H, W)
                if img.shape[0] == 1:  # Single channel
                    img = img.squeeze(0)  # Remove channel dimension
                else:  # Multiple channels
                    img = img.permute(1, 2, 0)  # Convert to (H, W, C)
            img = img.numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Heatmap
            heatmap = heatmap.cpu().detach()
            if heatmap.dim() == 3:  # (C, H, W)
                if heatmap.shape[0] == 1:  # Single channel
                    heatmap = heatmap.squeeze(0)  # Remove channel dimension
                else:  # Multiple channels
                    heatmap = heatmap.permute(1, 2, 0)  # Convert to (H, W, C)
            heatmap = heatmap.numpy()
            
            # Plot original image
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            
            # Plot heatmap with consistent color scaling
            if heatmap.ndim == 2:  # Single channel heatmap
                im = axes[1, i].imshow(heatmap, cmap='jet', vmin=global_min, vmax=global_max)
            else:  # RGB heatmap
                im = axes[1, i].imshow(heatmap)
            axes[1, i].set_title(f"Guided Grad-CAM {i+1}")
            axes[1, i].axis('off')
        
        # Add colorbar in a separate axes
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        if show:
            plt.show()
        plt.close()

    def visualize_gradcam_plus_plus(self, inputs, gradcam_plus_plus_maps, save_path=None, show=True):
        """
        Visualize Grad-CAM++ heatmaps with consistent color scaling
        
        Args:
            inputs: List of input tensors or single input tensor
            gradcam_plus_plus_maps: Grad-CAM++ heatmaps from compute_gradcam_plus_plus
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(gradcam_plus_plus_maps, list):
            gradcam_plus_plus_maps = [gradcam_plus_plus_maps]
            
        n_inputs = len(inputs)
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(4*n_inputs + 1.5, 8))
        # Create a grid with extra space on the right for colorbar
        gs = plt.GridSpec(2, n_inputs, figure=fig, right=0.85)
        axes = np.empty((2, n_inputs), dtype=object)
        
        # Create subplots
        for i in range(n_inputs):
            axes[0, i] = fig.add_subplot(gs[0, i])
            axes[1, i] = fig.add_subplot(gs[1, i])
        
        # Find global min and max for consistent color scaling
        global_min = min(heatmap.min().item() for heatmap in gradcam_plus_plus_maps)
        global_max = max(heatmap.max().item() for heatmap in gradcam_plus_plus_maps)
        
        for i, (input_tensor, heatmap) in enumerate(zip(inputs, gradcam_plus_plus_maps)):
            # Original image
            img = input_tensor.cpu().detach()
            if img.dim() == 4:  # (B, C, H, W)
                img = img.squeeze(0)  # Remove batch dimension
            if img.dim() == 3:  # (C, H, W)
                if img.shape[0] == 1:  # Single channel
                    img = img.squeeze(0)  # Remove channel dimension
                else:  # Multiple channels
                    img = img.permute(1, 2, 0)  # Convert to (H, W, C)
            img = img.numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Heatmap
            heatmap = heatmap.cpu().numpy()
            
            # Plot original image
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            
            # Plot heatmap with consistent color scaling
            im = axes[1, i].imshow(heatmap, cmap='jet', vmin=global_min, vmax=global_max)
            axes[1, i].set_title(f"Grad-CAM++ {i+1}")
            axes[1, i].axis('off')
        
        # Add colorbar in a separate axes
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        if show:
            plt.show()
        plt.close()

    def create_saliency_maps(self, inputs, target_class=None):
        """
        Create saliency maps for a given input
        
        Args:
            inputs: List of input tensors
        """
        # Forward pass to get the output

    def show(img, title=None):
        plt.axis('off')
        if title: plt.title(title)
        plt.imshow(img, cmap=None)
        
    def overlay_heatmap(img, sal, alpha=0.5):
        sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        heat = plt.get_cmap('jet')(sal_norm)[..., :3]          # RGB heat-map
        return (alpha * heat + (1 - alpha) * img).clip(0, 1)

    def compute_smoothgrad(self, inputs, target_class=None, target_dim=None):
        """
        Compute SmoothGrad visualization for the input
        
        Args:
            inputs: List of input tensors or single input tensor
            target_class: Index of target class for classification tasks, or None for regression
            target_dim: For regression tasks, which output dimension to visualize. If None, uses mean of all outputs.
            
        Returns:
            SmoothGrad heatmap for each input
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        smoothgrad_maps = []
        
        for input_tensor in inputs:
            # Ensure input has correct dimensions (B, C, H, W)
            if input_tensor.dim() == 2:  # (H, W)
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif input_tensor.dim() == 3:  # (C, H, W) or (B, H, W)
                if input_tensor.shape[0] == 1:  # (1, H, W) - single channel
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dim
                else:  # (B, H, W)
                    input_tensor = input_tensor.unsqueeze(1)  # Add channel dim
            
            # Initialize accumulated gradients with same shape as input
            accumulated_gradients = torch.zeros_like(input_tensor)
            
            # Compute gradients for multiple noisy samples
            for _ in range(self.n_samples):
                # Add noise to input (noise will have same shape as input)
                noise = torch.randn_like(input_tensor) * self.noise_level
                noisy_input = input_tensor + noise
                noisy_input.requires_grad_(True)
                
                # Forward pass
                output = self.model(noisy_input)
                
                # For regression models
                if target_class is None:
                    if target_dim is not None:
                        target = output[:, target_dim]
                    else:
                        target = output.mean()
                # For classification models
                else:
                    target = output[:, target_class]
                
                # Zero gradients
                self.model.zero_grad()
                
                # Backward pass
                target.backward(retain_graph=True)
                
                # Accumulate gradients
                if noisy_input.grad is not None:
                    accumulated_gradients += noisy_input.grad.detach()
            
            # Average the gradients
            smoothgrad = accumulated_gradients / self.n_samples
            
            # Take absolute value and normalize
            smoothgrad = torch.abs(smoothgrad)
            
            # Remove batch dimension and handle channels
            smoothgrad = smoothgrad.squeeze(0)  # Remove batch dimension
            if smoothgrad.shape[0] == 1:  # Single channel
                smoothgrad = smoothgrad.squeeze(0)  # Remove channel dimension
            else:  # Multiple channels
                smoothgrad = smoothgrad.mean(dim=0)  # Average across channels
            
            # Normalize heatmap
            smoothgrad = (smoothgrad - smoothgrad.min()) / (smoothgrad.max() - smoothgrad.min() + 1e-8)
            
            smoothgrad_maps.append(smoothgrad)
        
        return smoothgrad_maps[0] if len(smoothgrad_maps) == 1 else smoothgrad_maps

    def visualize_smoothgrad(self, inputs, smoothgrad_maps, save_path=None, show=True):
        """
        Visualize SmoothGrad heatmaps
        
        Args:
            inputs: List of input tensors or single input tensor
            smoothgrad_maps: SmoothGrad heatmaps from compute_smoothgrad
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(smoothgrad_maps, list):
            smoothgrad_maps = [smoothgrad_maps]
            
        n_inputs = len(inputs)
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(4*n_inputs + 1.5, 8))
        # Create a grid with extra space on the right for colorbar
        gs = plt.GridSpec(2, n_inputs, figure=fig, right=0.85)
        axes = np.empty((2, n_inputs), dtype=object)
        
        # Create subplots
        for i in range(n_inputs):
            axes[0, i] = fig.add_subplot(gs[0, i])
            axes[1, i] = fig.add_subplot(gs[1, i])
        
        # Find global min and max for consistent color scaling
        global_min = min(heatmap.min().item() for heatmap in smoothgrad_maps)
        global_max = max(heatmap.max().item() for heatmap in smoothgrad_maps)
        
        for i, (input_tensor, heatmap) in enumerate(zip(inputs, smoothgrad_maps)):
            # Original image
            img = input_tensor.cpu().detach()
            if img.dim() == 4:  # (B, C, H, W)
                img = img.squeeze(0)  # Remove batch dimension
            if img.dim() == 3:  # (C, H, W)
                if img.shape[0] == 1:  # Single channel
                    img = img.squeeze(0)  # Remove channel dimension
                else:  # Multiple channels
                    img = img.permute(1, 2, 0)  # Convert to (H, W, C)
            img = img.numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Heatmap
            heatmap = heatmap.cpu().numpy()
            
            # Plot original image
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            
            # Plot heatmap with consistent color scaling
            im = axes[1, i].imshow(heatmap, cmap='jet', vmin=global_min, vmax=global_max)
            axes[1, i].set_title(f"SmoothGrad {i+1}")
            axes[1, i].axis('off')
        
        # Add colorbar in a separate axes
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        if show:
            plt.show()
        plt.close()

    def compute_vanilla_gradient(self, inputs, target_class=None, target_dim=None):
        """
        Compute Vanilla Gradient visualization for the input
        
        Args:
            inputs: List of input tensors or single input tensor
            target_class: Index of target class for classification tasks, or None for regression
            target_dim: For regression tasks, which output dimension to visualize. If None, uses mean of all outputs.
            
        Returns:
            Vanilla gradient heatmap for each input
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        gradient_maps = []
        
        for input_tensor in inputs:
            # Ensure input has correct dimensions (B, C, H, W)
            if input_tensor.dim() == 2:  # (H, W)
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif input_tensor.dim() == 3:  # (C, H, W) or (B, H, W)
                if input_tensor.shape[0] == 1:  # (1, H, W) - single channel
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dim
                else:  # (B, H, W)
                    input_tensor = input_tensor.unsqueeze(1)  # Add channel dim
            
            # Make input require gradients
            input_tensor.requires_grad_(True)
            
            # Forward pass
            output = self.model(input_tensor)
            
            # For regression models
            if target_class is None:
                if target_dim is not None:
                    target = output[:, target_dim]
                else:
                    target = output.mean()
            # For classification models
            else:
                target = output[:, target_class]
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass
            target.backward(retain_graph=True)
            
            # Get gradients
            gradients = input_tensor.grad.detach()
            
            # Take absolute value and normalize
            gradients = torch.abs(gradients)
            
            # Remove batch dimension and handle channels
            gradients = gradients.squeeze(0)  # Remove batch dimension
            if gradients.shape[0] == 1:  # Single channel
                gradients = gradients.squeeze(0)  # Remove channel dimension
            else:  # Multiple channels
                gradients = gradients.mean(dim=0)  # Average across channels
            
            # Normalize heatmap
            gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
            
            gradient_maps.append(gradients)
        
        return gradient_maps[0] if len(gradient_maps) == 1 else gradient_maps

    def visualize_vanilla_gradient(self, inputs, gradient_maps, save_path=None, show=True):
        """
        Visualize Vanilla Gradient heatmaps
        
        Args:
            inputs: List of input tensors or single input tensor
            gradient_maps: Vanilla gradient heatmaps from compute_vanilla_gradient
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(gradient_maps, list):
            gradient_maps = [gradient_maps]
            
        n_inputs = len(inputs)
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(4*n_inputs + 1.5, 8))
        # Create a grid with extra space on the right for colorbar
        gs = plt.GridSpec(2, n_inputs, figure=fig, right=0.85)
        axes = np.empty((2, n_inputs), dtype=object)
        
        # Create subplots
        for i in range(n_inputs):
            axes[0, i] = fig.add_subplot(gs[0, i])
            axes[1, i] = fig.add_subplot(gs[1, i])
        
        # Find global min and max for consistent color scaling
        global_min = min(heatmap.min().item() for heatmap in gradient_maps)
        global_max = max(heatmap.max().item() for heatmap in gradient_maps)
        
        for i, (input_tensor, heatmap) in enumerate(zip(inputs, gradient_maps)):
            # Original image
            img = input_tensor.cpu().detach()
            if img.dim() == 4:  # (B, C, H, W)
                img = img.squeeze(0)  # Remove batch dimension
            if img.dim() == 3:  # (C, H, W)
                if img.shape[0] == 1:  # Single channel
                    img = img.squeeze(0)  # Remove channel dimension
                else:  # Multiple channels
                    img = img.permute(1, 2, 0)  # Convert to (H, W, C)
            img = img.numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Heatmap
            heatmap = heatmap.cpu().numpy()
            
            # Plot original image
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            
            # Plot heatmap with consistent color scaling
            im = axes[1, i].imshow(heatmap, cmap='jet', vmin=global_min, vmax=global_max)
            axes[1, i].set_title(f"Vanilla Gradient {i+1}")
            axes[1, i].axis('off')
        
        # Add colorbar in a separate axes
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        if show:
            plt.show()
        plt.close()

    def visualize_all_masks(self, inputs, masks_dict, save_path=None, show=True):
        """
        Visualize multiple interpretation masks side by side with a shared colorbar
        
        Args:
            inputs: List of input tensors or single input tensor
            masks_dict: Dictionary of interpretation masks, where keys are method names and values are the masks
                       e.g., {'Grad-CAM': gradcam_maps, 'SmoothGrad': smoothgrad_maps, 'Vanilla': vanilla_maps}
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        n_inputs = len(inputs)
        n_methods = len(masks_dict)
        
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(4*(n_methods + 1) + 1.5, 4*n_inputs))
        # Create a grid with extra space on the right for colorbar
        gs = plt.GridSpec(n_inputs, n_methods + 1, figure=fig, right=0.85)
        axes = np.empty((n_inputs, n_methods + 1), dtype=object)
        
        # Create subplots
        for i in range(n_inputs):
            for j in range(n_methods + 1):
                axes[i, j] = fig.add_subplot(gs[i, j])
        
        # Find global min and max for consistent color scaling across all masks
        all_masks = []
        for masks in masks_dict.values():
            if not isinstance(masks, list):
                masks = [masks]
            all_masks.extend(masks)
        
        global_min = min(mask.min().item() for mask in all_masks)
        global_max = max(mask.max().item() for mask in all_masks)
        
        # Plot for each input
        for i, input_tensor in enumerate(inputs):
            # Original image
            img = input_tensor.cpu().detach()
            if img.dim() == 4:  # (B, C, H, W)
                img = img.squeeze(0)  # Remove batch dimension
            if img.dim() == 3:  # (C, H, W)
                if img.shape[0] == 1:  # Single channel
                    img = img.squeeze(0)  # Remove channel dimension
                else:  # Multiple channels
                    img = img.permute(1, 2, 0)  # Convert to (H, W, C)
            img = img.numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Plot original image
            axes[i, 0].imshow(img, cmap='gray' if img.ndim == 2 else None)
            if i == 0:  # Only show title for first row
                axes[i, 0].set_title("Input")
            axes[i, 0].axis('off')
            
            # Plot each mask
            for j, (method_name, masks) in enumerate(masks_dict.items(), start=1):
                mask = masks[i] if isinstance(masks, list) else masks
                mask = mask.cpu().detach()
                
                # Handle different mask dimensions
                if mask.dim() == 3:  # (C, H, W) or (B, H, W)
                    if mask.shape[0] == 1:  # Single channel
                        mask = mask.squeeze(0)  # Remove channel dimension
                    else:  # Multiple channels
                        mask = mask.mean(dim=0)  # Average across channels
                elif mask.dim() == 4:  # (B, C, H, W)
                    mask = mask.squeeze(0)  # Remove batch dimension
                    if mask.shape[0] == 1:  # Single channel
                        mask = mask.squeeze(0)  # Remove channel dimension
                    else:  # Multiple channels
                        mask = mask.mean(dim=0)  # Average across channels
                
                # Convert to numpy and ensure 2D
                mask = mask.numpy()
                if mask.ndim != 2:
                    raise ValueError(f"Invalid mask shape {mask.shape} for {method_name}. Expected 2D array.")
                
                # Plot mask with consistent color scaling
                im = axes[i, j].imshow(mask, cmap='jet', vmin=global_min, vmax=global_max)
                if i == 0:  # Only show title for first row
                    axes[i, j].set_title(method_name)
                axes[i, j].axis('off')
        
        # Add colorbar in a separate axes
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        if show:
            plt.show()
        plt.close()

    def check_nonzero_percentage(tensor, percentage):
        """
        Check if a given percentage of tensor elements are nonzero
        
        Args:
            tensor: Input tensor
            percentage: Target percentage (0-100) of nonzero elements
            
        Returns:
            bool: True if the percentage of nonzero elements is >= the target percentage
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
            
        # Count nonzero elements
        nonzero_count = torch.count_nonzero(tensor).item()
        total_elements = tensor.numel()
        
        # Calculate actual percentage
        actual_percentage = (nonzero_count / total_elements) * 100
        
        return actual_percentage >= percentage

    def _get_scorecam_weights(self, input_tensor, target_class=None, target_dim=None, n_samples=32):
        """
        Compute Score-CAM weights using channel-wise importance scores
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (for classification tasks)
            target_dim: For regression tasks, which output dimension to visualize
            n_samples: Number of samples for importance score computation
            
        Returns:
            Channel-wise importance scores
        """
        if self.activations is None:
            raise ValueError("No activations captured. Run forward pass first.")
            
        # Get activations from the target layer
        activations = self.activations[0]  # Remove batch dimension
        n_channels = activations.shape[0]
        
        # Initialize importance scores
        importance_scores = torch.zeros(n_channels, device=input_tensor.device)
        
        # Create baseline (black image)
        baseline = torch.zeros_like(input_tensor)
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor)
            if target_class is not None:
                original_score = original_output[0, target_class]
            elif target_dim is not None:
                original_score = original_output[0, target_dim]
            else:
                original_score = original_output.mean()
        
        # Compute importance scores for each channel
        for k in range(n_channels):
            # Create masked input using upsampled activation map
            upsampled_activation = torch.nn.functional.interpolate(
                activations[k:k+1].unsqueeze(0),
                size=input_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Normalize activation map
            upsampled_activation = (upsampled_activation - upsampled_activation.min()) / \
                                 (upsampled_activation.max() - upsampled_activation.min() + 1e-8)
            
            # Create masked input
            masked_input = baseline + upsampled_activation * (input_tensor - baseline)
            
            # Get prediction for masked input
            with torch.no_grad():
                masked_output = self.model(masked_input.unsqueeze(0))
                if target_class is not None:
                    masked_score = masked_output[0, target_class]
                elif target_dim is not None:
                    masked_score = masked_output[0, target_dim]
                else:
                    masked_score = masked_output.mean()
            
            # Compute importance score
            importance_scores[k] = torch.relu(masked_score - original_score)
        
        return importance_scores

    def compute_scorecam(self, inputs, target_class=None, target_dim=None, n_samples=32):
        """
        Compute Score-CAM visualization for the input
        
        Args:
            inputs: List of input tensors or single input tensor
            target_class: Index of target class for classification tasks, or None for regression
            target_dim: For regression tasks, which output dimension to visualize. If None, uses mean of all outputs.
            n_samples: Number of samples for importance score computation
            
        Returns:
            Score-CAM heatmap for each input
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        scorecam_maps = []
        
        for input_tensor in inputs:
            # Forward pass to get activations
            _ = self.model(input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor)
            
            # Get importance scores
            importance_scores = self._get_scorecam_weights(
                input_tensor, 
                target_class=target_class,
                target_dim=target_dim,
                n_samples=n_samples
            )
            
            # Get activations
            activations = self.activations[0]  # Remove batch dimension
            
            # Weight the activations
            weighted_activations = torch.zeros_like(activations)
            for i, w in enumerate(importance_scores):
                weighted_activations[i] = w * activations[i]
            
            # Generate heatmap
            heatmap = torch.sum(weighted_activations, dim=0)
            heatmap = torch.relu(heatmap)  # Apply ReLU
            
            # Normalize heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Resize heatmap to match input size
            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=input_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            scorecam_maps.append(heatmap)
            
        return scorecam_maps[0] if len(scorecam_maps) == 1 else scorecam_maps

    def visualize_scorecam(self, inputs, scorecam_maps, save_path=None, show=True):
        """
        Visualize Score-CAM heatmaps with consistent color scaling
        
        Args:
            inputs: List of input tensors or single input tensor
            scorecam_maps: Score-CAM heatmaps from compute_scorecam
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(scorecam_maps, list):
            scorecam_maps = [scorecam_maps]
            
        n_inputs = len(inputs)
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(4*n_inputs + 1.5, 8))
        # Create a grid with extra space on the right for colorbar
        gs = plt.GridSpec(2, n_inputs, figure=fig, right=0.85)
        axes = np.empty((2, n_inputs), dtype=object)
        
        # Create subplots
        for i in range(n_inputs):
            axes[0, i] = fig.add_subplot(gs[0, i])
            axes[1, i] = fig.add_subplot(gs[1, i])
        
        # Find global min and max for consistent color scaling
        global_min = min(heatmap.min().item() for heatmap in scorecam_maps)
        global_max = max(heatmap.max().item() for heatmap in scorecam_maps)
        
        for i, (input_tensor, heatmap) in enumerate(zip(inputs, scorecam_maps)):
            # Original image
            img = input_tensor.cpu().detach()
            if img.dim() == 4:  # (B, C, H, W)
                img = img.squeeze(0)  # Remove batch dimension
            if img.dim() == 3:  # (C, H, W)
                if img.shape[0] == 1:  # Single channel
                    img = img.squeeze(0)  # Remove channel dimension
                else:  # Multiple channels
                    img = img.permute(1, 2, 0)  # Convert to (H, W, C)
            img = img.numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Heatmap
            heatmap = heatmap.cpu().numpy()
            
            # Plot original image
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            
            # Plot heatmap with consistent color scaling
            im = axes[1, i].imshow(heatmap, cmap='jet', vmin=global_min, vmax=global_max)
            axes[1, i].set_title(f"Score-CAM {i+1}")
            axes[1, i].axis('off')
        
        # Add colorbar in a separate axes
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        if show:
            plt.show()
        plt.close()


class IntegratedGradients:
    def __init__(self, model, steps=50):
        self.model = model
        self.steps = steps
        self.model.eval()  # Set model to evaluation mode
    
    def generate_baseline(self, input_tensor):
        """Generate a baseline (typically black images)"""
        return torch.zeros_like(input_tensor)
    
    def compute_integrated_gradients(self, inputs, target_class=None):
        """
        Compute integrated gradients for CNN models
        
        Args:
            inputs: List of 6 input tensors or a single input
            target_class: Index of target class for classification tasks, 
                         or None for regression
        
        Returns:
            List of attribution maps, one for each input branch
        """
        # Create baseline inputs (typically zeros)
        baselines = [self.generate_baseline(inp) for inp in inputs]
        
        # Store integrated gradients for each branch
        all_integrated_grads = []
        
        # We need to compute integrated gradients for each branch separately
        for branch_idx in range(len(inputs)):
            integrated_grad = torch.zeros_like(inputs[branch_idx]).float()
            
            # Compute integral approximation using Riemann sum
            for step in tqdm(range(self.steps), desc=f"Branch {branch_idx+1}"):
                alpha = step / self.steps
                
                # Create interpolated inputs
                interpolated_inputs = []
                for i, (inp, baseline) in enumerate(zip(inputs, baselines)):
                    if i == branch_idx:
                        # Only interpolate the current branch
                        interp = baseline + alpha * (inp - baseline)
                        interp.requires_grad_(True)
                    else:
                        # Keep other branches as they are
                        interp = inp.clone().detach().requires_grad_(True)
                    interpolated_inputs.append(interp)
                
                # Forward pass
                output = self.model(interpolated_inputs)
                
                # For regression models
                if target_class is None:
                    target = output
                # For classification models
                else:
                    target = output[:, target_class]
                
                # Zero all existing gradients
                self.model.zero_grad()
                
                # Calculate gradients
                target.backward(torch.ones_like(target), retain_graph=True)
                
                # Get gradients and accumulate
                grad = interpolated_inputs[branch_idx].grad.data
                integrated_grad += grad / self.steps
            
            # Scale the integrated gradients with the input-baseline difference
            integrated_grad *= (inputs[branch_idx] - baselines[branch_idx])
            all_integrated_grads.append(integrated_grad)
        
        return all_integrated_grads
    
    def visualize_attributions(self, inputs, attributions, save_path=None):
        """
        Visualize attributions for each branch
        
        Args:
            inputs: List of input tensors
            attributions: List of attribution tensors (output from compute_integrated_gradients)
            save_path: Optional path to save the visualization
        """
        fig, axs = plt.subplots(2, 6, figsize=(18, 6))
        
        for i in range(6):
            # Original image
            img = inputs[i].cpu().detach().squeeze().permute(1, 2, 0).numpy()
            # Normalize for visualization
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axs[0, i].imshow(img)
            axs[0, i].set_title(f"Input {i+1}")
            axs[0, i].axis('off')
            
            # Attribution map
            attr = attributions[i].cpu().detach().abs().sum(dim=1).squeeze().numpy()
            # Normalize for visualization
            attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
            im = axs[1, i].imshow(attr, cmap='hot')
            axs[1, i].set_title(f"Attribution {i+1}")
            axs[1, i].axis('off')
        
        plt.colorbar(im, ax=axs[1, -1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
        
        # Calculate and print contribution percentages
        total_attr = sum(attr.sum() for attr in attributions)
        contributions = [100 * attr.abs().sum().item() / total_attr for attr in attributions]
        
        for i, contrib in enumerate(contributions):
            print(f"Branch {i+1} contribution: {contrib:.2f}%")
        
        return contributions


class BranchContributionAnalyzer:
    def __init__(self, model):
        """
        Initializes the analyzer for a multi-branch model
        
        Args:
            model: PyTorch model with 6 branches
        """
        self.model = model
        self.model.eval()
        self.branch_outputs = [None] * 6
        self.register_hooks()
        
    def register_hooks(self):
        """Register forward hooks to capture branch outputs"""
        def make_hook(branch_idx):
            def hook(module, input, output):
                self.branch_outputs[branch_idx] = output.detach()
            return hook
        
        # Find CNN branch output modules and register hooks
        # This implementation assumes your model has accessible branch outputs
        # You may need to modify this to match your specific architecture
        for i, branch_module in enumerate(self.model.branches):
            # Assuming the final layer of each branch is what we want to hook
            branch_module[-1].register_forward_hook(make_hook(i))
    
    def compute_leave_one_out_contributions(self, inputs, target=None):
        """
        Calculate contribution by leaving out each branch one at a time
        
        Args:
            inputs: List of 6 input image tensors
            target: Optional target for supervised evaluation
            
        Returns:
            Contribution percentage for each branch
        """
        # Full prediction (all branches)
        with torch.no_grad():
            full_output = self.model(inputs)
            
        # If target is provided, calculate task-specific error
        if target is not None:
            full_error = torch.nn.functional.mse_loss(full_output, target).item()
        else:
            # Use the output magnitude as reference
            full_error = torch.norm(full_output).item()
            
        contributions = []
        
        for i in range(len(inputs)):
            # Create zeroed input for this branch
            modified_inputs = list(inputs)
            modified_inputs[i] = torch.zeros_like(inputs[i])
            
            # Get prediction without this branch
            with torch.no_grad():
                modified_output = self.model(modified_inputs)
            
            # Calculate error difference when branch is removed
            if target is not None:
                modified_error = torch.nn.functional.mse_loss(modified_output, target).item()
            else:
                modified_error = torch.norm(modified_output).item()
                
            # Contribution is proportional to how much performance degrades when branch is removed
            branch_contribution = abs(modified_error - full_error)
            contributions.append(branch_contribution)
            
        # Normalize to percentages
        total_contribution = sum(contributions)
        if total_contribution > 0:  # Avoid division by zero
            contributions = [100 * c / total_contribution for c in contributions]
        else:
            contributions = [100 / len(inputs)] * len(inputs)  # Equal distribution if no effect
            
        return contributions
    
    def analyze_feature_importance(self, inputs):
        """
        Analyze and visualize feature importance in each branch
        
        Args:
            inputs: List of 6 input image tensors
            
        Returns:
            Feature importance metrics and visualizations
        """
        # Forward pass to get branch outputs
        with torch.no_grad():
            _ = self.model(inputs)
            
        # Calculate feature importance based on activation magnitude
        importance_metrics = []
        
        for i, branch_output in enumerate(self.branch_outputs):
            # Calculate per-channel activation magnitude
            # Assuming branch_output shape is [batch_size, channels, height, width]
            channel_importance = branch_output.abs().mean(dim=(0, 2, 3))
            
            # Get top channels
            top_channels = torch.argsort(channel_importance, descending=True)
            top_importance = channel_importance[top_channels]
            
            importance_metrics.append({
                'branch_idx': i,
                'channel_importance': channel_importance,
                'top_channels': top_channels,
                'top_importance': top_importance
            })
            
        return importance_metrics
    
    def visualize_branch_contributions(self, inputs, target=None, save_path=None):
        """
        Compute and visualize branch contributions
        
        Args:
            inputs: List of 6 input tensors
            target: Optional target tensor for supervised evaluation
            save_path: Optional path to save visualization
            
        Returns:
            Branch contribution percentages
        """
        contributions = self.compute_leave_one_out_contributions(inputs, target)
        
        # Create horizontal bar chart
        branch_names = [f"Branch {i+1}" for i in range(len(contributions))]
        
        plt.figure(figsize=(10, 6))
        
        # Sort contributions for better visualization
        sorted_indices = np.argsort(contributions)
        sorted_contributions = [contributions[i] for i in sorted_indices]
        sorted_names = [branch_names[i] for i in sorted_indices]
        
        # Custom colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(contributions)))
        
        bars = plt.barh(sorted_names, sorted_contributions, color=colors)
        
        # Add percentage labels
        for bar, value in zip(bars, sorted_contributions):
            plt.text(
                bar.get_width() + 1, 
                bar.get_y() + bar.get_height()/2, 
                f"{value:.1f}%",
                va='center',
                fontweight='bold'
            )
        
        plt.xlabel('Contribution (%)', fontsize=14)
        plt.title('Branch Contribution to Final Prediction', fontsize=16)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.xlim(0, max(contributions) * 1.15)  # Add space for labels
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.show()
        
        return contributions
    
    def visualize_interactive_contributions(self, inputs, output=None, save_path=None):
        """
        Create interactive visualization of contributions using Plotly
        
        Args:
            inputs: List of 6 input tensors
            output: Model output (optional)
            save_path: Optional path to save HTML visualization
            
        Returns:
            None (displays interactive chart)
        """
        contributions = self.compute_leave_one_out_contributions(inputs)
        branch_names = [f"Branch {i+1}" for i in range(len(contributions))]
        
        # Basic contribution chart
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=("Branch Contributions", "Contribution Distribution")
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                y=branch_names,
                x=contributions,
                orientation='h',
                marker=dict(
                    color=contributions,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Contribution (%)"),
                ),
                text=[f"{c:.1f}%" for c in contributions],
                textposition='outside',
                name="Contribution"
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=branch_names,
                values=contributions,
                textinfo='label+percent',
                insidetextorientation='radial',
                marker=dict(
                    colors=px.colors.sequential.Viridis,
                    line=dict(color='white', width=2)
                )
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Branch Contribution Analysis",
            height=500,
            width=1000,
            showlegend=False,
            xaxis=dict(title="Contribution (%)"),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        if save_path:
            fig.write_html(save_path)
            
        fig.show()
        
    def visualize_feature_space(self, inputs, save_path=None):
        """
        Visualize the feature space using PCA to show how branches relate
        
        Args:
            inputs: List of 6 input tensors
            save_path: Optional path to save visualization
            
        Returns:
            PCA results and visualization
        """
        # Forward pass to get branch outputs
        with torch.no_grad():
            _ = self.model(inputs)
            
        # Extract feature vectors from each branch
        feature_vectors = []
        for branch_output in self.branch_outputs:
            # Flatten spatial dimensions to get feature vector
            # Assuming branch_output shape: [batch_size, channels, height, width]
            features = branch_output.mean(dim=(2, 3)).cpu().numpy()
            feature_vectors.append(features)
            
        # Concatenate all features
        all_features = np.vstack(feature_vectors)
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(all_features)
        
        # Split back by branch
        branch_pca = np.split(reduced_features, len(feature_vectors))
        
        # Plot PCA results
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10.colors[:len(branch_pca)]
        
        for i, (features, color) in enumerate(zip(branch_pca, colors)):
            plt.scatter(
                features[:, 0], 
                features[:, 1],
                color=color,
                label=f"Branch {i+1}",
                s=100,
                alpha=0.7
            )
            
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.title("PCA of Branch Feature Spaces")
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.show()
        
        return pca, branch_pca

class AttentionVisualizer:
    def __init__(self, model):
        """
        Initialize visualizer for a model with attention mechanisms
        
        Args:
            model: PyTorch model with attention mechanisms
        """
        self.model = model
        self.attention_scores = None
        self.register_hooks()
    
    def register_hooks(self):
        """
        Register forward hooks to capture attention scores
        This is an example implementation - you'll need to adapt to your specific architecture
        """
        def attention_hook(module, input, output):
            # Depending on your attention implementation, you might need to
            # extract the attention scores differently
            # This assumes output[1] contains attention weights
            self.attention_scores = output[1].detach()
        
        # Find the attention modules in your model
        # This is a simplified example - you'll need to find your actual attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                module.register_forward_hook(attention_hook)
    
    def get_attention_for_branches(self, inputs):
        """
        Get attention weights for each branch when processing the given inputs
        
        Args:
            inputs: List of 6 input tensors, one for each branch
        
        Returns:
            Attention weights for each branch and each head
        """
        # Forward pass to trigger hooks
        with torch.no_grad():
            _ = self.model(inputs)
        
        if self.attention_scores is None:
            raise ValueError("No attention scores were captured. Check your model architecture.")
            
        # Extract and reshape attention scores based on your architecture
        # This is a placeholder - adapt to your specific attention implementation
        # Assuming attention_scores shape: [batch_size, num_heads, seq_len, seq_len]
        # where seq_len corresponds to branches (6 in this case)
        
        # If your architecture has a different attention mechanism, you'll need to adapt this
        return self.attention_scores
    
    def visualize_attention_heatmap(self, attention_weights, save_path=None):
        """
        Visualize attention weights as heatmaps
        
        Args:
            attention_weights: Attention weights tensor from get_attention_for_branches
            save_path: Optional path to save the visualization
        """
        # Assuming attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
        # We'll visualize each attention head separately
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 4, 4))
        if num_heads == 1:
            axes = [axes]
            
        branch_labels = [f"Branch {i+1}" for i in range(seq_len)]
        
        # Custom colormap
        colors = ["#0000ff", "#4cc9f0", "#ffffff", "#f72585", "#ff0000"]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        
        for h, ax in enumerate(axes):
            head_weights = attention_weights[0, h].cpu().numpy()
            
            # Create heatmap
            sns.heatmap(
                head_weights, 
                ax=ax,
                cmap=cmap,
                vmin=0, 
                vmax=head_weights.max(),
                cbar=True,
                xticklabels=branch_labels,
                yticklabels=branch_labels,
                annot=True,
                fmt=".2f"
            )
            ax.set_title(f"Attention Head {h+1}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.show()
    
    def visualize_branch_contributions(self, attention_weights, save_path=None):
        """
        Calculate and visualize each branch's contribution based on attention
        
        Args:
            attention_weights: Attention weights tensor from get_attention_for_branches
            save_path: Optional path to save the visualization
        """
        # Sum attention across all heads for each branch
        # This is a simplification - actual contribution may be more complex
        branch_importance = attention_weights.sum(dim=1).mean(dim=1).cpu().numpy()[0]
        
        # Normalize to get percentages
        branch_contrib_percent = 100 * branch_importance / branch_importance.sum()
        
        # Plot as bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            [f"Branch {i+1}" for i in range(len(branch_contrib_percent))],
            branch_contrib_percent,
            color=plt.cm.viridis(np.linspace(0, 1, len(branch_contrib_percent)))
        )
        
        # Add values on top of bars
        for bar, value in zip(bars, branch_contrib_percent):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{value:.1f}%",
                ha='center',
                fontweight='bold'
            )
            
        plt.title('Branch Contribution Based on Attention Weights', fontsize=16)
        plt.ylabel('Contribution (%)', fontsize=14)
        plt.ylim(0, max(branch_contrib_percent) * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.show()
        
        return branch_contrib_percent
    
    def visualize_attention_flow(self, inputs, save_path=None):
        """
        Comprehensive visualization showing how attention flows between branches
        
        Args:
            inputs: List of 6 input tensors, one for each branch
            save_path: Optional path to save the visualization
        """
        attention_weights = self.get_attention_for_branches(inputs)
        avg_attention = attention_weights.mean(dim=1)[0].cpu().numpy()
        
        plt.figure(figsize=(12, 10))
        
        # Create a circular layout for branches
        n_branches = avg_attention.shape[0]
        angles = np.linspace(0, 2*np.pi, n_branches, endpoint=False).tolist()
        
        # Make it a full circle by repeating the first coordinate
        angles += angles[:1]
        
        # Branch positions
        branch_positions = []
        for angle in angles[:-1]:  # Exclude the repeated position
            x = np.cos(angle)
            y = np.sin(angle)
            branch_positions.append((x, y))
        
        # Draw branches as nodes
        for i, (x, y) in enumerate(branch_positions):
            circle = plt.Circle((x, y), 0.15, color=plt.cm.viridis(i/n_branches), alpha=0.8)
            plt.gca().add_patch(circle)
            plt.text(x, y, f"B{i+1}", ha='center', va='center', fontweight='bold')
        
        # Draw attention flow as arrows
        max_weight = avg_attention.max()
        min_thickness = 0.5
        max_thickness = 5
        
        for i, (x1, y1) in enumerate(branch_positions):
            for j, (x2, y2) in enumerate(branch_positions):
                if i != j:
                    weight = avg_attention[i, j]
                    # Skip negligible connections
                    if weight < 0.05 * max_weight:
                        continue
                        
                    # Calculate arrow properties
                    thickness = min_thickness + (max_thickness - min_thickness) * (weight / max_weight)
                    alpha = 0.6 * (weight / max_weight) + 0.2
                    
                    # Draw the arrow
                    dx = 0.8 * (x2 - x1)
                    dy = 0.8 * (y2 - y1)
                    plt.arrow(
                        x1, y1, dx, dy,
                        head_width=0.05,
                        head_length=0.1,
                        fc=plt.cm.plasma(weight/max_weight),
                        ec=plt.cm.plasma(weight/max_weight),
                        linewidth=thickness,
                        alpha=alpha,
                        length_includes_head=True
                    )
        
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.axis('off')
        plt.title('Attention Flow Between Branches', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.show()