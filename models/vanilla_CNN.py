import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import utilities.config as c

class VanillaCNN(nn.Module):
    def __init__(self, name, num_conv_layers=4, filters_per_layer=None, kernel_sizes=None, 
                 fc_layers=None, dropout_rate=0.5):
        """
        Args:
            name (str): Name of the model
            num_conv_layers (int): Number of convolutional layers (default: 4)
            filters_per_layer (list): Number of filters for each conv layer. If None, uses [32, 64, 128, 256]
            kernel_sizes (list): Kernel sizes for each conv layer. If None, uses [3] * num_conv_layers
            fc_layers (list): Number of neurons in fully connected layers. If None, uses [1024]
            dropout_rate (float): Dropout rate for fully connected layers
        """
        super(VanillaCNN, self).__init__()
        
        self.name = name
        output_size = c.K*(c.K+1)//2 + c.K**2

        # Set default values if not provided
        if filters_per_layer is None:
            filters_per_layer = [32, 64, 128, 256][:num_conv_layers]
        if len(filters_per_layer) != num_conv_layers:
            raise ValueError(f"Length of filters_per_layer ({len(filters_per_layer)}) must match num_conv_layers ({num_conv_layers})")

        if kernel_sizes is None:
            kernel_sizes = [3] * num_conv_layers
        if len(kernel_sizes) != num_conv_layers:
            raise ValueError(f"Length of kernel_sizes ({len(kernel_sizes)}) must match num_conv_layers ({num_conv_layers})")

        if fc_layers is None:
            fc_layers = [1024]

        # Build convolutional layers
        layers = []
        in_channels = 1
        current_resolution = c.RESOLUTION

        for i in range(num_conv_layers):
            layers.extend([
                nn.Conv2d(in_channels, filters_per_layer[i], 
                         kernel_size=kernel_sizes[i], stride=1, 
                         padding=kernel_sizes[i]//2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(filters_per_layer[i])
            ])
            in_channels = filters_per_layer[i]
            current_resolution //= 2

        # Calculate input size for first FC layer
        fc_input_size = filters_per_layer[-1] * current_resolution * current_resolution

        # Add fully connected layers
        layers.append(nn.Flatten())
        layers.append(nn.Dropout(p=dropout_rate))

        for i in range(len(fc_layers)):
            if i == 0:
                layers.append(nn.Linear(fc_input_size, fc_layers[i]))
            else:
                layers.append(nn.Linear(fc_layers[i-1], fc_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

        # Add final output layer
        layers.append(nn.Linear(fc_layers[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
