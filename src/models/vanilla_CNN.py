import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import src.utilities.config as c

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


class CustomCNN(nn.Module):
    def __init__(self, name,config_tuple, dropout=0.5, custom_head=None):
        super(CustomCNN, self).__init__()
        self.name = name

        K ,_ ,_ = config_tuple
        if c.MODE == 1:
            output_size = K * (K + 1) // 2 + K**2
        elif c.MODE == 2:
            output_size = 2*K
        elif c.MODE == 3:
            output_size = K

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 1 -> 16 filters
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # Downsample
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Downsample
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Downsample
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # No downsampling
        self.bn5 = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Downsample
        self.bn6 = nn.BatchNorm2d(256)  

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # Downsample
        self.bn7 = nn.BatchNorm2d(512)  

        self.conv_output_size = 512
        # Fully connected layers (custom head)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if custom_head is None:
            self.fc1 = nn.Linear(self.conv_output_size, 2048)  # Flattened output from conv layers
            self.fc2 = nn.Linear(2048, 1024)
            self.fc3 = nn.Linear(1024, output_size)  # Output size is 7
        elif len(custom_head) == 2:
            self.fc1 = nn.Linear(self.conv_output_size, custom_head[0])  # Flattened output from conv layers
            self.fc2 = nn.Linear(custom_head[0], custom_head[1])
            self.fc3 = nn.Linear(custom_head[1], output_size)  # Output size is 7
        else:
            raise ValueError(f"Custom head must be a list of length 2 or None, got {len(custom_head)}")
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Convolutional layers with ReLU and BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        x = self.avgpool(x)

        # Flatten before passing to the fully connected layers
        x = torch.flatten(x, 1) 
        # Fully connected layers with dropout and ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation for regression)
        
        return x
