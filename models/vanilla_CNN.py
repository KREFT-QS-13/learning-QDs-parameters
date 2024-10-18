import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import utilities.config as c

class VanillaCNN(nn.Module):
    def __init__(self, name):
        output_size = c.K*(c.K+1)//2 + c.K**2
        self.name = name

        super(VanillaCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),

            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256 * (c.RESOLUTION // 16) * (c.RESOLUTION // 16), 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        return self.network(x)
    

    # def __init__(self, input_channels, num_classes, conv_layers, fc_layers, pool_size, dropout_rate):
    #     super(VanillaCNN, self).__init__()
        
    #     # Build convolutional layers
    #     self.conv_layers = nn.ModuleList()
    #     in_channels = input_channels
    #     for conv in conv_layers:
    #         self.conv_layers.append(nn.Conv2d(in_channels, **conv))
    #         self.conv_layers.append(nn.ReLU())
    #         self.conv_layers.append(nn.MaxPool2d(pool_size))
    #         in_channels = conv['out_channels']
        
    #     # Build fully connected layers
    #     self.fc_layers = nn.ModuleList()
    #     for i in range(len(fc_layers)):
    #         if i == 0:
    #             self.fc_layers.append(nn.Linear(in_channels * (32 // (pool_size ** len(conv_layers)))**2, fc_layers[i]))
    #         else:
    #             self.fc_layers.append(nn.Linear(fc_layers[i-1], fc_layers[i]))
    #         self.fc_layers.append(nn.ReLU())
    #         self.fc_layers.append(nn.Dropout(dropout_rate))
        
    #     # Output layer
    #     self.fc_layers.append(nn.Linear(fc_layers[-1], num_classes))
        
    # def forward(self, x):
    #     # Convolutional layers
    #     for layer in self.conv_layers:
    #         x = layer(x)
        
    #     # Flatten
    #     x = x.view(x.size(0), -1)
        
    #     # Fully connected layers
    #     for layer in self.fc_layers:
    #         x = layer(x)
        
    #     return x