import torch
import torch.nn as nn
from torchvision import models
import utilities.config as c

class ResNet(nn.Module):
    def __init__(self, K, name="transfer_model", base_model='resnet18', pretrained=True, 
                 dropout:float=None, custom_head:list=None):
        super(ResNet, self).__init__()
        self.name = name
        self.base_model = self._get_base_model(base_model, pretrained)
        self.custom_head = custom_head if custom_head is not None else None
        # Modify the first convolutional layer to accept single-channel input
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove the last fully connected layer
        
        if c.MODE == 1:
            output_size = K * (K + 1) // 2 + K**2
        elif c.MODE == 2:
            output_size = 2*K
        elif c.MODE == 3:
            output_size = K
 
        
        if custom_head is not None:
            layers = []
            in_features = num_features
            
            # Build layers based on the provided list
            for out_features in custom_head[:-1]:  # All layers except the last one
                layers.extend([
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                in_features = out_features
            
            # Add final layer to match required output size
            layers.append(nn.Linear(in_features, output_size))
            
            self.custom_head = nn.Sequential(*layers)
        else:
            # Default architecture if no custom_head is provided
            self.custom_head = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, output_size)
            )
        
    def _get_base_model(self, model_name, pretrained):
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet18(weights=weights)
        elif model_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet34(weights=weights)
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported base model: {model_name}")
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.custom_head(x)
        return x

    def __str__(self):
        return self.name
