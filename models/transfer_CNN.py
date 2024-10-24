import torch
import torch.nn as nn
from torchvision import models
import utilities.config as c

class TransferLearningCNN(nn.Module):
    def __init__(self, name="transfer_model", base_model='resnet18', pretrained=True):
        super(TransferLearningCNN, self).__init__()
        self.name = name
        self.base_model = self._get_base_model(base_model, pretrained)
        
        # Modify the first convolutional layer to accept single-channel input
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove the last fully connected layer
        
        if c.MODE == 1:
            output_size = c.K * (c.K + 1) // 2 + c.K**2
        elif c.MODE == 2:
            output_size = 2*c.K
        elif c.MODE == 3:
            output_size = c.K
 
        
        self.custom_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
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
