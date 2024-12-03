import torch
import torch.nn as nn
from torchvision import models
import utilities.config as c
    
class ResNet(nn.Module):
    def __init__(self, config_tuple, name="transfer_model", base_model='resnet18', pretrained=True, 
                 dropout:float=None, custom_head:list=None, filters_per_layer:list=[16,32,64,128]):
        super(ResNet, self).__init__()
        self.name = name
        self.base_model = self._get_base_model(base_model, pretrained, filters_per_layer)
        self.custom_head = custom_head if custom_head is not None else None
        
        # Modify the first convolutional layer to accept single-channel input
        if base_model == 'resnet10' or base_model == 'resnet12':
            self.base_model.conv1 = nn.Conv2d(1, filters_per_layer[0], kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove the last fully connected layer
        
        K, N, S = config_tuple
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
        
    def _get_base_model(self, model_name, pretrained, filters_per_layer=None):
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet18(weights=weights)
        elif model_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet34(weights=weights)
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet50(weights=weights)
        # Custom ResNets
        elif model_name == 'resnet10' or model_name == 'resnet12': 
            num_layers = int(model_name.split('resnet')[-1])
            print(f"Using custom ResNet with {num_layers} layers. No pretrained weights used in the model.")
            return CustomResNet(num_layers, filters_per_layer) if filters_per_layer is not None else CustomResNet(num_layers)
        # elif 
        #     num_layers = int(model_name.split('resnet')[-1])
        #     print(f"Using custom ResNet with {num_layers} layers. No pretrained weights used in the model.")
        #     return CustomResNet(num_layers)
        else:
            raise ValueError(f"Unsupported base model: {model_name}")
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.custom_head(x)
        return x

    def __str__(self):
        return self.name

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample

    def forward(self, x):
        res = x
        if self.downsample is not None:
            res = self.downsample(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out += res
        out = nn.ReLU(inplace=True)(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, num_layers, filters_per_layer=[16,32,64,128]):
        super(CustomResNet, self).__init__()
        self.in_channels = 16 if filters_per_layer is None else filters_per_layer[0]
        self.num_layers = num_layers
        self.filters_per_layer = filters_per_layer
        
        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        if self.num_layers == 10:
            # ResNet10: 2 blocks per stage, filter sizes [16,32,64,128]
            self.layer1 = self._make_layer(self.filters_per_layer[0], 2, stride=1)
            self.layer2 = self._make_layer(self.filters_per_layer[1], 2, stride=2)
            self.layer3 = self._make_layer(self.filters_per_layer[2], 2, stride=2)
            self.layer4 = self._make_layer(self.filters_per_layer[3], 2, stride=2)
            self.fc_features = self.filters_per_layer[3]
        elif self.num_layers == 12:
            self.layer1 = self._make_layer(self.filters_per_layer[0], 2, stride=1)
            self.layer2 = self._make_layer(self.filters_per_layer[1], 2, stride=2)
            self.layer3 = self._make_layer(self.filters_per_layer[2],  3, stride=2)
            self.layer4 = self._make_layer(self.filters_per_layer[3], 3, stride=2)
            self.fc_features = self.filters_per_layer[3]
        else:
            raise NotImplementedError(f"ResNet{self.num_layers} is not implemented yet")
        
        # Average pooling and flatten
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.fc_features , 2)

    def _make_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        # First block might need downsampling
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
