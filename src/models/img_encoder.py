import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ContextEncoder(nn.Module):
    """
    Context encoder that processes context vectors through a small neural network.
    
    Args:
        context_vector_size (int): Size of input context vectors
        context_embedding_dim (int): Output dimension of context embedding
        dropout (float): Dropout rate
        hidden_dims (list, optional): List of hidden layer dimensions. If None, uses default 2-layer network.
    """
    def __init__(self, context_vector_size=13, context_embedding_dim=64, dropout=0.5, hidden_dims=None):
        super(ContextEncoder, self).__init__()
        self.context_vector_size = context_vector_size
        self.context_embedding_dim = context_embedding_dim
        
        # Normalize context vectors
        self.context_norm = nn.LayerNorm(context_vector_size)
        
        # Build context processing network (2-3 layers)
        if hidden_dims is None:
            # Default: 2-layer network
            hidden_dims = [64]
        
        layers = []
        in_features = context_vector_size
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(in_features, context_embedding_dim))
        
        self.context_mlp = nn.Sequential(*layers)

    def forward(self, context_vector):
        """
        Forward pass of context encoder.
        
        Args:
            context_vector (torch.Tensor): Input context vector of shape (batch_size, context_vector_size)
                                           or (batch_size, num_branches, context_vector_size)
        
        Returns:
            torch.Tensor: Processed context embedding of shape (batch_size, context_embedding_dim)
                         or (batch_size, num_branches, context_embedding_dim)
        """
        # Handle different input shapes
        original_shape = context_vector.shape
        if context_vector.dim() == 1:
            context_vector = context_vector.unsqueeze(0)
        elif context_vector.dim() == 3:
            # Reshape to (batch_size * num_branches, context_vector_size)
            batch_size, num_branches, ctx_size = context_vector.shape
            context_vector = context_vector.view(-1, ctx_size)
            needs_reshape = True
        else:
            needs_reshape = False
        
        # Normalize and process
        context_vector = self.context_norm(context_vector)
        context_embedding = self.context_mlp(context_vector)
        
        # Reshape back if needed
        if needs_reshape:
            context_embedding = context_embedding.view(batch_size, num_branches, -1)
        
        return context_embedding

class CNN(nn.Module):
    def __init__(self, name, dropout=0.5, custom_cnn_layers:list=None):
        super(CNN, self).__init__()
        self.name = name

        # Make the convolutional layers dependnt on custom_cnn_layers: [(filters, kernel_size, stride, padding)]
        assert custom_cnn_layers is not None, "Custom CNN layers must be provided. Each layer must be a tuple of (filters, kernel_size, stride, padding)."
        assert len(custom_cnn_layers) > 0, "Custom CNN layers must be a non-empty list. Each layer must be a tuple of (filters, kernel_size, stride, padding)."
        assert all(isinstance(layer, tuple) and len(layer) == 4 for layer in custom_cnn_layers), "Each layer must be a tuple of (filters, kernel_size, stride, padding)."
    
        layers = []
        in_channels = 1
        for filters, kernel_size, stride, padding in custom_cnn_layers:
            layers.extend([nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding),
                           nn.BatchNorm2d(filters),
                           nn.ReLU(inplace=True)])
            in_channels = filters
        self.conv_layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Store the output feature dimension (number of filters in last conv layer)
        self.feature_dim = custom_cnn_layers[-1][0]

    def forward(self, x):
        """
        Forward pass of CNN encoder.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: Raw image features of shape (batch_size, feature_dim)
        """
        # Convolutional layers with ReLU and BatchNorm
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Return raw features without any projection
        return x

class ResNet(nn.Module):
    """
    ResNet architecture with configurable depth
    Possible models:
    - resnet18 (defualt, pytorch implementation)
    - resnet34 (pytorch implementation)
    - resnet50 (pytorch implementation)
    and 
    - resnet10 (custom implementation)
    - resnet12 (custom implementation)
    - resnet14 (custom implementation)
    - resnet16 (custom implementation)

    Args:
        name (str): Name of the model
        base_model (str): Base model to use
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout rate
        filters_per_layer (list): Filters per layer
        branch_output_dim (int): Output dimension for image features
    """
    def __init__(self, name="transfer_model", base_model='resnet18', pretrained=True, 
                 dropout:float=None, filters_per_layer:list=[16,32,64,128]):
        super(ResNet, self).__init__()
        self.name = name
        self.base_model = self._get_base_model(base_model, pretrained, filters_per_layer)
        
        # Modify the first convolutional layer to accept single-channel input
        if base_model == 'resnet10' or base_model == 'resnet12' or base_model == 'resnet16' or base_model == 'resnet14':
            self.base_model.conv1 = nn.Conv2d(1, filters_per_layer[0], kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Store the output feature dimension (before removing fc layer)
        num_features = self.base_model.fc.in_features
        self.feature_dim = num_features
        self.base_model.fc = nn.Identity()  # Remove the last fully connected layer
        
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
        elif model_name == 'resnet16' or model_name == 'resnet10' or model_name == 'resnet12' or model_name == 'resnet14':
            num_layers = int(model_name.split('resnet')[-1])
            print(f"Using custom ResNet with {num_layers} layers. No pretrained weights used in the model.")
            return CustomResNet(num_layers, filters_per_layer) if filters_per_layer is not None else CustomResNet(num_layers)
        else:
            raise ValueError(f"Unsupported base model: {model_name}")
    
    def forward(self, x):
        """
        Forward pass of ResNet encoder.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: Raw image features of shape (batch_size, feature_dim)
        """
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        # Return raw features without any projection
        return x

    def __str__(self):
        return self.name

class ResidualBlock(nn.Module):
    """
    Basic building block for ResNet architectures
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer    
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (identity mapping or projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    
    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add shortcut connection
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

class CustomResNet(nn.Module):
    """
    ResNet architecture with configurable depth
    """
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
            self.layer1 = self._make_layer(ResidualBlock, self.filters_per_layer[0], 1, stride=1)
            self.layer2 = self._make_layer(ResidualBlock, self.filters_per_layer[1], 1, stride=2)
            self.layer3 = self._make_layer(ResidualBlock, self.filters_per_layer[2], 1, stride=2)
            self.layer4 = self._make_layer(ResidualBlock, self.filters_per_layer[3], 1, stride=2)
            self.fc_features = self.filters_per_layer[3]
        elif self.num_layers == 12:
            self.layer1 = self._make_layer(ResidualBlock, self.filters_per_layer[0], 2, stride=1)
            self.layer2 = self._make_layer(ResidualBlock, self.filters_per_layer[1], 1, stride=2)
            self.layer3 = self._make_layer(ResidualBlock, self.filters_per_layer[2], 1, stride=2)
            self.layer4 = self._make_layer(ResidualBlock, self.filters_per_layer[3], 1, stride=2)
            self.fc_features = self.filters_per_layer[3]
        elif self.num_layers == 14:
            self.layer1 = self._make_layer(ResidualBlock, self.filters_per_layer[0], 2, stride=1)
            self.layer2 = self._make_layer(ResidualBlock, self.filters_per_layer[1], 2, stride=2)
            self.layer3 = self._make_layer(ResidualBlock, self.filters_per_layer[2], 1, stride=2)
            self.layer4 = self._make_layer(ResidualBlock, self.filters_per_layer[3], 1, stride=2)
            self.fc_features = self.filters_per_layer[3]
        elif self.num_layers == 16:
            # ResNet10: 2 blocks per stage, filter sizes [16,32,64,128]
            self.layer1 = self._make_layer(ResidualBlock, self.filters_per_layer[0], 2, stride=1)
            self.layer2 = self._make_layer(ResidualBlock, self.filters_per_layer[1], 2, stride=2)
            self.layer3 = self._make_layer(ResidualBlock, self.filters_per_layer[2], 2, stride=2)
            self.layer4 = self._make_layer(ResidualBlock, self.filters_per_layer[3], 1, stride=2)
            self.fc_features = self.filters_per_layer[3]
        else:
            raise NotImplementedError(f"ResNet-{self.num_layers} is not implemented.")
        
        # Average pooling and flatten
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.fc_features, 2) # this output size is irrelevant it is still replaced by correct one, see ResNet class when prediction head is defined

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            
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
