import torch
import torch.nn as nn
from torchvision import models
import utilities.config as c

from transfer_CNN import ResNet


class MultiheadCNN(nn.Module):
    def __init__(self, config_tuple, name="multihead_model", base_model='resnet18', pretrained=True, 
                 dropout:float=None, custom_head:list=None, filters_per_layer:list=[16,32,64,128]):
        super(MultiheadCNN, self).__init__()
        self.name = name
        self.cnn_branch = ResNet(config_tuple, name="base_model", base_model=base_model, pretrained=pretrained, 
                                dropout=dropout, custom_head=custom_head, filters_per_layer=filters_per_layer)
        
        self.env_branch = nn.Sequential(
            nn.Linear(self.cnn_branch.fc.out_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1)
        )
