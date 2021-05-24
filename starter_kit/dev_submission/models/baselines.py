import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
import numpy as np


__all__ = [
    'MLP', 'CNN'
]


class MLP(nn.Module):

    def __init__(
        self,
        input_size: int=486,
        hidden_size: Union[int, List[int]] = 10,
        num_classes: int = 6,
        drop_rate: bool = False,
        init_weights: bool = True
    ) -> None:
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        self.num_classes = num_classes
        self.dropout = dropout
        if self.dropout:
            self.dropout = nn.Dropout()

        self.input_linear = nn.Linear(self.input_size, self.hidden_size[0])
        self.linears = nn.ModuleList(
            [nn.Linear(self.hidden_size[i], self.hidden_size[i+1]) for i in range(len(self.hidden_size)-1)]
        )
        self.output_linear = nn.Linear(self.hidden_size[-1], self.num_classes)

        self.relu = nn.ReLU()

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.input_linear(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.output_linear(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class CNN(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)