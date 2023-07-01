import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers: int = 64):
        super(DQN, self).__init__()
        # self.fc1 = nn.Linear(input_dim, hidden_layers)
        # self.fc2 = nn.Linear(hidden_layers, hidden_layers // 2)
        # self.fc3 = nn.Linear(hidden_layers // 2, output_dim)
        # self.vgg_model = models.vgg11(pretrained=False)

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding = 1),
            nn.Conv2d(32, 32,3, padding = 1),
            nn.Conv2d(32, 32,3, padding = 1),
            nn.MaxPool2d(2)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 32,3, padding = 1),
            nn.Conv2d(32, 32,3, padding = 1),
            nn.Conv2d(32, 32,3, padding = 1),
            nn.MaxPool2d(2)
        )
        self.pred = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        # x = x.view(x.size()[0], -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)
        x = self.cnn(x)
        x = self.cnn2(x)
        x = x.view(x.shape[0], -1)
        x = self.pred(x)
        return x