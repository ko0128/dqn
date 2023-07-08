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
        self.fc_layer = nn.Sequential(
            nn.Linear(130, 130),
            nn.ReLU(),
            nn.Linear(130, 130),            
        )
        self.rnn = nn.LSTM(1, 1, 2, batch_first=True)

        

        self.pred = nn.Sequential(
            nn.Linear(130, 130),
            nn.ReLU(),
            nn.Linear(130, 130),
            nn.ReLU(),
            nn.Linear(130, 5),
        )



        self.pos_layer = nn.Sequential(
            nn.Linear(2, 6),
            nn.ReLU(),
        )

    def forward(self, input):
        # x = x.view(x.size()[0], -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)
        # pos = torch.zeros((x.shape[0],0))
        # for i in len(pos):
        #     pos[i] =
        x = input[:,:4,:,:]
        pos = input[:,4,0,:2]
        x = self.cnn(x)
        x = self.cnn2(x)
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, pos),dim=1)
        y = self.fc_layer(x) 
        res = x + y
        res = res.unsqueeze(-1)
        res, _ = self.rnn(res) 
        res = res.squeeze(-1)
        x = self.pred(x)
        return x