# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class WASDCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (1, 120, 160)
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)  # halves H/W each time

        # Flattened size after conv+pool: 64 x 30 x 40 = 76800
        self.fc1 = nn.Linear(64*30*40, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 outputs: w,a,s,d

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 120x160
        x = self.pool(F.relu(self.conv2(x)))  # 60x80
        x = self.pool(F.relu(self.conv3(x)))  # 30x40
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # multi-label sigmoid
        return x

def get_model():
    return WASDCNN()
