import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Strato di Max Pooling
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Input: B x 3 x 64 x 64
        x = self.pool_layer(self.conv1(x).relu()) # B x 64 x 32 x 32
        x = self.pool_layer(self.conv2(x).relu()) # B x 128 x 16 x 16
        x = self.conv3(x).relu()                  # B x 256 x 16 x 16

        x = self.pool(x)                          # B x 256 x 1 x 1
        x = x.flatten(1)                          # B x 256
        x = self.fc1(x)
        return x