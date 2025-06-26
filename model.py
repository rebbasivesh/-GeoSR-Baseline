import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4), nn.ReLU(),
            nn.Conv2d(64, 32, 5, padding=2), nn.ReLU(),
            nn.Conv2d(32, 1, 5, padding=2)
        )

    def forward(self, x):
        return self.net(x)
