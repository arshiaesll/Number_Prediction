import torch
import torch.nn as nn

class OGLeNet5(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Define the layers
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(400, 120)
        self.linear2 = torch.nn.Linear(120, 84)
        self.linear3 = torch.nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.functional.F.tanh(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = torch.functional.F.tanh(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = torch.functional.F.tanh(x)
        x = self.linear2(x)
        x = torch.functional.F.tanh(x)
        x = self.linear3(x)
        return x