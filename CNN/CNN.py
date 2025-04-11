import torch.nn as nn
import torch.nn.functional as F
import torch
"""
    CNN模型
"""
class CNNModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(CNNModel, self).__init__()

        in_channels, height, width = input_shape

        self.conv1 = nn.Conv2d(in_channels, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(25, 1, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        in_features = 1 * height * width
        self.output_shape = output_shape
        output_units = output_shape[0] * output_shape[1]
        self.fc1 = nn.Linear(in_features, output_units)
        self.fc2 = nn.Linear(in_features, output_units)
        self.fc3 = nn.Linear(in_features, output_units)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        out1 = torch.sigmoid(self.fc1(x)).reshape(-1, self.output_shape[0], self.output_shape[1])
        out2 = self.fc2(x).reshape(-1, self.output_shape[0], self.output_shape[1])
        out3 = self.fc3(x).reshape(-1, self.output_shape[0], self.output_shape[1])
        return torch.stack((out1, out2, out3), dim=0)
