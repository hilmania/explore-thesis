import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, p_drop=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=16, p_drop=0.2):
        super(ResidualBlock, self).__init__()
        self.block1 = ConvBlock(channels, channels, kernel_size, p_drop)
        self.block2 = ConvBlock(channels, channels, kernel_size, p_drop)

    def forward(self, x):
        return x + self.block2(self.block1(x))

# EEGNet (Final Model)
class EEGNet(nn.Module):
    def __init__(self, input_channels=20, num_classes=2, p_drop=0.2):
        super(EEGNet, self).__init__()
        self.initial_conv = ConvBlock(input_channels, 64, p_drop=p_drop)

        self.res_blocks = nn.Sequential(
            ResidualBlock(64, p_drop=p_drop),
            ResidualBlock(64, p_drop=p_drop),
            ResidualBlock(64, p_drop=p_drop),
            ResidualBlock(64, p_drop=p_drop)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):  # [B, 20, 1280]
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
