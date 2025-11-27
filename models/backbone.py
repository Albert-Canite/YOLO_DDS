import torch
import torch.nn as nn


def conv_block(in_channels: int, out_channels: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    )


class TinyBackbone(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            conv_block(in_channels, 16, k=3, s=1, p=1),
            nn.MaxPool2d(2, 2),
            conv_block(16, 32, k=3, s=1, p=1),
            nn.MaxPool2d(2, 2),
            conv_block(32, 64, k=3, s=1, p=1),
            nn.MaxPool2d(2, 2),
            conv_block(64, 128, k=3, s=1, p=1),
            nn.MaxPool2d(2, 2),
            conv_block(128, 256, k=3, s=1, p=1),
            # Keep final stride at 1/16 to improve localization granularity
            # for narrow, tall tooth boxes.
            conv_block(256, 512, k=3, s=1, p=1),
            conv_block(512, 1024, k=3, s=1, p=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
