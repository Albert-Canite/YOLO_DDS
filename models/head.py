import torch
import torch.nn as nn

import config


def conv_block(in_channels: int, out_channels: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    )


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = conv_block(in_channels, 512, k=3, s=1, p=1)
        out_channels = len(config.ANCHORS) * (5 + config.NUM_CLASSES)
        self.pred = nn.Conv2d(512, out_channels, kernel_size=1)
        with torch.no_grad():
            # Lower initial objectness/class logits to reduce early false positives
            bias = self.pred.bias.view(len(config.ANCHORS), 5 + config.NUM_CLASSES)
            bias[:, 4] = -4.0
            if config.NUM_CLASSES > 0:
                bias[:, 5:] = -4.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pred(x)
        b, _, h, w = x.shape
        x = x.view(b, len(config.ANCHORS), 5 + config.NUM_CLASSES, h, w)
        return x.permute(0, 1, 3, 4, 2).contiguous()
