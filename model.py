import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super(YOLOv1, self).__init__()
        self.darknet = self._create_conv_layers()
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes)

    def forward(self, x):
        x = self.darknet(x)
        # Flattening to: (Batch, 1024 * 7 * 7)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self):
        # Full architecture based on the paper (Simplified representation)
        return nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(192, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Adding more layers to reduce size to 7x7
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
        )

    def _create_fcs(self, S, B, C):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096), # This matches 50176 (1024*7*7)
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )