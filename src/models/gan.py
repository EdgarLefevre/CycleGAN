import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual Block.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.double_conv(x)
        return skip + x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.up_path = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_path(x)
        x = self.res_blocks(x)
        return self.up_path(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)
