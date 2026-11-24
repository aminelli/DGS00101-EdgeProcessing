from __future__ import annotations

import torch
from torch import nn


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UNet(nn.Module):
    """U-Net binaria compatta per input SAR VV/VH in forma NCHW."""

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 2,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes != 2 or base_channels <= 0:
            raise ValueError("U-Net richiede canali positivi e due classi.")

        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.encoders = nn.ModuleList()
        previous_channels = in_channels
        for channel_count in channels:
            self.encoders.append(DoubleConvolution(previous_channels, channel_count))
            previous_channels = channel_count

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConvolution(channels[-1], channels[-1] * 2)
        self.upconvolutions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_input_channels = channels[-1] * 2
        for channel_count in reversed(channels):
            self.upconvolutions.append(
                nn.ConvTranspose2d(
                    decoder_input_channels, channel_count, kernel_size=2, stride=2
                )
            )
            self.decoders.append(
                DoubleConvolution(channel_count * 2, channel_count)
            )
            decoder_input_channels = channel_count
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []
        current = inputs
        for encoder in self.encoders:
            current = encoder(current)
            skip_connections.append(current)
            current = self.pool(current)

        current = self.bottleneck(current)
        for upconvolution, decoder, skip in zip(
            self.upconvolutions, self.decoders, reversed(skip_connections)
        ):
            current = upconvolution(current)
            if current.shape[-2:] != skip.shape[-2:]:
                current = nn.functional.interpolate(
                    current, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            current = decoder(torch.cat((skip, current), dim=1))
        return self.classifier(current)
