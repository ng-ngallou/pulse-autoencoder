import torch
from torch import nn


class PulseAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=7, stride=4, padding=3
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=5, stride=4, padding=2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=16, out_channels=8, kernel_size=5, stride=4, padding=2
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=4,
                padding=2,
                output_padding=3,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=4,
                padding=2,
                output_padding=3,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=8,
                out_channels=1,
                kernel_size=7,
                stride=4,
                padding=3,
                output_padding=3,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return nn.functional.interpolate(out, size=4096, mode="linear")
