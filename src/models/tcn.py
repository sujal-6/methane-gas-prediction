from typing import List

import torch
import torch.nn as nn

class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv1(x)
        out = out[:, :, : x.size(2)]
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(2)]
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TCN(nn.Module):

    def __init__(
        self,
        num_features: int,
        channels: List[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        layers: List[nn.Module] = []
        in_ch = num_features
        for i, out_ch in enumerate(channels):
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.network(x)
        return self.head(x)


class TCNWithCropEmbedding(nn.Module):

    def __init__(
        self,
        num_crops: int,
        num_features: int,
        channels: List[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.crop_emb = nn.Embedding(num_crops, 8)
        self.tcn = TCN(
            num_features=num_features + 8,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x_num: torch.Tensor, crop_idx: torch.Tensor) -> torch.Tensor:
        # x_num: (B, T, F)
        crop_emb = self.crop_emb(crop_idx)  # (B, 8)
        crop_emb = crop_emb.unsqueeze(1).repeat(1, x_num.size(1), 1)
        x = torch.cat([x_num, crop_emb], dim=2)
        return self.tcn(x)
