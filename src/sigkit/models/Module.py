"""Module for constructing the Neural Network."""

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUnit1d(nn.Module):
    """A single ResNet‐style 1D residual unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

        self.act2 = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.act2(out)


class SigKitClassifier(pl.LightningModule):
    """ResNet‐style Signal classifier for complex I/Q signals."""

    def __init__(
        self, num_classes: int, lr: float = 1e-3, leaky_relu_slope: float = 0.01
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=32,
                kernel_size=3,
                stride=4,
                padding=1,
                bias=False,
            ),  # 4096->1024
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 1024->512
        )

        # Residual Block 1 (4 × [32->32, stride=1])
        block1_units = [
            ResidualUnit1d(32, 32, stride=1, leaky_relu_slope=leaky_relu_slope)
            for _ in range(4)
        ]
        self.block1 = nn.Sequential(*block1_units)  # (512 -> 512, channels remain 32)

        # Residual Block 2
        #   - First unit downsamples (32->64, stride=2): time 512->256
        #   - Second unit (64->64, stride=1)
        self.block2 = nn.Sequential(
            ResidualUnit1d(
                32, 64, stride=2, leaky_relu_slope=leaky_relu_slope
            ),  # (512->256, 32->64)
            ResidualUnit1d(64, 64, stride=1, leaky_relu_slope=leaky_relu_slope),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(
            output_size=1
        )  # (256-channels length->1)
        self.flatten = nn.Flatten()  # (B, 64, 1) -> (B, 64)
        self.head = nn.Sequential(
            nn.Linear(64, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)  # -> (B,32,512)
        out = self.block1(out)  # -> (B,32,512)
        out = self.block2(out)  # -> (B,64,256)
        out = self.global_pool(out)  # -> (B,64,1)
        out = self.flatten(out)  # -> (B,64)
        return self.head(out)  # -> (B,num_classes)

    def training_step(self, batch, batch_idx):
        signals, labels = batch
        logits = self(signals)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        signals, labels = batch
        logits = self(signals)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
