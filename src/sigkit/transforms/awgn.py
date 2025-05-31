"""Module for AWGN Torch Transform."""

import torch
from torch import nn

from sigkit.core.base import SigKitError


class ApplyAWGN(nn.Module):
    """Additive White Gaussian Noise Torch Transform with optional SNR range.

    Args:
        snr_db:
            - If float or int: use that fixed SNR (in dB) on every forward().
            - If tuple/list of two floats: (min_snr_db, max_snr_db), sample
              uniformly from [min_snr_db, max_snr_db] each all.
    """

    def __init__(self, snr_db: float | tuple[float, float]):
        super().__init__()
        if isinstance(snr_db, (int, float)):
            self.min_snr = float(snr_db)
            self.max_snr = float(snr_db)
        elif (
            isinstance(snr_db, (tuple, list))
            and len(snr_db) == 2
            and all(isinstance(v, (int, float)) for v in snr_db)
        ):
            self.min_snr = float(snr_db[0])
            self.max_snr = float(snr_db[1])
        else:
            raise SigKitError(
                "ApplyAWGN: snr_db must be a number or a tuple of two numbers, "
                f"got {snr_db!r}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies AWGN to the input to reach the target SNR.

        Args:
            x: torch.Tensor of shape [N], dtype=torch.complex64.

        Returns:
            torch.Tensor of shape [N], dtype=torch.complex64 with AWGN.
        """
        if x.dtype != torch.complex64 or x.ndim != 1:
            raise SigKitError("Expected input of shape [N] and dtype=torch.complex64")

        if self.min_snr == self.max_snr:
            snr_db = self.min_snr
        else:
            r = torch.rand(1).item()
            snr_db = self.min_snr + (self.max_snr - self.min_snr) * r

        sig_power = (x.abs() ** 2).mean()
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / snr_lin
        std_dev = torch.sqrt(noise_power / 2.0)

        real_noise = std_dev * torch.randn_like(x.real)
        imag_noise = std_dev * torch.randn_like(x.real)
        noise = (real_noise + 1j * imag_noise).to(torch.complex64)

        return x + noise
