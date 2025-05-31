"""Utility module for SigKit transforms."""

import torch
import torch.nn as nn

from sigkit.core.base import SigKitError


class ComplexTo2D(nn.Module):
    """Convert to the expected input of the model for training.

    Transform a 1D torch.Tensor of dtype=torch.complex64 and shape (N,)
    into a 2×N torch.Tensor of dtype=torch.float32:
      - Row 0 = real part
      - Row 1 = imaginary part

    Example:
        x = torch.randn(4096) + 1j * torch.randn(4096)
        x = x.to(torch.complex64)
        iq = ComplexTo2D(x)
    """

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise SigKitError(f"Expected a torch.Tensor, got {type(x)}")
        if x.dtype != torch.complex64:
            raise SigKitError(f"Expected dtype=torch.complex64, got {x.dtype}")
        if x.ndim != 1:
            raise SigKitError(f"Expected a 1D tensor, got shape {tuple(x.shape)}")

        real = x.real.to(torch.float32)  # shape (N,), dtype float32
        imag = x.imag.to(torch.float32)  # shape (N,), dtype float32
        return torch.stack([real, imag], dim=0)  # shape (2, N), dtype float32


class Normalize(nn.Module):
    """Normalize the input data."""

    def __init__(self, norm=float("inf")):
        super().__init__()
        self.norm_order = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(x, ord=self.norm_order, dim=-1, keepdim=True)
        return (x / norm).to(torch.complex64)
