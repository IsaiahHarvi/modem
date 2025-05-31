"""Module for PhaseShift Torch Transform."""

import math
from typing import Tuple, Union

import torch
from torch import nn

from sigkit.core.base import SigKitError


class ApplyPhaseShift(nn.Module):
    """Apply a constant or random phase offset to a 1D complex64 torch.Tensor.

    Args:
        phase_offset:
            - If a single float or int: apply that fixed phase (radians).
            - If a tuple/list of two floats: (min_phase, max_phase), pick
              random uniform phi from [min_phase, max_phase] per call
    """

    def __init__(self, phase_offset: Union[float, Tuple[float, float]]):
        super().__init__()
        if isinstance(phase_offset, (int, float)):
            self.min_phi = float(phase_offset)
            self.max_phi = float(phase_offset)
        elif (
            isinstance(phase_offset, (tuple, list))
            and len(phase_offset) == 2
            and all(isinstance(p, (int, float)) for p in phase_offset)
        ):
            self.min_phi = float(phase_offset[0])
            self.max_phi = float(phase_offset[1])
        else:
            raise SigKitError(
                f"ApplyPhaseShift: phase_offset must be a single number or a tuple of "
                f"two numbers, got {phase_offset!r}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the PhaseShift to the Tensor."""
        if x.dtype != torch.complex64 or x.ndim != 1:
            raise SigKitError(
                f"ApplyPhaseShift expects a 1D tensor of dtype=torch.complex64, "
                f"got {x.shape=}, {x.dtype=}"
            )

        if self.min_phi == self.max_phi:
            phi = self.min_phi
        else:
            r = torch.rand(1).item()
            phi = self.min_phi + (self.max_phi - self.min_phi) * r

        c = math.cos(phi)
        s = math.sin(phi)
        phase_factor = torch.complex(
            torch.tensor(c, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)
        )
        return x * phase_factor
