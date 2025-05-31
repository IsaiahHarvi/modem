import pytest
import numpy as np
import torch

from sigkit.metrics.integrity import estimate_snr
from sigkit.transforms.awgn import ApplyAWGN
from sigkit.impairments.awgn import AWGN
from sigkit.core.base import Signal


@pytest.mark.parametrize("snr_db", [0, 10, 20, 30])
def test_awgn_torch(snr_db):
    awgn = ApplyAWGN(snr_db=snr_db)

    N = 4096
    theta = 2 * torch.pi * torch.arange(N, dtype=torch.float32) / N
    real = torch.cos(theta)
    imag = torch.sin(theta)
    x_complex = (real + 1j * imag).to(torch.complex64)

    y_complex = awgn(x_complex)

    y_i = y_complex.real
    y_q = y_complex.imag
    y_iq = torch.stack([y_i, y_q], dim=0)

    x_iq = torch.stack([real, imag], dim=0)

    measured = estimate_snr(x_iq, y_iq)
    assert abs(measured - snr_db) < 1.0, f"measured {measured:.2f} dB != target {snr_db} dB"


@pytest.mark.parametrize("snr_db", [0, 10, 20, 30])
def test_awgn_np(snr_db):
    awgn = AWGN(snr_db=snr_db)

    samples = (
        np.exp(1j * 2 * np.pi * np.arange(4096) / 4096)
        .astype(np.complex64)
    )
    signal = Signal(
        samples=samples,
        sample_rate=1e6,
        carrier_frequency=0.0,
    )

    noisy = awgn.apply(signal)
    measured = estimate_snr(signal.samples, noisy.samples)

    assert abs(measured - snr_db) < 1.0, f"measured {measured:.2f} dB != target {snr_db} dB"

