# noqa
import numpy as np
import torch
from icecream import ic
from torchvision.transforms import Compose

from sigkit.models.Module import SigKitClassifier
from sigkit.modem.psk import PSK
from sigkit.transforms.utils import ComplexTo2D, Normalize


def main():
    ckpt_path = "data/checkpoints/best.ckpt"
    model = SigKitClassifier.load_from_checkpoint(ckpt_path, num_classes=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    transform = Compose(
        [
            Normalize(norm=np.inf),
            ComplexTo2D(),
        ]
    )

    modem = PSK(1024, 32, 4)
    num_symbols = 4096 // modem.sps
    bitstreams = [
        np.random.randint(
            0, 2, size=(num_symbols * modem.bits_per_symbol,), dtype=np.uint8
        )
        for _ in range(32)
    ]

    waveforms = [modem.modulate(bits).samples for bits in bitstreams]

    tensors = [torch.tensor(w, dtype=torch.complex64) for w in waveforms]
    transformed = [transform(w) for w in tensors]
    batch = torch.stack(transformed).to(device)

    with torch.no_grad():
        preds = model(batch)
        predicted_classes = torch.argmax(preds, dim=1)

    ic("Predicted class indices:", predicted_classes.tolist())


if __name__ == "__main__":
    main()
