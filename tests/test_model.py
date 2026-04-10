from pathlib import Path

import torch

from squeakout import SqueakOut, load_model


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "squeakout_weights.ckpt"


def test_model_forward_shape() -> None:
    model = SqueakOut()
    output = model(torch.zeros(2, 1, 512, 512))
    assert output.shape == (2, 1, 512, 512)


def test_repo_checkpoint_loads_for_inference() -> None:
    model = load_model(CHECKPOINT_PATH, device="cpu")
    assert not model.training

    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 512, 512))

    assert output.shape == (1, 1, 512, 512)
