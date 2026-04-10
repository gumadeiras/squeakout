import torch

from squeakout import SqueakOut, SqueakOut_autoencoder, load_model
from tests.support import CHECKPOINT_PATH


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


def test_legacy_autoencoder_import_still_loads_repo_checkpoint() -> None:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = SqueakOut_autoencoder()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 512, 512))

    assert output.shape == (1, 1, 512, 512)
