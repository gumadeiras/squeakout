from pathlib import Path

import numpy as np
import torch
from PIL import Image

from squeakout import segment_directory
from tests.support import CHECKPOINT_PATH, write_grayscale_image
from utils import create_montage


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, logit_value: float) -> None:
        super().__init__()
        self.logit_value = logit_value

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (batch.size(0), 1, batch.size(2), batch.size(3)),
            self.logit_value,
            dtype=batch.dtype,
            device=batch.device,
        )

def test_segment_directory_streams_outputs_in_natural_order(tmp_path: Path) -> None:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    write_grayscale_image(source_dir / "10.png", 64)
    write_grayscale_image(source_dir / "2.png", 192)

    outputs = segment_directory(
        source_dir,
        mask_root=tmp_path / "masks",
        montage_root=tmp_path / "montages",
        model=ConstantLogitModel(logit_value=10.0),
        device="cpu",
        batch_size=1,
    )

    assert [output.input_name for output in outputs] == ["2.png", "10.png"]

    first_mask = np.asarray(Image.open(outputs[0].mask_path))
    first_montage = Image.open(outputs[0].montage_path)

    assert first_mask.shape == (512, 512)
    assert np.all(first_mask == 255)
    assert first_montage.size == (512 * 3, 512)


def test_segment_directory_loads_repo_checkpoint_end_to_end(tmp_path: Path) -> None:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    write_grayscale_image(source_dir / "sample.png", 128)

    outputs = segment_directory(
        source_dir,
        mask_root=tmp_path / "masks",
        montage_root=tmp_path / "montages",
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
        batch_size=1,
    )

    assert len(outputs) == 1
    assert outputs[0].mask_path.exists()
    assert outputs[0].montage_path.exists()

    mask = np.asarray(Image.open(outputs[0].mask_path))
    montage = Image.open(outputs[0].montage_path)

    assert mask.shape == (512, 512)
    assert montage.size == (512 * 3, 512)


def test_legacy_utils_create_montage_still_writes_expected_artifacts(tmp_path: Path) -> None:
    image = torch.ones((1, 32, 32), dtype=torch.float32)
    gt = torch.ones((1, 32, 32), dtype=torch.float32)
    mask = torch.ones((1, 32, 32), dtype=torch.float32)

    create_montage(
        image,
        gt,
        mask,
        show_plot=False,
        model_dir=tmp_path,
        image_name="sample",
    )

    assert (tmp_path / "sample.png").exists()
    assert (tmp_path / "sample_img.png").exists()
    assert (tmp_path / "sample_mask.png").exists()
