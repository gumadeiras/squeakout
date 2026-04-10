from pathlib import Path

import numpy as np
import torch
from PIL import Image

from squeakout import segment_directory


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


def _write_image(path: Path, value: int) -> None:
    array = np.full((64, 64), value, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def test_segment_directory_streams_outputs_in_natural_order(tmp_path: Path) -> None:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    _write_image(source_dir / "10.png", 64)
    _write_image(source_dir / "2.png", 192)

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
