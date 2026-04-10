from __future__ import annotations

from pathlib import Path

import natsort
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_IMAGE_SIZE = (512, 512)
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
RESAMPLING_LANCZOS = Image.Resampling.LANCZOS


def list_spectrogram_paths(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Spectrogram directory does not exist: {root_path}")

    image_paths = [
        path
        for path in root_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return list(natsort.natsorted(image_paths, key=lambda path: path.name))


def load_spectrogram_tensor(
    path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    image_path = Path(path)
    with Image.open(image_path) as image:
        resized = image.convert("L").resize(image_size, RESAMPLING_LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class SpectrogramDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.image_size = image_size
        self.image_paths = list_spectrogram_paths(self.root)
        if not self.image_paths:
            raise ValueError(f"No supported spectrogram images found in {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[index]
        tensor = load_spectrogram_tensor(image_path, image_size=self.image_size)
        return tensor, image_path.name
