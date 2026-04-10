from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "squeakout_weights.ckpt"


def write_grayscale_image(path: Path, value: int, *, size: tuple[int, int] = (64, 64)) -> None:
    array = np.full(size, value, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)
