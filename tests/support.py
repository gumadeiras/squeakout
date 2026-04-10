from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "squeakout_weights.ckpt"
PYTHON = sys.executable


def write_grayscale_image(path: Path, value: int, *, size: tuple[int, int] = (64, 64)) -> None:
    array = np.full(size, value, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def run_cli(*args: str | Path, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, "-m", "squeakout", *(str(arg) for arg in args)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT if cwd is None else cwd,
    )
