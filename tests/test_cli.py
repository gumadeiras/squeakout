import subprocess
import sys
from pathlib import Path

from PIL import Image

from tests.support import CHECKPOINT_PATH, write_grayscale_image


def test_cli_segments_directory_end_to_end(tmp_path: Path) -> None:
    source_dir = tmp_path / "inputs"
    mask_root = tmp_path / "masks"
    montage_root = tmp_path / "montages"
    source_dir.mkdir()
    write_grayscale_image(source_dir / "sample.png", 128)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "squeakout",
            str(source_dir),
            "--checkpoint",
            str(CHECKPOINT_PATH),
            "--mask-root",
            str(mask_root),
            "--montage-root",
            str(montage_root),
            "--device",
            "cpu",
            "--batch-size",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mask_path = mask_root / source_dir.name / "sample.png"
    montage_path = montage_root / source_dir.name / "sample_montage.png"
    expected_stdout = (
        f"Saved 1 mask to {mask_path.parent}\n"
        f"Saved 1 montage to {montage_path.parent}\n"
    )

    assert result.stdout == expected_stdout
    assert result.stderr == ""
    assert mask_path.exists()
    assert montage_path.exists()

    mask = Image.open(mask_path)
    montage = Image.open(montage_path)

    assert mask.size == (512, 512)
    assert montage.size == (512 * 3, 512)
