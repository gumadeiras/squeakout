from pathlib import Path

from PIL import Image

from tests.support import CHECKPOINT_PATH, run_cli, write_grayscale_image


def test_cli_segments_directory_end_to_end(tmp_path: Path) -> None:
    source_dir = tmp_path / "inputs"
    mask_root = tmp_path / "masks"
    montage_root = tmp_path / "montages"
    source_dir.mkdir()
    write_grayscale_image(source_dir / "sample.png", 128)

    result = run_cli(
        source_dir,
        "--checkpoint",
        CHECKPOINT_PATH,
        "--mask-root",
        mask_root,
        "--montage-root",
        montage_root,
        "--device",
        "cpu",
        "--batch-size",
        "1",
    )

    mask_path = mask_root / source_dir.name / "sample.png"
    montage_path = montage_root / source_dir.name / "sample_montage.png"
    expected_stdout = (
        f"Saved 1 mask to {mask_path.parent}\n"
        f"Saved 1 montage to {montage_path.parent}\n"
    )

    assert result.returncode == 0
    assert result.stdout == expected_stdout
    assert result.stderr == ""
    assert mask_path.exists()
    assert montage_path.exists()

    mask = Image.open(mask_path)
    montage = Image.open(montage_path)

    assert mask.size == (512, 512)
    assert montage.size == (512 * 3, 512)


def test_cli_reports_missing_input_directory_cleanly(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    result = run_cli(missing_dir)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"error: Spectrogram directory does not exist: {missing_dir.resolve()}\n"


def test_cli_reports_empty_input_directory_cleanly(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = run_cli(empty_dir)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"error: No supported spectrogram images found in {empty_dir.resolve()}\n"
