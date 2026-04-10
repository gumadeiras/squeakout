from __future__ import annotations

import argparse
from pathlib import Path

from .inference import SegmentationOutput, segment_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SqueakOut segmentation on a directory of spectrograms.")
    parser.add_argument("source_dir", type=Path, help="Directory containing input spectrogram images")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./squeakout_weights.ckpt"),
        help="Checkpoint path",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path("./outputs/segmentation"),
        help="Root directory for output masks",
    )
    parser.add_argument(
        "--montage-root",
        type=Path,
        default=Path("./outputs/montages"),
        help="Root directory for output montage images",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu, cuda, mps")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    return parser


def _default_output_dir(root: Path, source_dir: Path) -> Path:
    return root / source_dir.name


def _artifact_dir(
    outputs: list[SegmentationOutput],
    *,
    kind: str,
    root: Path,
    source_dir: Path,
) -> Path:
    if outputs:
        artifact_path = outputs[0].mask_path if kind == "mask" else outputs[0].montage_path
        return artifact_path.parent
    return _default_output_dir(root, source_dir)


def _artifact_summary(label: str, count: int, directory: Path) -> str:
    noun = label if count == 1 else f"{label}s"
    return f"Saved {count} {noun} to {directory}"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        outputs = segment_directory(
            args.source_dir,
            mask_root=args.mask_root,
            montage_root=args.montage_root,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    output_count = len(outputs)
    mask_dir = _artifact_dir(outputs, kind="mask", root=args.mask_root, source_dir=args.source_dir)
    montage_dir = _artifact_dir(outputs, kind="montage", root=args.montage_root, source_dir=args.source_dir)
    print(_artifact_summary("mask", output_count, mask_dir))
    print(_artifact_summary("montage", output_count, montage_dir))


if __name__ == "__main__":
    main()
