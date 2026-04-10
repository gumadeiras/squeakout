from __future__ import annotations

import argparse
from pathlib import Path

from .inference import segment_directory


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


def main() -> None:
    args = build_parser().parse_args()
    outputs = segment_directory(
        args.source_dir,
        mask_root=args.mask_root,
        montage_root=args.montage_root,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(f"Saved {len(outputs)} masks to {args.mask_root / args.source_dir.name}")
    print(f"Saved {len(outputs)} montages to {args.montage_root / args.source_dir.name}")


if __name__ == "__main__":
    main()
