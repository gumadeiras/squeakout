from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from .data import DEFAULT_IMAGE_SIZE, SpectrogramDataset
from .model import SqueakOut, load_checkpoint

DEFAULT_BATCH_SIZE = 8
DEFAULT_MASK_THRESHOLD = 0.51


@dataclass(frozen=True)
class SegmentationOutput:
    input_name: str
    mask_path: Path
    montage_path: Path


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> SqueakOut:
    resolved_device = resolve_device(None if device is None else str(device))
    return load_checkpoint(checkpoint_path, device=resolved_device)


def logits_to_mask(logits: torch.Tensor, threshold: float = DEFAULT_MASK_THRESHOLD) -> np.ndarray:
    probabilities = torch.sigmoid(logits.detach()).squeeze(0).cpu().numpy()
    return (probabilities >= threshold).astype(np.uint8) * 255


def build_montage_image(
    spectrogram: torch.Tensor,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    spectrogram_array = (spectrogram.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    overlay = np.clip(
        spectrogram_array.astype(np.float32) * (1.0 - alpha) + mask.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    montage = np.hstack([spectrogram_array, mask, overlay])
    return Image.fromarray(montage, mode="L")


def _prepare_output_dirs(
    source_dir: str | Path,
    mask_root: str | Path,
    montage_root: str | Path,
) -> tuple[Path, Path]:
    source_path = Path(source_dir).expanduser().resolve()
    mask_dir = Path(mask_root).expanduser().resolve() / source_path.name
    montage_dir = Path(montage_root).expanduser().resolve() / source_path.name
    mask_dir.mkdir(parents=True, exist_ok=True)
    montage_dir.mkdir(parents=True, exist_ok=True)
    return mask_dir, montage_dir


def _resolve_model(
    *,
    model: nn.Module | None,
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> nn.Module:
    if model is None and checkpoint_path is None:
        raise ValueError("Pass either `model` or `checkpoint_path`")
    if model is not None and checkpoint_path is not None:
        raise ValueError("Pass only one of `model` or `checkpoint_path`")

    resolved_model = model if model is not None else load_model(checkpoint_path, device=device)
    resolved_model.to(device)
    resolved_model.eval()
    return resolved_model


def segment_directory(
    source_dir: str | Path,
    *,
    mask_root: str | Path,
    montage_root: str | Path,
    checkpoint_path: str | Path | None = None,
    model: nn.Module | None = None,
    device: str | torch.device | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    num_workers: int = 0,
    threshold: float = DEFAULT_MASK_THRESHOLD,
) -> list[SegmentationOutput]:
    resolved_device = resolve_device(None if device is None else str(device))
    resolved_model = _resolve_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
    )
    dataset = SpectrogramDataset(source_dir, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
    )
    mask_dir, montage_dir = _prepare_output_dirs(source_dir, mask_root, montage_root)

    outputs: list[SegmentationOutput] = []
    with torch.inference_mode():
        for spectrograms, file_names in loader:
            logits = resolved_model(spectrograms.to(resolved_device))
            for spectrogram, logit, file_name in zip(spectrograms, logits, file_names):
                mask = logits_to_mask(logit, threshold=threshold)
                mask_path = mask_dir / file_name
                montage_path = montage_dir / f"{Path(file_name).stem}_montage.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                build_montage_image(spectrogram, mask).save(montage_path)
                outputs.append(
                    SegmentationOutput(
                        input_name=file_name,
                        mask_path=mask_path,
                        montage_path=montage_path,
                    )
                )
    return outputs
