from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _to_numpy_2d(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeezing, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def to_uint8_image(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = _to_numpy_2d(value)
    if array.max(initial=0.0) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def threshold_mask(
    value: torch.Tensor | np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    array = _to_numpy_2d(value)
    return (array > threshold).astype(np.uint8) * 255


def save_grayscale_image(value: torch.Tensor | np.ndarray, output_path: str | Path) -> Path:
    path = Path(output_path)
    Image.fromarray(to_uint8_image(value), mode="L").save(path)
    return path


def save_mask_image(mask: torch.Tensor | np.ndarray, output_path: str | Path) -> Path:
    path = Path(output_path)
    Image.fromarray(to_uint8_image(mask), mode="L").save(path)
    return path


def build_montage_from_mask(
    spectrogram: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    *,
    alpha: float = 0.5,
) -> Image.Image:
    spectrogram_array = to_uint8_image(spectrogram)
    mask_array = to_uint8_image(mask)
    overlay = np.clip(
        spectrogram_array.astype(np.float32) * (1.0 - alpha) + mask_array.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    montage = np.hstack([spectrogram_array, mask_array, overlay])
    return Image.fromarray(montage, mode="L")


def build_montage_image(
    spectrogram: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    *,
    alpha: float = 0.5,
    threshold: float = 0.5,
) -> Image.Image:
    mask_array = threshold_mask(mask, threshold=threshold)
    return build_montage_from_mask(spectrogram, mask_array, alpha=alpha)


def create_montage(
    image: torch.Tensor | np.ndarray,
    gt: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    alpha: float = 0.5,
    show_plot: bool = True,
    model_dir: str | Path | None = None,
    image_name: str | None = None,
) -> None:
    """Legacy montage helper retained for external imports."""

    import matplotlib.pyplot as plt

    image_array = to_uint8_image(image)
    mask_array = threshold_mask(mask, threshold=0.5)
    gt_array = threshold_mask(gt, threshold=0.0)
    blend = image_array * (mask_array > 0).astype(np.uint8)

    plt.figure()

    plt.subplot(231)
    plt.imshow(image_array, cmap="gray")
    plt.title("image")

    plt.subplot(232)
    plt.imshow(mask_array, cmap="jet")
    plt.title("mask")

    plt.subplot(233)
    plt.imshow(gt_array, cmap="jet")
    plt.title("ground truth")

    plt.subplot(234)
    plt.imshow(image_array, cmap="gray")
    plt.imshow(mask_array, cmap="jet", alpha=alpha)
    plt.title("overlay mask")

    plt.subplot(235)
    plt.imshow(image_array, cmap="gray")
    plt.imshow(gt_array, cmap="jet", alpha=alpha)
    plt.title("overlay gt")

    plt.subplot(236)
    plt.imshow(blend, cmap="gray")
    plt.title("image*mask")

    if show_plot:
        plt.show()

    if model_dir is not None and image_name is not None:
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_grayscale_image(image_array, output_dir / f"{image_name}_img.png")
        save_mask_image(mask_array, output_dir / f"{image_name}_mask.png")
        plt.savefig(output_dir / f"{image_name}.png")

    plt.close()
