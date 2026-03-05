"""
visualization.py — Heatmap rendering and image utilities for XAI.

Functions:
    overlay_heatmap     → overlay colored heatmap on original image → base64 PNG
    tensor_to_base64    → convert torch tensor to base64 PNG
    volume_to_slices    → extract axial/coronal/sagittal slices from 3D volume
    normalize_image     → normalize numpy array to [0, 255]
"""

import base64
import io
import logging
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Colormap: jet-like (blue=low attention, red=high attention)
# Precomputed for performance — no matplotlib dependency
def _jet_colormap(value: float) -> tuple:
    """Map value in [0,1] to RGB using jet colormap."""
    value = max(0.0, min(1.0, value))
    if value < 0.25:
        r, g, b = 0, int(255 * (value / 0.25)), 255
    elif value < 0.5:
        r, g, b = 0, 255, int(255 * (1 - (value - 0.25) / 0.25))
    elif value < 0.75:
        r, g, b = int(255 * ((value - 0.5) / 0.25)), 255, 0
    else:
        r, g, b = 255, int(255 * (1 - (value - 0.75) / 0.25)), 0
    return r, g, b


def _apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Apply jet colormap to normalized heatmap → RGB array."""
    h, w = heatmap.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            rgb[i, j] = _jet_colormap(float(heatmap[i, j]))
    return rgb


def _apply_colormap_fast(heatmap: np.ndarray) -> np.ndarray:
    """Fast vectorized jet colormap using numpy."""
    v = np.clip(heatmap, 0.0, 1.0)
    r = np.where(v < 0.5, 0.0, np.where(v < 0.75, (v - 0.5) / 0.25, 1.0))
    g = np.where(
        v < 0.25,
        v / 0.25,
        np.where(v < 0.75, 1.0, np.where(v < 1.0, 1.0 - (v - 0.75) / 0.25, 0.0)),
    )
    b = np.where(v < 0.25, 1.0, np.where(v < 0.5, 1.0 - (v - 0.25) / 0.25, 0.0))
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize numpy array to uint8 [0, 255]."""
    img = img.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def overlay_heatmap(
    original: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    return_base64: bool = True,
    colorbar: bool = True,
) -> Union[str, Image.Image]:
    """
    Overlay a heatmap on an original image.

    Args:
        original:      PIL Image or numpy array (H, W) or (H, W, 3)
        heatmap:       Normalized numpy array (H, W) with values in [0, 1]
        alpha:         Heatmap opacity (0=invisible, 1=full overlay)
        return_base64: Return base64 PNG string if True, else PIL Image
        colorbar:      Add a colorbar legend (Low → High attention)

    Returns:
        base64 PNG string or PIL Image
    """
    # Convert original to PIL RGB
    if isinstance(original, np.ndarray):
        orig_norm = normalize_image(original)
        if orig_norm.ndim == 2:
            orig_pil = Image.fromarray(orig_norm, mode="L").convert("RGB")
        elif orig_norm.ndim == 3 and orig_norm.shape[2] == 3:
            orig_pil = Image.fromarray(orig_norm, mode="RGB")
        else:
            orig_pil = Image.fromarray(orig_norm[:, :, 0], mode="L").convert("RGB")
    else:
        orig_pil = original.convert("RGB")

    # Resize heatmap to match original
    orig_w, orig_h = orig_pil.size
    heatmap_resized = (
        np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.BILINEAR
            )
        )
        / 255.0
    )

    # Apply colormap
    heatmap_rgb = _apply_colormap_fast(heatmap_resized)
    heatmap_pil = Image.fromarray(heatmap_rgb, mode="RGB")

    # Blend
    blended = Image.blend(orig_pil, heatmap_pil, alpha=alpha)

    # Add colorbar
    if colorbar:
        blended = _add_colorbar(blended)

    if return_base64:
        return image_to_base64(blended)
    return blended


def _add_colorbar(image: Image.Image, height: int = 20) -> Image.Image:
    """Add a colorbar legend below the image."""
    w, h = image.size
    bar = np.linspace(0, 1, w)
    bar_rgb = _apply_colormap_fast(bar[np.newaxis, :].repeat(height, axis=0))
    bar_pil = Image.fromarray(bar_rgb, mode="RGB")

    # Add text labels
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(bar_pil)
        draw.text((2, 2), "Low", fill=(255, 255, 255))
        draw.text((w - 30, 2), "High", fill=(255, 255, 255))
    except Exception:
        pass

    # Stack vertically
    combined = Image.new("RGB", (w, h + height))
    combined.paste(image, (0, 0))
    combined.paste(bar_pil, (0, h))
    return combined


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def tensor_to_base64(tensor, channel: int = 0) -> str:
    """
    Convert a PyTorch tensor to base64 PNG.

    Args:
        tensor:  Tensor of shape (C, H, W) or (B, C, H, W)
        channel: Which channel to visualize
    """

    if tensor.dim() == 4:
        tensor = tensor[0]
    img_np = tensor[channel].cpu().numpy()
    img_norm = normalize_image(img_np)
    pil = Image.fromarray(img_norm, mode="L").convert("RGB")
    return image_to_base64(pil)


def volume_to_slices(
    volume: np.ndarray,
    axis: int = 0,
    idx: Optional[int] = None,
) -> np.ndarray:
    """
    Extract a 2D slice from a 3D volume.

    Args:
        volume: 3D numpy array (D, H, W)
        axis:   0=axial, 1=coronal, 2=sagittal
        idx:    Slice index (None = middle)

    Returns:
        2D numpy array
    """
    if idx is None:
        idx = volume.shape[axis] // 2

    if axis == 0:
        return volume[idx, :, :]
    elif axis == 1:
        return volume[:, idx, :]
    else:
        return volume[:, :, idx]


def create_side_by_side(
    original: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    title_original: str = "Input",
    title_heatmap: str = "Attention",
) -> str:
    """
    Create a side-by-side comparison: original image | heatmap overlay.
    Returns base64 PNG.
    """
    if isinstance(original, np.ndarray):
        orig_norm = normalize_image(original)
        if orig_norm.ndim == 2:
            orig_pil = Image.fromarray(orig_norm, mode="L").convert("RGB")
        else:
            orig_pil = Image.fromarray(orig_norm, mode="RGB")
    else:
        orig_pil = original.convert("RGB")

    overlay = overlay_heatmap(
        orig_pil, heatmap, alpha=alpha, return_base64=False, colorbar=False
    )

    w, h = orig_pil.size
    padding = 4
    combined = Image.new("RGB", (w * 2 + padding, h + 20), color=(30, 30, 30))

    # Paste images
    combined.paste(orig_pil, (0, 20))
    combined.paste(overlay, (w + padding, 20))

    # Add titles
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(combined)
        draw.text((w // 2 - 20, 4), title_original, fill=(200, 200, 200))
        draw.text((w + padding + w // 2 - 20, 4), title_heatmap, fill=(200, 200, 200))
    except Exception:
        pass

    return image_to_base64(combined)
