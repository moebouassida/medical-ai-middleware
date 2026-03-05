"""
gradcam.py — Grad-CAM implementation for medical imaging CNNs.

Supports:
    - U-Net (Breast Cancer Segmentation)
    - Any CNN with configurable target layer
    - 2D images (H x W) and 3D volumes (D x H x W)

Grad-CAM (Gradient-weighted Class Activation Mapping):
    1. Forward pass → get feature maps from target layer
    2. Backward pass → get gradients w.r.t. target layer
    3. Weight feature maps by global average of gradients
    4. ReLU + normalize → heatmap

Reference: Selvaraju et al. (2017) https://arxiv.org/abs/1610.02391
"""

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .visualization import overlay_heatmap

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM explanation for CNN-based segmentation models (e.g. U-Net).

    Args:
        model:        PyTorch model
        target_layer: nn.Module — layer to compute CAM from.
                      For U-Net: typically the last encoder block
                      e.g. model.encoder4 or model.bottleneck

    Example (Breast Cancer U-Net):
        cam = GradCAM(model, target_layer=model.encoder4)
        result = cam.explain(image_tensor)
        # result["heatmap_b64"] → base64 PNG of overlay
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.device = next(model.parameters()).device

        self._feature_maps: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks = []

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""

        def forward_hook(module, input, output):
            self._feature_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Clean up hooks — call when done with GradCAM."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        original_image: Optional[Union[Image.Image, np.ndarray]] = None,
        slice_axis: int = 0,
        slice_idx: Optional[int] = None,
        return_base64: bool = True,
        alpha: float = 0.5,
    ) -> dict:
        """
        Generate Grad-CAM explanation.

        Args:
            image_tensor:   Input tensor (B, C, H, W) or (B, C, D, H, W) for 3D
            target_class:   Class index to explain. None = uses predicted class.
            original_image: PIL Image or numpy array for overlay. If None, uses
                            first channel of input.
            slice_axis:     For 3D volumes — axis to slice (0=axial, 1=coronal, 2=sagittal)
            slice_idx:      For 3D volumes — which slice. None = middle slice.
            return_base64:  Return base64 PNG if True, else return raw numpy heatmap
            alpha:          Heatmap overlay transparency (0-1)

        Returns:
            dict with keys:
                heatmap_b64:   base64 PNG of heatmap overlay (if return_base64=True)
                heatmap_raw:   numpy array of raw heatmap (H, W) values 0-1
                target_class:  class that was explained
                method:        "gradcam"
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        is_3d = image_tensor.dim() == 5

        # Forward pass
        self.model.zero_grad()
        output = self.model(image_tensor)

        # Handle segmentation output (B, C, H, W) or (B, C, D, H, W)
        if isinstance(output, torch.Tensor):
            logits = output
        elif hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output[0]

        # Select target class
        if target_class is None:
            if logits.dim() > 2:
                # Segmentation: use class with highest mean activation
                target_class = (
                    logits.mean(dim=tuple(range(2, logits.dim()))).argmax(dim=1).item()
                )
            else:
                target_class = logits.argmax(dim=1).item()

        # Backward pass on target class
        if logits.dim() > 2:
            score = logits[:, target_class].mean()
        else:
            score = logits[:, target_class]

        score.backward()

        # Compute Grad-CAM
        gradients = self._gradients  # (B, C, ...)
        feature_maps = self._feature_maps  # (B, C, ...)

        if gradients is None or feature_maps is None:
            raise RuntimeError(
                "Hooks did not capture gradients/features. Check target_layer."
            )

        # Global average pooling of gradients
        if is_3d:
            weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        else:
            weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # For 3D: extract 2D slice
        if is_3d:
            D, H, W = cam.shape[2], cam.shape[3], cam.shape[4]
            if slice_idx is None:
                slice_indices = {"axial": D // 2, "coronal": H // 2, "sagittal": W // 2}
                slice_idx = list(slice_indices.values())[slice_axis]

            if slice_axis == 0:
                cam_2d = cam[0, 0, slice_idx, :, :]
            elif slice_axis == 1:
                cam_2d = cam[0, 0, :, slice_idx, :]
            else:
                cam_2d = cam[0, 0, :, :, slice_idx]
        else:
            cam_2d = cam[0, 0]

        # Normalize to [0, 1]
        cam_np = cam_2d.cpu().numpy()
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_np = np.zeros_like(cam_np)

        result = {
            "heatmap_raw": cam_np,
            "target_class": target_class,
            "method": "gradcam",
            "model_type": "3d_segmentation" if is_3d else "2d_segmentation",
        }

        if return_base64:
            # Get original image for overlay
            if original_image is None:
                if is_3d:
                    vol = image_tensor[0, 0].cpu().numpy()
                    if slice_axis == 0:
                        orig_slice = vol[slice_idx, :, :]
                    elif slice_axis == 1:
                        orig_slice = vol[:, slice_idx, :]
                    else:
                        orig_slice = vol[:, :, slice_idx]
                    original_image = orig_slice
                else:
                    original_image = image_tensor[0, 0].cpu().numpy()

            result["heatmap_b64"] = overlay_heatmap(
                original_image, cam_np, alpha=alpha, return_base64=True
            )

        return result

    def explain_3d_multiview(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
    ) -> dict:
        """
        Generate Grad-CAM for all three views of a 3D volume.
        Returns axial, coronal, sagittal heatmap overlays.

        Perfect for SwinUNETR brain tumor visualization.
        """
        results = {}
        axes = {"axial": 0, "coronal": 1, "sagittal": 2}

        for view_name, axis in axes.items():
            result = self.explain(
                image_tensor,
                target_class=target_class,
                slice_axis=axis,
                return_base64=True,
                alpha=alpha,
            )
            results[view_name] = result["heatmap_b64"]
            if target_class is None:
                target_class = result["target_class"]

        return {
            "axial": results["axial"],
            "coronal": results["coronal"],
            "sagittal": results["sagittal"],
            "target_class": target_class,
            "method": "gradcam_3d_multiview",
        }

    def __del__(self):
        self.remove_hooks()
