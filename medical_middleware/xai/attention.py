"""
attention.py — Attention map extraction for transformer-based medical AI models.

Supports:
    - SwinUNETR  → Swin Transformer window attention weights (3D brain segmentation)
    - Med-GaMMa  → Vision encoder cross-attention (Path-VQA)

For transformers, attention weights directly show which input patches
the model focused on — more interpretable than gradient methods for
clinical explanation.
"""

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .visualization import overlay_heatmap

logger = logging.getLogger(__name__)


class AttentionMap:
    """
    Attention map extractor for transformer-based medical AI models.

    Args:
        model:       PyTorch model (SwinUNETR or Med-GaMMa)
        model_type:  "swinunetr" | "medgemma"
        layer_idx:   Which attention layer to extract from (-1 = last)

    Example (SwinUNETR):
        attn = AttentionMap(model, model_type="swinunetr")
        result = attn.explain(image_tensor)

    Example (Med-GaMMa):
        attn = AttentionMap(model, model_type="medgemma")
        result = attn.explain(
            image_tensor,
            question="Is this malignant?",
            processor=processor
        )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str = "swinunetr",
        layer_idx: int = -1,
    ):
        self.model = model
        self.model_type = model_type
        self.layer_idx = layer_idx
        self.device = next(model.parameters()).device

        self._attention_weights = []
        self._hooks = []

        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on attention layers based on model type."""
        self._attention_weights = []

        if self.model_type == "swinunetr":
            self._register_swinunetr_hooks()
        elif self.model_type == "medgemma":
            self._register_medgemma_hooks()
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. Use 'swinunetr' or 'medgemma'"
            )

    def _register_swinunetr_hooks(self):
        """Hook into Swin Transformer attention layers."""
        hooked = 0
        for name, module in self.model.named_modules():
            # MONAI SwinUNETR attention module names
            if any(x in name for x in ["attn", "attention", "WindowAttention"]):
                if hasattr(module, "forward"):
                    hook = module.register_forward_hook(self._capture_attention)
                    self._hooks.append(hook)
                    hooked += 1

        if hooked == 0:
            logger.warning(
                "[xai] No attention layers found in SwinUNETR — trying generic hooks"
            )
            self._register_generic_hooks()

        logger.debug(f"[xai] Registered hooks on {hooked} SwinUNETR attention layers")

    def _register_medgemma_hooks(self):
        """Hook into Med-GaMMa vision encoder attention layers."""
        hooked = 0
        for name, module in self.model.named_modules():
            # Gemma3 / PaliGemma attention module names
            if any(
                x in name
                for x in [
                    "self_attn",
                    "cross_attn",
                    "SiglipAttention",
                    "Gemma3Attention",
                ]
            ):
                if hasattr(module, "forward"):
                    hook = module.register_forward_hook(self._capture_attention)
                    self._hooks.append(hook)
                    hooked += 1

        logger.debug(f"[xai] Registered hooks on {hooked} Med-GaMMa attention layers")

    def _register_generic_hooks(self):
        """Fallback: hook into any module with 'attn' in name."""
        for name, module in self.model.named_modules():
            if "attn" in name.lower() and len(list(module.children())) == 0:
                hook = module.register_forward_hook(self._capture_attention)
                self._hooks.append(hook)

    def _capture_attention(self, module, input, output):
        """Capture attention weights from module output."""
        # Attention modules return (output, attn_weights) or just output
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[1]
            if attn is not None and isinstance(attn, torch.Tensor):
                self._attention_weights.append(attn.detach())
        elif isinstance(output, torch.Tensor):
            # Some modules return attention weights directly
            if output.dim() >= 3:
                self._attention_weights.append(output.detach())

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def explain(
        self,
        image_tensor: torch.Tensor,
        question: Optional[str] = None,
        processor=None,
        original_image: Optional[Union[Image.Image, np.ndarray]] = None,
        slice_axis: int = 0,
        slice_idx: Optional[int] = None,
        return_base64: bool = True,
        alpha: float = 0.5,
    ) -> dict:
        """
        Generate attention map explanation.

        Args:
            image_tensor:   Image tensor (B, C, H, W) or (B, C, D, H, W)
            question:       For Med-GaMMa — the clinical question being asked
            processor:      For Med-GaMMa — the model processor
            original_image: PIL Image or numpy array for overlay
            slice_axis:     For 3D — 0=axial, 1=coronal, 2=sagittal
            slice_idx:      For 3D — which slice (None = middle)
            return_base64:  Return base64 PNG
            alpha:          Overlay transparency

        Returns:
            dict with heatmap_b64, heatmap_raw, method, explanation_text
        """
        self._attention_weights = []
        self.model.eval()
        image_tensor = image_tensor.to(self.device)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        is_3d = image_tensor.dim() == 5

        with torch.no_grad():
            if self.model_type == "medgemma" and processor and question:
                self._forward_medgemma(image_tensor, question, processor)
            else:
                self.model(image_tensor)

        # Extract attention weights from the target layer
        if not self._attention_weights:
            logger.warning(
                "[xai] No attention weights captured — returning uniform map"
            )
            if is_3d:
                h, w = image_tensor.shape[3], image_tensor.shape[4]
            else:
                h, w = image_tensor.shape[2], image_tensor.shape[3]
            attn_map = np.ones((h, w)) * 0.5
        else:
            attn = self._attention_weights[self.layer_idx]
            attn_map = self._process_attention(
                attn, image_tensor.shape, is_3d, slice_axis, slice_idx
            )

        result = {
            "heatmap_raw": attn_map,
            "method": f"attention_{self.model_type}",
            "layers_captured": len(self._attention_weights),
        }

        # Generate explanation text for Med-GaMMa
        if self.model_type == "medgemma" and question:
            result["explanation_text"] = self._generate_explanation_text(
                attn_map, question
            )

        if return_base64:
            if original_image is None:
                if is_3d:
                    vol = image_tensor[0, 0].cpu().numpy()
                    if slice_axis == 0:
                        si = slice_idx or vol.shape[0] // 2
                        original_image = vol[si]
                    elif slice_axis == 1:
                        si = slice_idx or vol.shape[1] // 2
                        original_image = vol[:, si, :]
                    else:
                        si = slice_idx or vol.shape[2] // 2
                        original_image = vol[:, :, si]
                else:
                    original_image = image_tensor[0, 0].cpu().numpy()

            result["heatmap_b64"] = overlay_heatmap(
                original_image, attn_map, alpha=alpha, return_base64=True
            )

        return result

    def _forward_medgemma(self, image_tensor, question, processor):
        """Forward pass for Med-GaMMa with question."""
        try:
            from PIL import Image as PILImage

            # Convert tensor back to PIL for processor
            img_np = image_tensor[0, 0].cpu().numpy()
            img_np = (
                (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255
            ).astype(np.uint8)
            pil_img = PILImage.fromarray(img_np).convert("RGB")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are an expert pathologist."},
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

            return self.model(**inputs, output_attentions=True)

        except Exception as e:
            logger.error(f"[xai] Med-GaMMa forward failed: {e}")
            return self.model(image_tensor, output_attentions=True)

    def _process_attention(
        self,
        attn: torch.Tensor,
        input_shape: torch.Size,
        is_3d: bool,
        slice_axis: int,
        slice_idx: Optional[int],
    ) -> np.ndarray:
        """Convert raw attention tensor to 2D heatmap."""
        attn_np = attn.cpu().float().numpy()

        # Average over heads and batch
        while attn_np.ndim > 2:
            attn_np = attn_np.mean(axis=0)

        # Average over query dimension if still 2D
        if attn_np.ndim == 2:
            attn_np = attn_np.mean(axis=0)

        # Get target spatial size
        if is_3d:
            D, H, W = input_shape[2], input_shape[3], input_shape[4]
            target_sizes = {0: (H, W), 1: (D, W), 2: (D, H)}
            target_h, target_w = target_sizes[slice_axis]
        else:
            target_h, target_w = input_shape[2], input_shape[3]

        # Reshape to spatial map (approximate patch grid)
        n_patches = attn_np.shape[0]
        patch_h = patch_w = int(np.sqrt(n_patches))

        if patch_h * patch_w == n_patches:
            attn_2d = attn_np.reshape(patch_h, patch_w)
        else:
            # Non-square: just reshape to closest square
            side = int(np.sqrt(n_patches))
            attn_2d = attn_np[: side * side].reshape(side, side)

        # Resize to image dimensions
        attn_tensor = torch.from_numpy(attn_2d).unsqueeze(0).unsqueeze(0).float()
        attn_resized = F.interpolate(
            attn_tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        # Normalize
        if attn_resized.max() > attn_resized.min():
            attn_resized = (attn_resized - attn_resized.min()) / (
                attn_resized.max() - attn_resized.min()
            )

        return attn_resized

    def _generate_explanation_text(self, attn_map: np.ndarray, question: str) -> str:
        """Generate human-readable explanation for Med-GaMMa."""
        # Find regions with highest attention
        threshold = attn_map.mean() + attn_map.std()
        high_attn_pct = (attn_map > threshold).mean() * 100

        h, w = attn_map.shape
        # Find centroid of high-attention region
        high_mask = attn_map > threshold
        if high_mask.any():
            rows, cols = np.where(high_mask)
            center_y = rows.mean() / h
            center_x = cols.mean() / w
            location = _describe_location(center_x, center_y)
        else:
            location = "distributed across the image"

        return (
            f"To answer '{question}', the model focused primarily on "
            f"the {location} of the image ({high_attn_pct:.1f}% of patches "
            f"received above-average attention)."
        )

    def explain_multiview(
        self,
        image_tensor: torch.Tensor,
        alpha: float = 0.5,
    ) -> dict:
        """Generate attention maps for axial, coronal, and sagittal views (3D only)."""
        views = {}
        for name, axis in [("axial", 0), ("coronal", 1), ("sagittal", 2)]:
            result = self.explain(
                image_tensor, slice_axis=axis, return_base64=True, alpha=alpha
            )
            views[name] = result["heatmap_b64"]

        return {
            "axial": views["axial"],
            "coronal": views["coronal"],
            "sagittal": views["sagittal"],
            "method": f"attention_{self.model_type}_multiview",
        }

    def __del__(self):
        self.remove_hooks()


def _describe_location(cx: float, cy: float) -> str:
    """Convert normalized (x,y) center to human-readable location."""
    v = "upper" if cy < 0.33 else ("lower" if cy > 0.66 else "middle")
    h = "left" if cx < 0.33 else ("right" if cx > 0.66 else "central")
    if h == "central" and v == "middle":
        return "central region"
    return f"{v}-{h} region"
