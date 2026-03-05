"""
medical_middleware.xai — Explainable AI for medical imaging models.

Supported methods:
    - Grad-CAM        → U-Net (Breast Cancer), CNN encoders
    - Attention Maps  → SwinUNETR (3D Brain), Med-GaMMa (Path-VQA)

Usage:
    from medical_middleware.xai import GradCAM, AttentionMap, explain_router

    # Breast Cancer / U-Net
    cam = GradCAM(model, target_layer=model.encoder4)
    heatmap_b64 = cam.explain(image_tensor, return_base64=True)

    # SwinUNETR
    attn = AttentionMap(model, model_type="swinunetr")
    heatmap_b64 = attn.explain(image_tensor, return_base64=True)

    # Path-VQA / Med-GaMMa
    attn = AttentionMap(model, model_type="medgemma")
    heatmap_b64 = attn.explain(image_tensor, question="Is this malignant?", return_base64=True)
"""

from .gradcam import GradCAM
from .attention import AttentionMap
from .visualization import overlay_heatmap, tensor_to_base64
from .router import make_explain_router

__all__ = [
    "GradCAM",
    "AttentionMap",
    "overlay_heatmap",
    "tensor_to_base64",
    "make_explain_router",
]
