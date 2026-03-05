"""
router.py — XAI explanation endpoints for medical AI APIs.

Adds to any FastAPI app:
    POST /explain/predict   → prediction + explanation in one call
    POST /explain/heatmap   → explanation only (if prediction already done)
    GET  /explain/methods   → list available XAI methods for this model

Usage:
    from medical_middleware.xai import make_explain_router

    # In your project's main.py
    explain_router = make_explain_router(
        model=model,
        model_type="unet",       # "unet" | "swinunetr" | "medgemma"
        predict_fn=predict_fn,   # your existing prediction function
        target_layer=model.encoder4,  # for Grad-CAM (U-Net / SwinUNETR)
        processor=processor,     # for Med-GaMMa only
    )
    app.include_router(explain_router, prefix="/explain", tags=["XAI"])
"""

import io
import logging
from typing import Callable, Optional

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

logger = logging.getLogger(__name__)


def make_explain_router(
    model: torch.nn.Module,
    model_type: str,
    predict_fn: Callable,
    target_layer=None,
    processor=None,
) -> APIRouter:
    """
    Factory function — creates an XAI router configured for your model.

    Args:
        model:        PyTorch model
        model_type:   "unet" | "swinunetr" | "medgemma"
        predict_fn:   Your existing prediction function
                      Signature: predict_fn(image: PIL.Image) -> dict
        target_layer: For Grad-CAM — the CNN layer to hook into
                      U-Net:      model.encoder4 (or last encoder block)
                      SwinUNETR:  model.swinViT.layers[-1] (optional, uses attention by default)
        processor:    For Med-GaMMa — the HuggingFace processor

    Returns:
        APIRouter with /explain endpoints
    """
    router = APIRouter()

    # Initialize XAI method based on model type
    explainer = _build_explainer(model, model_type, target_layer, processor)

    @router.get("/methods")
    def get_methods():
        """List available XAI methods for this model."""
        methods = {
            "unet": {
                "primary": "Grad-CAM",
                "description": "Highlights which image regions drove the segmentation decision",
                "output": "Heatmap overlay on input image",
                "reference": "Selvaraju et al. (2017) https://arxiv.org/abs/1610.02391",
            },
            "swinunetr": {
                "primary": "Attention Maps + Grad-CAM",
                "description": "Shows which 3D patches the Swin Transformer attended to",
                "output": "Axial, coronal, and sagittal heatmap overlays",
                "reference": "Hatamizadeh et al. (2022) https://arxiv.org/abs/2201.01266",
            },
            "medgemma": {
                "primary": "Vision Encoder Attention Maps",
                "description": "Shows which image regions influenced the VLM's answer",
                "output": "Heatmap overlay + natural language explanation",
                "reference": "Google MedGemma (2024)",
            },
        }
        return {
            "model_type": model_type,
            "xai_method": methods.get(model_type, {"primary": "Unknown"}),
            "endpoints": {
                "POST /explain/predict": "Get prediction + explanation in one call",
                "POST /explain/heatmap": "Get explanation for a specific image",
                "GET  /explain/methods": "This endpoint",
            },
        }

    @router.post("/predict")
    async def predict_with_explanation(
        request: Request,
        file: UploadFile = File(...),
        question: Optional[str] = Form(None),
        alpha: float = Form(0.5),
        multiview: bool = Form(False),
    ):
        """
        Run prediction AND generate XAI explanation in one call.

        Args:
            file:       Medical image file
            question:   For Med-GaMMa — the clinical question
            alpha:      Heatmap transparency (0.0 - 1.0)
            multiview:  For 3D models — return axial/coronal/sagittal views

        Returns:
            JSON with prediction result + base64 heatmap(s)

        Headers required:
            X-Data-Consent: true
        """
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Run prediction
            prediction = predict_fn(image)

            # Run XAI
            explanation = _run_explainer(
                explainer=explainer,
                model_type=model_type,
                image=image,
                question=question,
                processor=processor,
                alpha=alpha,
                multiview=multiview,
            )

            return JSONResponse(
                {
                    "prediction": prediction,
                    "explanation": explanation,
                    "xai_method": explanation.get("method"),
                    "request_id": getattr(request.state, "request_id", None),
                }
            )

        except Exception as e:
            logger.error(f"[xai] predict_with_explanation error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"XAI explanation failed: {str(e)}"
            )

    @router.post("/heatmap")
    async def get_heatmap(
        request: Request,
        file: UploadFile = File(...),
        question: Optional[str] = Form(None),
        alpha: float = Form(0.5),
        multiview: bool = Form(False),
    ):
        """
        Generate XAI explanation only (no prediction).
        Useful when you already have the prediction and just want the heatmap.
        """
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            explanation = _run_explainer(
                explainer=explainer,
                model_type=model_type,
                image=image,
                question=question,
                processor=processor,
                alpha=alpha,
                multiview=multiview,
            )

            return JSONResponse(
                {
                    "explanation": explanation,
                    "xai_method": explanation.get("method"),
                    "request_id": getattr(request.state, "request_id", None),
                }
            )

        except Exception as e:
            logger.error(f"[xai] get_heatmap error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Heatmap generation failed: {str(e)}"
            )

    return router


def _build_explainer(model, model_type, target_layer, processor):
    """Initialize the right explainer for the model type."""
    if model_type == "unet":
        from .gradcam import GradCAM

        if target_layer is None:
            # Try to auto-detect last encoder layer
            target_layer = _auto_detect_unet_layer(model)
        return GradCAM(model, target_layer)

    elif model_type == "swinunetr":
        from .attention import AttentionMap

        return AttentionMap(model, model_type="swinunetr")

    elif model_type == "medgemma":
        from .attention import AttentionMap

        return AttentionMap(model, model_type="medgemma")

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Use 'unet', 'swinunetr', or 'medgemma'"
        )


def _run_explainer(explainer, model_type, image, question, processor, alpha, multiview):
    """Run the explainer and return results dict."""
    import torchvision.transforms as T

    # Convert PIL to tensor
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    tensor = transform(image.convert("L")).unsqueeze(0)

    if model_type == "unet":
        result = explainer.explain(
            tensor,
            original_image=image,
            return_base64=True,
            alpha=alpha,
        )
        return {
            "method": result["method"],
            "heatmap_b64": result["heatmap_b64"],
            "target_class": result.get("target_class"),
            "clinical_note": (
                "Red regions indicate areas that most strongly influenced "
                "the segmentation. Blue regions had minimal influence."
            ),
        }

    elif model_type == "swinunetr":
        if multiview:
            result = explainer.explain_multiview(tensor, alpha=alpha)
        else:
            result = explainer.explain(
                tensor, original_image=image, return_base64=True, alpha=alpha
            )
        result["clinical_note"] = (
            "Highlighted regions show which anatomical patches the model "
            "attended to when generating the segmentation mask."
        )
        return result

    elif model_type == "medgemma":
        result = explainer.explain(
            tensor,
            question=question,
            processor=processor,
            original_image=image,
            return_base64=True,
            alpha=alpha,
        )
        return {
            "method": result["method"],
            "heatmap_b64": result.get("heatmap_b64"),
            "explanation_text": result.get("explanation_text"),
            "clinical_note": (
                "The highlighted regions show which parts of the pathology image "
                "the model used to formulate its answer."
            ),
        }


def _auto_detect_unet_layer(model):
    """Try to auto-detect the last encoder layer of a U-Net."""
    candidates = ["encoder4", "encoder3", "down4", "down3", "layer4", "features"]
    for name in candidates:
        if hasattr(model, name):
            logger.info(f"[xai] Auto-detected target layer: {name}")
            return getattr(model, name)

    # Fall back to last named child
    children = list(model.named_children())
    if children:
        name, layer = children[len(children) // 2]
        logger.warning(f"[xai] Could not auto-detect U-Net layer, using: {name}")
        return layer

    raise ValueError(
        "Could not auto-detect U-Net target layer. "
        "Pass target_layer explicitly to make_explain_router()."
    )
