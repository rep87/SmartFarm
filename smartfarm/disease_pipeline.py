"""Disease-risk analysis pipeline for top-down plant imagery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoModelForObjectDetection

import os
import urllib.request

DEFAULT_SAM_CKPT = "/content/sam_vit_b_01ec64.pth"
SAM_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def _cfg(config: Dict[str, object], key: str, default):
    return config.get(key, default)


def _ensure_sam_checkpoint(path=DEFAULT_SAM_CKPT):
    if os.path.exists(path) and os.path.getsize(path) > 10_000_000:
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[SAM] Downloading checkpoint to {path} ...")

    urllib.request.urlretrieve(SAM_CKPT_URL, path)

    if not os.path.exists(path):
        raise RuntimeError("SAM checkpoint download failed")

    return path


@dataclass
class LeafCandidate:
    bbox: Tuple[int, int, int, int]
    crop_rgb: np.ndarray
    mask: np.ndarray
    area_ratio: float
    green_ratio: float


@dataclass
class LeafRiskResult:
    bbox: Tuple[int, int, int, int]
    crop_rgb: np.ndarray
    risk_score: float
    healthy_probability: float
    entropy: float
    top_k_labels: List[str]
    top_k_probs: List[float]
    area_ratio: float
    green_ratio: float


@dataclass
class DetectionRegion:
    bbox: Tuple[int, int, int, int]
    score: float
    label: str


@dataclass
class DiseaseAnalysisResult:
    model_name: str
    leaf_results: List[LeafRiskResult] = field(default_factory=list)
    detections: List[DetectionRegion] = field(default_factory=list)


def _get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_sam(config: Dict[str, object]):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam_type = _cfg(config, "sam_model_type", "vit_b")
    sam_ckpt = _cfg(config, "sam_checkpoint", DEFAULT_SAM_CKPT)
    sam_ckpt = _ensure_sam_checkpoint(sam_ckpt)
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)

    sam.to(device=_get_device())
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=_cfg(config, "sam_points_per_side", 32),
        pred_iou_thresh=_cfg(config, "sam_pred_iou_thresh", 0.88),
        stability_score_thresh=_cfg(config, "sam_stability_score_thresh", 0.92),
        crop_n_layers=_cfg(config, "sam_crop_n_layers", 0),
        crop_n_points_downscale_factor=_cfg(
            config, "sam_crop_n_points_downscale_factor", 1
        ),
        min_mask_region_area=_cfg(config, "sam_min_mask_region_area", 300),
    )

    return mask_generator


def _compute_green_ratio(crop_rgb: np.ndarray, crop_mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    green_mask = (
        (h >= 35)
        & (h <= 85)
        & (s >= 40)
        & (v >= 40)
        & (crop_mask > 0)
    )
    total = np.count_nonzero(crop_mask)
    if total == 0:
        return 0.0
    return float(np.count_nonzero(green_mask)) / float(total)


def _is_shape_sane(bbox: Tuple[int, int, int, int], config: Dict[str, object]) -> bool:
    _, _, w, h = bbox
    if min(w, h) < _cfg(config, "min_bbox_size", 32):
        return False
    if h == 0:
        return False
    aspect_ratio = w / float(h)
    min_aspect_ratio = _cfg(config, "min_aspect_ratio", 0.2)
    max_aspect_ratio = _cfg(config, "max_aspect_ratio", 5.0)
    return min_aspect_ratio <= aspect_ratio <= max_aspect_ratio


def extract_leaf_candidates(
    image_bgr: np.ndarray, config: Dict[str, object]
) -> List[LeafCandidate]:
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("image_bgr must be a HxWx3 array")

    config = config or {}
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    image_area = float(height * width)

    mask_generator = _load_sam(config)
    masks = mask_generator.generate(image_rgb)

    candidates: List[LeafCandidate] = []
    max_candidates = int(_cfg(config, "max_candidates", 40))
    crop_pad = int(_cfg(config, "crop_pad", 12))
    for mask_info in masks:
        segmentation = mask_info.get("segmentation")
        if segmentation is None:
            continue
        mask = segmentation.astype(np.uint8)
        area_ratio = float(mask_info.get("area", np.sum(mask))) / image_area
        if area_ratio < _cfg(config, "min_area_ratio", 0.01):
            continue
        bbox_raw = mask_info.get("bbox")
        if not bbox_raw:
            continue
        x, y, w, h = map(int, bbox_raw)
        bbox = (x, y, w, h)
        if not _is_shape_sane(bbox, config):
            continue
        x0 = max(x - crop_pad, 0)
        y0 = max(y - crop_pad, 0)
        x1 = min(x + w + crop_pad, width)
        y1 = min(y + h + crop_pad, height)
        padded_bbox = (x0, y0, x1 - x0, y1 - y0)
        crop_rgb = image_rgb[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        green_ratio = _compute_green_ratio(crop_rgb, crop_mask)
        if green_ratio < _cfg(config, "min_green_ratio", 0.25):
            continue
        candidates.append(
            LeafCandidate(
                bbox=padded_bbox,
                crop_rgb=crop_rgb,
                mask=mask,
                area_ratio=area_ratio,
                green_ratio=green_ratio,
            )
        )
        if len(candidates) >= max_candidates:
            break
    return candidates


class _HFClassifier:
    def __init__(self, model_id: str, device: torch.device) -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        self.device = device
        config = self.model.config
        self.id2label = {int(k): v for k, v in config.id2label.items()}

    def predict(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits


def _load_classifier(model_name: str, config: Dict[str, object]) -> _HFClassifier:
    model_ids = _cfg(
        config,
        "model_ids",
        {
            "mobilenet": "nateraw/plantvillage-mobilenetv2",
            "efficientnet": "nateraw/plantvillage-efficientnet-b0",
        },
    )
    if model_name not in model_ids:
        available = ", ".join(sorted(model_ids))
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {available}")
    model_id = model_ids[model_name]
    device = _get_device(_cfg(config, "device", None))
    return _HFClassifier(model_id, device)


def _softmax_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)


def score_leaf_risk(
    crops: Sequence[np.ndarray],
    model_name: str,
    config: Optional[Dict[str, object]] = None,
) -> List[LeafRiskResult]:
    if config is None:
        config = {}

    if not crops:
        return []

    classifier = _load_classifier(model_name, config)
    logits = classifier.predict(crops)
    probs = torch.softmax(logits, dim=-1)

    top_k = min(int(_cfg(config, "top_k", 5)), probs.shape[-1])
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    entropy = _softmax_entropy(probs)

    healthy_label_keywords = _cfg(config, "healthy_label_keywords", ("healthy",))
    results: List[LeafRiskResult] = []
    for idx in range(len(crops)):
        labels = [classifier.id2label[int(i)] for i in top_indices[idx].tolist()]
        probs_list = top_probs[idx].tolist()
        healthy_probs = [
            float(probs[idx, class_idx].item())
            for class_idx, label in classifier.id2label.items()
            if any(keyword in label.lower() for keyword in healthy_label_keywords)
        ]
        healthy_probability = max(healthy_probs) if healthy_probs else 0.0
        risk_score = 1.0 - healthy_probability
        results.append(
            LeafRiskResult(
                bbox=(0, 0, 0, 0),
                crop_rgb=crops[idx],
                risk_score=float(risk_score),
                healthy_probability=float(healthy_probability),
                entropy=float(entropy[idx].item()),
                top_k_labels=labels,
                top_k_probs=[float(p) for p in probs_list],
                area_ratio=0.0,
                green_ratio=0.0,
            )
        )
    return results


def rank_risky_leaves(
    candidates: Sequence[LeafCandidate],
    risks: Sequence[LeafRiskResult],
    config: Optional[Dict[str, object]] = None,
) -> List[LeafRiskResult]:
    if config is None:
        config = {}
    if len(candidates) != len(risks):
        raise ValueError("candidates and risks must have the same length")

    enriched: List[LeafRiskResult] = []
    for candidate, risk in zip(candidates, risks):
        enriched.append(
            LeafRiskResult(
                bbox=candidate.bbox,
                crop_rgb=candidate.crop_rgb,
                risk_score=risk.risk_score,
                healthy_probability=risk.healthy_probability,
                entropy=risk.entropy,
                top_k_labels=risk.top_k_labels,
                top_k_probs=risk.top_k_probs,
                area_ratio=candidate.area_ratio,
                green_ratio=candidate.green_ratio,
            )
        )

    enriched.sort(key=lambda item: (item.risk_score, item.entropy), reverse=True)
    top_n = int(_cfg(config, "top_n_risky", 8))
    return enriched[:top_n]


class _HFDetector:
    def __init__(self, model_id: str, device: torch.device) -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        self.device = device
        config = self.model.config
        self.id2label = {int(k): v for k, v in config.id2label.items()}

    def detect(self, image_rgb: np.ndarray, score_threshold: float) -> List[DetectionRegion]:
        inputs = self.processor(images=[image_rgb], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        height, width = image_rgb.shape[:2]
        target_sizes = torch.tensor([[height, width]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes
        )[0]

        detections: List[DetectionRegion] = []
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x_min, y_min, x_max, y_max = box.tolist()
            bbox = (
                int(x_min),
                int(y_min),
                int(x_max - x_min),
                int(y_max - y_min),
            )
            label = self.id2label.get(int(label_id), str(int(label_id)))
            detections.append(
                DetectionRegion(
                    bbox=bbox, score=float(score.item()), label=label
                )
            )
        return detections


def _run_classification_pipeline(
    image_bgr: np.ndarray,
    model_name: str,
    config: Dict[str, object],
) -> DiseaseAnalysisResult:
    config = config or {}
    candidates = extract_leaf_candidates(image_bgr, _cfg(config, "candidate", {}))
    crops = [candidate.crop_rgb for candidate in candidates]
    risks = score_leaf_risk(crops, model_name, _cfg(config, "scoring", {}))
    ranked = rank_risky_leaves(candidates, risks, _cfg(config, "ranking", {}))
    return DiseaseAnalysisResult(model_name=model_name, leaf_results=ranked)


def _run_detection_pipeline(
    image_bgr: np.ndarray,
    config: Dict[str, object],
) -> DiseaseAnalysisResult:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    config = config or {}
    device = _get_device(_cfg(config, "device", None))
    model_id = _cfg(
        config, "model_id", "nickmuchi/yolos-small-plant-disease-detection"
    )
    score_threshold = float(_cfg(config, "score_threshold", 0.3))
    detector = _HFDetector(model_id, device)
    detections = detector.detect(image_rgb, score_threshold)
    return DiseaseAnalysisResult(model_name="yolos", detections=detections)


def run_disease_analysis(
    image_bgr: np.ndarray,
    model_name: str,
    config: Dict[str, object],
) -> DiseaseAnalysisResult:
    """
    model_name: 'mobilenet' | 'efficientnet' | 'yolos'
    returns a unified result object for visualization
    """

    if model_name in ("mobilenet", "efficientnet"):
        return _run_classification_pipeline(image_bgr, model_name, config)
    if model_name == "yolos":
        return _run_detection_pipeline(image_bgr, _cfg(config or {}, "detection", {}))
    raise ValueError("model_name must be one of: 'mobilenet', 'efficientnet', 'yolos'")


__all__ = [
    "LeafCandidate",
    "LeafRiskResult",
    "DetectionRegion",
    "DiseaseAnalysisResult",
    "extract_leaf_candidates",
    "score_leaf_risk",
    "rank_risky_leaves",
    "run_disease_analysis",
]
