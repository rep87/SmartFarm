"""Disease-risk analysis pipeline for top-down plant imagery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoModelForObjectDetection


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


@dataclass
class LeafCandidateConfig:
    sam_checkpoint: str
    sam_model_type: str = "vit_b"
    min_area_ratio: float = 0.01
    min_green_ratio: float = 0.15
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0
    min_bbox_size: int = 32
    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.86
    sam_stability_score_thresh: float = 0.92
    sam_crop_n_layers: int = 1
    sam_crop_n_points_downscale_factor: int = 2
    sam_min_mask_region_area: int = 100


@dataclass
class RiskScoringConfig:
    device: Optional[str] = None
    top_k: int = 5
    healthy_label_keywords: Tuple[str, ...] = ("healthy",)
    model_ids: Dict[str, str] = field(
        default_factory=lambda: {
            "mobilenet": "nateraw/plantvillage-mobilenetv2",
            "efficientnet": "nateraw/plantvillage-efficientnet-b0",
        }
    )


@dataclass
class RankingConfig:
    top_n: int = 5


@dataclass
class DetectionConfig:
    device: Optional[str] = None
    model_id: str = "nickmuchi/yolos-small-plant-disease-detection"
    score_threshold: float = 0.3


@dataclass
class DiseasePipelineConfig:
    candidate: LeafCandidateConfig
    scoring: RiskScoringConfig = field(default_factory=RiskScoringConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)


def _get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_sam(config: LeafCandidateConfig):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[config.sam_model_type](checkpoint=config.sam_checkpoint)
    sam.to(device=_get_device())
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=config.sam_points_per_side,
        pred_iou_thresh=config.sam_pred_iou_thresh,
        stability_score_thresh=config.sam_stability_score_thresh,
        crop_n_layers=config.sam_crop_n_layers,
        crop_n_points_downscale_factor=config.sam_crop_n_points_downscale_factor,
        min_mask_region_area=config.sam_min_mask_region_area,
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


def _is_shape_sane(bbox: Tuple[int, int, int, int], config: LeafCandidateConfig) -> bool:
    _, _, w, h = bbox
    if min(w, h) < config.min_bbox_size:
        return False
    if h == 0:
        return False
    aspect_ratio = w / float(h)
    return config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio


def extract_leaf_candidates(
    image_bgr: np.ndarray, config: LeafCandidateConfig
) -> List[LeafCandidate]:
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("image_bgr must be a HxWx3 array")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    image_area = float(height * width)

    mask_generator = _load_sam(config)
    masks = mask_generator.generate(image_rgb)

    candidates: List[LeafCandidate] = []
    for mask_info in masks:
        segmentation = mask_info.get("segmentation")
        if segmentation is None:
            continue
        mask = segmentation.astype(np.uint8)
        area_ratio = float(mask_info.get("area", np.sum(mask))) / image_area
        if area_ratio < config.min_area_ratio:
            continue
        bbox_raw = mask_info.get("bbox")
        if not bbox_raw:
            continue
        x, y, w, h = map(int, bbox_raw)
        bbox = (x, y, w, h)
        if not _is_shape_sane(bbox, config):
            continue
        crop_rgb = image_rgb[y : y + h, x : x + w]
        crop_mask = mask[y : y + h, x : x + w]
        green_ratio = _compute_green_ratio(crop_rgb, crop_mask)
        if green_ratio < config.min_green_ratio:
            continue
        candidates.append(
            LeafCandidate(
                bbox=bbox,
                crop_rgb=crop_rgb,
                mask=mask,
                area_ratio=area_ratio,
                green_ratio=green_ratio,
            )
        )
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


def _load_classifier(model_name: str, config: RiskScoringConfig) -> _HFClassifier:
    if model_name not in config.model_ids:
        available = ", ".join(sorted(config.model_ids))
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {available}")
    model_id = config.model_ids[model_name]
    device = _get_device(config.device)
    return _HFClassifier(model_id, device)


def _softmax_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)


def score_leaf_risk(
    crops: Sequence[np.ndarray],
    model_name: str,
    config: Optional[RiskScoringConfig] = None,
) -> List[LeafRiskResult]:
    if config is None:
        config = RiskScoringConfig()

    if not crops:
        return []

    classifier = _load_classifier(model_name, config)
    logits = classifier.predict(crops)
    probs = torch.softmax(logits, dim=-1)

    top_k = min(config.top_k, probs.shape[-1])
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    entropy = _softmax_entropy(probs)

    results: List[LeafRiskResult] = []
    for idx in range(len(crops)):
        labels = [classifier.id2label[int(i)] for i in top_indices[idx].tolist()]
        probs_list = top_probs[idx].tolist()
        healthy_probs = [
            float(probs[idx, class_idx].item())
            for class_idx, label in classifier.id2label.items()
            if any(keyword in label.lower() for keyword in config.healthy_label_keywords)
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
    config: Optional[RankingConfig] = None,
) -> List[LeafRiskResult]:
    if config is None:
        config = RankingConfig()
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
    return enriched[: config.top_n]


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
    config: DiseasePipelineConfig,
) -> DiseaseAnalysisResult:
    candidates = extract_leaf_candidates(image_bgr, config.candidate)
    crops = [candidate.crop_rgb for candidate in candidates]
    risks = score_leaf_risk(crops, model_name, config.scoring)
    ranked = rank_risky_leaves(candidates, risks, config.ranking)
    return DiseaseAnalysisResult(model_name=model_name, leaf_results=ranked)


def _run_detection_pipeline(
    image_bgr: np.ndarray,
    config: DetectionConfig,
) -> DiseaseAnalysisResult:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    device = _get_device(config.device)
    detector = _HFDetector(config.model_id, device)
    detections = detector.detect(image_rgb, config.score_threshold)
    return DiseaseAnalysisResult(model_name="yolos", detections=detections)


def run_disease_analysis(
    image_bgr: np.ndarray,
    model_name: str,
    config: DiseasePipelineConfig,
) -> DiseaseAnalysisResult:
    """
    model_name: 'mobilenet' | 'efficientnet' | 'yolos'
    returns a unified result object for visualization
    """

    if model_name in ("mobilenet", "efficientnet"):
        return _run_classification_pipeline(image_bgr, model_name, config)
    if model_name == "yolos":
        return _run_detection_pipeline(image_bgr, config.detection)
    raise ValueError("model_name must be one of: 'mobilenet', 'efficientnet', 'yolos'")


__all__ = [
    "LeafCandidate",
    "LeafCandidateConfig",
    "LeafRiskResult",
    "DetectionRegion",
    "DiseaseAnalysisResult",
    "RiskScoringConfig",
    "RankingConfig",
    "DetectionConfig",
    "DiseasePipelineConfig",
    "extract_leaf_candidates",
    "score_leaf_risk",
    "rank_risky_leaves",
    "run_disease_analysis",
]
