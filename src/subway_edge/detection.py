"""6단계: 사람 검출 + 결과 필터.

검출기는 Detector 인터페이스로 분리해서 YOLO 없이도 가짜 검출기로 테스트할 수 있다.
검출기는 입력 영상(ROI) 픽셀 좌표를 돌려주고, 필터 단계에서 전체 프레임 정규화 좌표로 바꾼다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

import numpy as np

from .config import DetectionConfig, Polygon
from .preprocess import PreprocessResult


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int = 0

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]:
        """입력 영상 픽셀 좌표 기준 검출 결과를 돌려준다."""
        ...


class YoloDetector:
    """ultralytics YOLO 검출기. ultralytics는 이 클래스를 만들 때만 import한다."""

    def __init__(
        self,
        model_path: str,
        classes: Sequence[int],
        conf_floor: float = 0.1,
        imgsz: int = 640,
        device: str = "",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.classes = list(classes)
        self.conf_floor = conf_floor
        self.imgsz = imgsz
        self.device = device or None

    @classmethod
    def from_config(cls, cfg: DetectionConfig, imgsz: int) -> "YoloDetector":
        classes = [cfg.person_class, *cfg.aux_classes]
        return cls(cfg.model_path, classes, cfg.low_conf_threshold, imgsz, cfg.device)

    def detect(self, image: np.ndarray) -> list[Detection]:
        # NMS는 설정값(혼잡도별로 바뀜)으로 filter_detections에서 다시 하므로 여기서는 느슨하게 둔다
        result = self.model.predict(
            image,
            conf=self.conf_floor,
            iou=0.9,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return [Detection(*map(float, b), float(s), int(c)) for b, s, c in zip(xyxy, conf, cls)]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4)와 (M,4) 박스의 IoU 행렬."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(dets: list[Detection], iou_threshold: float) -> list[Detection]:
    """점수 순 greedy NMS. 혼잡할수록 iou_threshold를 높여 겹친 사람을 지우지 않게 한다."""
    if len(dets) <= 1:
        return list(dets)
    order = sorted(range(len(dets)), key=lambda i: dets[i].score, reverse=True)
    boxes = np.array([dets[i].box for i in order], np.float64)
    ious = iou_matrix(boxes, boxes)
    keep: list[int] = []
    suppressed = np.zeros(len(order), bool)
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        suppressed |= ious[i] > iou_threshold
    return [dets[i] for i in keep]


def point_in_polygon(x: float, y: float, poly: Polygon) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _size_ok(det: Detection, cfg: DetectionConfig) -> bool:
    top = cfg.ref_person_area_top
    if top <= 0:
        return True
    bottom = cfg.ref_person_area_bottom if cfg.ref_person_area_bottom > 0 else top
    t = min(max(det.center[1], 0.0), 1.0)
    ref = top + (bottom - top) * t  # 화면 위치(y)별 기준 면적
    return ref * cfg.min_area_ratio <= det.area <= ref * cfg.max_area_ratio


def normalize_detections(dets: list[Detection], pre: PreprocessResult) -> list[Detection]:
    """ROI 픽셀 좌표 → 전체 프레임 정규화 좌표."""
    out = []
    for d in dets:
        x1, y1 = pre.to_normalized(d.x1, d.y1)
        x2, y2 = pre.to_normalized(d.x2, d.y2)
        out.append(replace(d, x1=x1, y1=y1, x2=x2, y2=y2))
    return out


def filter_detections(
    dets: list[Detection],
    pre: PreprocessResult,
    cfg: DetectionConfig,
) -> tuple[list[Detection], list[Detection]]:
    """신뢰도·마스크·클래스·크기·NMS 필터. (사람, 보조 객체)를 정규화 좌표로 돌려준다.

    사람은 low_conf_threshold 이상을 모두 남긴다. 고/저 신뢰도 구분은 추적 단계에서 한다.
    보조 객체(캐리어 등)는 카운트에 쓰지 않고 확인용으로만 돌려준다.
    """
    persons: list[Detection] = []
    aux: list[Detection] = []
    for d in normalize_detections(dets, pre):
        if d.score < cfg.low_conf_threshold:
            continue
        cx, cy = d.center
        if any(point_in_polygon(cx, cy, poly) for poly in pre.masks):
            continue
        if d.cls == cfg.person_class:
            if _size_ok(d, cfg):
                persons.append(d)
        elif d.cls in cfg.aux_classes and d.score >= cfg.conf_threshold:
            aux.append(d)
    return nms(persons, cfg.nms_iou), aux
