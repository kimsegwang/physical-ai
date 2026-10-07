"""--show용 디버그 오버레이. 화면 표시만 하고 파일로 저장하지 않는다."""

from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig
from .pipeline import FrameResult

_ROI = (255, 200, 0)
_MASK = (80, 80, 80)
_LINE_A = (0, 220, 0)
_LINE_B = (0, 0, 255)
_TRACK = (0, 255, 255)
_AUX = (255, 0, 255)


def _px(p, w: int, h: int) -> tuple[int, int]:
    return int(round(p[0] * w)), int(round(p[1] * h))


def draw_overlay(result: FrameResult, cfg: PipelineConfig) -> np.ndarray:
    img = result.pre.frame.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]

    x0, y0, x1, y1 = result.pre.roi_px
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), _ROI, 1)
    for poly in result.pre.masks:
        pts = np.array([_px(p, w, h) for p in poly], np.int32)
        cv2.polylines(img, [pts], True, _MASK, 2)

    for gate in cfg.counting.gates:
        cv2.line(img, _px(gate.line_a[0], w, h), _px(gate.line_a[1], w, h), _LINE_A, 2)
        cv2.line(img, _px(gate.line_b[0], w, h), _px(gate.line_b[1], w, h), _LINE_B, 2)
        cv2.putText(img, gate.name, _px(gate.line_a[0], w, h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _LINE_A, 1)

    for t in result.tracks:
        bx1, by1 = _px(t.box[:2], w, h)
        bx2, by2 = _px(t.box[2:], w, h)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), _TRACK, 2)
        cv2.putText(img, f"{t.track_id}", (bx1, max(12, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TRACK, 1)
    for d in result.aux:
        cv2.rectangle(img, _px((d.x1, d.y1), w, h), _px((d.x2, d.y2), w, h), _AUX, 1)

    info = (
        f"IN {result.totals['in']}  OUT {result.totals['out']}  "
        f"mode={result.mode}  clahe={'on' if result.pre.clahe_on else 'off'}"
    )
    cv2.putText(img, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


def show(result: FrameResult, cfg: PipelineConfig, window: str = "subway_edge") -> bool:
    """오버레이를 띄운다. q 또는 ESC를 누르면 False."""
    cv2.imshow(window, draw_overlay(result, cfg))
    return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)
