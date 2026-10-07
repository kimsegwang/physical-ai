"""전처리 1~5단계.

1. 리사이즈·샘플링  2. 영상 안정화(조건부)  3. ROI 크롭 + 고정 마스크
4. 밝기 깜빡임 보정(조건부)  5. CLAHE(조건부/자동)

ROI로 먼저 자르고 마스크를 칠한 뒤에 보정을 하므로, 보정 연산은 ROI 안에서만 돈다.
광고판처럼 밝은 영역이 밝기 계산을 왜곡하지 않도록 밝기 통계는 마스크 밖 픽셀로만 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Polygon, PreprocessConfig


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_frame(frame: np.ndarray, input_size: int) -> np.ndarray:
    """긴 변을 input_size에 맞춰 줄인다. 작은 영상은 키우지 않는다(연산만 늘어남)."""
    h, w = frame.shape[:2]
    scale = input_size / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


class FrameSampler:
    """타임스탬프 기준으로 처리할 프레임만 고른다 (원본 fps → 처리 fps)."""

    def __init__(self, process_fps: float) -> None:
        self.interval = 1.0 / process_fps if process_fps > 0 else 0.0
        self._next: float | None = None

    def accept(self, ts: float) -> bool:
        if self._next is not None and ts < self._next - 1e-6:
            return False
        # 늦게 도착한 프레임이 있어도 평균 처리 fps가 유지되도록 예정 시각 기준으로 다음 시각을 잡는다
        if self._next is None or ts - self._next >= self.interval:
            self._next = ts + self.interval
        else:
            self._next += self.interval
        return True


class Stabilizer:
    """평행이동 흔들림 보정 (위상 상관). 축소한 흑백 영상으로 이동량만 추정한다."""

    def __init__(self, width: int, max_shift: float) -> None:
        self.width = width
        self.max_shift = max_shift
        self._ref: np.ndarray | None = None
        self._window: np.ndarray | None = None

    def run(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        h, w = frame.shape[:2]
        sw = min(self.width, w)
        sh = max(1, round(h * sw / w))
        small = cv2.resize(to_gray(frame), (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._ref is None or self._ref.shape != small.shape:
            # 첫 프레임(또는 해상도 변경)을 기준 영상으로 삼는다
            self._ref = small
            self._window = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
            return frame, (0.0, 0.0)
        (dx, dy), _ = cv2.phaseCorrelate(self._ref, small, self._window)
        dx, dy = dx * w / sw, dy * h / sh
        if abs(dx) > self.max_shift * w or abs(dy) > self.max_shift * h:
            return frame, (0.0, 0.0)
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return frame, (0.0, 0.0)
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        out = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return out, (dx, dy)


@dataclass
class PreprocessResult:
    image: np.ndarray  # 검출기에 넣을 ROI 영상 (마스크·보정 적용)
    frame: np.ndarray  # 리사이즈·안정화된 전체 프레임 (디버그 표시용)
    roi_px: tuple[int, int, int, int]  # frame 기준 ROI 픽셀 좌표 (x0, y0, x1, y1)
    shift: tuple[float, float]  # 안정화로 보정한 이동량(px)
    clahe_on: bool
    masks: list[Polygon]  # 적용된 마스크 (정규화 좌표)

    @property
    def frame_size(self) -> tuple[int, int]:
        h, w = self.frame.shape[:2]
        return w, h

    def to_normalized(self, x: float, y: float) -> tuple[float, float]:
        """ROI 영상의 픽셀 좌표 → 전체 프레임 정규화 좌표."""
        w, h = self.frame_size
        return (x + self.roi_px[0]) / w, (y + self.roi_px[1]) / h


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self._stabilizer: Stabilizer | None = None
        self._flicker_ref: float | None = None
        self._clahe_on = False
        self._frame_idx = 0
        self.apply_config(cfg)

    def apply_config(self, cfg: PreprocessConfig) -> None:
        """설정 교체 (시간대·날씨 전환 시 호출). 무거운 객체는 켜질 때만 만든다."""
        self.cfg = cfg
        self._mask_key = None
        self._mask: np.ndarray | None = None
        if cfg.stabilization and self._stabilizer is None:
            self._stabilizer = Stabilizer(cfg.stab_width, cfg.stab_max_shift)
        elif not cfg.stabilization:
            self._stabilizer = None
        if not cfg.flicker_correction:
            self._flicker_ref = None
        self._clahe = (
            cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
            if cfg.clahe_mode != "off"
            else None
        )

    def active_masks(self) -> list[Polygon]:
        masks = list(self.cfg.masks)
        if self.cfg.apply_wet_floor_masks:
            masks += self.cfg.wet_floor_masks
        return masks

    def run(self, frame: np.ndarray) -> PreprocessResult:
        cfg = self.cfg
        self._frame_idx += 1

        # 1. 리사이즈 (샘플링은 FrameSampler가 앞에서 처리)
        frame = resize_frame(frame, cfg.input_size)

        # 2. 영상 안정화
        shift = (0.0, 0.0)
        if self._stabilizer is not None:
            frame, shift = self._stabilizer.run(frame)

        # 3. ROI 크롭 + 고정 마스크
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = cfg.roi
        x0, y0 = int(round(rx1 * w)), int(round(ry1 * h))
        x1, y1 = max(x0 + 1, int(round(rx2 * w))), max(y0 + 1, int(round(ry2 * h)))
        roi_px = (x0, y0, x1, y1)
        image = frame[y0:y1, x0:x1]
        mask = self._roi_mask(w, h, roi_px)
        if mask is not None:
            image = cv2.bitwise_and(image, image, mask=mask)

        # 4. 밝기 깜빡임 보정
        if cfg.flicker_correction:
            image = self._correct_flicker(image, mask)

        # 5. CLAHE
        clahe_on = self._decide_clahe(image, mask)
        if clahe_on:
            image = self._apply_clahe(image)

        return PreprocessResult(image, frame, roi_px, shift, clahe_on, self.active_masks())

    def _roi_mask(self, w: int, h: int, roi_px: tuple[int, int, int, int]) -> np.ndarray | None:
        """ROI 영상 크기의 마스크 (255=사용, 0=가림). 크기·설정이 같으면 재사용한다."""
        key = (w, h, roi_px)
        if key == self._mask_key:
            return self._mask
        polygons = self.active_masks()
        mask = None
        if polygons:
            x0, y0, x1, y1 = roi_px
            mask = np.full((y1 - y0, x1 - x0), 255, np.uint8)
            for poly in polygons:
                pts = np.array([[x * w - x0, y * h - y0] for x, y in poly], np.float32)
                cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 0)
        self._mask_key, self._mask = key, mask
        return mask

    def _correct_flicker(self, image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """프레임 평균 밝기를 직전 프레임들의 이동평균에 맞춘다."""
        small = to_gray(image[::4, ::4])
        small_mask = mask[::4, ::4] if mask is not None else None
        current = cv2.mean(small, mask=small_mask)[0]
        if self._flicker_ref is None:
            self._flicker_ref = current
            return image
        gain = float(np.clip(self._flicker_ref / max(current, 1.0), self.cfg.flicker_gain_min, self.cfg.flicker_gain_max))
        a = self.cfg.flicker_ema
        self._flicker_ref = (1 - a) * self._flicker_ref + a * current
        if abs(gain - 1.0) < 0.02:
            return image
        return cv2.convertScaleAbs(image, alpha=gain)

    def _decide_clahe(self, image: np.ndarray, mask: np.ndarray | None) -> bool:
        mode = self.cfg.clahe_mode
        if mode != "auto":
            return mode == "on"
        # 시계가 아니라 프레임 밝기로 판단 (계절 변화에도 안전). 매 프레임이 아니라 주기적으로만 계산
        if (self._frame_idx - 1) % max(1, self.cfg.clahe_check_interval) != 0:
            return self._clahe_on
        gray = to_gray(image)
        h, w = gray.shape
        sw = min(64, w)
        sh = max(1, round(h * sw / w))
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        if mask is not None:
            small_mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            values = small[small_mask > 0]
        else:
            values = small.ravel()
        if values.size == 0:
            self._clahe_on = False
            return False
        dark = np.count_nonzero(values < self.cfg.clahe_dark_level) / values.size
        bright = np.count_nonzero(values > self.cfg.clahe_bright_level) / values.size
        ratio = self.cfg.clahe_backlight_ratio
        self._clahe_on = bool(values.mean() < self.cfg.clahe_dark_mean or (dark >= ratio and bright >= ratio))
        return self._clahe_on

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return self._clahe.apply(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
