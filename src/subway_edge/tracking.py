"""7단계: ByteTrack 방식 추적 (외부 추적 라이브러리 없이 직접 구현).

- 1차: 고신뢰도 검출 ↔ 확정 트랙(놓친 트랙 포함) IoU 매칭
- 2차: 저신뢰도 검출 ↔ 1차에서 남은, 직전 프레임까지 추적 중이던 트랙 매칭
  (가려져서 점수가 낮아진 사람을 놓치지 않게 함)
- 3차: 남은 고신뢰도 검출 ↔ 미확정 트랙. 미확정 트랙은 한 번만 놓쳐도 지운다
- 남은 고신뢰도 검출은 새 트랙. 놓친 트랙은 track_buffer 프레임 동안 유지
위치 예측은 칼만 필터 대신 등속 모델(이동평균 속도)로 단순화했다.
매칭은 IoU가 큰 쌍부터 고르는 greedy 방식이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import TrackingConfig
from .detection import Detection, iou_matrix


@dataclass
class Track:
    track_id: int
    box: np.ndarray  # 정규화 좌표 x1, y1, x2, y2 (놓친 동안은 예측 위치)
    score: float
    hits: int = 1  # 검출과 매칭된 프레임 수 (궤적 길이)
    misses: int = 0  # 연속으로 놓친 프레임 수
    confirmed: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    _last_obs: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._last_obs = self.box.copy()

    def anchor(self, mode: str) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, y2) if mode == "bottom" else ((x1 + x2) / 2, (y1 + y2) / 2)

    def predict(self) -> None:
        self.box = self.box + self.velocity

    def update(self, det: Detection, smoothing: float) -> None:
        obs = np.array(det.box, np.float64)
        step = (obs - self._last_obs) / (self.misses + 1)
        self.velocity = smoothing * self.velocity + (1 - smoothing) * step
        self.box = obs
        self._last_obs = obs
        self.score = det.score
        self.hits += 1
        self.misses = 0
        if self.hits >= 2:
            self.confirmed = True


def _match(
    tracks: list[Track], dets: list[Detection], iou_min: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    ious = iou_matrix(np.array([t.box for t in tracks]), np.array([d.box for d in dets], np.float64))
    pairs = sorted(zip(*np.nonzero(ious >= iou_min)), key=lambda p: ious[p], reverse=True)
    used_t: set[int] = set()
    used_d: set[int] = set()
    matches = []
    for ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((int(ti), int(di)))
    return (
        matches,
        [i for i in range(len(tracks)) if i not in used_t],
        [i for i in range(len(dets)) if i not in used_d],
    )


class ByteTracker:
    def __init__(self, cfg: TrackingConfig, high_thresh: float) -> None:
        self.tracks: list[Track] = []
        self._next_id = 1
        self.apply_config(cfg, high_thresh)

    def apply_config(self, cfg: TrackingConfig, high_thresh: float) -> None:
        self.cfg = cfg
        self.high_thresh = high_thresh

    @property
    def alive_ids(self) -> set[int]:
        return {t.track_id for t in self.tracks}

    def update(self, dets: list[Detection]) -> list[Track]:
        """한 프레임의 검출로 트랙을 갱신하고, 이번 프레임에 검출과 매칭된 트랙을 돌려준다."""
        cfg = self.cfg
        for t in self.tracks:
            t.predict()

        high = [d for d in dets if d.score >= self.high_thresh]
        low = [d for d in dets if d.score < self.high_thresh]
        confirmed = [t for t in self.tracks if t.confirmed]
        tentative = [t for t in self.tracks if not t.confirmed]
        was_tracking = {t.track_id for t in confirmed if t.misses == 0}

        # 1차: 고신뢰도 ↔ 확정 트랙
        m1, ut1, ud1 = _match(confirmed, high, cfg.match_iou_min)
        for ti, di in m1:
            confirmed[ti].update(high[di], cfg.velocity_smoothing)

        # 2차: 저신뢰도 ↔ 직전까지 추적 중이던 남은 트랙
        rest = [confirmed[i] for i in ut1 if confirmed[i].track_id in was_tracking]
        m2, _, _ = _match(rest, low, cfg.low_match_iou_min)
        for ti, di in m2:
            rest[ti].update(low[di], cfg.velocity_smoothing)

        # 3차: 남은 고신뢰도 ↔ 미확정 트랙
        high_rest = [high[i] for i in ud1]
        m3, ut3, ud3 = _match(tentative, high_rest, cfg.match_iou_min)
        for ti, di in m3:
            tentative[ti].update(high_rest[di], cfg.velocity_smoothing)
        dropped = {tentative[i].track_id for i in ut3}

        updated_ids = {confirmed[ti].track_id for ti, _ in m1}
        updated_ids |= {rest[ti].track_id for ti, _ in m2}
        updated_ids |= {tentative[ti].track_id for ti, _ in m3}

        alive: list[Track] = []
        for t in self.tracks:
            if t.track_id in dropped:
                continue
            if t.track_id not in updated_ids:
                t.misses += 1
                if t.misses > cfg.track_buffer:
                    continue
            alive.append(t)

        for di in ud3:
            det = high_rest[di]
            alive.append(Track(self._next_id, np.array(det.box, np.float64), det.score))
            self._next_id += 1

        self.tracks = alive
        return [t for t in self.tracks if t.misses == 0]
