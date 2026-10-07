"""9단계: 1분 집계 + 이상치 처리.

- 개찰구 물리 한계(게이트 수 × gate_limit_per_min)를 넘는 값은 잘라내고 clipped_* 플래그를 붙인다
- 직전 분들의 중앙값보다 크게 튄 값은 바꾸지 않고 spike_* 플래그만 붙인다
  (경기 종료처럼 진짜 급증일 수 있으므로 판단은 클라우드에 맡긴다)
"""

from __future__ import annotations

from collections import deque
from statistics import median

from .config import AggregationConfig, DIRECTIONS
from .counting import CountEvent


class MinuteAggregator:
    def __init__(self, cfg: AggregationConfig, num_gates: int) -> None:
        self.num_gates = num_gates
        self._bucket: int | None = None
        self._counts = {d: 0 for d in DIRECTIONS}
        self.apply_config(cfg)

    def apply_config(self, cfg: AggregationConfig) -> None:
        self.cfg = cfg
        old = getattr(self, "_history", None)
        self._history = {d: deque(old[d] if old else [], maxlen=cfg.spike_history_min) for d in DIRECTIONS}

    def tick(self, ts: float) -> list[dict]:
        """ts가 새 분에 들어서면 지난 분(빈 분 포함)의 집계 레코드를 돌려준다."""
        bucket = int(ts // 60)
        if self._bucket is None:
            self._bucket = bucket
            return []
        records = []
        while self._bucket < bucket:
            records.append(self._close())
            self._bucket += 1
        return records

    def add(self, event: CountEvent) -> None:
        if self._bucket is None:
            self._bucket = int(event.ts // 60)
        self._counts[event.direction] += 1

    def flush(self) -> list[dict]:
        """진행 중인 분을 강제로 마감한다 (종료 시)."""
        if self._bucket is None:
            return []
        record = self._close()
        self._bucket = None
        return [record]

    def _close(self) -> dict:
        record: dict = {"minute_start": self._bucket * 60, "flags": []}
        limit = self.cfg.gate_limit_per_min * self.num_gates
        for d in DIRECTIONS:
            raw = self._counts[d]
            value = min(raw, limit)
            if raw > limit:
                record["flags"].append(f"clipped_{d}")
            history = self._history[d]
            if len(history) >= self.cfg.spike_min_history:
                base = median(history)
                if value >= base * self.cfg.spike_ratio and value - base >= self.cfg.spike_min_diff:
                    record["flags"].append(f"spike_{d}")
            history.append(value)
            record[d] = value
            record[f"raw_{d}"] = raw
        self._counts = {d: 0 for d in DIRECTIONS}
        return record
