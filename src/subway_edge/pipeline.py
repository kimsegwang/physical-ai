"""1~9단계를 순서대로 묶은 엣지 파이프라인.

출력은 숫자(집계 레코드)뿐이다. 영상·프레임은 저장하거나 외부로 보내지 않는다.
1분 집계가 마감되면 Publisher로 바로 전송한다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .aggregation import MinuteAggregator
from .config import PipelineConfig
from .controller import AdaptiveController
from .counting import CountEvent, LineCounter
from .detection import Detection, Detector, filter_detections
from .preprocess import FrameSampler, Preprocessor, PreprocessResult
from .publisher import Publisher
from .tracking import ByteTracker, Track


@dataclass
class FrameResult:
    ts: float
    tracks: list[Track]  # 이번 프레임에 검출과 매칭된 트랙
    events: list[CountEvent]  # 이번 프레임에 센 통과
    records: list[dict]  # 이번 프레임에 마감된 1분 집계 (클라우드 전송 대상)
    aux: list[Detection]  # 카운트 제외 확인용 보조 객체
    pre: PreprocessResult
    mode: str
    totals: dict[str, int]


class EdgePipeline:
    def __init__(self, cfg: PipelineConfig, detector: Detector, publisher: Publisher | None = None) -> None:
        self.detector = detector
        self.publisher = publisher  # None이면 전송하지 않고 FrameResult.records로만 돌려준다
        self.controller = AdaptiveController(cfg)
        cfg = self.controller.config
        self.num_gates = len(cfg.counting.gates)
        self.sampler = FrameSampler(cfg.preprocess.process_fps)
        self.preprocessor = Preprocessor(cfg.preprocess)
        self.tracker = ByteTracker(cfg.tracking, cfg.detection.conf_threshold)
        self.counter = LineCounter(cfg.counting)
        self.aggregator = MinuteAggregator(cfg.aggregation, self.num_gates)

    @property
    def config(self) -> PipelineConfig:
        return self.controller.config

    def set_weather(self, condition: str, heavy_coat: bool = False) -> None:
        if self.controller.set_weather(condition, heavy_coat):
            self._apply_config()

    def _apply_config(self) -> None:
        cfg = self.controller.config
        self.preprocessor.apply_config(cfg.preprocess)
        self.tracker.apply_config(cfg.tracking, cfg.detection.conf_threshold)
        self.counter.apply_config(cfg.counting)
        self.aggregator.apply_config(cfg.aggregation)

    def process(self, frame: np.ndarray, ts: float) -> FrameResult | None:
        """프레임 하나를 처리한다. 샘플링에서 건너뛴 프레임이면 None."""
        if not self.sampler.accept(ts):
            return None

        # 분이 바뀌었으면 먼저 집계를 마감하고, 혼잡도 모드가 바뀌면 이번 프레임부터 새 설정 적용
        records = self.aggregator.tick(ts)
        for record in records:
            self._finalize(record)
            if self.controller.on_minute(record, self.num_gates):
                self._apply_config()

        cfg = self.config
        pre = self.preprocessor.run(frame)  # 1~5단계
        persons, aux = filter_detections(self.detector.detect(pre.image), pre, cfg.detection)  # 6단계
        tracks = self.tracker.update(persons)  # 7단계
        events = self.counter.update(tracks, ts, self.tracker.alive_ids)  # 8단계
        for event in events:  # 9단계 (1분 단위로 쌓음)
            self.aggregator.add(event)
        # 트랙은 다음 프레임에서 계속 갱신되므로 이 시점의 상태를 복사해 둔다
        # (box·velocity는 제자리 수정 없이 새 배열로 교체되므로 얕은 복사로 충분)
        snapshot = [copy.copy(t) for t in tracks]
        return FrameResult(ts, snapshot, events, records, aux, pre, self.controller.mode, dict(self.counter.totals))

    def flush(self) -> list[dict]:
        """진행 중인 1분 집계를 마감한다 (스트림 종료 시)."""
        records = self.aggregator.flush()
        for record in records:
            self._finalize(record)
        return records

    def _finalize(self, record: dict) -> None:
        # 레코드에 담기는 모드는 그 1분 동안 적용된 설정
        record["camera_id"] = self.config.camera_id
        record["mode"] = self.controller.mode
        record["weather"] = self.controller.weather
        if self.publisher is not None:
            self.publisher.publish(record)
