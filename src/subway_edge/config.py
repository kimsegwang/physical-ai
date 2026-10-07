"""파이프라인 설정.

가이드(report/preprocess.md) 2장의 파라미터 기본값을 PipelineConfig 하나에 모은다.
configs/*.json 에는 바꾸고 싶은 값만 적으면 되고, 적지 않은 값은 여기 기본값을 쓴다.
ROI·마스크·통과선 좌표는 모두 원본 프레임 기준 0~1 정규화 좌표다.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Point = list[float]  # [x, y]
Polygon = list[Point]

CLAHE_MODES = ("off", "on", "auto")
ANCHORS = ("center", "bottom")
DIRECTIONS = ("in", "out")
WEATHER_CONDITIONS = ("clear", "cloudy", "rain", "snow")


@dataclass(frozen=True)
class PreprocessConfig:
    """1~5단계: 리사이즈·샘플링, 안정화, ROI·마스크, 깜빡임 보정, CLAHE."""

    # 1. 리사이즈·샘플링
    input_size: int = 640  # 긴 변 기준 px. 사람이 작게 보이면(1-1이 4m 이상) 960
    process_fps: float = 10.0  # 카메라당 처리 fps (8~10)

    # 2. 영상 안정화 (1-5 "자주 흔들림"일 때만 켬, 평행이동만 보정)
    stabilization: bool = False
    stab_width: int = 160  # 흔들림 추정용 축소 영상 너비
    stab_max_shift: float = 0.1  # 이보다 큰 이동(화면 비율)은 장면 변화로 보고 무시

    # 3. ROI 크롭 + 고정 마스크
    roi: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])  # x1, y1, x2, y2
    masks: list[Polygon] = field(default_factory=list)  # 광고판·유리·에스컬레이터 등
    wet_floor_masks: list[Polygon] = field(default_factory=list)  # 비·눈 올 때 추가할 바닥 반사 영역
    apply_wet_floor_masks: bool = False  # 날씨 프로필이 켬

    # 4. 밝기 깜빡임 보정 (2-1이 "형광등" 또는 "혼합"일 때 켬)
    flicker_correction: bool = False
    flicker_ema: float = 0.1  # 기준 밝기 이동평균 계수
    flicker_gain_min: float = 0.8
    flicker_gain_max: float = 1.25

    # 5. CLAHE: off / on / auto(프레임 밝기를 보고 자동 전환)
    clahe_mode: str = "off"
    clahe_clip_limit: float = 2.0  # 2-3 "항상 역광"이면 3.0
    clahe_tile: int = 8
    clahe_check_interval: int = 10  # auto 판단 주기(프레임)
    clahe_dark_mean: float = 70.0  # 평균 밝기가 이보다 낮으면 켬
    clahe_backlight_ratio: float = 0.2  # 어두운 픽셀·밝은 픽셀이 각각 이 비율 이상이면 역광으로 보고 켬
    clahe_dark_level: int = 40
    clahe_bright_level: int = 220

    def __post_init__(self) -> None:
        if self.clahe_mode not in CLAHE_MODES:
            raise ValueError(f"clahe_mode는 {CLAHE_MODES} 중 하나여야 합니다: {self.clahe_mode}")
        x1, y1, x2, y2 = self.roi
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"roi는 0~1 범위의 [x1, y1, x2, y2]여야 합니다: {self.roi}")


@dataclass(frozen=True)
class DetectionConfig:
    """6단계: 사람 검출 + 결과 필터."""

    model_path: str = "models/yolov8n.pt"
    device: str = ""  # ""이면 ultralytics가 자동 선택
    conf_threshold: float = 0.4  # 고신뢰도 기준 (추적 1차 매칭·새 트랙 생성)
    low_conf_threshold: float = 0.1  # 이 값 이상은 추적 2차 매칭에만 사용 (ByteTrack 방식)
    nms_iou: float = 0.5
    person_class: int = 0
    aux_classes: list[int] = field(default_factory=lambda: [28])  # 카운트 제외 확인용 (COCO 28=캐리어)

    # 박스 크기 필터: 성인 평균 박스 면적(정규화) 대비 비율. 0이면 필터를 쓰지 않음
    # 비스듬한 각도면 화면 위쪽·아래쪽 기준을 따로 주고 y 위치에 따라 선형 보간한다
    ref_person_area_top: float = 0.0
    ref_person_area_bottom: float = 0.0  # 0이면 top과 같은 값 사용
    min_area_ratio: float = 0.35
    max_area_ratio: float = 2.5


@dataclass(frozen=True)
class TrackingConfig:
    """7단계: ByteTrack 방식 추적."""

    track_buffer: int = 30  # 놓친 트랙을 유지하는 프레임 수
    match_iou_min: float = 0.2  # 1차(고신뢰도) 매칭 IoU 하한. 낮을수록 느슨함
    low_match_iou_min: float = 0.5  # 2차(저신뢰도) 매칭 IoU 하한
    velocity_smoothing: float = 0.5  # 속도 추정 이동평균 계수(이전 값 비중)


@dataclass(frozen=True)
class GateConfig:
    """개찰구 하나의 이중 통과선."""

    name: str
    line_a: list[Point]  # [[x1, y1], [x2, y2]]
    line_b: list[Point]

    def __post_init__(self) -> None:
        for line in (self.line_a, self.line_b):
            if len(line) != 2 or any(len(p) != 2 for p in line):
                raise ValueError(f"통과선은 [[x1, y1], [x2, y2]] 형식이어야 합니다: {self.name}")


def _default_gates() -> list[GateConfig]:
    return [GateConfig("gate1", [[0.05, 0.45], [0.95, 0.45]], [[0.05, 0.55], [0.95, 0.55]])]


@dataclass(frozen=True)
class CountingConfig:
    """8단계: 이중 통과선 카운팅."""

    gates: list[GateConfig] = field(default_factory=_default_gates)
    a_to_b: str = "out"  # 선 A → 선 B 순서로 지날 때의 방향 ("in" 입장 / "out" 퇴장)
    anchor: str = "center"  # 탑뷰는 center, 비스듬한 각도는 bottom(발 위치)
    min_track_len: int = 5  # 이보다 짧은 궤적은 세지 않음 (반사·광고판 오검출 제거)
    recount_block_sec: float = 5.0  # 같은 ID 재카운트 금지 시간

    def __post_init__(self) -> None:
        gates = [GateConfig(**g) if isinstance(g, dict) else g for g in self.gates]
        object.__setattr__(self, "gates", gates)
        if not gates:
            raise ValueError("gates가 비어 있습니다")
        if self.a_to_b not in DIRECTIONS:
            raise ValueError(f"a_to_b는 {DIRECTIONS} 중 하나여야 합니다: {self.a_to_b}")
        if self.anchor not in ANCHORS:
            raise ValueError(f"anchor는 {ANCHORS} 중 하나여야 합니다: {self.anchor}")


@dataclass(frozen=True)
class AggregationConfig:
    """9단계: 1분 집계 + 이상치 처리.

    물리 한계를 넘는 값만 잘라내고, 급증은 값을 그대로 두고 플래그만 붙인다.
    """

    gate_limit_per_min: int = 60  # 게이트 1개·방향 1개당 분당 최대 통과 인원
    spike_ratio: float = 3.0  # 직전 중앙값의 몇 배 이상이면 급증
    spike_min_diff: int = 10  # 중앙값과의 차이가 이 값 이상이어야 급증
    spike_history_min: int = 15  # 중앙값 계산에 쓰는 직전 분 수
    spike_min_history: int = 5  # 이력이 이보다 적으면 급증 판단을 하지 않음


def _default_mode_profiles() -> dict[str, dict]:
    return {
        "peak": {
            "detection": {"conf_threshold": 0.35, "nms_iou": 0.6},
            "tracking": {"track_buffer": 40, "match_iou_min": 0.15},
        },
        "normal": {},
        "quiet": {"detection": {"conf_threshold": 0.45}},
    }


def _default_weather_profiles() -> dict[str, dict]:
    wet = {"preprocess": {"apply_wet_floor_masks": True}, "counting": {"min_track_len": 8}}
    return {"clear": {}, "cloudy": {}, "rain": copy.deepcopy(wet), "snow": copy.deepcopy(wet)}


@dataclass(frozen=True)
class AdaptiveConfig:
    """시간대(혼잡도)·날씨별 설정 자동 전환 (가이드 5장).

    혼잡도는 직전 1분 집계의 게이트당 인원(입장+퇴장)으로 판단한다.
    """

    enabled: bool = True
    peak_enter_per_gate: float = 10.0  # 이 값 이상이면 peak
    peak_exit_per_gate: float = 7.0  # peak 상태에서 이 값 미만이면 해제
    quiet_enter_per_gate: float = 1.0  # 이 값 이하면 quiet
    quiet_exit_per_gate: float = 2.0  # quiet 상태에서 이 값 이상이면 해제
    mode_profiles: dict[str, dict] = field(default_factory=_default_mode_profiles)
    weather_profiles: dict[str, dict] = field(default_factory=_default_weather_profiles)
    heavy_coat_profile: dict = field(default_factory=lambda: {"detection": {"max_area_ratio": 3.0}})


@dataclass(frozen=True)
class PipelineConfig:
    camera_id: str = "cam0"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)


def _merge_dict(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(obj: Any, overrides: dict, _path: str = "") -> Any:
    """설정 객체에 일부 값을 덮어쓴 새 객체를 돌려준다. 모르는 키는 오타로 보고 에러를 낸다."""
    if not isinstance(overrides, dict):
        raise TypeError(f"{_path or '설정'}에는 객체(dict)가 와야 합니다")
    names = {f.name for f in dataclasses.fields(obj)}
    changes: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in names:
            raise KeyError(f"알 수 없는 설정 키: {_path}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            changes[key] = apply_overrides(current, value, f"{_path}{key}.")
        elif isinstance(current, dict) and isinstance(value, dict):
            changes[key] = _merge_dict(current, value)
        else:
            changes[key] = copy.deepcopy(value)
    return dataclasses.replace(obj, **changes)


def config_to_dict(cfg: PipelineConfig) -> dict:
    return dataclasses.asdict(cfg)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> PipelineConfig:
    """기본값 위에 JSON 파일과 overrides를 차례로 덮어쓴다."""
    cfg = PipelineConfig()
    if path is not None:
        with open(path, encoding="utf-8") as f:
            cfg = apply_overrides(cfg, json.load(f))
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg
