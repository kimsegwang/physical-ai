"""가짜 검출기로 파이프라인 전체를 검증한다 (YOLO 불필요).

검은 프레임에 흰 사각형(사람)을 그리고, BlobDetector가 흰 덩어리를 찾아 검출로 돌려준다.
그래서 리사이즈·ROI 크롭·마스크·좌표 변환까지 실제 경로를 그대로 지난다.
통과선 기본값: 선 A y=0.45, 선 B y=0.55 (x 0.05~0.95), A→B(아래로) = 퇴장.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from subway_edge.aggregation import MinuteAggregator
from subway_edge.config import (
    AggregationConfig,
    PipelineConfig,
    apply_overrides,
    config_to_dict,
    load_config,
)
from subway_edge.controller import AdaptiveController
from subway_edge.counting import CountEvent
from subway_edge.detection import Detection, filter_detections
from subway_edge.pipeline import EdgePipeline
from subway_edge.preprocess import FrameSampler, Preprocessor, Stabilizer

ROOT = Path(__file__).resolve().parents[1]
W, H = 640, 480
BOX_W, BOX_H = 0.08, 0.15
FPS = 10.0


# ---------------------------------------------------------------- 도우미


class BlobDetector:
    """흰 사각형을 사람으로 검출하는 가짜 검출기."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            dets.append(Detection(x, y, x + w, y + h, 0.9, 0))
        return dets


def person(cx: float, cy: float) -> tuple[float, float, float, float]:
    return cx - BOX_W / 2, cy - BOX_H / 2, cx + BOX_W / 2, cy + BOX_H / 2


def ys(start: float, end: float, step: float = 0.03) -> list[float]:
    n = max(1, int(round(abs(end - start) / step)))
    return [start + (end - start) * i / n for i in range(n + 1)]


def walk(x: float, *legs: tuple[float, float]) -> list[list[tuple]]:
    """x 위치에서 (시작 y, 끝 y) 구간들을 차례로 걷는 한 사람의 프레임별 박스."""
    frames: list[list[tuple]] = []
    for i, (a, b) in enumerate(legs):
        points = ys(a, b)
        frames += [[person(x, y)] for y in (points if i == 0 else points[1:])]
    return frames


def together(*people: list[list[tuple]]) -> list[list[tuple]]:
    n = max(len(p) for p in people)
    return [sum((p[i] for p in people if i < len(p)), []) for i in range(n)]


def draw(boxes: list[tuple]) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (round(x1 * W), round(y1 * H)), (round(x2 * W) - 1, round(y2 * H) - 1), (255, 255, 255), -1)
    return img


def make_config(extra: dict | None = None) -> PipelineConfig:
    cfg = load_config(overrides={"detection": {"ref_person_area_top": BOX_W * BOX_H}})
    return apply_overrides(cfg, extra) if extra else cfg


def run(frames: list[list[tuple]], extra: dict | None = None, weather: str | None = None):
    pipe = EdgePipeline(make_config(extra), BlobDetector())
    if weather:
        pipe.set_weather(weather)
    results = [pipe.process(draw(boxes), i / FPS) for i, boxes in enumerate(frames)]
    records = pipe.flush()
    return pipe, results, records


def totals(records: list[dict]) -> tuple[int, int]:
    return sum(r["in"] for r in records), sum(r["out"] for r in records)


# ---------------------------------------------------------------- 카운팅 시나리오


def test_one_person_exit():
    pipe, _, records = run(walk(0.5, (0.2, 0.8)))
    assert pipe.counter.totals == {"in": 0, "out": 1}
    assert totals(records) == (0, 1)


def test_one_person_enter():
    pipe, _, records = run(walk(0.5, (0.8, 0.2)))
    assert pipe.counter.totals == {"in": 1, "out": 0}
    assert totals(records) == (1, 0)


def test_cross_first_line_then_turn_back():
    pipe, _, _ = run(walk(0.5, (0.2, 0.5), (0.5, 0.2)))
    assert pipe.counter.totals == {"in": 0, "out": 0}


def test_jitter_near_second_line_counts_once():
    pipe, _, _ = run(walk(0.5, (0.2, 0.62), (0.62, 0.5), (0.5, 0.75)))
    assert pipe.counter.totals == {"in": 0, "out": 1}


@pytest.mark.parametrize(
    "frames",
    [
        # 앞뒤로 붙어서 (박스 간격 약 10px)
        together(walk(0.5, (0.37, 0.91)), walk(0.5, (0.2, 0.74))),
        # 나란히 붙어서
        together(walk(0.44, (0.2, 0.8)), walk(0.56, (0.2, 0.8))),
    ],
    ids=["tandem", "side_by_side"],
)
def test_tailgating_counts_two(frames):
    pipe, _, _ = run(frames)
    assert pipe.counter.totals == {"in": 0, "out": 2}


def test_three_frame_false_detection_is_ignored():
    frames = [[person(0.5, y)] for y in (0.44, 0.50, 0.56)] + [[]] * 5
    pipe, _, _ = run(frames)
    assert pipe.counter.totals == {"in": 0, "out": 0}
    # 대조군: 최소 궤적 길이를 1로 낮추면 같은 궤적이 실제로 두 선을 넘는 것이 세어진다
    pipe, _, _ = run(frames, {"counting": {"min_track_len": 1}})
    assert pipe.counter.totals == {"in": 0, "out": 1}


# ---------------------------------------------------------------- ROI·마스크


def test_roi_crop_keeps_coordinates():
    frames = together(walk(0.5, (0.2, 0.8)), walk(0.12, (0.2, 0.8)))
    pipe, results, _ = run(frames, {"preprocess": {"roi": [0.3, 0.1, 0.7, 0.9]}})
    # ROI 밖(x=0.12)의 사람은 잘려서 안 세고, ROI 안의 사람만 센다
    assert pipe.counter.totals == {"in": 0, "out": 1}
    assert results[0].pre.image.shape[:2] == (round(0.8 * H), round(0.4 * W))
    # 잘라낸 영상에서 검출했어도 트랙 좌표는 전체 프레임 정규화 좌표와 일치
    for i in (3, 10, 18):
        (track,) = results[i].tracks
        expected = person(0.5, ys(0.2, 0.8)[i])
        assert np.allclose(track.box, expected, atol=1.5 / H)


def test_masked_region_is_ignored():
    mask = [[0.0, 0.0], [0.3, 0.0], [0.3, 1.0], [0.0, 1.0]]
    frames = together(walk(0.15, (0.2, 0.8)), walk(0.6, (0.2, 0.8)))
    pipe, _, _ = run(frames, {"preprocess": {"masks": [mask]}})
    assert pipe.counter.totals == {"in": 0, "out": 1}


def test_filter_detections_mask_and_coordinates():
    """마스크 영역 검출 제거와 좌표 변환을 검출 필터 단독으로 확인 (영상을 칠하지 않는 경로)."""
    mask = [[0.25, 0.25], [0.5, 0.25], [0.5, 0.75], [0.25, 0.75]]
    cfg = make_config({"preprocess": {"roi": [0.25, 0.25, 0.75, 0.75], "masks": [mask]}})
    pre = Preprocessor(cfg.preprocess).run(np.zeros((H, W, 3), np.uint8))
    assert pre.image.shape[:2] == (240, 320)

    bw, bh = BOX_W * W, BOX_H * H
    in_mask = Detection(80 - bw / 2, 120 - bh / 2, 80 + bw / 2, 120 + bh / 2, 0.9)
    kept = Detection(240 - bw / 2, 120 - bh / 2, 240 + bw / 2, 120 + bh / 2, 0.9)
    duplicate = Detection(kept.x1 + 2, kept.y1 + 2, kept.x2 + 2, kept.y2 + 2, 0.8)
    too_low = Detection(240 - bw / 2, 20, 240 + bw / 2, 20 + bh, 0.05)
    too_small = Detection(200, 200, 205, 205, 0.9)
    suitcase = Detection(150, 150, 190, 200, 0.9, cls=28)

    persons, aux = filter_detections([in_mask, kept, duplicate, too_low, too_small, suitcase], pre, cfg.detection)
    assert len(persons) == 1
    assert np.allclose(persons[0].center, (0.625, 0.5))
    assert np.allclose(persons[0].box, person(0.625, 0.5))
    assert [d.cls for d in aux] == [28]


def test_wet_floor_mask_follows_weather():
    wet = [[0.0, 0.0], [0.3, 0.0], [0.3, 1.0], [0.0, 1.0]]
    frames = walk(0.15, (0.2, 0.8))
    extra = {"preprocess": {"wet_floor_masks": [wet]}}
    pipe, _, _ = run(frames, extra)
    assert pipe.counter.totals["out"] == 1
    pipe, _, _ = run(frames, extra, weather="rain")
    assert pipe.counter.totals["out"] == 0
    assert pipe.config.counting.min_track_len == 8


# ---------------------------------------------------------------- 집계 이상치


def close_minute(agg: MinuteAggregator, minute: int, n_in: int = 0, n_out: int = 0) -> dict:
    if minute == 0:
        agg.tick(0.0)
    for i in range(n_in):
        agg.add(CountEvent(minute * 60 + 1.0, i, "g", "in"))
    for i in range(n_out):
        agg.add(CountEvent(minute * 60 + 1.0, i, "g", "out"))
    (record,) = agg.tick((minute + 1) * 60.0)
    return record


def test_aggregation_clips_over_physical_limit():
    agg = MinuteAggregator(AggregationConfig(gate_limit_per_min=60), num_gates=2)
    record = close_minute(agg, 0, n_in=30, n_out=150)
    assert record["out"] == 120 and record["raw_out"] == 150
    assert record["in"] == 30 and record["raw_in"] == 30
    assert record["flags"] == ["clipped_out"]


def test_aggregation_flags_spike_without_changing_value():
    agg = MinuteAggregator(AggregationConfig(), num_gates=2)
    for m in range(15):
        assert close_minute(agg, m, n_in=5, n_out=5)["flags"] == []
    record = close_minute(agg, 15, n_in=5, n_out=40)  # 경기 종료 같은 급증
    assert record["out"] == 40 and record["raw_out"] == 40
    assert record["flags"] == ["spike_out"]


def test_aggregation_small_jump_or_short_history_is_not_spike():
    agg = MinuteAggregator(AggregationConfig(), num_gates=1)
    assert close_minute(agg, 0, n_out=0)["flags"] == []
    assert close_minute(agg, 1, n_out=30)["flags"] == []  # 이력 부족
    agg = MinuteAggregator(AggregationConfig(), num_gates=1)
    for m in range(5):
        close_minute(agg, m)
    assert close_minute(agg, 5, n_out=3)["flags"] == []  # 0 → 3명은 차이가 작음


def test_aggregation_emits_empty_minutes():
    agg = MinuteAggregator(AggregationConfig(), num_gates=1)
    agg.tick(0.0)
    records = agg.tick(185.0)
    assert [r["minute_start"] for r in records] == [0, 60, 120]


# ---------------------------------------------------------------- 시간대·날씨 자동 전환


def minute(n_in: int, n_out: int = 0) -> dict:
    return {"in": n_in, "out": n_out}


def test_controller_congestion_modes_with_hysteresis():
    ctrl = AdaptiveController(make_config())
    assert ctrl.mode == "normal"
    assert ctrl.on_minute(minute(12, 8), num_gates=2)  # 게이트당 10명 → peak
    cfg = ctrl.config
    assert ctrl.mode == "peak"
    assert cfg.detection.conf_threshold == 0.35 and cfg.detection.nms_iou == 0.6
    assert cfg.tracking.track_buffer == 40 and cfg.tracking.match_iou_min == 0.15
    assert not ctrl.on_minute(minute(16), num_gates=2)  # 8명/게이트: 복귀 기준(7) 이상 → 유지
    assert ctrl.on_minute(minute(13), num_gates=2) and ctrl.mode == "normal"  # 6.5명 → normal
    assert ctrl.config.tracking.match_iou_min == 0.2

    assert ctrl.on_minute(minute(2), num_gates=2) and ctrl.mode == "quiet"  # 1명 → quiet
    assert ctrl.config.detection.conf_threshold == 0.45
    assert not ctrl.on_minute(minute(3), num_gates=2)  # 1.5명: 복귀 기준(2) 미만 → 유지
    assert ctrl.on_minute(minute(4), num_gates=2) and ctrl.mode == "normal"


def test_controller_weather_and_heavy_coat():
    ctrl = AdaptiveController(make_config())
    ctrl.on_minute(minute(10), num_gates=1)  # peak
    assert ctrl.set_weather("snow", heavy_coat=True)
    cfg = ctrl.config
    assert cfg.preprocess.apply_wet_floor_masks and cfg.counting.min_track_len == 8
    assert cfg.detection.max_area_ratio == 3.0
    assert cfg.detection.conf_threshold == 0.35  # 혼잡도 프로필과 함께 적용
    assert ctrl.set_weather("clear")
    assert not ctrl.config.preprocess.apply_wet_floor_masks
    assert ctrl.config.detection.max_area_ratio == 2.5
    with pytest.raises(ValueError):
        ctrl.set_weather("hail")


def test_pipeline_switches_tracker_settings_after_busy_minute():
    pipe = EdgePipeline(make_config(), BlobDetector())
    blank = draw([])
    pipe.process(blank, 0.0)
    for i in range(10):
        pipe.aggregator.add(CountEvent(1.0, i, "gate1", "out"))
    result = pipe.process(blank, 60.0)
    assert result.records[0]["out"] == 10 and result.records[0]["mode"] == "normal"
    assert result.mode == "peak"
    assert pipe.tracker.cfg.match_iou_min == 0.15 and pipe.tracker.high_thresh == 0.35


# ---------------------------------------------------------------- 전처리 단위


def test_sampler_reduces_30fps_to_10fps():
    sampler = FrameSampler(10.0)
    accepted = sum(sampler.accept(i / 30) for i in range(300))
    assert accepted == 100


def test_resize_long_side():
    cfg = make_config()
    pre = Preprocessor(cfg.preprocess).run(np.zeros((1080, 1920, 3), np.uint8))
    assert pre.frame.shape[:2] == (360, 640)


def textured(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((H, W)) * 255).astype(np.uint8), (0, 0), 3)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_stabilizer_compensates_shift():
    base = textured()
    shifted = cv2.warpAffine(base, np.float32([[1, 0, 8], [0, 1, -6]]), (W, H), borderMode=cv2.BORDER_REFLECT)
    stab = Stabilizer(160, 0.1)
    stab.run(base)
    out, (dx, dy) = stab.run(shifted)
    assert abs(dx - 8) < 1.5 and abs(dy + 6) < 1.5
    inner = (slice(60, H - 60), slice(60, W - 60))
    before = np.abs(shifted[inner].astype(int) - base[inner]).mean()
    after = np.abs(out[inner].astype(int) - base[inner]).mean()
    assert after < before * 0.3


def test_flicker_correction_matches_recent_brightness():
    cfg = make_config({"preprocess": {"flicker_correction": True}})
    pre = Preprocessor(cfg.preprocess)
    base = (textured() * 0.5 + 60).astype(np.uint8)
    for _ in range(10):
        pre.run(base)
    bright = cv2.convertScaleAbs(base, alpha=1.15)
    corrected = pre.run(bright).image
    assert abs(corrected.mean() - base.mean()) < 0.03 * base.mean()


def test_clahe_auto_by_brightness():
    cfg = make_config({"preprocess": {"clahe_mode": "auto"}})
    rng = np.random.default_rng(1)
    dark = rng.integers(10, 50, (H, W, 3), dtype=np.uint8)
    normal = rng.integers(100, 160, (H, W, 3), dtype=np.uint8)
    backlit = normal.copy()
    backlit[:, : W // 3] = 245
    backlit[:, W // 3 : 2 * W // 3] = 15
    assert Preprocessor(cfg.preprocess).run(dark).clahe_on
    assert not Preprocessor(cfg.preprocess).run(normal).clahe_on
    assert Preprocessor(cfg.preprocess).run(backlit).clahe_on
    off = make_config()
    assert not Preprocessor(off.preprocess).run(dark).clahe_on


# ---------------------------------------------------------------- 설정 파일


def test_default_json_matches_code_defaults():
    with open(ROOT / "configs" / "default.json", encoding="utf-8") as f:
        assert json.load(f) == json.loads(json.dumps(config_to_dict(PipelineConfig())))


def test_example_site_config_overrides_partially():
    cfg = load_config(ROOT / "configs" / "example_site.json")
    assert [g.name for g in cfg.counting.gates] == ["gate1", "gate2"]
    assert cfg.counting.anchor == "bottom"
    assert cfg.detection.conf_threshold == 0.4  # 적지 않은 값은 기본값 유지
    peak = cfg.adaptive.mode_profiles["peak"]
    assert peak["tracking"] == {"track_buffer": 50, "match_iou_min": 0.15}
    assert peak["detection"]["conf_threshold"] == 0.35


def test_unknown_config_key_is_rejected():
    with pytest.raises(KeyError):
        load_config(overrides={"detection": {"conf_treshold": 0.3}})
    with pytest.raises(ValueError):
        load_config(overrides={"preprocess": {"clahe_mode": "sometimes"}})
