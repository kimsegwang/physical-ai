"""엣지 파이프라인 실행.

예) python scripts/run.py --source data/sample.mp4 --config configs/example_site.json --weather rain --show
    python scripts/run.py --source 0            (웹캠)
    python scripts/run.py --source rtsp://...   (IP 카메라)

1분 집계 레코드를 JSON 한 줄씩 표준 출력으로 낸다 (클라우드 전송 대상, 숫자만).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subway_edge.config import WEATHER_CONDITIONS, load_config  # noqa: E402
from subway_edge.detection import YoloDetector  # noqa: E402
from subway_edge.pipeline import EdgePipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="지하철 개찰구 엣지 전처리·카운팅")
    p.add_argument("--source", required=True, help="영상 파일 경로, 카메라 번호, RTSP 주소")
    p.add_argument("--config", help="설정 JSON (기본값 위에 덮어씀)")
    p.add_argument("--weather", choices=WEATHER_CONDITIONS, default="clear")
    p.add_argument("--heavy-coat", action="store_true", help="두꺼운 외투 계절 (최대 박스 크기 상향)")
    p.add_argument("--show", action="store_true", help="디버그 화면 표시 (저장하지 않음)")
    return p.parse_args()


def emit(records: list[dict]) -> None:
    for record in records:
        print(json.dumps(record, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    detector = YoloDetector.from_config(cfg.detection, cfg.preprocess.input_size)
    pipeline = EdgePipeline(cfg, detector)
    pipeline.set_weather(args.weather, args.heavy_coat)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"영상을 열 수 없습니다: {args.source}", file=sys.stderr)
        return 1

    # 파일은 프레임 번호로 시각을 계산하고, 실시간 소스는 현재 시각을 쓴다
    is_file = isinstance(source, str) and Path(source).exists()
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    show = None
    if args.show:
        from subway_edge.debug import show

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = frame_idx / src_fps if is_file else time.time()
            frame_idx += 1
            result = pipeline.process(frame, ts)
            if result is None:
                continue
            emit(result.records)
            if show is not None and not show(result, pipeline.config):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show is not None:
            cv2.destroyAllWindows()
    emit(pipeline.flush())
    return 0


if __name__ == "__main__":
    sys.exit(main())
