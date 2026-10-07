# FlowSight – 지하철 개찰구 혼잡도 엣지 파이프라인

개찰구 카메라 영상을 역사 안의 엣지 AI 박스(Jetson급)에서 처리해 **입장·퇴장 인원을 1분 단위로 세는** 파이프라인입니다.
클라우드에는 숫자(1분 집계)만 보내고, 영상은 저장하거나 외부로 보내지 않습니다.

- 사람 검출: YOLO 경량 모델 (ultralytics)
- 추적: ByteTrack 방식 직접 구현 (외부 추적 라이브러리 버전에 의존하지 않음)
- 카운팅: 이중 통과선. 두 선을 순서대로 지나야 1명으로 셈
- 혼잡도·날씨에 따라 설정 자동 전환

기획은 [PRD/PRD.md](PRD/PRD.md), 전처리 설계와 파라미터 근거는 [report/preprocess.md](report/preprocess.md)에 있습니다.

## 처리 흐름

```
프레임 → 리사이즈·샘플링 → (흔들림) 안정화 → ROI 크롭 + 마스크 → (형광등) 깜빡임 보정 → (어두움·역광) CLAHE
      → YOLO 검출 + 필터 → 추적 → 이중 통과선 카운팅 → 1분 집계 + 이상치 처리 → 전송 (숫자만)
```

## 폴더 구조

```
├── src/subway_edge/       엣지 파이프라인 패키지
│   ├── config.py          모든 파라미터 기본값 (PipelineConfig), JSON 덮어쓰기
│   ├── preprocess.py      1~5단계: 리사이즈·샘플링, 안정화, ROI·마스크, 깜빡임 보정, CLAHE
│   ├── detection.py       6단계: 검출기 인터페이스, YOLO 검출기, 결과 필터
│   ├── tracking.py        7단계: ByteTrack 방식 추적
│   ├── counting.py        8단계: 이중 통과선 카운팅
│   ├── aggregation.py     9단계: 1분 집계, 물리 한계 잘라내기, 급증 플래그
│   ├── controller.py      혼잡도·날씨별 설정 자동 전환
│   ├── pipeline.py        1~9단계 연결
│   ├── publisher.py       결과 전송 인터페이스 (지금은 표준 출력)
│   └── debug.py           --show 디버그 화면 (저장 안 함)
├── scripts/run.py         실행 진입점
├── configs/
│   ├── default.json       전체 기본값 (코드 기본값과 같은지 테스트로 확인)
│   └── example_site.json  현장 설정 예시 (바꿀 값만 적음)
├── tests/                 테스트 (YOLO 없이 가짜 검출기로 실행)
├── report/                현장 조사 체크리스트, 전처리 설계 가이드
├── PRD/                   제품 요구사항 문서
├── index.html, images/, stitch/   제품 소개 홈페이지 (GitHub Pages)
├── data/                  테스트 영상·이미지 (Git에 올리지 않음)
└── models/                모델 파일 (Git에 올리지 않음)
```

## 설치 (conda 환경 `physical_ai`)

```powershell
conda create -n physical_ai python=3.11
conda activate physical_ai
pip install numpy opencv-python pytest
pip install ultralytics          # 실제 검출에 필요 (torch가 함께 설치됨)
```

모델 파일은 `models/`에 받습니다.

```powershell
python -c "from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset('models/yolov8n.pt')"
```

> Jetson에서는 PyPI의 torch 대신 NVIDIA가 배포하는 Jetson용 torch를 설치해야 GPU를 씁니다.

## 테스트

```powershell
conda activate physical_ai
python -m pytest
```

YOLO 없이 흰 사각형을 사람으로 검출하는 가짜 검출기로 전체 경로를 검증합니다.
한 명 입·퇴장, 되돌아감, 꼬리물기, 짧은 오검출, ROI·마스크 좌표, 집계 이상치, 설정 전환 등을 확인합니다.

## 실행

```powershell
# 영상 파일
python scripts/run.py --source data/sample.mp4 --config configs/example_site.json

# 비 오는 날 + 외투 계절, 디버그 화면 표시 (q 또는 ESC로 종료)
python scripts/run.py --source data/sample.mp4 --config configs/example_site.json --weather rain --heavy-coat --show

# 웹캠 / IP 카메라
python scripts/run.py --source 0
python scripts/run.py --source rtsp://<카메라 주소>
```

| 옵션 | 설명 |
|---|---|
| `--source` | 영상 파일 경로, 카메라 번호, RTSP 주소 |
| `--config` | 설정 JSON. 기본값 위에 적힌 값만 덮어씀 |
| `--weather` | `clear` / `cloudy` / `rain` / `snow` (기본 `clear`) |
| `--heavy-coat` | 두꺼운 외투 계절 (최대 박스 크기 상향) |
| `--show` | 디버그 화면 표시 (ROI·마스크·통과선·트랙). 화면에만 띄우고 저장하지 않음 |

1분마다 집계 레코드가 JSON 한 줄씩 출력됩니다.

```json
{"minute_start": 0, "flags": [], "in": 0, "raw_in": 0, "out": 3, "raw_out": 3, "camera_id": "stationA_cam03", "mode": "normal", "weather": "rain"}
```

- `in` / `out`: 전송 값 (물리 한계를 넘으면 잘라낸 값), `raw_*`: 잘라내기 전 값
- `flags`: `clipped_in/out` (한계 초과로 잘라냄), `spike_in/out` (급증. 값은 그대로)
- `mode`: 그 1분 동안 적용된 혼잡도 설정 (`peak` / `normal` / `quiet`)

전송 방식은 `Publisher` 인터페이스(`publish`, `close`)로 분리되어 있어, MQTT 등으로 바꿀 때는 같은 메서드를 가진 클래스를 `EdgePipeline`에 넘기면 됩니다.

## 설정

- 현장 설정 JSON에는 **바꿀 값만** 적습니다. 적지 않은 값은 [configs/default.json](configs/default.json)의 기본값을 씁니다. 모르는 키는 오타로 보고 오류를 냅니다.
- ROI·마스크·통과선 좌표는 모두 **0~1 정규화 좌표**입니다.
- 혼잡도(peak/quiet)·날씨 프로필은 `at_least` / `at_most`로 조정 방향만 정하므로, 현장에서 더 강하게 맞춘 값을 약하게 되돌리지 않습니다.

## 개인정보 원칙

- **영상·프레임을 파일로 저장하지 않습니다.** 처리한 프레임은 메모리에서만 쓰고 버립니다.
- **영상을 외부로 보내지 않습니다.** 전송하는 것은 1분 단위 인원 수와 상태 값뿐입니다.
- 얼굴 인식을 하지 않습니다. 트랙 ID는 카운트 중복을 막기 위한 임시 번호이며 전송하지 않습니다.
- `--show` 디버그 화면은 현장 점검용으로 화면에만 표시하고 녹화하지 않습니다.
- `data/`(영상·이미지)와 `models/`는 Git에 올리지 않습니다. 실제 역 영상 녹화는 운영 기관의 허가와 개인정보 처리 절차가 필요합니다.
