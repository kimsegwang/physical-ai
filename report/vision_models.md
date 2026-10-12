# 비전 모델 조사 – YOLOE · DINO-X · SAM 3 · SigLIP · OCR · MediaPipe

> 개찰구 혼잡도 프로젝트에서 쓸 수 있는 비전 모델 6종의 **기능, 사용 예시, 프로젝트 적용 방안**을 정리한 문서입니다.
> 현재 엣지 파이프라인(YOLO + ByteTrack + 이중 통과선)은 `report/preprocess.md`를 따릅니다.
>
> 작성 기준: 2026-10. 모델 버전·API는 자주 바뀌므로 실제 적용 전에 공식 저장소 문서를 다시 확인합니다.

**검토 시 전제 (CLAUDE.md 규칙)**
- 영상·프레임은 파일로 저장하거나 외부로 전송하지 않는다 → **클라우드 API 모델은 운영 영상에 쓰지 않는다**
- 좌표(ROI·마스크·통과선)는 0~1 정규화 좌표로 받는다
- 테스트 영상·이미지는 `data/`, 모델 파일은 `models/`에만 둔다
- API 키는 코드에 적지 않고 환경 변수로 읽는다

---

## 1. 한눈에 보기

| 모델 | 한 줄 요약 | 입력 → 출력 | 실행 위치 | Jetson 실시간 | 라이선스 |
|---|---|---|---|---|---|
| **YOLOE** | 텍스트·이미지로 클래스를 정하는 실시간 YOLO | 이미지 + 클래스 이름 → 박스·마스크 | 엣지 | ○ (클래스 고정 후 일반 YOLO 속도) | AGPL-3.0 (ultralytics) |
| **DINO-X** | 거의 모든 물체를 찾는 오픈월드 검출 (클라우드 API) | 이미지 + 텍스트/예시 → 박스·마스크·키포인트·캡션 | 클라우드 | × | 상용 API (유료) |
| **SAM 3** | 개념(텍스트·예시)으로 모든 인스턴스를 분할·추적 | 이미지/영상 + 명사구 → 마스크·박스·ID | 서버 GPU | × (대형 모델) | SAM License (Meta) |
| **SigLIP** | 이미지·텍스트를 같은 공간에 임베딩 | 이미지 + 문장 후보 → 유사도 점수 | 엣지/서버 | △ (저빈도 실행) | Apache-2.0 |
| **OCR** | 이미지 속 글자 읽기 (EasyOCR, PaddleOCR 등) | 이미지 → 글자 + 위치 + 신뢰도 | 엣지/서버 | △ (작은 영역만) | Apache-2.0 |
| **MediaPipe** | 구글의 온디바이스 비전 솔루션 모음 | 이미지 → 얼굴·포즈·손 랜드마크 등 | 엣지/PC | ○ (CPU에서도 동작) | Apache-2.0 |

**프로젝트 내 역할 요약**

| 역할 | 추천 모델 |
|---|---|
| 엣지 실시간 사람 검출 (현재) | YOLO (기존 유지) |
| COCO에 없는 클래스(유모차·휠체어 등) 검출 | YOLOE |
| 학습용 라벨 자동 생성 (오프라인) | SAM 3, DINO-X(허가된 공개 데이터만), YOLOE |
| 카메라 상태·현장 상황 분류 (가림, 젖은 바닥 등) | SigLIP |
| 검증 영상의 화면 시각(OSD) 읽기 | OCR |
| 검증 영상 검토 시 얼굴 비식별화 | MediaPipe |

---

## 2. YOLOE (Real-Time Seeing Anything)

### 2.1 기능
- 칭화대(THU-MIG)가 2025년에 발표한 **오픈 어휘(open-vocabulary) 검출·분할** 모델. YOLOv8/YOLO11 구조 기반
- 프롬프트 방식 3가지
  | 방식 | 설명 |
  |---|---|
  | 텍스트 프롬프트 (RepRTA) | `["person", "stroller"]`처럼 클래스 이름을 주면 그 클래스만 검출 |
  | 비주얼 프롬프트 (SAVPE) | 예시 이미지에 박스를 그려 "이런 물체"를 찾게 함 |
  | 프롬프트 없음 (LRPC) | 내장 어휘(약 4,500개)로 보이는 물체를 모두 검출 |
- 텍스트 임베딩을 모델에 **재파라미터화(re-parameterize)**해 넣기 때문에, 클래스를 정한 뒤에는 **일반 YOLO와 같은 속도**로 동작하고 ONNX/TensorRT로 내보낼 수 있음
- ultralytics 패키지에 포함되어 있어 현재 코드베이스와 바로 연결 가능

### 2.2 사용 예시

**텍스트 프롬프트로 사람·유모차·휠체어 검출**
```python
import cv2
from ultralytics import YOLOE

# 모델 파일은 models/ 폴더에 둔다
model = YOLOE("models/yoloe-11s-seg.pt")

# 검출할 클래스를 텍스트로 지정 (COCO에 없는 stroller, wheelchair도 가능)
names = ["person", "suitcase", "stroller", "wheelchair"]
model.set_classes(names, model.get_text_pe(names))

frame = cv2.imread("data/sample_frame.jpg")
results = model.predict(frame, conf=0.4, verbose=False)

h, w = frame.shape[:2]
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    # 프로젝트 규칙에 맞춰 0~1 정규화 좌표로 변환
    print(names[int(box.cls)], float(box.conf), (x1 / w, y1 / h, x2 / w, y2 / h))
```

**클래스를 고정해 엣지용으로 내보내기**
```python
# set_classes 이후 export 하면 텍스트 인코더 없이 일반 YOLO처럼 동작
model.export(format="engine", half=True)  # Jetson TensorRT
```

**비주얼 프롬프트 (예시 박스로 찾기)**
```python
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

model = YOLOE("models/yoloe-11s-seg.pt")
visual_prompts = dict(
    bboxes=np.array([[221.5, 405.8, 344.9, 857.5]]),  # 예시 이미지 속 유모차 박스 (픽셀)
    cls=np.array([0]),
)
results = model.predict("data/sample_frame.jpg", visual_prompts=visual_prompts,
                        predictor=YOLOEVPSegPredictor)
```

### 2.3 프로젝트 적용
- `preprocess.md` 2장에서 "COCO에 유모차 클래스는 없음"이라고 한 부분을 해결할 수 있음 → 유모차·휠체어를 **카운트 제외 확인용 보조 클래스**로 검출
- 현장 데이터로 YOLO를 다시 학습하기 전, 새 클래스를 빠르게 시험해 보는 용도로 적합
- 주의: 오픈 어휘 성능은 학습된 전용 모델보다 낮을 수 있으므로 `preprocess.md` 4장 방식으로 정확도를 따로 검증

---

## 3. DINO-X

### 3.1 기능
- IDEA Research가 2024년 말 발표한 **오픈월드 객체 이해 통합 모델** (Grounding DINO 후속)
- 하나의 모델로 여러 출력을 냄
  | 기능 | 설명 |
  |---|---|
  | 프롬프트 없는 검출 | 이름을 주지 않아도 화면 속 물체를 모두 찾음 |
  | 텍스트 프롬프트 검출 | `"person . stroller"`처럼 이름으로 검출 |
  | 비주얼 프롬프트 | 예시 박스·점으로 비슷한 물체 검출 (드문 물체에 강함) |
  | 분할·키포인트·캡션 | 박스와 함께 마스크, 사람 포즈 키포인트, 영역 설명 문장 생성 |
- Pro(정확도 우선)와 Edge(경량) 버전이 있으나, **일반 사용은 DeepDataSpace 클라우드 API로만 제공**(가중치 비공개)
- 장시간 꼬리(long-tail) 물체, 밀집 장면에서 강한 제로샷 성능

### 3.2 사용 예시
API 토큰은 환경 변수(`DDS_API_TOKEN`)로 읽는다. 실제 코드에 넣게 되면 `.env.example`에 변수 이름만 추가한다.

```python
import os
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

client = Client(Config(os.environ["DDS_API_TOKEN"]))

# 주의: 이미지가 외부 서버로 업로드된다 → 허가된 공개 데이터셋에만 사용
image_url = client.upload_file("data/public_dataset/sample.jpg")

task = V2Task(
    api_path="/v2/task/dinox/detection",
    api_body={
        "model": "DINO-X-1.0",
        "image": image_url,
        "prompt": {"type": "text", "text": "person.suitcase.stroller"},
        "targets": ["bbox", "mask"],
        "bbox_threshold": 0.25,
        "iou_threshold": 0.8,
    },
)
client.run_task(task)

for obj in task.result["objects"]:
    print(obj["category"], obj["score"], obj["bbox"])
```
> SDK 버전에 따라 `api_path`, 결과 형식이 달라질 수 있으므로 공식 문서를 확인한다.

### 3.3 프로젝트 적용
- **운영 개찰구 영상에는 사용 불가** (프레임을 외부로 전송하므로 개인정보 규칙 위반)
- 쓸 수 있는 경우: 라이선스상 허용된 **공개 보행자 데이터셋**의 라벨 생성, YOLOE·SAM 3 결과와의 비교 기준(상한선) 측정
- 비용이 호출당 과금이라 대량 라벨링에는 SAM 3·YOLOE가 더 현실적

---

## 4. SAM 3 (Segment Anything with Concepts)

### 4.1 기능
- Meta가 2025년 11월 공개한 Segment Anything 3세대
- 기존 SAM 1·2가 "점·박스로 찍은 **하나의** 물체"를 분할했다면, SAM 3는 **개념 프롬프트(PCS, Promptable Concept Segmentation)**를 지원
  | 프롬프트 | 결과 |
  |---|---|
  | 짧은 명사구 (`"person"`, `"yellow suitcase"`) | 해당하는 **모든** 인스턴스의 마스크 + ID |
  | 예시 박스 (양성/음성) | 예시와 같은 개념의 물체 전부 |
  | 점·박스 (SAM 2 방식) | 지정한 물체 하나 |
- 영상 입력 시 인스턴스를 프레임 간 **추적**(같은 ID 유지)
- 모델이 크고(약 8억 파라미터대) 서버 GPU 기준이라 Jetson 실시간에는 맞지 않음
- 가중치는 Hugging Face에서 **접근 승인 후** 다운로드

### 4.2 사용 예시

**이미지에서 텍스트로 모든 사람 분할 (공식 저장소 `facebookresearch/sam3`)**
```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)

image = Image.open("data/sample_frame.jpg")
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="person")

masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
w, h = image.size
for box, score in zip(boxes.tolist(), scores.tolist()):
    x1, y1, x2, y2 = box
    print(round(score, 3), (x1 / w, y1 / h, x2 / w, y2 / h))  # 정규화 좌표
```

**라벨 자동 생성 흐름 (오프라인, 메모리에서만 처리)**
```
data/ 의 허가된 검증 영상
  → SAM 3 "person" 프롬프트로 박스·ID 생성
  → 사람이 검토·수정 (라벨링 도구)
  → YOLO 형식 라벨(좌표 텍스트)만 저장
  → 엣지용 YOLO 파인튜닝
```

### 4.3 프로젝트 적용
- **현장 데이터 라벨링 비용 절감**이 가장 큰 효용. 밀집 장면에서 사람마다 박스를 그리는 작업을 대부분 자동화
- 영상 추적 기능으로 **검증용 정답 궤적**을 만들어 ByteTrack의 ID 끊김을 비교할 수 있음
- 사람 검토 없이 그대로 학습에 쓰면 SAM 3의 실수가 YOLO로 옮겨가므로 반드시 샘플 검수

---

## 5. SigLIP / SigLIP 2

### 5.1 기능
- 구글의 **이미지-텍스트 대조 학습 모델** (CLIP 계열). 손실 함수를 softmax 대신 sigmoid로 바꿔 작은 배치에서도 잘 학습됨
- SigLIP 2(2025)는 다국어, 위치 정보(localization), 가변 해상도(NaFlex)가 개선됨
- 주요 기능
  | 기능 | 설명 |
  |---|---|
  | 제로샷 이미지 분류 | 학습 없이 "문장 후보" 중 이미지와 가장 맞는 것 선택 |
  | 이미지·텍스트 검색 | 임베딩 유사도로 원하는 장면 찾기 |
  | 비전 인코더 | VLM(PaliGemma 등)의 이미지 인코더로 사용 |
- 출력이 sigmoid 점수라 여러 후보가 동시에 참일 수 있음(다중 라벨에 유리)

### 5.2 사용 예시

**카메라 상태 점검 (제로샷 분류)**
```python
import cv2
from PIL import Image
from transformers import pipeline

clf = pipeline("zero-shot-image-classification",
               model="google/siglip2-base-patch16-224")

labels = [
    "a clear view of subway ticket gates",
    "a camera view blocked by an object",
    "a very dark camera image",
    "a wet reflective floor",
    "a crowded subway station",
]

frame = cv2.imread("data/sample_frame.jpg")
image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
for r in clf(image, candidate_labels=labels):
    print(f"{r['score']:.3f}  {r['label']}")
```

**임베딩 직접 계산 (여러 프레임 비교용)**
```python
import torch
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval()
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    emb = model.get_image_features(**inputs)
emb = emb / emb.norm(dim=-1, keepdim=True)  # 코사인 유사도용 정규화
```

### 5.3 프로젝트 적용
- **카메라 이상 감지**: 1분에 1번 정도 프레임을 분류해 가림·암전·각도 틀어짐을 알림 → 카운트 0이 "사람이 없어서"인지 "카메라 문제"인지 구분
- **젖은 바닥 자동 판단**: 날씨 API 대신 영상으로 `wet_floor_masks` 적용 여부 보조 판단
- 기준 프레임(설치 직후) 임베딩과 현재 프레임 임베딩의 유사도로 **카메라 각도 변화** 감지
- 주의: 사람을 구분하는 임베딩(재식별, ReID)으로 쓰거나 저장하지 않는다. 장면 단위 판단에만 사용

---

## 6. OCR (광학 문자 인식)

### 6.1 기능
- 이미지에서 글자 영역을 찾고(검출) 읽는(인식) 기술. 대표 오픈소스
  | 도구 | 특징 |
  |---|---|
  | EasyOCR | 설치·사용이 가장 간단, 한국어 지원, PyTorch 기반 |
  | PaddleOCR (PP-OCRv5) | 정확도·속도 우수, 경량 모델(mobile) 제공, 한국어 지원 |
  | Tesseract | 오래된 표준, 깨끗한 인쇄체에 적합, 영상 속 글자에는 약함 |
- 출력: 글자 박스(4점), 텍스트, 신뢰도

### 6.2 사용 예시

**CCTV 화면 시각(OSD) 읽기 – 정규화 좌표로 영역 지정**
```python
import re
import cv2
import easyocr

reader = easyocr.Reader(["ko", "en"], gpu=True)

# 화면 왼쪽 위 시각 표시 영역 (0~1 정규화 좌표)
OSD_ROI = (0.0, 0.0, 0.35, 0.08)

def read_osd_time(frame):
    """프레임의 OSD 영역에서 'YYYY-MM-DD HH:MM:SS' 형식 시각을 읽는다."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = OSD_ROI
    crop = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    text = " ".join(t for _, t, conf in reader.readtext(gray) if conf > 0.5)
    m = re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", text)
    return m.group(0) if m else None

frame = cv2.imread("data/sample_frame.jpg")
print(read_osd_time(frame))
```

**PaddleOCR (3.x)**
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="korean",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)
for res in ocr.predict("data/gate_sign.jpg"):
    for text, score in zip(res["rec_texts"], res["rec_scores"]):
        print(text, round(score, 3))
```

### 6.3 프로젝트 적용
- **검증 시간 동기화**: 수동 카운트와 AI 카운트를 분 단위로 맞출 때, 녹화 영상의 OSD 시각을 자동으로 읽음 (`preprocess.md` 4장)
- **설치 정보 확인**: 개찰구 번호·출구 표지판을 읽어 카메라-게이트 매핑 설정 검증
- OSD 영역만 잘라서 처리하므로 엣지에서도 가볍게 실행 가능 (예: 1분에 1회)
- 주의: 교통카드·휴대폰 화면 등 **개인 정보가 담긴 글자는 읽지 않도록** ROI를 OSD·표지판으로 한정

---

## 7. MediaPipe

### 7.1 기능
- 구글의 **온디바이스 ML 솔루션 모음**. TFLite 기반으로 CPU·모바일에서도 빠름
- Tasks API로 제공되는 주요 비전 기능
  | 태스크 | 출력 |
  |---|---|
  | Face Detector (BlazeFace) | 얼굴 박스 + 6개 키포인트 |
  | Pose Landmarker | 사람 33개 관절 좌표 (2D + 3D) |
  | Hand Landmarker / Gesture Recognizer | 손 21개 관절, 제스처 |
  | Object Detector | EfficientDet-Lite 기반 일반 물체 검출 |
  | Image Segmenter | 사람/배경 분리 등 |
- 실행 모드: IMAGE / VIDEO / LIVE_STREAM

### 7.2 사용 예시

**얼굴 비식별화 (메모리에서만 블러, 저장하지 않음)**
```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

options = vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path="models/blaze_face_short_range.tflite"),
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.5,
)
detector = vision.FaceDetector.create_from_options(options)

def blur_faces(frame):
    """프레임 속 얼굴을 흐리게 만든 복사본을 반환한다 (파일 저장 없음)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    out = frame.copy()
    for det in result.detections:
        b = det.bounding_box
        x, y = max(b.origin_x, 0), max(b.origin_y, 0)
        roi = out[y:y + b.height, x:x + b.width]
        if roi.size:
            out[y:y + b.height, x:x + b.width] = cv2.GaussianBlur(roi, (51, 51), 0)
    return out

frame = cv2.imread("data/sample_frame.jpg")
cv2.imshow("anonymized", blur_faces(frame))
cv2.waitKey(0)
```

**포즈 랜드마크**
```python
options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path="models/pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=5,
)
landmarker = vision.PoseLandmarker.create_from_options(options)
result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
for pose in result.pose_landmarks:
    nose = pose[0]
    print(nose.x, nose.y)  # 이미 0~1 정규화 좌표
```

### 7.3 프로젝트 적용
- **검증 영상 검토 시 얼굴 블러**: 수동 카운트 작업자 화면에 비식별화된 영상을 보여주는 용도 (`--show` 화면 등)
- 포즈 랜드마크는 쓰러짐 감지 같은 확장 기능 후보이나, 개찰구 카운팅 자체에는 불필요
- 한계: BlazeFace 단거리 모델은 약 2m 이내 정면 얼굴 기준이라 **천장 CCTV의 작은 얼굴은 놓칠 수 있음** → 비식별화 용도라면 YOLO 사람 박스 상단을 함께 블러하는 방식이 더 안전. 포즈도 멀리 있는 다수 인원에는 정확도가 낮음

---

## 8. 종합 제안

### 8.1 엣지(운영) vs 오프라인(개발) 배치
```
[엣지 박스 – 운영, 실시간]
  YOLO(현행) 또는 클래스 고정 YOLOE  → 사람·보조 클래스 검출
  SigLIP (1분 1회)                    → 카메라 이상·젖은 바닥 판단
  OCR (1분 1회, OSD 영역만)            → 시각 확인 (선택)

[개발 PC / 서버 – 오프라인]
  SAM 3, YOLOE(프롬프트 없음)          → 라벨 자동 생성 → 사람 검수 → YOLO 파인튜닝
  MediaPipe                            → 검증 영상 검토 화면 얼굴 블러
  DINO-X (공개 데이터만)                → 성능 비교 기준선
```

### 8.2 우선순위
| 순위 | 작업 | 기대 효과 | 난이도 |
|---|---|---|---|
| 1 | YOLOE로 유모차·휠체어 보조 클래스 검출 시험 | 카운트 제외 대상 확인, 기존 코드와 바로 연결 | 낮음 |
| 2 | SAM 3 기반 라벨 자동 생성 | 현장 파인튜닝 데이터 확보 속도 향상 | 중간 |
| 3 | SigLIP 카메라 상태 점검 | 카운트 0의 원인 구분, 운영 안정성 | 낮음 |
| 4 | OCR로 검증 영상 시각 동기화 | 정확도 검증 작업 시간 단축 | 낮음 |
| 5 | MediaPipe 얼굴 블러 | 검증 과정의 개인정보 보호 강화 | 낮음 |

### 8.3 공통 주의사항
- 새 모델 파일은 `models/`에만 두고 Git에 올리지 않는다
- 외부 API(DINO-X)는 운영 영상에 쓰지 않는다
- 라벨 생성 시에도 프레임 이미지는 저장하지 않고, 좌표 텍스트(라벨)만 남긴다. 학습용 이미지 보관이 꼭 필요하면 운영 기관 허가와 개인정보 처리 절차를 먼저 거친다
- 각 모델을 도입하면 `preprocess.md` 4.4 표처럼 **도입 전후 정확도를 숫자로 기록**한다
- 라이선스 확인: YOLOE(ultralytics)는 AGPL-3.0, SAM 3는 Meta 자체 라이선스 → 상용 배포 시 검토 필요
