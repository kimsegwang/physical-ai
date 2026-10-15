# 추론 엔진 선정: ONNX Runtime

> 후보 4개(ONNX Runtime, NCNN, OpenVINO, TFLite(LiteRT)) 중 엣지 AI 박스(Jetson Orin Nano)에서 YOLO를 돌릴 추론 엔진을 고르고, 그 이유를 정리한 문서.

## 0. 결론

- **ONNX Runtime을 선택한다.** Jetson에서는 GPU 실행 경로(TensorRT 실행 공급자 → CUDA 실행 공급자 → CPU 순서로 대체)를 쓴다.
- 이유는 한 줄로 **"후보 4개 중 Jetson의 GPU를 쓸 수 있는 엔진은 ONNX Runtime뿐"** 이기 때문이다. NCNN·OpenVINO·TFLite는 Jetson에서 사실상 CPU로 돌아 초당 100프레임 조건을 맞출 수 없다.
- 아직 Jetson 실측 전이다. 아래 속도는 Ultralytics 공식 Jetson 가이드의 공개 벤치마크이며, 우리 영상·우리 모델로 잰 값이 아니다.

---

## 1. 판단 기준

| 기준 | 내용 | 근거 |
|---|---|---|
| 처리량 | 박스 1대가 카메라 10대 × 10fps = **초당 100프레임** → 프레임당 추론 예산 **약 10ms** (전처리·추적 시간 제외 전) | `report/model_plan.md` 1.2 |
| 하드웨어 | Jetson Orin Nano 8GB. 연산 대부분이 **GPU(CUDA 코어·Tensor 코어)** 에 있고 CPU는 ARM 6코어 | `report/hardware.md` 3.1 |
| 정밀도 | FP16 우선, 부족하면 INT8 (현장 이미지로 보정) | `report/model_plan.md` 2.7 |
| 보드 교체 | 비용 절감 시 RK3588(RKNN)·Hailo(HEF)로 바꿀 수 있어야 함 | `report/hardware.md` 6장 |
| 개발 환경 | 개발 PC는 CUDA가 없음. 같은 모델 파일로 PC(CPU)에서도 테스트할 수 있으면 좋음 | `report/model_plan.md` 2.6 |
| 개인정보 | 엔진은 메모리 안에서만 추론. 프레임 저장·외부 전송 없음 (네 엔진 모두 해당) | CLAUDE.md |

---

## 2. 후보 비교

### 2.1 엔진별 특징

| 엔진 | 만든 곳 | 주로 맞는 하드웨어 | Jetson GPU 사용 | 모델 형식 |
|---|---|---|---|---|
| **ONNX Runtime** | Microsoft | 범용 (CPU, NVIDIA GPU, 그 밖의 가속기를 "실행 공급자"로 연결) | **○** CUDA·TensorRT 실행 공급자 | `.onnx` |
| NCNN | Tencent | 휴대폰·ARM CPU, Vulkan GPU | △ Vulkan 경로만, CUDA·Tensor 코어는 못 씀 | `.param` + `.bin` |
| OpenVINO | Intel | Intel CPU·내장 GPU·NPU | ✕ NVIDIA GPU 미지원 (ARM CPU 플러그인만) | `.xml` + `.bin` |
| TFLite (LiteRT) | Google | 휴대폰, 마이크로컨트롤러, Coral Edge TPU | ✕ 사실상 CPU(XNNPACK) | `.tflite` |

### 2.2 Jetson Orin Nano Super 공개 벤치마크

Ultralytics 공식 Jetson 가이드, YOLO26n, 640px, 이미지 1장당 추론 시간(전·후처리 제외).

| 형식 | 추론 시간 | 초당 처리 (단순 환산) | 카메라 10대 × 10fps 가능? |
|---|---|---|---|
| TensorRT INT8 | 3.80ms | 약 263 | ○ |
| TensorRT FP16 | 4.57ms | 약 219 | ○ |
| **ONNX (ONNX Runtime)** | **15.76ms** | 약 63 | △ 이 상태로는 부족, GPU 실행 공급자 필요 |
| NCNN | 47.50ms | 약 21 | ✕ 카메라 2대 수준 |
| OpenVINO | 56.23ms | 약 18 | ✕ |
| TFLite | 286.00ms | 약 3.5 | ✕ |

- 가이드는 TensorRT 외의 형식이 **CPU만 사용한다**고 적고 있다. 즉 위 ONNX 수치는 GPU 가속 없이 잰 값에 가깝다.
- ONNX Runtime은 GPU 버전(onnxruntime-gpu)을 설치하고 **TensorRT 실행 공급자**를 켜면 내부에서 TensorRT로 추론한다. 그래서 위 TensorRT 수치에 가까워질 수 있는 **유일한 후보**다. (실측으로 확인할 항목)
- NCNN·OpenVINO·TFLite는 엔진을 아무리 조정해도 Jetson의 GPU를 제대로 쓰지 못해 10ms 예산과 차이가 크다.

---

## 3. ONNX Runtime을 고른 이유

### 3.1 Jetson의 GPU를 쓰는 유일한 후보

- Jetson Orin Nano의 성능은 대부분 GPU에서 나온다. GPU를 못 쓰는 엔진은 보드 값의 대부분을 버리는 셈이다.
- ONNX Runtime은 실행 공급자를 순서대로 지정해, **TensorRT로 되는 연산은 TensorRT로, 안 되는 연산은 CUDA로, 그래도 안 되면 CPU로** 자동으로 나눠 돌린다. 모델 일부 연산이 TensorRT에서 지원되지 않아도 전체가 멈추지 않는다.

### 3.2 ONNX는 모든 대안 보드의 공통 출발점

- TensorRT 엔진, RKNN(RK3588), HEF(Hailo) 모두 **ONNX 파일을 입력으로 받아 변환**한다.
- ONNX를 기준 모델 파일로 두면, 보드를 바꿔도 "학습 → ONNX 내보내기"까지는 그대로이고 마지막 변환만 바뀐다. (`report/hardware.md` 6.2)

### 3.3 개발 PC와 Jetson에서 같은 파일

- 개발 PC에는 CUDA가 없지만, 같은 `.onnx` 파일을 ONNX Runtime CPU 버전으로 그대로 돌릴 수 있다.
- 반면 TensorRT 엔진(`.engine`)은 GPU·TensorRT 버전에 묶여 Jetson에서 만든 엔진만 Jetson에서 쓸 수 있다. (`report/model_plan.md` 2.7)
- 그래서 PC에서 정확도를 검증한 모델 파일과 Jetson에 올라가는 모델 파일이 **같은 파일**이 된다.

### 3.4 지금 코드에 붙이기 쉬움

- `src/subway_edge/detection.py`의 `YoloDetector`는 ultralytics `YOLO(model_path)`로 모델을 연다. ultralytics는 `.onnx` 파일을 받으면 ONNX Runtime으로 실행하므로, **설정의 `model_path`만 `.onnx`로 바꾸면** 바로 돌아간다.
- 검출기는 `Detector` 인터페이스로 분리돼 있어, TensorRT 실행 공급자를 쓰는 전용 검출기를 추가해도 추적·카운팅 코드는 바꾸지 않는다.

### 3.5 다른 후보를 고르지 않은 이유

| 엔진 | 제외 이유 |
|---|---|
| NCNN | ARM CPU·휴대폰에 최적화된 엔진이다. Jetson에서는 CUDA·Tensor 코어를 못 쓰고, 공개 벤치마크로 초당 약 21프레임이라 카메라 2대분이다. 라즈베리파이처럼 GPU 가속이 없는 보드라면 좋은 선택이다. |
| OpenVINO | Intel CPU·내장 GPU·NPU용이다. NVIDIA GPU를 지원하지 않아 Jetson에서는 ARM CPU로만 돈다. 엣지 박스를 Intel 미니 PC로 바꿀 때만 다시 검토한다. |
| TFLite (LiteRT) | 휴대폰·마이크로컨트롤러·Coral용이다. Jetson에서 가장 느렸다(초당 약 3.5프레임). Coral은 `report/hardware.md` 6장에서 이미 제외했다. |

---

## 4. 쓸 때 주의할 점

1. **ultralytics로 열면 CUDA 실행 공급자까지만 쓴다.** ultralytics의 ONNX 실행 코드는 CUDA → CPU 순서만 지정하고 TensorRT 실행 공급자는 켜지 않는다. TensorRT 속도가 필요하면 `onnxruntime.InferenceSession`을 직접 만드는 검출기를 `Detector` 인터페이스로 추가한다. (코드 작성은 Jetson 실측 후 결정)
   ```python
   providers = [
       ("TensorrtExecutionProvider", {
           "trt_fp16_enable": True,
           "trt_engine_cache_enable": True,
           "trt_engine_cache_path": "models/trt_cache",  # models/는 Git 제외
       }),
       "CUDAExecutionProvider",
       "CPUExecutionProvider",
   ]
   session = onnxruntime.InferenceSession("models/best.onnx", providers=providers)
   ```
2. **첫 실행이 느리다.** TensorRT 실행 공급자는 처음 실행할 때 엔진을 만드느라 몇 분이 걸릴 수 있다. 엔진 캐시를 켜고 캐시는 `models/`(Git 제외)에 둔다. 박스 설치 직후 한 번 돌려 캐시를 만들어 둔다.
3. **Jetson용 설치 파일을 따로 받는다.** PyPI의 일반 `onnxruntime-gpu`가 아니라 JetPack·Python·CUDA 버전에 맞는 Jetson용 wheel을 설치해야 한다. (Ultralytics Jetson 가이드, Jetson Zoo 호환표 참고)
4. **INT8 보정은 현장 이미지로.** TensorRT 실행 공급자로 INT8을 쓸 때도 보정 데이터는 우리 현장 이미지를 쓴다. (`report/model_plan.md` 2.7과 같은 원칙)
5. **동적 입력 크기를 피한다.** 입력 크기가 바뀌면 TensorRT 엔진을 다시 만든다. 전처리에서 크기를 고정(`imgsz`)한 상태로 내보낸다.

```bash
yolo export model=models/best.pt format=onnx imgsz=640   # 개발 PC·Colab 어디서 만들어도 됨
```

---

## 5. TensorRT 직접 실행과의 관계

- TensorRT 자체는 후보 4개에 없지만, 기존 계획(`report/model_plan.md` 2.7)은 ultralytics로 TensorRT 엔진(`.engine`)을 만들어 직접 실행하는 것이었다.
- ONNX Runtime + TensorRT 실행 공급자는 **안에서 같은 TensorRT를 쓰면서**, 지원되지 않는 연산의 대체 실행, 개발 PC와 같은 모델 파일, 다른 보드로의 이동성을 더한 방식이다.
- Jetson 실측에서 두 방식의 속도 차이가 작으면 ONNX Runtime으로 통일하고, 차이가 커서 초당 100프레임을 못 맞추면 TensorRT 직접 실행을 쓴다.

---

## 6. 보드를 바꾸면

| 보드 | 추론 엔진 | 비고 |
|---|---|---|
| Jetson Orin Nano (현재) | **ONNX Runtime** (TensorRT·CUDA 실행 공급자) | 이 문서의 선택 |
| RK3588 | RKNN Toolkit (ONNX → RKNN 변환) | 후보 4개 밖. ONNX Runtime은 CPU로만 돌아 부족 |
| 라즈베리파이 5 + Hailo | HailoRT (ONNX → HEF 변환) | 후보 4개 밖 |
| 라즈베리파이 5 단독 | NCNN | 후보 중 ARM CPU에서 가장 유리. 카메라 2~3대 역만 |
| Intel 미니 PC | OpenVINO | 후보 중 Intel 하드웨어 전용 |

---

## 7. 확인이 필요한 항목

- [ ] Jetson에서 같은 모델로 ONNX Runtime(CUDA / TensorRT 실행 공급자)과 TensorRT 직접 실행 속도 비교
- [ ] 카메라 10대 동시 처리 시 전처리·디코딩·추적을 포함한 전체 초당 프레임
- [ ] FP16 → INT8 전환 시 카운팅 정확도 변화 (`report/model_plan.md` 3장 기준)
- [ ] 설치할 JetPack 버전에 맞는 onnxruntime-gpu wheel 확인

## 참고

- Ultralytics, NVIDIA Jetson 가이드 (Orin Nano Super 벤치마크, onnxruntime-gpu 설치): https://docs.ultralytics.com/guides/nvidia-jetson/
- ultralytics ONNX 실행 코드: `ultralytics/nn/backends/onnx.py` (CUDA·CPU 실행 공급자 지정)
