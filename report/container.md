# 엣지 AI 박스 컨테이너 구성

> 엣지 AI 박스(Jetson Orin Nano) 안의 프로그램을 Docker 컨테이너 몇 개로 나누는 방법과, 지금 코드(`src/subway_edge/`)를 어떻게 나눠 넣을지 정리한 문서.

## 0. 결론

- **컨테이너로 나눌 수 있다.** 지금 코드는 검출기(`Detector`)와 전송(`Publisher`)이 이미 인터페이스로 분리돼 있어서, 전송 부분만 MQTT로 바꾸면 컨테이너 사이를 브로커로 이을 수 있다.
- 컨테이너는 **4개**로 나눈다: ① 영상 처리(vision), ② 브로커(broker), ③ 클라우드 전송(uplink), ④ 상태 감시(watchdog, 선택)
- 나누는 기준은 **"영상 프레임이 컨테이너 밖으로 나가지 않는다"** 이다. 디코딩부터 1분 집계까지는 한 컨테이너 안 메모리에서 끝내고, 컨테이너 사이에는 **숫자(집계 레코드)와 명령만** 오간다.
- 이 시스템에는 모터·비상정지 같은 구동부가 없고 카메라도 네트워크(PoE IP 카메라)로 받으므로, **privileged 컨테이너가 필요 없다.** GPU 사용만 NVIDIA 런타임으로 허용한다.
- 아직 Jetson에서 실제로 띄워 보지 않았다. 아래 메모리 수치와 이미지 태그는 실측·확인이 필요하다. (7장)

---

## 1. 전체 구성

```
┌──────────────────────────── Jetson Orin Nano (JetPack + Docker Engine + NVIDIA Container Runtime) ────────────────────────────┐
│                                                                                                                               │
│  [HW] GbE (PoE 스위치 ← IP 카메라 10대) · GPU(CUDA/TensorRT) · NVDEC · NVMe SSD                                                │
│        │ RTSP (카메라 LAN)          │ runtime: nvidia                                                                          │
│        ▼                            ▼                                                                                         │
│  ┌───────────────────────────────────────────────┐                                                                            │
│  │ ① vision                                      │   💥 "죽어도 되는" 구역                                                     │
│  │  RTSP 디코딩 → 전처리 → YOLO → 추적 → 카운팅 → 1분 집계 │   unprivileged, mem_limit                                           │
│  │  (scripts/run.py + subway_edge, 카메라 N대)      │   read_only, 쓰기 볼륨 없음 (프레임 저장 불가)                        │
│  │  프레임은 이 컨테이너 메모리 밖으로 나가지 않음          │   restart: on-failure                                              │
│  └───────────────────────────────────────────────┘                                                                            │
│        │ pub: edge/{camera_id}/minute, edge/{camera_id}/health          ▲ sub: edge/cmd/weather                               │
│        ▼                                                                │                                                     │
│  ┌───────────────────────────────────────────────┐                                                                            │
│  │ ② broker  (Mosquitto, 박스 내부 전용)              │   🧱 시스템의 척추                                                    │
│  │  unprivileged · 자원 거의 안 씀 · 외부 포트 안 열음     │   restart: always                                                    │
│  └───────────────────────────────────────────────┘                                                                            │
│        │ sub: edge/+/minute, edge/+/health                              ▲ pub: edge/cmd/weather                               │
│        ▼                                                                │                                                     │
│  ┌───────────────────────────────────────────────┐   ┌───────────────────────────────────┐                                    │
│  │ ③ uplink                                       │   │ ④ watchdog (선택)                    │                                   │
│  │  로컬 MQTT ↔ AWS IoT Core (MQTT over TLS 8883)   │   │  health 끊긴 카메라 감지 → 알림         │                                   │
│  │  끊기면 레코드를 디스크 큐에 쌓았다가 재전송 (숫자만)     │   │  unprivileged · restart: always      │                                   │
│  │  🔒 "데이터를 잃으면 안 되는" 구역 · restart: always   │   └───────────────────────────────────┘                                    │
│  └───────────────────────────────────────────────┘                                                                            │
│        │ LTE 라우터 / 역사 유선망                                                                                                │
└────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
         ▼
   [AWS IoT Core]  ── 날씨 정보 → edge/cmd/weather 로 내려보냄
```

### 1.1 예시 구성(라즈베리파이 로봇)과 다른 점

| 항목 | 예시 (로봇) | 이 프로젝트 |
|---|---|---|
| 하드웨어 접근 | GPIO·CSI·모터 드라이버 → privileged / `--device` | IP 카메라는 네트워크로 받음 → **privileged 없음**, GPU만 `runtime: nvidia` |
| "절대 죽으면 안 되는" 곳 | 모터 제어·비상정지 (사람 안전) | **uplink** (집계 숫자를 잃지 않는 것). 안전에 직결되는 구동부는 없음 |
| "죽어도 되는" 곳 | AI 추론 | **vision** (재시작 동안 그 분의 카운트만 빠짐, 숫자 손실은 7장 기준으로 허용 범위 확인) |
| 컨테이너 사이 데이터 | 카메라·IMU 원본 데이터 | **집계 숫자와 명령만**. 영상 프레임은 개인정보 원칙상 컨테이너 밖으로 보내지 않음 |

---

## 2. 컨테이너별 역할

### 2.1 ① vision — 영상 처리

- 지금 코드 그대로: `scripts/run.py` → `EdgePipeline` (1~9단계: 전처리, 검출, 추적, 카운팅, 1분 집계)
- 입력: 카메라 RTSP 주소. 출력: `Publisher`로 1분 집계 레코드(JSON)를 브로커에 보냄
- 날씨 명령(`edge/cmd/weather`)을 받아 `EdgePipeline.set_weather()`를 부른다. 지금은 `--weather` 실행 인자로만 받는다.
- 1분마다 상태(`health`: 처리 fps, 마지막 프레임 시각, 혼잡도 모드)를 보낸다. 영상 정보는 넣지 않는다.
- **디코딩과 추론을 같은 컨테이너에 둔다.** 예시처럼 "센서 수집"과 "추론"을 나누면 프레임이 컨테이너 사이를 오가야 한다. 이는
  - 개인정보 원칙(프레임 외부 전송 금지)의 경계를 흐리고,
  - 640×360 프레임 초당 100장이면 약 70MB/s를 복사해야 해서 브로커로는 감당할 수 없다.

### 2.2 ② broker — Mosquitto

- 박스 내부 컨테이너끼리만 쓰는 MQTT 브로커. 호스트 포트를 열지 않는다(`ports:` 없음).
- 메모리 수 MB 수준. 다른 컨테이너가 모두 여기에 기대므로 `restart: always`.
- 예시의 ROS 2 Discovery Server는 필요 없다. 이 프로젝트는 ROS를 쓰지 않는다.

### 2.3 ③ uplink — 클라우드 전송

- 로컬 `edge/+/minute`를 받아 AWS IoT Core로 MQTT over TLS(8883)로 보낸다. (`report/hardware.md` 3.5)
- LTE가 끊기면 레코드를 **디스크 큐(SQLite 등)** 에 쌓았다가 다시 보낸다. 쌓는 것은 숫자뿐이라 개인정보 원칙에 걸리지 않는다.
- 클라우드에서 내려온 날씨 정보를 `edge/cmd/weather`로 다시 보낸다.
- 인증서·엔드포인트는 코드에 적지 않고 환경 변수와 읽기 전용 볼륨으로 받는다. 새 변수 이름은 `.env.example`에만 추가한다.
- vision과 분리하는 이유: vision이 메모리 부족(OOM)으로 죽어도 **이미 집계된 레코드와 전송 큐는 살아 있다.**

### 2.4 ④ watchdog — 상태 감시 (선택)

- `health`가 일정 시간(예: 3분) 끊긴 카메라를 찾아 클라우드로 알린다.
- Docker 자체 `healthcheck`와 `restart`로 대부분 해결되므로, 처음에는 빼고 시작해도 된다.

---

## 3. 카메라 10대를 컨테이너 몇 개로 돌릴까

지금 `scripts/run.py`는 **실행 하나에 카메라 하나**다. 그래서 두 가지 방법이 있다.

| 방법 | 구성 | 장점 | 걸리는 점 |
|---|---|---|---|
| A. 카메라마다 컨테이너 | vision 컨테이너 10개 (같은 이미지, `--source`·`--config`만 다름) | **코드 수정 거의 없음.** 카메라 하나가 죽어도 나머지는 영향 없음 | 컨테이너마다 모델·CUDA 컨텍스트를 따로 올림. Jetson은 CPU·GPU가 8GB 메모리를 같이 써서 10개면 부족할 수 있음. 배치 추론 불가 |
| B. 컨테이너 하나에 카메라 여러 대 | vision 컨테이너 1개(또는 2~3개), 안에서 카메라별 디코딩 스레드 + 검출기 공유 | 모델을 한 번만 올림, 프레임을 모아 배치 추론 가능 → 초당 100프레임에 유리 | 여러 소스를 받는 실행 스크립트를 새로 만들어야 함. 한 프로세스가 죽으면 묶인 카메라가 같이 멈춤 |

- **권장: 처음엔 A로 2~3대 띄워 메모리를 재고, 10대에서 부족하면 B로 간다.** B로 가도 카메라 3~4대씩 묶은 컨테이너 2~3개로 나누면 장애 범위를 줄일 수 있다.
- `EdgePipeline`은 카메라 하나의 상태(추적·집계)만 가지므로, B에서도 **카메라마다 `EdgePipeline` 하나 + 검출기는 공유**하는 구조가 된다. 추적·카운팅 코드는 바꾸지 않는다.

---

## 4. docker-compose 예시 (방법 A, 카메라 2대)

> 구성을 보여주기 위한 초안이다. 이미지 태그·경로는 JetPack 버전을 정한 뒤 확인한다.

```yaml
services:
  broker:
    image: eclipse-mosquitto:2
    restart: always
    volumes:
      - ./deploy/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro
    networks: [edge]
    # ports 없음: 박스 밖에서 접속 불가

  vision-cam01: &vision
    build: { context: ., dockerfile: deploy/vision.Dockerfile }
    runtime: nvidia                       # GPU만 허용, privileged 아님
    restart: on-failure
    mem_limit: 1500m                      # 실측 후 조정 (7장)
    read_only: true                       # 파일 쓰기 불가 → 프레임 저장 원천 차단
    tmpfs: [/tmp]
    volumes:
      - ./models:/app/models:ro           # 모델은 models/에서 읽기 전용
      - ./configs:/app/configs:ro
    environment:
      MQTT_HOST: broker
      CAMERA_RTSP_URL: ${CAM01_RTSP_URL}  # 값은 .env에만
    command: >
      python scripts/run.py --source ${CAM01_RTSP_URL}
      --config configs/stationA_cam01.json --publisher mqtt
    networks: [edge]
    depends_on: [broker]

  vision-cam02:
    <<: *vision
    command: >
      python scripts/run.py --source ${CAM02_RTSP_URL}
      --config configs/stationA_cam02.json --publisher mqtt

  uplink:
    build: { context: ., dockerfile: deploy/uplink.Dockerfile }
    restart: always
    mem_limit: 128m
    environment:
      MQTT_HOST: broker
      AWS_IOT_ENDPOINT: ${AWS_IOT_ENDPOINT}
    volumes:
      - ./certs:/app/certs:ro             # 인증서는 Git 제외 경로
      - uplink-queue:/app/queue           # 전송 대기 숫자 레코드만
    networks: [edge]
    depends_on: [broker]

networks:
  edge: {}

volumes:
  uplink-queue: {}
```

- vision 이미지는 Jetson용 베이스(예: `ultralytics/ultralytics:latest-jetson-jetpack6`, 또는 `nvcr.io/nvidia/l4t-jetpack`)에서 시작한다. 일반 x86 이미지는 Jetson에서 GPU를 못 쓴다.
- 개발 PC(Windows, CUDA 없음)에서는 `runtime: nvidia`를 뺀 override 파일로 같은 구성을 CPU로 띄워 테스트한다. 모델은 `.onnx`로 두면 PC·Jetson에서 같은 파일을 쓴다. (`report/inference_engine.md` 3.3)

---

## 5. 개인정보·보안 장치

| 장치 | 효과 |
|---|---|
| vision `read_only: true` + 쓰기 볼륨 없음 | 코드에 실수가 있어도 프레임을 디스크에 저장할 수 없다 |
| 프레임은 vision 프로세스 메모리 안에서만 | 컨테이너 사이에 영상이 흐르는 경로 자체가 없다 |
| broker 외부 포트 없음 | 박스 밖에서 내부 토픽을 엿보거나 명령을 넣을 수 없다 |
| 인터넷으로 나가는 것은 uplink만 | vision은 카메라 LAN과 broker만 필요하다. 호스트 방화벽(DOCKER-USER 체인)으로 vision의 외부 접속을 카메라 대역으로 제한한다 |
| privileged 없음, `--device` 없음 | 컨테이너가 뚫려도 호스트 장치에 접근하기 어렵다 |
| 비밀값은 `.env`·인증서 볼륨으로만 | Git·이미지에 키가 들어가지 않는다 |

---

## 6. 지금 코드에서 바꿔야 할 것

| 항목 | 위치 | 내용 |
|---|---|---|
| MQTT 전송 | `src/subway_edge/publisher.py` | `Publisher` 인터페이스를 구현한 `MqttPublisher` 추가. 파이프라인 코드는 그대로 |
| 전송 방식 선택 | `scripts/run.py` | `--publisher stdout|mqtt` 인자 추가. 브로커 주소는 환경 변수 `MQTT_HOST` |
| 날씨 명령 수신 | `scripts/run.py` (또는 새 모듈) | `edge/cmd/weather` 구독 → `pipeline.set_weather()` 호출. 프레임 처리 루프와 스레드가 겹치지 않게 큐로 넘김 |
| 상태 보고 | 새 모듈 | 1분마다 `health` 토픽으로 처리 fps·모드 전송 |
| RTSP 재연결 | `scripts/run.py` | 지금은 프레임을 못 읽으면 종료한다. 실시간 소스는 재연결을 몇 번 시도하고, 그래도 안 되면 종료해서 `restart: on-failure`에 맡긴다 |
| uplink 서비스 | 새 패키지 (예: `src/subway_uplink/`) | 로컬 MQTT ↔ AWS IoT Core 중계, 디스크 큐 |
| 배포 파일 | 새 폴더 (예: `deploy/`) | Dockerfile, compose, mosquitto.conf |
| (방법 B로 갈 때) 다중 카메라 실행 | 새 스크립트 | 카메라별 `EdgePipeline` + 검출기 공유 + 배치 추론 |

---

## 7. 확인이 필요한 항목

- [ ] Jetson에서 vision 컨테이너 1개의 실제 메모리 사용량 (모델 + CUDA 컨텍스트) → 방법 A로 몇 대까지 되는지
- [ ] 설치할 JetPack 버전과 그에 맞는 베이스 이미지 태그. JetPack 6은 Ubuntu 22.04·Python 3.10이라, 프로젝트 규칙(Python 3.11)과 맞출지 결정 필요
- [ ] 컨테이너 안에서 NVDEC 하드웨어 디코딩이 되는지 (OpenCV `VideoCapture`가 GStreamer 경로를 쓰는지)
- [ ] vision 재시작에 걸리는 시간(모델 로드·TensorRT 엔진 캐시 포함)과, 그동안 빠지는 카운트를 어떻게 표시할지 (예: 레코드에 `partial` 표시)
- [ ] uplink 디스크 큐 보관 기간 (LTE가 며칠 끊기는 경우)

## 참고

- 전송 인터페이스: `src/subway_edge/publisher.py` (`Publisher`, `StdoutPublisher`)
- 파이프라인: `src/subway_edge/pipeline.py` (`EdgePipeline.process`, `set_weather`, `flush`)
- 하드웨어·네트워크: `report/hardware.md`
- 추론 엔진·모델 파일 형식: `report/inference_engine.md`
