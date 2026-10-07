# 프로젝트: 지하철 개찰구 혼잡도 예측 (엣지 전처리)

## 개요
- 개찰구 카메라 영상을 엣지 AI 박스(Jetson급)에서 처리해 입·퇴장 인원을 센다
- 기획 문서: PRD/PRD.md, 전처리 설계: report/preprocess.md

## 규칙
- Python 3.11, OpenCV, numpy, ultralytics(YOLO)
- 코드는 src/subway_edge/ 패키지, 실행은 scripts/, 테스트는 tests/
- 주석과 docstring은 한국어로 작성
- ROI·마스크·통과선 좌표는 모두 0~1 정규화 좌표로 받는다
- 개인정보: 영상·프레임을 파일로 저장하거나 외부로 전송하는 코드를 만들지 않는다
- data/, models/ 폴더 내용은 Git에 올리지 않는다
- 코드를 수정하면 tests/를 실행해서 통과하는지 확인한다

## Git·보안 규칙
- .gitignore에 있는 파일(.env, data/, models/, .claude/settings.local.json)은 커밋하지 않는다
- `git add -f`로 무시된 파일을 강제로 추가하지 않는다
- .gitignore에서 위 항목을 빼거나 수정하지 않는다
- .env의 값(API 키, 비밀번호 등)을 코드·로그·문서에 직접 적지 않는다.
  필요한 값은 환경 변수로 읽고, 새 변수가 생기면 .env.example에 이름만 추가한다
- 테스트 영상·이미지는 data/에, 모델 파일은 models/에만 저장한다
- 커밋 전에 git status로 위 파일이 포함되지 않았는지 확인한다