# 배경 제거 도구 (Background Remover)

이미지에서 배경을 자동으로 제거하는 프로그램입니다. [transparent-background](https://github.com/plemeri/transparent-background) 라이브러리를 사용하여 고품질의 배경 제거 기능을 제공합니다.

<img width="904" alt="Image" src="https://github.com/user-attachments/assets/d9073622-cf9f-48f0-b975-33ca10b2add0" />

## 시스템 요구사항
- Windows 10 이상
- Python 3.8 이상
- pip (파이썬 패키지 관리자)

## 프로젝트 구조
```
BackgroundRemover/
├── main.py           # 메인 프로그램
├── run.bat          # 실행 스크립트
└── requirements.txt  # 의존성 목록
```

## 프로그램 실행 방법

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 프로그램 실행:
```bash
run.bat
```
또는 직접 Python으로 실행:
```bash
python main.py
```

## 사용 방법
1. "이미지 선택" 버튼을 클릭하여 처리할 이미지를 선택합니다.
2. 원하는 모드를 선택합니다:
   - Fast: 빠른 처리 (품질 다소 낮음)
   - Base: 기본 모드 (균형잡힌 처리)
   - Simple: 단순한 이미지에 적합
   - Legacy: 이전 버전 호환성 모드
3. 필요한 경우 설정을 조정합니다:
   - 임계값: 배경 감지 민감도 (0-100)
   - 강도: 배경 제거 강도 (0-100)
4. "배경 제거" 버튼을 클릭하여 처리를 시작합니다.
5. 처리가 완료되면 결과 이미지가 자동으로 저장됩니다.

## 기술 정보
- 이 프로그램은 transparent-background 라이브러리를 사용합니다.
- 딥러닝 기반의 고성능 배경 제거 알고리즘을 사용합니다.
- GPU가 있는 경우 자동으로 GPU 가속을 지원합니다.

## 결과물 저장 위치
- 처리된 이미지는 원본 이미지와 동일한 폴더에 저장됩니다.
- 파일명 형식: `nobg_원본파일명.png`
  예) 원본: `photo.jpg` → 결과: `nobg_photo.png`

## 주의사항
- 큰 이미지의 경우 처리 시간이 오래 걸릴 수 있습니다.
- GPU가 있는 경우 자동으로 GPU를 사용하여 처리 속도가 향상됩니다.
- 처리 중에는 프로그램을 종료하지 마세요.
- 처리 실패 시 콘솔 창의 오류 메시지를 확인해주세요.

## 문제 해결
- 프로그램 실행 오류 시:
  1. Python 버전 확인 (3.8 이상인지 확인)
  2. pip로 의존성이 정상 설치되었는지 확인
  3. 콘솔에서 `python --version` 명령어로 Python이 정상 설치되었는지 확인
- 이미지 처리 오류 발생 시:
  1. 다른 모드로 시도
  2. 이미지 크기를 줄여서 시도
  3. 임계값과 강도 조절 시도 
