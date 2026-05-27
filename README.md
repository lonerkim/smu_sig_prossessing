# 아날로그 영상 잡음 완화 파이프라인

영상처리 기말 프로젝트 — 6팀 (김승민, 김원석)

## 프로젝트 개요
아날로그 FPV DVR 영상에서 시인성을 해치는 잡음·색왜곡·대비저하를 기본 영상처리 기법으로 완화하는 범용 파이프라인.

## 구조
```
input/        — 기준 원본 이미지 (5종)
output/       — 처리 결과 (비교 이미지, 히스토그램, 비디오)
scripts/
  pipeline.py       — Phase 0~2 정적 이미지 파이프라인
  video_pipeline.py — 비디오 기반 파이프라인 테스트
```

## 파이프라인 단계

| Phase | 내용 | 기법 |
|-------|------|------|
| 0 | 실험 데이터 준비 | 합성 이미지 + 인위 열화 (Gaussian, Impulse, Color bias, Brightness↓, Periodic) |
| 1 | 잡음 제거 | 미디언 필터, 가우시안 로우패스, 위너 필터, FFT Notch |
| 2 | 색상/대비 보정 | 감마 보정, 로그 연산, 히스토그램 평활화, 채널별 보정 |

## 실행
```bash
# 정적 이미지 파이프라인
python scripts/pipeline.py

# 비디오 파이프라인
python scripts/video_pipeline.py
```

## 의존성
- Python 3.12+
- opencv-python-headless
- scikit-image
- numpy
