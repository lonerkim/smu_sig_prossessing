#!/usr/bin/env python3
"""
아날로그 영상 잡음 및 색 왜곡 완화를 위한 범용 영상처리 파이프라인
메인 진입점 — 모듈식 파이프라인 실행 및 평가

Usage:
    python main.py                     # 정적 이미지 파이프라인 실행
    python main.py --video             # 비디오 파이프라인 실행
    python main.py --pipeline wiener   # 특정 파이프라인 선택
    python main.py --list-filters      # 사용 가능한 필터 목록
"""
from __future__ import annotations

import sys
import os

# 패키지 임포트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smu_sig_prossessing.config import PipelineConfig
from smu_sig_prossessing.filters import list_filters
from smu_sig_prossessing import pipeline


def main():
    args = sys.argv[1:]

    if "--list-filters" in args or "-l" in args:
        print("📋 Available Filters:")
        print(pipeline.list_available_filters())
        return

    if "--video" in args or "-v" in args:
        from scripts.run_video import main as video_main
        video_main()
        return

    # Default: run static image pipeline
    from scripts.run_pipeline import main as image_main
    image_main()


if __name__ == "__main__":
    main()
