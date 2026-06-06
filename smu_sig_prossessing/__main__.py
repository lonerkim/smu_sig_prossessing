"""
smu_sig_prossessing 패키지 진입점.

python -m smu_sig_prossessing <command> [options]

Commands:
  process       — 이미지/비디오 처리 (main.py와 동일)
  eval          — 자동 정량+정성 평가 (run_auto_eval.py와 동일)
  list-filters  — 등록된 필터 목록
"""
from __future__ import annotations

import sys
import os

# Ensure package root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd in ("process", "run", "main"):
        # Forward to main.py
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from main import main as _main
        _main()

    elif cmd in ("eval", "evaluate", "auto-eval"):
        # Forward to run_auto_eval.py
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from run_auto_eval import main as _main
        _main()

    elif cmd in ("list-filters", "filters", "list"):
        from smu_sig_prossessing import pipeline as pl
        print("📋 Registered Filters:")
        print(pl.list_available_filters())

    elif cmd in ("--help", "-h", "help"):
        print(__doc__)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
