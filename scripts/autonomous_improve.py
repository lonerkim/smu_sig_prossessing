#!/usr/bin/env python3
"""
Autonomous improvement script for smu_sig_prossessing v3.3+.
Runs benchmark iterations, tries new presets, pushes improvements.
Outputs a short status report to stdout.

Usage:
    python scripts/autonomous_improve.py [--repo-dir PATH]

Default REPO_DIR: same directory as this script's parent (../)
Supports both direct execution (from repo root) and cron (from anywhere).
"""
import subprocess, sys, os, json, time, traceback
from datetime import datetime

# Resolve repo directory relative to this script's location
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) > 2 and sys.argv[1] == '--repo-dir':
    REPO_DIR = os.path.abspath(sys.argv[2])

os.chdir(REPO_DIR)

VENV_PYTHON = sys.executable  # use the python that's running this script

def run(cmd, timeout=120):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def run_fast_benchmark():
    """Run the fast v33 benchmark and return output"""
    code, out, err = run(f"{VENV_PYTHON} run_v33_benchmark.py 2>&1 | tail -60", timeout=300)
    return code == 0, out

def get_git_status():
    code, out, _ = run("git log --oneline -3")
    return out

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

# ─── Main ───
log("=== smu_sig autonomous improvement ===")
log(f"Repo: {REPO_DIR}")
log(f"Python: {VENV_PYTHON}")

# 1. Check git state
git_log = get_git_status()
log(f"Recent commits:\n{git_log}")

# 2. Run fast benchmark
ok, output = run_fast_benchmark()
if ok and output:
    log(f"Benchmark output:\n{output}")
else:
    log("⚠ Benchmark failed or no output")

# 3. Check for uncommitted changes
code, changes, _ = run("git status --short")
if changes.strip():
    n_files = len(changes.strip().split('\n'))
    log(f"Uncommitted changes: {n_files} files")
    log(f"Changes:\n{changes}")

# 4. Report
report_lines = [
    f"📊 **smu_sig v3.3+ 자율 개선** ({get_timestamp()})",
    f"",
    f"**Git:** {git_log.split(chr(10))[0] if git_log else 'N/A'}",
    f"",
]
if ok and output:
    report_lines.append("**Benchmark Top 10:**")
    report_lines.append(f"```\n{output}\n```")

n_uncommitted = len(changes.strip().split('\n')) if changes.strip() else 0
report_lines.append(f"**Uncommitted:** {n_uncommitted} files")
print("\n".join(report_lines))
