#!/usr/bin/env python3
"""
smu_sig_prossessing v3.4 — Comprehensive 2-hour improvement loop.

Runs every 2 hours: git check, benchmark, experiments, commit, push.
"""
import subprocess, sys, os, json, time, traceback
from datetime import datetime

BASE = "/home/openclaw/smu_sig_prossessing"
os.chdir(BASE)
VENV = f"{BASE}/.venv/bin/python3"

def run(cmd, timeout=120):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def get_ts():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

# --- Steps ---
log("=" * 60)
log(f"=== smu_sig v3.4 improvement loop start: {get_ts()} ===")
log("=" * 60)

# 1. Git state
log("--- Git ---")
code, out, _ = run("git log --oneline -3")
log(f"Recent:\n{out}")
code, changes, _ = run("git status --short")
if changes.strip():
    log(f"Uncommitted: {len(changes.strip().split(chr(10)))} files")

# 2. Benchmark
log("--- Benchmark ---")
code, out, _ = run(f"{VENV} run_v33_benchmark.py 2>&1 | tail -15", timeout=300)
if code == 0:
    log(f"Top scores:\n{out}")
else:
    log(f"Benchmark failed (code={code})")

# 3. GitHub check
log("--- GitHub ---")
code, out, _ = run(
    'curl -s "https://api.github.com/repos/lonerkim/smu_sig_prossessing/issues?state=all&per_page=5" 2>&1 '
    '| python3 -c "import sys,json; data=json.load(sys.stdin); '
    '[print(f\'#{i[\\\"number\\\"]} [{i[\\\"state\\\"]}] {i[\\\"title\\\"]}\') for i in (data if isinstance(data,list) else [])]" 2>&1',
    timeout=15
)
if out.strip():
    log(f"Issues:\n{out}")
else:
    log("No issues found")

# 4. Simple improvement experiments (Python inline)
log("--- Experiments ---")
exp_script = os.path.join(BASE, "scripts", "run_experiments.py")
code, out, _ = run(f"{VENV} {exp_script} 2>&1 | tail -20", timeout=180)
log(f"Results:\n{out}")

# 5. Commit if changes
log("--- Commit ---")
code, changes, _ = run("git status --short")
if changes.strip():
    run("git add -A")
    commit_msg = f"perf: auto-improve {get_ts()}"
    code, out, _ = run(f"git commit -m '{commit_msg}'")
    log(f"Commit: {out[:200]}")
    code, out, _ = run("git push origin main 2>&1")
    log(f"Push: {out[:200]}")
else:
    log("No changes to commit")

log("=" * 60)
log(f"=== Loop complete: {get_ts()} ===")
