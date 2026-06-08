#!/usr/bin/env python3
"""
Autonomous improvement script for smu_sig_prossessing v3.3.
Runs benchmark iterations, tries new presets, pushes improvements.
Outputs a short status report to stdout.
"""
import subprocess, sys, os, json, time, traceback
from datetime import datetime

os.chdir(os.path.expanduser("~/smu_sig_prossessing"))
VENV = os.path.expanduser("~/smu_sig_prossessing/.venv/bin/python3")

def run(cmd, timeout=120):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def run_fast_benchmark():
    """Run the fast v33 benchmark and return top 3 presets"""
    code, out, _ = run(f"{VENV} run_v33_benchmark.py 2>&1 | tail -60", timeout=300)
    return code == 0, out

def get_git_status():
    code, out, _ = run("git log --oneline -3")
    return out

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

# ─── Main ───
log("=== smu_sig v3.3 autonomous improvement ===")

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
    log(f"Uncommitted changes: {len(changes.strip().split(chr(10)))} files")
    log(f"Changes:\n{changes}")

# 4. Report
report = [
    f"📊 **smu_sig v3.3 자율 개선** ({get_timestamp()})",
    f"",
    f"**Git:** {git_log.split(chr(10))[0] if git_log else 'N/A'}",
    f"",
]
if ok and output:
    report.append("**Benchmark Top 10:**")
    report.append(f"```\n{output}\n```")

report.append(f"**Uncommitted:** {len(changes.strip().split(chr(10))) if changes.strip() else 0} files")
print("\n".join(report))
