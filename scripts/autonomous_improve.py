#!/usr/bin/env python3
"""
Autonomous improvement script for smu_sig_prossessing.
Runs a single improvement iteration: test current state, try improvements, evaluate.
Outputs a short status report to stdout for the cron job to deliver.
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

def check_api():
    """Check if the API is available (not rate-limited)"""
    code, out, err = run(f"{VENV} -c \"import requests; r=requests.get('https://api.z.ai/api/coding/paas/v4/models', headers={{'Authorization':'Bearer 776ed3bd04b34a55aeeed844dc4ff0e2.xaOqdEdCtFh1T9B8'}}, timeout=10); print(r.status_code)\"", timeout=20)
    return "200" in out

def run_eval(preset, degrade="none", sample=3):
    """Run auto_eval and return results"""
    cmd = f"{VENV} main.py -p input/analog_whoop_footage.mp4 --preset {preset} --degrade {degrade} --sample {sample} 2>&1"
    code, out, err = run(cmd, timeout=180)
    return code == 0, out[-500:] if out else err[-500:]

def run_full_eval():
    """Run run_auto_eval.py for quantitative metrics"""
    cmd = f"{VENV} run_auto_eval.py -i input/analog_whoop_footage.mp4 --degrade none --sample 3 2>&1 | tail -40"
    code, out, err = run(cmd, timeout=300)
    return code == 0, out if out else err

def get_git_status():
    code, out, _ = run("git log --oneline -3")
    return out

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

# ─── Main ───
log("=== smu_sig autonomous improvement ===")

# 1. Check git state
git_log = get_git_status()
log(f"Recent commits:\n{git_log}")

# 2. Check what improvements have been made
code, out, _ = run(f"grep -c 'def ' smu_sig_prossessing/filters.py")
filter_count = out.strip() if code == 0 else "?"
log(f"Filter functions: {filter_count}")

# 3. Check if new files exist
new_files = []
for f in ["smu_sig_prossessing/noise_estimator.py", "smu_sig_prossessing/adaptive.py"]:
    if os.path.exists(f):
        new_files.append(f.split("/")[-1])
log(f"New modules: {', '.join(new_files) if new_files else 'none'}")

# 4. Test current state with different presets
results = {}
for preset in ["adaptive", "analog-clean", "video-enhanced", "wavelet-denoise"]:
    ok, output = run_eval(preset)
    results[preset] = "✅ OK" if ok else "❌ FAIL"
    log(f"  {preset}: {results[preset]}")

# 5. Report
report_lines = [
    f"📊 **smu_sig 자율 개선 상태 리포트** ({get_timestamp()})",
    f"",
    f"**Git:** {git_log.split(chr(10))[0] if git_log else 'N/A'}",
    f"**필터 수:** {filter_count}",
    f"**신규 모듈:** {', '.join(new_files) if new_files else '없음'}",
    f"",
    f"**Preset 테스트:**",
]
for p, r in results.items():
    report_lines.append(f"  • {p}: {r}")

# 6. Check for uncommitted changes
code, changes, _ = run("git status --short")
if changes.strip():
    report_lines.append(f"")
    report_lines.append(f"**미커밋 변경:** {len(changes.strip().split(chr(10)))}개 파일")

print("\n".join(report_lines))
