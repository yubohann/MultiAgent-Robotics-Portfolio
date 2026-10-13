"""实验矩阵串行调度器，逐配置重复运行 fc_align 并汇总判据。

用法
  python run_matrix.py --group grasp        # B/C/D 组
  python run_matrix.py --group all
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = r"C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe"
LOGDIR = HERE / "_matrix"
LOGDIR.mkdir(exist_ok=True)
RESULTS = HERE / "matrix_results.csv"

GROUPS = {
    "grasp": [
        ("B1_base", ["--base-only"], 20),
        ("B3_full", [], 20),
        ("C2_nocontact", ["--no-contact"], 20),
        ("D2_rigid", ["--rigid"], 20),
    ],
    "quick": [
        ("B3_full", [], 3),
    ],
}


def run_one(name, extra, rep, timeout_s=2700):
    tag = "%s_r%02d" % (name, rep)
    log = LOGDIR / (tag + ".log")
    cmd = [PY, "fc_align.py", "--headless", "--enable_cameras",
           "--out", "runs/matrix_" + tag] + extra
    t0 = time.time()
    with open(log, "wb") as f:
        proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=f, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -9
    dt = time.time() - t0
    verdict = "no-verdict"
    try:
        text = log.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"\[al\] 判据 (.*?)-> (PASS|FAIL)", text)
        if m:
            verdict = m.group(2) + " " + m.group(1).strip()
    except Exception:
        pass
    return tag, verdict, round(dt, 1), rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="grasp")
    args = ap.parse_args()
    todo = GROUPS.get(args.group, GROUPS["quick"])
    new_file = not RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tag", "verdict", "seconds", "rc"])
        for name, extra, reps in todo:
            for rep in range(1, reps + 1):
                tag, verdict, dt, rc = run_one(name, extra, rep)
                w.writerow([tag, verdict, dt, rc])
                f.flush()
                print("[mx] %s -> %s (%.0fs rc=%d)" % (tag, verdict, dt, rc), flush=True)
    print("[mx] 完成，结果在 %s" % RESULTS, flush=True)


if __name__ == "__main__":
    main()
