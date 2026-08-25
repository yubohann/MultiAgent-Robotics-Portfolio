"""Run missing gate54 and gate60 mainline evaluations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(os.environ.get("PYTHON", sys.executable))
DEFAULT_CHECKPOINT = (
    ROOT
    / "runtime"
    / "gate_density_oldmethod_resume_fixed_dyn_20260510_151811"
    / "bc_k02_specialized_seed_splits_v1"
    / "bc_k02_g18_g24_seed5_9"
    / "checkpoints"
    / "best_agent.pt"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "runtime"
    / "gate_density_oldmethod_resume_fixed_dyn_20260510_151811"
    / "axis_48_54_60_inference_to_60_20260515_60_svg"
)


def run_one(
    *,
    python_exe: Path,
    checkpoint: Path,
    output_root: Path,
    gate_count: int,
    seed: int,
    force: bool,
) -> dict[str, object]:
    out_dir = output_root / f"gate{gate_count}_seed{seed}" / f"seed_{seed}"
    summary = out_dir / "stage_summary.json"
    if summary.exists() and not force:
        return {"gate_count": gate_count, "seed": seed, "status": "skipped_existing", "summary": str(summary)}
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_combined.log"
    cmd = [
        str(python_exe),
        str(ROOT / "gate_density_single" / "scripts" / "run_gate_density_eval.py"),
        "--checkpoint",
        str(checkpoint),
        "--gate-count",
        str(gate_count),
        "--seed",
        str(seed),
        "--random-yaw",
        "--gate-layout-version",
        "irregular_centerline_v7_large_arena_dynamic",
        "--enable-agent-policy",
        "--enable-path-planner",
        "--dynamic-controller-profile",
        "density_adaptive_v1",
        "--moving-gates",
        "--moving-gate-amplitude-m",
        "1.2",
        "--moving-gate-speed-mps",
        "2.0",
        "--drone-speed-mps",
        "3.5",
        "--drone-accel-mps2",
        "2.45",
        "--episodes",
        "1",
        "--max-steps",
        "800",
        "--output-dir",
        str(out_dir),
    ]
    start = time.time()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    status = "ok" if proc.returncode == 0 and summary.exists() else f"failed_returncode_{proc.returncode}"
    return {
        "gate_count": gate_count,
        "seed": seed,
        "status": status,
        "elapsed_s": round(elapsed, 2),
        "summary": str(summary),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gates", type=int, nargs="+", default=[54, 60])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    jobs = [(gate, seed) for gate in args.gates for seed in args.seeds]
    results: list[dict[str, object]] = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"jobs": jobs, "workers": args.workers, "output_root": str(args.output_root)}, ensure_ascii=False), flush=True)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(
                run_one,
                python_exe=args.python_exe,
                checkpoint=args.checkpoint,
                output_root=args.output_root,
                gate_count=gate,
                seed=seed,
                force=bool(args.force),
            )
            for gate, seed in jobs
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            (args.output_root / "run_results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    failures = [row for row in results if not str(row.get("status", "")).startswith(("ok", "skipped"))]
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()



