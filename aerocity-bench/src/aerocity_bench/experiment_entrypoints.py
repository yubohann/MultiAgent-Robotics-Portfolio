"""Static guard for every executable path that consumes public experiment input.

This is deliberately a source-level complement to the runtime boundary audit:
it stops a later tool from being added to an experimental pipeline without
calling the common public-artifact validator.  It cannot establish scientific
validity, but it prevents the precise class of interface drift that let legacy
CF2X evidence remain runnable after the public contract changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import content_hash

ENTRYPOINT_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "tools/audit_baseline_opportunity.py": ("public-layout consumer", "audit_public_layout"),
    "tools/cf2x_l1_fleet_preflight.py": ("public L1 consumer", "audit_public_layout"),
    "tools/run_external_cf2x_l1_preflight.py": (
        "guarded external-process L1 scheduler",
        "audit_public_layout",
    ),
    "tools/materialize_g2_i_l1_layout.py": ("public-layout producer", "audit_public_layout"),
    "tools/quadrotor_l1_vertical_slice.py": ("public L1 consumer", "audit_public_layout"),
    "tools/run_cf2x_b_gate_l0_pairing.py": ("public L0 consumer", "audit_public_layout"),
    "tools/run_cf2x_b_gate_replays.py": ("public L1 panel scheduler", "audit_public_layout"),
    "tools/run_marvel_g2i_l0_smoke.py": ("external-process consumer", "audit_public_layout"),
    "tools/trace_public_route.py": ("public route diagnostic", "audit_public_layout"),
    "tools/diagnose_g2_i_policy_deadline.py": (
        "detached public-input diagnostic",
        "validate_public_task_spec",
    ),
    "src/aerocity_bench/marvel_g2i_projection.py": (
        "external-process payload projection",
        "assert_public_fields",
    ),
    "src/aerocity_bench/native_gate_contract.py": (
        "native public-input loader",
        "validate_public_task_spec",
    ),
    "src/aerocity_bench/compiler.py": ("public task compiler", "validate_public_task_spec"),
    "src/aerocity_bench/builder_v3.py": (
        "authority-release validator",
        "validate_public_task_spec",
    ),
}


def audit_experiment_entrypoints(repository_root: Path) -> dict[str, Any]:
    """Return a deterministic PASS/FAIL inventory of public-input entrypoints."""

    root = repository_root.resolve()
    records: list[dict[str, Any]] = []
    for relative, (role, required_marker) in sorted(ENTRYPOINT_REQUIREMENTS.items()):
        path = root / relative
        if not path.is_file():
            records.append(
                {
                    "path": relative,
                    "role": role,
                    "required_marker": required_marker,
                    "status": "FAIL_MISSING_SOURCE",
                }
            )
            continue
        source = path.read_text(encoding="utf-8")
        records.append(
            {
                "path": relative,
                "role": role,
                "required_marker": required_marker,
                "status": "PASS" if required_marker in source else "FAIL_MISSING_GUARD",
            }
        )
    passed = all(record["status"] == "PASS" for record in records)
    report: dict[str, Any] = {
        "schema": "org.aerocity.bench.experiment-entrypoint-audit.v1",
        "formal_score_eligible": False,
        "status": "PASS" if passed else "FAIL",
        "entrypoint_count": len(records),
        "records": records,
    }
    report["report_hash"] = content_hash(report)
    return report
