"""Fail closed if a screened external candidate is misrepresented as G2-I-ready."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "org.aerocity.bench.external-candidate-input-semantics.v1"
REQUIRED_TOP_LEVEL = {
    "schema",
    "generated_on",
    "purpose",
    "g2_i_public_inputs",
    "g2_i_forbidden_inputs",
    "candidates",
}
REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "upstream_url",
    "upstream_commit",
    "license",
    "task_class",
    "required_upstream_inputs",
    "g2_i_mapping",
    "missing_or_forbidden_inputs",
    "license_status",
    "integration_status",
    "c_gate_eligible",
    "disposition",
}
FULL_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


def audit(path: Path) -> dict[str, Any]:
    """Validate the decision registry without changing the candidate set."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != REQUIRED_TOP_LEVEL:
        raise ValueError("candidate input-semantics registry has unexpected top-level fields")
    if raw["schema"] != SCHEMA:
        raise ValueError("candidate input-semantics registry schema differs")
    public_inputs = raw["g2_i_public_inputs"]
    forbidden_inputs = raw["g2_i_forbidden_inputs"]
    candidates = raw["candidates"]
    if not all(isinstance(value, str) and value for value in public_inputs):
        raise ValueError("G2-I public inputs must be non-empty strings")
    if not all(isinstance(value, str) and value for value in forbidden_inputs):
        raise ValueError("G2-I forbidden inputs must be non-empty strings")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate input-semantics registry needs at least one candidate")

    candidate_ids: set[str] = set()
    eligible: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != REQUIRED_CANDIDATE_FIELDS:
            raise ValueError("candidate input-semantics fields differ")
        candidate_id = candidate["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_ids:
            raise ValueError("candidate IDs must be unique, non-empty strings")
        candidate_ids.add(candidate_id)
        if not isinstance(candidate["c_gate_eligible"], bool):
            raise ValueError(f"{candidate_id}: c_gate_eligible must be boolean")
        for field in (
            "upstream_url",
            "license",
            "task_class",
            "g2_i_mapping",
            "license_status",
            "integration_status",
            "disposition",
        ):
            if not isinstance(candidate[field], str) or not candidate[field]:
                raise ValueError(f"{candidate_id}: {field} must be a non-empty string")
        for field in ("required_upstream_inputs", "missing_or_forbidden_inputs"):
            value = candidate[field]
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError(f"{candidate_id}: {field} must be a string list")
        if candidate["c_gate_eligible"]:
            revision = candidate["upstream_commit"]
            if not isinstance(revision, str) or not FULL_GIT_REVISION.fullmatch(revision):
                raise ValueError(
                    f"{candidate_id}: a C-gate candidate needs a locked full upstream revision"
                )
            if "three-dimensional" not in candidate["task_class"]:
                raise ValueError(
                    f"{candidate_id}: a C-gate candidate must be a substantive "
                    "three-dimensional inspection or search method"
                )
            if "geometry-search" not in candidate["task_class"]:
                raise ValueError(
                    f"{candidate_id}: a C-gate candidate must explicitly be mapped "
                    "to the geometry-search track"
                )
            if "perception" in candidate["task_class"]:
                raise ValueError(
                    f"{candidate_id}: a perception-search method cannot close the "
                    "geometry-search C gate"
                )
            if candidate["missing_or_forbidden_inputs"]:
                raise ValueError(
                    f"{candidate_id}: a C-gate candidate cannot have unresolved input gaps"
                )
            if candidate["license_status"] != "verified":
                raise ValueError(f"{candidate_id}: a C-gate candidate needs a verified license")
            if "three-city L0 and L1 closed" not in candidate["integration_status"]:
                raise ValueError(
                    f"{candidate_id}: a C-gate candidate needs a three-city L0 and L1 "
                    "closed integration"
                )
            eligible.append(candidate_id)

    return {
        "schema": "org.aerocity.bench.external-candidate-input-semantics-audit.v1",
        "registry": path.as_posix(),
        "candidate_count": len(candidate_ids),
        "c_gate_eligible": eligible,
        "status": "BLOCKED_NO_SUBSTANTIVE_EXTERNAL_G2_I_METHOD"
        if not eligible
        else "CANDIDATE_REQUIRES_REPLAY_VERIFICATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("external/candidate-input-semantics.json"),
        help="candidate input-semantics JSON registry",
    )
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.registry), indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"EXTERNAL_CANDIDATE_SEMANTICS_AUDIT_REJECTED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
