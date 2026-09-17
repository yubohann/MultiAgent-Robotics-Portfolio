"""Core identifying, geometry, and scene-boundary helpers for City-Lite."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .._identity import IdentityAccumulator
from .aabb import (
    AABB,
    CITY_LITE_COMMAND_VOLUME_W_M,
    CITY_LITE_FLIGHT_VOLUME_W_M,
    FORMAL_SCORING_VOLUME_W_M,
    TARGET_FREE_SAFE_STARTS_W_M,
    coerce_aabb,
)
from .constants import (
    FORBIDDEN_DECORATION_COMPONENTS,
    FORBIDDEN_PRIM_PREFIXES,
    ROUTE_CLEARANCE_M,
)


class CityLiteAuthorityError(ValueError):
    """Raised when the local City-Lite authority stops the run."""


class CityLiteRouteError(ValueError):
    """Raised when a public City-Lite route violates its frozen contract."""

def canonical_payload_identity(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return IdentityAccumulator(encoded).hexdigest()

def aabb_geometry_identity(aabbs: Sequence[AABB | Mapping[str, Any]]) -> str:
    rows = [coerce_aabb(value).as_dict() for value in aabbs]
    rows.sort(
        key=lambda row: (
            str(row.get("category", "")),
            str(row.get("source_prim", "")),
            tuple(row["minimum"]),
            tuple(row["maximum"]),
        )
    )
    return canonical_payload_identity(rows)

def flight_contract_payload() -> dict[str, Any]:
    return {
        "formal_scoring_volume_w_m": FORMAL_SCORING_VOLUME_W_M.as_dict(),
        "flight_volume_w_m": CITY_LITE_FLIGHT_VOLUME_W_M.as_dict(),
        "command_volume_w_m": CITY_LITE_COMMAND_VOLUME_W_M.as_dict(),
        "target_free_safe_starts_w_m": [list(value) for value in TARGET_FREE_SAFE_STARTS_W_M],
        "target_free_safe_starts_identity": canonical_payload_identity(
            TARGET_FREE_SAFE_STARTS_W_M
        ),
        "route_clearance_m": ROUTE_CLEARANCE_M,
    }

def identity_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = IdentityAccumulator()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def forbidden_scene_paths(prim_paths: Sequence[str]) -> list[str]:
    """Return exact legacy/decorative prim paths in a composed new stage."""

    violations: list[str] = []
    for raw_path in prim_paths:
        path = str(raw_path).rstrip("/") or "/"
        if any(
            path == prefix or path.startswith(prefix + "/")
            for prefix in FORBIDDEN_PRIM_PREFIXES
        ):
            violations.append(path)
            continue
        # The City-Lite derive script removes these exact named sublayers. A
        # path-component match catches them if a future source nests the
        # component below a different transform without treating generic words
        # such as "sign" as forbidden.
        for root in (
            "/World/City/Rivermark",
            "/World/StaticScene/City/Rivermark",
        ):
            if path == root or not path.startswith(root + "/"):
                continue
            components = {
                component.lower()
                for component in path[len(root) + 1 :].split("/")
                if component
            }
            if components & FORBIDDEN_DECORATION_COMPONENTS:
                violations.append(path)
            break
    return sorted(set(violations))
