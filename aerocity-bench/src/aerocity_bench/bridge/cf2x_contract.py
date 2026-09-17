"""Local-runtime contract for the reviewed CF2X USD asset structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CF2X_FILENAME = "cf2x.usd"
CF2X_SCHEMA_RELATIVE_PATH = Path("configuration") / "cf2x_robot_schema.usd"
CF2X_DEFAULT_PRIM = "/crazyflie"
CF2X_BODY_PRIM = "/crazyflie/body"
CF2X_THRUSTER_BODY_NAMES = ("m1_prop", "m2_prop", "m3_prop", "m4_prop")
CF2X_ROTOR_JOINT_NAMES = ("m1_joint", "m2_joint", "m3_joint", "m4_joint")

# These names are prohibited as *sources*.  They may appear in historic audit
# documents, but no executable local runtime path may resolve through them.
_FORBIDDEN_PATH_TOKENS = (
    "5_in_drone",
    "five_in_drone",
    "nucleus",
    "omniverse",
    "official_isaacsim_assets",
    "isaacsim_assets",
)


@dataclass(frozen=True)
class VerifiedCF2XAsset:
    """The only robot-asset handle that native benchmark code may consume."""

    usd_path: Path
    schema_path: Path
    usd_bytes: int
    schema_bytes: int
    redistribution_status: str = "local_runtime_only_license_clearance_pending"

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "schema": "org.aerocity.bench.cf2x-local-asset.v1",
            "asset_kind": "cf2x_local_runtime_dependency",
            "usd_filename": self.usd_path.name,
            "usd_bytes": self.usd_bytes,
            "schema_relative_path": CF2X_SCHEMA_RELATIVE_PATH.as_posix(),
            "schema_bytes": self.schema_bytes,
            "default_prim": CF2X_DEFAULT_PRIM,
            "body_prim": CF2X_BODY_PRIM,
            "thruster_body_names": list(CF2X_THRUSTER_BODY_NAMES),
            "rotor_joint_names": list(CF2X_ROTOR_JOINT_NAMES),
            "redistribution_status": self.redistribution_status,
        }


def _validate_runtime_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    normalized_parts = "/".join(part.lower() for part in resolved.parts)
    if any(token in normalized_parts for token in _FORBIDDEN_PATH_TOKENS):
        raise ValueError("CF2X local asset path resolves through a prohibited asset source")
    return resolved


def verify_local_cf2x_asset(path: Path) -> VerifiedCF2XAsset:
    """Validate the CF2X root layer name and its mandatory relative layer."""

    usd_path = _validate_runtime_path(path)
    if usd_path.name.lower() != CF2X_FILENAME:
        raise ValueError(f"CF2X asset must be named {CF2X_FILENAME}")
    if not usd_path.is_file():
        raise FileNotFoundError(f"CF2X USD is missing: {usd_path}")
    schema_path = _validate_runtime_path(usd_path.parent / CF2X_SCHEMA_RELATIVE_PATH)
    if not schema_path.is_file():
        raise FileNotFoundError("CF2X USD relative schema layer is missing")
    return VerifiedCF2XAsset(
        usd_path=usd_path,
        schema_path=schema_path,
        usd_bytes=usd_path.stat().st_size,
        schema_bytes=schema_path.stat().st_size,
    )


def inspect_verified_cf2x_structure(asset: VerifiedCF2XAsset) -> dict[str, object]:
    """Check USD structure when the optional USD runtime is available."""

    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover - exercised in Isaac release jobs.
        raise RuntimeError("pxr is required for native CF2X USD structure inspection") from exc
    stage = Usd.Stage.Open(str(asset.usd_path))
    if stage is None:
        raise ValueError("USD runtime failed to open the verified CF2X layer")
    default_prim = stage.GetDefaultPrim()
    if str(default_prim.GetPath()) != CF2X_DEFAULT_PRIM:
        raise ValueError("CF2X default prim differs from the reviewed contract")
    required_body_paths = (
        CF2X_BODY_PRIM,
        *(f"{CF2X_DEFAULT_PRIM}/{name}" for name in CF2X_THRUSTER_BODY_NAMES),
    )
    missing = [
        path for path in required_body_paths if not stage.GetPrimAtPath(path).IsValid()
    ]
    if missing:
        raise ValueError(f"CF2X required body prims are missing: {missing}")
    missing_joints = [
        name
        for name in CF2X_ROTOR_JOINT_NAMES
        if not stage.GetPrimAtPath(f"{CF2X_BODY_PRIM}/{name}").IsValid()
    ]
    if missing_joints:
        raise ValueError(f"CF2X required rotor joints are missing: {missing_joints}")
    return {
        **asset.fingerprint_payload(),
        "usd_structure_inspected": True,
        "required_body_count": 1 + len(CF2X_THRUSTER_BODY_NAMES),
        "required_joint_count": len(CF2X_ROTOR_JOINT_NAMES),
    }
