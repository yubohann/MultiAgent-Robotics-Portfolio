"""Fail-closed local-runtime contract for the reviewed CF2X USD asset.

The benchmark does not redistribute a robot USD.  A native runner receives a
user-supplied local path and accepts it only when both the root layer and its
required relative schema layer match the reviewed digests.  This separates a
reproducible execution dependency from a redistribution claim: the source and
licence of the local asset still require a release-time clearance decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .canonical import file_hash

CF2X_FILENAME = "cf2x.usd"
CF2X_SHA256 = "7372ac0786312c47a92603da3fcd412d560b21c3757a8f0d5e7c2bfb2233d2f4"
CF2X_SCHEMA_RELATIVE_PATH = Path("configuration") / "cf2x_robot_schema.usd"
CF2X_SCHEMA_SHA256 = "c7a63f78ce3937c25cd05936ee73348bfdbbd0a10e82c0b8a37250730a3cbb9c"
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
    usd_sha256: str
    schema_sha256: str
    redistribution_status: str = "local_runtime_only_license_clearance_pending"

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "schema": "org.aerocity.bench.cf2x-local-asset.v1",
            "asset_kind": "cf2x_local_runtime_dependency",
            "usd_filename": self.usd_path.name,
            "usd_sha256": self.usd_sha256,
            "schema_relative_path": CF2X_SCHEMA_RELATIVE_PATH.as_posix(),
            "schema_sha256": self.schema_sha256,
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
    """Validate the reviewed CF2X root layer and its mandatory relative layer.

    The function intentionally does not locate an asset automatically and does
    not offer a fallback.  A missing, changed, or unreviewed USD must stop the
    native runner before Isaac starts.
    """

    usd_path = _validate_runtime_path(path)
    if usd_path.name.lower() != CF2X_FILENAME:
        raise ValueError(f"CF2X asset must be named {CF2X_FILENAME}")
    if not usd_path.is_file():
        raise FileNotFoundError(f"CF2X USD is missing: {usd_path}")
    usd_sha256 = file_hash(usd_path)
    if usd_sha256.lower() != CF2X_SHA256:
        raise ValueError("CF2X USD SHA-256 does not match the reviewed local-runtime asset")
    schema_path = _validate_runtime_path(usd_path.parent / CF2X_SCHEMA_RELATIVE_PATH)
    if not schema_path.is_file():
        raise FileNotFoundError("CF2X USD relative schema layer is missing")
    schema_sha256 = file_hash(schema_path)
    if schema_sha256.lower() != CF2X_SCHEMA_SHA256:
        raise ValueError("CF2X schema SHA-256 does not match the reviewed local-runtime asset")
    return VerifiedCF2XAsset(
        usd_path=usd_path,
        schema_path=schema_path,
        usd_sha256=usd_sha256,
        schema_sha256=schema_sha256,
    )


def inspect_verified_cf2x_structure(asset: VerifiedCF2XAsset) -> dict[str, object]:
    """Check USD structure when the optional USD runtime is available.

    This complements digest locking.  It is deliberately not a core package
    dependency because normal generator, evaluator, and adapter tests run
    without Isaac or ``pxr`` installed.
    """

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
