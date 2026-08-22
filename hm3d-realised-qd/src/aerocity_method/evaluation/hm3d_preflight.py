"""Fail-closed formal preflight for HM3D-derived multi-UAV experiments.

The module audits evidence; it does not launch Habitat, Isaac, training, or a
holdout run.  Unit tests can prove that malformed evidence is rejected, but
only artifacts explicitly produced by a real runtime may close runtime gates.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerocity_method.contracts import FORMAL_FLEET_SIZE
from aerocity_method.contracts.io import (
    canonical_sha256,
    finite_number,
    read_json_object,
    require_identifier,
    require_sha256,
)
from aerocity_method.evaluation.hm3d_exploration_contract import (
    load_exploration_observation_contract,
)
from aerocity_method.evaluation.hm3d_exploration_metrics import (
    evaluation_denominator_sha256,
)
from aerocity_method.runtime.hm3d_realised_qd import (
    HM3D_REALISED_QD_ARCHIVE_SPEC,
    HM3D_REALISED_QD_SCHEMA_VERSION,
    MAXIMUM_REALISED_QD_AXIS_ABSOLUTE_CORRELATION,
    MINIMUM_REALISED_QD_AXIS_CORRELATION_DETERMINANT,
    MINIMUM_REALISED_QD_JOINT_CELLS,
    MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION,
    MINIMUM_REALISED_QD_SHANNON_EFFECTIVE_CELLS,
)
from aerocity_method.runtime.sensors import (
    FORMAL_H15_SENSOR_PILOT_MODES,
    SensorProfile,
    SensorThroughputRecord,
    audit_sensor_throughput_pilot,
)

PREFLIGHT_PROTOCOL_SCHEMA_VERSION = "hm3d-formal-preflight-protocol-v4"
PREFLIGHT_EVIDENCE_SCHEMA_VERSION = "hm3d-formal-preflight-evidence-v1"
PREFLIGHT_ARTIFACT_SCHEMA_VERSION = "hm3d-formal-preflight-artifact-v1"
PRIMARY_METRIC = "Explored-Free-Flight-Volume-AUC_time"
METHOD_CORE = "realised-QD+OGFR-enhanced-RB-SF-SAC"
METHOD_CORE_ALIASES = {
    "realised-QD+RFG-enhanced-RB-SF-SAC": METHOD_CORE,
}
MECHANISM_VARIANT_ALIASES = {
    "no_rfg": "no_ogfr",
    "rfg": "ogfr",
    "realised_qd_rfg_rb_sf_sac": "realised_qd_ogfr_rb_sf_sac",
}

PHASE_SPECS = (
    ("P01", "asset_lock", "source_license_audit"),
    ("P02", "runtime_aba_reset", "real_runtime"),
    ("P03", "flight_space_3d", "real_runtime"),
    ("P04", "public_observation_contract", "real_runtime"),
    ("P05", "scene_split_freeze", "source_license_audit"),
    ("P06", "sensor_h15_pilot", "real_runtime"),
    ("P07", "task_validity_matrix", "real_runtime"),
    ("P08", "mechanism_pilot", "real_runtime"),
    ("P09", "formal_protocol_freeze", "protocol_audit"),
    ("P10", "formal_holdout_matrix", "real_runtime"),
)
PHASE_IDS = tuple(row[0] for row in PHASE_SPECS)
TASK_VALIDITY_METHODS = (
    "random",
    "frontier_3d",
    "auction",
)
MECHANISM_VARIANTS = (
    "no_qd",
    "planned_qd",
    "realised_qd",
    "no_ogfr",
    "ogfr",
    "rb_sf_sac_reference",
    "rb_sf_sac_selected",
)
FORMAL_MATRIX_METHODS = (
    *TASK_VALIDITY_METHODS,
    "gvp_mrep_port",
    "single_rl",
    "realised_qd_ogfr_rb_sf_sac",
)
ALLOWED_FLIGHT_REPRESENTATIONS = (
    "voxel_esdf_3d",
    "mesh_distance_field_3d",
)
_SPLITS = frozenset({"train", "validation", "test"})
_FORBIDDEN_EVIDENCE_TOKENS = ("aerocity-bench", "synthetic", "fixture", "mock")


def _exact(payload: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        missing = sorted(expected - set(payload)) if isinstance(payload, dict) else sorted(expected)
        extra = sorted(set(payload) - expected) if isinstance(payload, dict) else []
        raise ValueError(f"{label} fields mismatch; missing={missing}, extra={extra}")


def _positive(value: Any, name: str) -> float:
    resolved = finite_number(value, name)
    if resolved <= 0.0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _nonnegative(value: Any, name: str) -> float:
    resolved = finite_number(value, name)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _one_sided_sign_test_p_value(wins: int, losses: int) -> float:
    """Return an exact, conservative paired sign-test p-value.

    Ties have no directional evidence and are excluded.  This deliberately
    simple test makes the P08 gate auditable without assuming normal AUC
    differences or trusting a reported p-value that cannot be recomputed from
    the paired records.
    """

    observations = wins + losses
    if observations < 1:
        return 1.0
    return sum(math.comb(observations, positive) for positive in range(wins, observations + 1)) / (
        2**observations
    )


def _identifier_list(value: Any, name: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    rows = tuple(require_identifier(row, name) for row in value)
    if nonempty and not rows:
        raise ValueError(f"{name} cannot be empty")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{name} must contain unique values")
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_measured_file(base: Path, raw_path: Any, expected_hash: Any, name: str) -> Path:
    require_identifier(raw_path, f"{name}.path")
    require_sha256(expected_hash, f"{name}.sha256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base / path).resolve()
    lowered = str(path).replace("\\", "/").casefold()
    if "aerocity-bench" in lowered:
        raise ValueError(f"{name} must not access AeroCityBench")
    if not path.is_file():
        raise ValueError(f"{name} file is missing")
    if _file_sha256(path) != expected_hash:
        raise ValueError(f"{name} hash mismatch")
    return path


def _require_real_identity(payload: dict[str, Any]) -> None:
    if payload.get("evidence_class") != "real_runtime_measurement":
        raise ValueError("runtime artifact must declare real_runtime_measurement")
    run_id = require_identifier(payload.get("runtime_run_id"), "runtime_run_id")
    command_sha256 = payload.get("runtime_command_sha256")
    require_sha256(command_sha256, "runtime_command_sha256")
    lowered = run_id.casefold()
    if any(token in lowered for token in _FORBIDDEN_EVIDENCE_TOKENS[1:]):
        raise ValueError("synthetic/mock/fixture runtime identity is forbidden")


@dataclass(frozen=True, slots=True)
class PreflightPhaseRequirement:
    phase_id: str
    kind: str
    required_origin: str

    def __post_init__(self) -> None:
        require_identifier(self.phase_id, "phase_id")
        require_identifier(self.kind, "kind")
        require_identifier(self.required_origin, "required_origin")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreflightPhaseRequirement:
        _exact(payload, {"phase_id", "kind", "required_origin"}, "phase requirement")
        return cls(**payload)

    def to_dict(self) -> dict[str, str]:
        return {
            "phase_id": self.phase_id,
            "kind": self.kind,
            "required_origin": self.required_origin,
        }


@dataclass(frozen=True, slots=True)
class HM3DFormalPreflightProtocol:
    dataset: str
    task_interface: str
    method_core: str
    phases: tuple[PreflightPhaseRequirement, ...]
    fleet_size: int
    sensor_pilot_modes: tuple[str, ...]
    task_validity_methods: tuple[str, ...]
    mechanism_variants: tuple[str, ...]
    formal_matrix_methods: tuple[str, ...]
    primary_metric: str
    split_policy: str
    allowed_flight_representations: tuple[str, ...]
    allowed_evidence_origins: tuple[str, ...]
    three_d_thresholds: dict[str, float]
    aerocity_bench_mode: str
    status: str
    schema_version: str = PREFLIGHT_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PREFLIGHT_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("HM3D preflight protocol schema mismatch")
        if self.dataset != "HM3D":
            raise ValueError("preflight dataset must be HM3D")
        if self.task_interface != "hm3d-derived-multi-uav-exploration-3d":
            raise ValueError("HM3D task interface mismatch")
        if self.method_core != METHOD_CORE:
            raise ValueError("QD/OGFR/RB-SF-SAC method core cannot be removed or renamed")
        actual_phases = tuple((row.phase_id, row.kind, row.required_origin) for row in self.phases)
        if actual_phases != PHASE_SPECS:
            raise ValueError("preflight phases must exactly cover P01 through P10")
        if self.fleet_size != FORMAL_FLEET_SIZE:
            raise ValueError(f"formal fleet size must be N={FORMAL_FLEET_SIZE}")
        if self.sensor_pilot_modes != FORMAL_H15_SENSOR_PILOT_MODES:
            raise ValueError("formal H15 must retain the camera-free physics/range pair")
        if self.task_validity_methods != TASK_VALIDITY_METHODS:
            raise ValueError("task validity baseline matrix is incomplete")
        if self.mechanism_variants != MECHANISM_VARIANTS:
            raise ValueError("QD/OGFR/RB-SF-SAC mechanism matrix is incomplete")
        if self.formal_matrix_methods != FORMAL_MATRIX_METHODS:
            raise ValueError("formal method matrix is incomplete")
        if self.primary_metric != PRIMARY_METRIC:
            raise ValueError("primary metric must remain explored free-flight volume AUC over time")
        if self.split_policy != "official_scene_split_no_overlap":
            raise ValueError("HM3D scene split policy mismatch")
        if self.allowed_flight_representations != ALLOWED_FLIGHT_REPRESENTATIONS:
            raise ValueError("only explicit 3D flight-space representations are allowed")
        if set(self.allowed_evidence_origins) != {
            "real_runtime",
            "source_license_audit",
            "protocol_audit",
        }:
            raise ValueError("synthetic evidence cannot close an HM3D preflight phase")
        _exact(
            self.three_d_thresholds,
            {
                "min_vertical_span_m",
                "min_connected_height_bands",
                "min_vertical_opportunity_fraction",
                "min_fixed_altitude_relative_auc_gain",
            },
            "three_d_thresholds",
        )
        for name, value in self.three_d_thresholds.items():
            _nonnegative(value, f"three_d_thresholds.{name}")
        if self.three_d_thresholds["min_connected_height_bands"] < 2:
            raise ValueError("3D admission needs at least two connected height bands")
        if self.aerocity_bench_mode != "read_only":
            raise ValueError("AeroCityBench must remain read_only")
        if self.status != "protocol_only":
            raise ValueError("repository protocol cannot claim a runtime result")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HM3DFormalPreflightProtocol:
        expected = {
            "schema_version",
            "dataset",
            "task_interface",
            "method_core",
            "phases",
            "fleet_size",
            "sensor_pilot_modes",
            "task_validity_methods",
            "mechanism_variants",
            "formal_matrix_methods",
            "primary_metric",
            "split_policy",
            "allowed_flight_representations",
            "allowed_evidence_origins",
            "three_d_thresholds",
            "aerocity_bench_mode",
            "status",
        }
        _exact(payload, expected, "HM3D formal preflight protocol")
        raw_phases = payload["phases"]
        if not isinstance(raw_phases, list):
            raise ValueError("preflight phases must be a list")
        thresholds = payload["three_d_thresholds"]
        if not isinstance(thresholds, dict):
            raise ValueError("three_d_thresholds must be an object")
        return cls(
            schema_version=str(payload["schema_version"]),
            dataset=str(payload["dataset"]),
            task_interface=str(payload["task_interface"]),
            method_core=METHOD_CORE_ALIASES.get(
                str(payload["method_core"]), str(payload["method_core"])
            ),
            phases=tuple(PreflightPhaseRequirement.from_dict(row) for row in raw_phases),
            fleet_size=_integer(payload["fleet_size"], "fleet_size", minimum=1),
            sensor_pilot_modes=tuple(payload["sensor_pilot_modes"]),
            task_validity_methods=tuple(payload["task_validity_methods"]),
            mechanism_variants=tuple(
                MECHANISM_VARIANT_ALIASES.get(variant, variant)
                for variant in payload["mechanism_variants"]
            ),
            formal_matrix_methods=tuple(
                MECHANISM_VARIANT_ALIASES.get(variant, variant)
                for variant in payload["formal_matrix_methods"]
            ),
            primary_metric=str(payload["primary_metric"]),
            split_policy=str(payload["split_policy"]),
            allowed_flight_representations=tuple(payload["allowed_flight_representations"]),
            allowed_evidence_origins=tuple(payload["allowed_evidence_origins"]),
            three_d_thresholds={key: float(value) for key, value in thresholds.items()},
            aerocity_bench_mode=str(payload["aerocity_bench_mode"]),
            status=str(payload["status"]),
        )

    @property
    def protocol_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "task_interface": self.task_interface,
            "method_core": self.method_core,
            "phases": [row.to_dict() for row in self.phases],
            "fleet_size": self.fleet_size,
            "sensor_pilot_modes": self.sensor_pilot_modes,
            "task_validity_methods": self.task_validity_methods,
            "mechanism_variants": self.mechanism_variants,
            "formal_matrix_methods": self.formal_matrix_methods,
            "primary_metric": self.primary_metric,
            "split_policy": self.split_policy,
            "allowed_flight_representations": self.allowed_flight_representations,
            "allowed_evidence_origins": self.allowed_evidence_origins,
            "three_d_thresholds": self.three_d_thresholds,
            "aerocity_bench_mode": self.aerocity_bench_mode,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class PreflightEvidenceArtifact:
    phase_id: str
    kind: str
    origin: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        for name in ("phase_id", "kind", "origin", "path"):
            require_identifier(getattr(self, name), name)
        require_sha256(self.sha256, "artifact sha256")
        lowered = self.path.replace("\\", "/").casefold()
        if "aerocity-bench" in lowered:
            raise ValueError("preflight evidence must not access AeroCityBench")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreflightEvidenceArtifact:
        _exact(payload, {"phase_id", "kind", "origin", "path", "sha256"}, "artifact")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class HM3DFormalPreflightEvidence:
    protocol_hash: str
    requested_gate: str
    method_core: str
    aerocity_bench_accesses: tuple[str, ...]
    artifacts: tuple[PreflightEvidenceArtifact, ...]
    schema_version: str = PREFLIGHT_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PREFLIGHT_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("HM3D preflight evidence schema mismatch")
        require_sha256(self.protocol_hash, "protocol_hash")
        if self.requested_gate not in {"formal_experiment_start", "formal_results"}:
            raise ValueError("requested_gate must be formal_experiment_start or formal_results")
        if self.method_core != METHOD_CORE:
            raise ValueError("evidence method core does not match the fixed mainline")
        if self.aerocity_bench_accesses:
            raise ValueError("AeroCityBench access is forbidden in HM3D preflight")
        phases = [row.phase_id for row in self.artifacts]
        if len(phases) != len(set(phases)):
            raise ValueError("preflight evidence contains duplicate phase artifacts")
        if any(phase not in PHASE_IDS for phase in phases):
            raise ValueError("preflight evidence contains an unknown phase")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HM3DFormalPreflightEvidence:
        expected = {
            "schema_version",
            "protocol_hash",
            "requested_gate",
            "method_core",
            "aerocity_bench_accesses",
            "artifacts",
        }
        _exact(payload, expected, "HM3D formal preflight evidence")
        if not isinstance(payload["artifacts"], list):
            raise ValueError("preflight artifacts must be a list")
        return cls(
            schema_version=str(payload["schema_version"]),
            protocol_hash=str(payload["protocol_hash"]),
            requested_gate=str(payload["requested_gate"]),
            method_core=METHOD_CORE_ALIASES.get(
                str(payload["method_core"]), str(payload["method_core"])
            ),
            aerocity_bench_accesses=tuple(payload["aerocity_bench_accesses"]),
            artifacts=tuple(
                PreflightEvidenceArtifact.from_dict(row) for row in payload["artifacts"]
            ),
        )


def _failure_denominator(row: dict[str, Any], label: str) -> None:
    names = ("planned", "executed", "failed", "timeout", "oom", "other_failed")
    values = {name: _integer(row.get(name), f"{label}.{name}") for name in names}
    if values["planned"] < 1:
        raise ValueError(f"{label} must contain at least one planned episode")
    denominator = sum(values[name] for name in names if name != "planned")
    if denominator != values["planned"]:
        raise ValueError(f"{label} failure denominator is incomplete")


def _validate_p01(payload: dict[str, Any], base: Path, context: dict[str, Any]) -> None:
    _exact(
        payload,
        {
            "evidence_class",
            "dataset_version",
            "license_id",
            "license_record_path",
            "license_sha256",
            "source_url",
            "raw_assets_redistributed",
            "repository_included",
            "conversion_tool",
            "scenes",
        },
        "P01 payload",
    )
    if payload["evidence_class"] != "source_license_audit":
        raise ValueError("P01 must be a source/license audit")
    for name in ("dataset_version", "license_id", "source_url"):
        require_identifier(payload[name], name)
    if payload["raw_assets_redistributed"] is not False:
        raise ValueError("HM3D raw/converted assets must not be redistributed")
    if payload["repository_included"] is not False:
        raise ValueError("HM3D assets must not be committed in this repository")
    _resolve_measured_file(
        base, payload["license_record_path"], payload["license_sha256"], "license_record"
    )
    tool = payload["conversion_tool"]
    _exact(tool, {"tool_id", "version", "sha256"}, "conversion_tool")
    require_identifier(tool["tool_id"], "conversion tool ID")
    require_identifier(tool["version"], "conversion tool version")
    require_sha256(tool["sha256"], "conversion tool hash")
    scenes = payload["scenes"]
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("P01 needs at least one official scene")
    scene_map: dict[str, dict[str, Any]] = {}
    for index, scene in enumerate(scenes):
        _exact(
            scene,
            {"scene_id", "split", "asset_origin", "path", "sha256"},
            f"P01 scene[{index}]",
        )
        scene_id = require_identifier(scene["scene_id"], "scene_id")
        if any(token in scene_id.casefold() for token in _FORBIDDEN_EVIDENCE_TOKENS[1:]):
            raise ValueError("synthetic/mock/fixture scene cannot close P01")
        if scene_id in scene_map:
            raise ValueError("P01 scene IDs must be unique")
        if scene["split"] not in _SPLITS:
            raise ValueError("P01 scene split is unsupported")
        if scene["asset_origin"] != "official_hm3d":
            raise ValueError("P01 accepts only official HM3D assets")
        _resolve_measured_file(base, scene["path"], scene["sha256"], f"scene {scene_id}")
        scene_map[scene_id] = scene
    if {scene["split"] for scene in scenes} != _SPLITS:
        raise ValueError("P01 scene manifest must contain train, validation and test scenes")
    context["dataset_version"] = payload["dataset_version"]
    context["scenes"] = scene_map


def _validate_p02(payload: dict[str, Any], context: dict[str, Any]) -> None:
    _require_real_identity(payload)
    _exact(
        payload,
        {
            "evidence_class",
            "runtime_run_id",
            "runtime_command_sha256",
            "length_unit_m",
            "source_up_axis",
            "runtime_up_axis",
            "coordinate_transform_sha256",
            "gravity_m_s2",
            "vehicle_envelope_m",
            "simulator_id",
            "simulator_version",
            "controller_sha256",
            "dynamics_sha256",
            "vehicle_collider_sha256",
            "aba_reset",
        },
        "P02 payload",
    )
    if not math.isclose(finite_number(payload["length_unit_m"], "length_unit_m"), 1.0):
        raise ValueError("HM3D runtime length unit must be meters")
    if payload["source_up_axis"] not in {"Y", "Z"} or payload["runtime_up_axis"] not in {
        "Y",
        "Z",
    }:
        raise ValueError("runtime up axis must be explicitly Y or Z")
    for name in (
        "coordinate_transform_sha256",
        "controller_sha256",
        "dynamics_sha256",
        "vehicle_collider_sha256",
    ):
        require_sha256(payload[name], name)
    gravity = payload["gravity_m_s2"]
    if not isinstance(gravity, list) or len(gravity) != 3:
        raise ValueError("gravity_m_s2 must be a three-vector")
    magnitude = math.sqrt(sum(finite_number(value, "gravity") ** 2 for value in gravity))
    if not 9.0 <= magnitude <= 10.0:
        raise ValueError("runtime gravity magnitude is not physically plausible")
    envelope = payload["vehicle_envelope_m"]
    if not isinstance(envelope, list) or len(envelope) != 3:
        raise ValueError("vehicle_envelope_m must be a three-vector")
    if any(_positive(value, "vehicle_envelope_m") <= 0.0 for value in envelope):
        raise ValueError("vehicle envelope must be positive")
    require_identifier(payload["simulator_id"], "simulator_id")
    require_identifier(payload["simulator_version"], "simulator_version")
    reset = payload["aba_reset"]
    _exact(
        reset,
        {
            "scene_a_id",
            "scene_b_id",
            "a1_fingerprint",
            "b_fingerprint",
            "a2_fingerprint",
            "components",
            "passed",
        },
        "A-B-A reset",
    )
    if reset["scene_a_id"] == reset["scene_b_id"]:
        raise ValueError("A-B-A reset needs two distinct scenes")
    scenes = context.get("scenes", {})
    if reset["scene_a_id"] not in scenes or reset["scene_b_id"] not in scenes:
        raise ValueError("A-B-A reset scenes are not locked by P01")
    for name in ("a1_fingerprint", "b_fingerprint", "a2_fingerprint"):
        require_sha256(reset[name], name)
    if reset["a1_fingerprint"] != reset["a2_fingerprint"]:
        raise ValueError("A1 and A2 reset fingerprints differ")
    if reset["b_fingerprint"] == reset["a1_fingerprint"]:
        raise ValueError("B reset fingerprint is not independent from A")
    required_components = {
        "scene",
        "collider",
        "contact",
        "sensor",
        "rng",
        "controller",
        "reset_state",
    }
    if set(_identifier_list(reset["components"], "reset components")) != required_components:
        raise ValueError("A-B-A reset fingerprint components are incomplete")
    if reset["passed"] is not True:
        raise ValueError("A-B-A reset did not pass")
    context["controller_sha256"] = payload["controller_sha256"]
    context["dynamics_sha256"] = payload["dynamics_sha256"]


def _validate_p03(
    payload: dict[str, Any], protocol: HM3DFormalPreflightProtocol, context: dict[str, Any]
) -> None:
    _require_real_identity(payload)
    _exact(
        payload,
        {
            "evidence_class",
            "runtime_run_id",
            "runtime_command_sha256",
            "navmesh_authorizes_flight",
            "admission_scope",
            "scenes",
        },
        "P03 payload",
    )
    if payload["navmesh_authorizes_flight"] is not False:
        raise ValueError("Habitat 2D navmesh cannot authorize UAV free flight")
    if payload["admission_scope"] != "stratified_development_cohort":
        raise ValueError("P03 must declare a stratified development cohort")
    rows = payload["scenes"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("P03 needs per-scene 3D evidence")
    locked = context.get("scenes", {})
    development_scene_ids = {
        scene_id for scene_id, scene in locked.items() if scene["split"] in {"train", "validation"}
    }
    cohort_scene_ids = {row.get("scene_id") for row in rows}
    if not cohort_scene_ids <= development_scene_ids:
        raise ValueError("P03 cohort contains a test or unknown scene")
    if len(rows) != len(cohort_scene_ids):
        raise ValueError("P03 contains duplicate scene rows")
    cohort_splits = {locked[scene_id]["split"] for scene_id in cohort_scene_ids}
    if cohort_splits != {"train", "validation"}:
        raise ValueError("P03 cohort must contain both train and validation scenes")
    thresholds = protocol.three_d_thresholds
    flight_hashes: dict[str, str] = {}
    for index, row in enumerate(rows):
        _exact(
            row,
            {
                "scene_id",
                "source_geometry_sha256",
                "flight_space_manifest_hash",
                "representation",
                "dimension",
                "resolution_m",
                "collision_geometry_sha256",
                "free_flight_validated",
                "generator_version",
                "vehicle_clearance_m",
                "vertical_span_m",
                "free_flight_volume_m3",
                "connected_height_band_count",
                "vertical_opportunity_fraction",
                "fixed_altitude_control_run",
                "fixed_altitude_control_delta",
                "fixed_altitude_control_relative_gain",
                "vertical_counterfactual_sha256",
                "collision_replay_passed",
                "flight_space_evidence_sha256",
                "collision_replay_evidence_sha256",
                "collision_derivative_sha256",
            },
            f"P03 scene[{index}]",
        )
        scene_id = row["scene_id"]
        if row["source_geometry_sha256"] != locked[scene_id]["sha256"]:
            raise ValueError("P03 source geometry hash does not match P01")
        require_sha256(row["flight_space_manifest_hash"], "flight_space_manifest_hash")
        require_sha256(row["collision_geometry_sha256"], "collision_geometry_sha256")
        if row["representation"] not in protocol.allowed_flight_representations:
            raise ValueError("P03 representation is not a 3D occupancy/distance field")
        if row["dimension"] != 3 or row["free_flight_validated"] is not True:
            raise ValueError("P03 flight space is not validated three-dimensional free flight")
        _positive(row["resolution_m"], "resolution_m")
        _positive(row["vehicle_clearance_m"], "vehicle_clearance_m")
        _positive(row["free_flight_volume_m3"], "free_flight_volume_m3")
        if row["vertical_span_m"] < thresholds["min_vertical_span_m"]:
            raise ValueError("P03 vertical span is below the frozen threshold")
        if row["connected_height_band_count"] < thresholds["min_connected_height_bands"]:
            raise ValueError("P03 connected height bands are insufficient")
        if row["vertical_opportunity_fraction"] < thresholds["min_vertical_opportunity_fraction"]:
            raise ValueError("P03 lacks vertical observation pressure")
        if (
            row["fixed_altitude_control_run"] is not True
            or row["fixed_altitude_control_delta"] <= 0.0
            or row["fixed_altitude_control_relative_gain"]
            < thresholds["min_fixed_altitude_relative_auc_gain"]
        ):
            raise ValueError("P03 lacks a material fixed-altitude counterfactual")
        require_sha256(row["vertical_counterfactual_sha256"], "vertical_counterfactual_sha256")
        if row["collision_replay_passed"] is not True:
            raise ValueError("P03 collision replay failed")
        for name in (
            "flight_space_evidence_sha256",
            "collision_replay_evidence_sha256",
            "collision_derivative_sha256",
        ):
            require_sha256(row[name], f"P03 {name}")
        require_identifier(row["generator_version"], "flight-space generator version")
        flight_hashes[scene_id] = row["flight_space_manifest_hash"]
    context["flight_hashes"] = flight_hashes
    context["p03_cohort_scene_ids"] = frozenset(flight_hashes)
    context["p03_evaluation_denominator_sha256"] = evaluation_denominator_sha256(tuple(rows))


def _validate_p04(payload: dict[str, Any], context: dict[str, Any]) -> None:
    _require_real_identity(payload)
    _exact(
        payload,
        {
            "evidence_class",
            "runtime_run_id",
            "runtime_command_sha256",
            "public_contract_sha256",
            "evaluation_denominator_sha256",
            "split_manifest_sha256",
            "episodes",
        },
        "P04 payload",
    )
    require_sha256(payload["public_contract_sha256"], "public_contract_sha256")
    frozen_contract = load_exploration_observation_contract()
    if payload["public_contract_sha256"] != frozen_contract.digest:
        raise ValueError("P04 public observation contract hash does not match the frozen contract")
    require_sha256(payload["evaluation_denominator_sha256"], "evaluation_denominator_sha256")
    if payload["evaluation_denominator_sha256"] != context.get("p03_evaluation_denominator_sha256"):
        raise ValueError("P04 evaluator denominator does not match the frozen P03 cohort")
    require_sha256(payload["split_manifest_sha256"], "split_manifest_sha256")
    episodes = payload["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("P04 needs measured public-observation episodes")
    scene_ids: set[str] = set()
    for index, row in enumerate(episodes):
        _exact(
            row,
            {
                "episode_id",
                "scene_id",
                "source_geometry_sha256",
                "flight_space_manifest_hash",
                "source_observation_ids_total",
                "observed_free_voxels_total",
                "observation_voxel_resolution_m",
                "source_observation_binding",
                "method_private_truth_fields",
            },
            f"P04 episode[{index}]",
        )
        require_identifier(row["episode_id"], "episode_id")
        scene_id = require_identifier(row["scene_id"], "scene_id")
        locked = context.get("scenes", {})
        if scene_id not in locked:
            raise ValueError("P04 episode scene is not locked by P01")
        if locked[scene_id]["split"] == "test":
            raise ValueError("P04 development opportunity calibration cannot inspect test scenes")
        if row["source_geometry_sha256"] != locked[scene_id]["sha256"]:
            raise ValueError("P04 scene hash does not match P01")
        if row["flight_space_manifest_hash"] != context.get("flight_hashes", {}).get(scene_id):
            raise ValueError("P04 flight-space hash does not match P03")
        _integer(row["source_observation_ids_total"], "source observations", minimum=1)
        if _integer(row["observed_free_voxels_total"], "observed free voxels") < 1:
            raise ValueError("P04 must contain real public observations")
        _positive(row["observation_voxel_resolution_m"], "observation voxel resolution")
        if row["source_observation_binding"] is not True:
            raise ValueError("P04 observations must bind source_observation_id")
        if row["method_private_truth_fields"] != []:
            raise ValueError("method-visible evaluator truth is forbidden")
        scene_ids.add(scene_id)
    if scene_ids != set(context.get("p03_cohort_scene_ids", ())):
        raise ValueError("P04 must cover the P03 stratified development cohort exactly")
    context["public_contract_sha256"] = payload["public_contract_sha256"]
    context["evaluation_denominator_sha256"] = payload["evaluation_denominator_sha256"]
    context["p04_split_manifest_sha256"] = payload["split_manifest_sha256"]


def _validate_p05(payload: dict[str, Any], context: dict[str, Any]) -> None:
    _exact(
        payload,
        {
            "evidence_class",
            "official_split_provenance",
            "dataset_version",
            "scene_assignments",
            "split_manifest_sha256",
            "public_contract_sha256",
            "evaluation_denominator_sha256",
            "episode_seed_manifest_sha256",
            "difficulty_distribution_sha256",
            "run_partition",
            "test_used_for_development",
            "test_access_count_before_freeze",
        },
        "P05 payload",
    )
    if payload["evidence_class"] != "source_license_audit":
        raise ValueError("P05 must be a source split audit")
    require_identifier(payload["official_split_provenance"], "official split provenance")
    if payload["dataset_version"] != context.get("dataset_version"):
        raise ValueError("P05 dataset version does not match P01")
    assignments = payload["scene_assignments"]
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("P05 scene assignments cannot be empty")
    expected = [
        {
            "scene_id": scene_id,
            "split": scene["split"],
            "asset_sha256": scene["sha256"],
        }
        for scene_id, scene in sorted(context.get("scenes", {}).items())
    ]
    if assignments != expected:
        raise ValueError("P05 assignments must exactly match P01 scene IDs, splits and hashes")
    computed = canonical_sha256(assignments)
    if payload["split_manifest_sha256"] != computed:
        raise ValueError("P05 split manifest hash is not canonical")
    if payload["split_manifest_sha256"] != context.get("p04_split_manifest_sha256"):
        raise ValueError("P05 split hash does not match P04 public contract")
    if payload["public_contract_sha256"] != context.get("public_contract_sha256"):
        raise ValueError("P05 public observation contract hash drifted from P04")
    if payload["evaluation_denominator_sha256"] != context.get("evaluation_denominator_sha256"):
        raise ValueError("P05 evaluator denominator hash drifted from P04")
    for name in ("episode_seed_manifest_sha256", "difficulty_distribution_sha256"):
        require_sha256(payload[name], name)
    if payload["run_partition"] != "development":
        raise ValueError("P05 preflight must remain in the development partition")
    if payload["test_used_for_development"] is not False:
        raise ValueError("test scenes cannot be used for development")
    if payload["test_access_count_before_freeze"] != 0:
        raise ValueError("test scenes were accessed before protocol freeze")
    context["split_manifest_sha256"] = computed


def _validate_p06(
    payload: dict[str, Any], protocol: HM3DFormalPreflightProtocol, context: dict[str, Any]
) -> None:
    _require_real_identity(payload)
    _exact(
        payload,
        {
            "evidence_class",
            "runtime_run_id",
            "runtime_command_sha256",
            "source_observation_binding",
            "selection_partition",
            "records",
            "selected_profile",
            "entitlements",
        },
        "P06 payload",
    )
    if payload["source_observation_binding"] is not True:
        raise ValueError("H15 sensing must preserve source observation outcomes")
    if payload["selection_partition"] not in {"train", "validation"}:
        raise ValueError("H15 sensor selection may only use train/validation")
    if not isinstance(payload["records"], list):
        raise ValueError("P06 records must be a list")
    records = tuple(SensorThroughputRecord.from_dict(row) for row in payload["records"])
    pilot = audit_sensor_throughput_pilot(records, modes=FORMAL_H15_SENSOR_PILOT_MODES)
    if pilot["status"] != "PASS":
        raise ValueError(f"H15 pilot is incomplete: {pilot['reasons']}")
    selected_payload = payload["selected_profile"]
    if not isinstance(selected_payload, dict):
        raise ValueError("selected_profile must be an object")
    selected = SensorProfile.from_dict(selected_payload)
    if selected.mode == "physics_only":
        raise ValueError("physics_only cannot be the formal search sensor")
    matching = [row for row in records if row.profile.entitlement_hash == selected.entitlement_hash]
    if {row.fleet_size for row in matching} != {protocol.fleet_size}:
        raise ValueError(f"selected sensor profile was not measured for N={protocol.fleet_size}")
    expected_methods = set(protocol.formal_matrix_methods) | set(protocol.mechanism_variants)
    entitlements = payload["entitlements"]
    if not isinstance(entitlements, list):
        raise ValueError("P06 entitlements must be a list")
    method_ids: set[str] = set()
    for row in entitlements:
        _exact(row, {"method_id", "profile_hash"}, "P06 entitlement")
        method_id = require_identifier(row["method_id"], "entitlement method_id")
        if method_id in method_ids:
            raise ValueError("P06 entitlement methods must be unique")
        if row["profile_hash"] != selected.entitlement_hash:
            raise ValueError("methods have unequal sensor entitlements")
        method_ids.add(method_id)
    if method_ids != expected_methods:
        raise ValueError("P06 entitlements do not cover every compared method")
    context["sensor_profile_sha256"] = selected.entitlement_hash


def _validate_p07(
    payload: dict[str, Any], protocol: HM3DFormalPreflightProtocol, context: dict[str, Any]
) -> None:
    _require_real_identity(payload)
    _exact(
        payload,
        {
            "evidence_class",
            "runtime_run_id",
            "runtime_command_sha256",
            "partition",
            "budget_sha256",
            "sensor_profile_sha256",
            "public_contract_sha256",
            "evaluation_denominator_sha256",
            "evaluation_geometry_denominator_sha256",
            "primary_metric",
            "rows",
            "task_validity_passed",
        },
        "P07 payload",
    )
    if payload["partition"] not in {"train", "validation"}:
        raise ValueError("task validity matrix cannot use test scenes")
    require_sha256(payload["budget_sha256"], "P07 budget hash")
    if payload["sensor_profile_sha256"] != context.get("sensor_profile_sha256"):
        raise ValueError("P07 sensor profile differs from H15 selection")
    if payload["public_contract_sha256"] != context.get("public_contract_sha256"):
        raise ValueError("P07 public observation contract differs from P04")
    require_sha256(payload["evaluation_denominator_sha256"], "P07 episode evaluator denominator")
    require_sha256(
        payload["evaluation_geometry_denominator_sha256"],
        "P07 geometry evaluator denominator",
    )
    if (
        payload["evaluation_geometry_denominator_sha256"]
        != context.get("evaluation_denominator_sha256")
    ):
        raise ValueError("P07 geometry evaluator denominator differs from P04")
    if payload["primary_metric"] != protocol.primary_metric:
        raise ValueError("P07 primary metric mismatch")
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != len(protocol.task_validity_methods):
        raise ValueError("P07 task validity matrix is incomplete")
    by_method: dict[str, dict[str, Any]] = {}
    for row in rows:
        required = {
            "method_id",
            "deployed",
            "reads_private_truth",
            "oracle_only",
            "ranked",
            "budget_sha256",
            "sensor_profile_sha256",
            "public_contract_sha256",
            "evaluation_denominator_sha256",
            "evaluation_geometry_denominator_sha256",
            "planned",
            "executed",
            "failed",
            "timeout",
            "oom",
            "other_failed",
            "explored_free_flight_volume_auc_time",
            "final_coverage_at_budget",
            "collision_count",
            "communication_failure_count",
            "energy_used_j",
        }
        _exact(row, required, "P07 method row")
        method_id = require_identifier(row["method_id"], "P07 method_id")
        if method_id in by_method:
            raise ValueError("P07 method rows must be unique")
        if row["deployed"] is not True:
            raise ValueError("P07 cannot mark an undeployed baseline as completed")
        if row["reads_private_truth"] is not False or row["ranked"] is not True:
            raise ValueError("P07 ranked methods may only use public observations")
        if row["budget_sha256"] != payload["budget_sha256"]:
            raise ValueError("P07 methods have unequal budgets")
        if row["sensor_profile_sha256"] != payload["sensor_profile_sha256"]:
            raise ValueError("P07 methods have unequal sensor entitlements")
        if row["public_contract_sha256"] != payload["public_contract_sha256"]:
            raise ValueError("P07 methods have unequal public observation contracts")
        if row["evaluation_denominator_sha256"] != payload["evaluation_denominator_sha256"]:
            raise ValueError("P07 methods have unequal reachable evaluator denominators")
        if (
            row["evaluation_geometry_denominator_sha256"]
            != payload["evaluation_geometry_denominator_sha256"]
        ):
            raise ValueError("P07 methods have unequal geometry evaluator denominators")
        _failure_denominator(row, f"P07 {method_id}")
        for name in ("explored_free_flight_volume_auc_time", "final_coverage_at_budget"):
            value = finite_number(row[name], name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("collision_count", "communication_failure_count"):
            _integer(row[name], f"P07 {name}")
        _nonnegative(row["energy_used_j"], "P07 energy_used_j")
        by_method[method_id] = row
    if tuple(by_method) != protocol.task_validity_methods:
        raise ValueError("P07 methods must use the frozen order and complete set")
    public_rows = list(by_method.values())
    if max(row["final_coverage_at_budget"] for row in public_rows) <= 0.0:
        raise ValueError("P07 has no public-method exploration progress")
    if by_method["random"]["final_coverage_at_budget"] >= 0.95:
        raise ValueError("P07 task is trivial under the random baseline")
    if payload["task_validity_passed"] is not True:
        raise ValueError("P07 task validity was not approved")
    context["budget_sha256"] = payload["budget_sha256"]
    context["p07_evaluation_geometry_denominator_sha256"] = payload[
        "evaluation_geometry_denominator_sha256"
    ]


def _validate_p08(
    payload: dict[str, Any], protocol: HM3DFormalPreflightProtocol, context: dict[str, Any]
) -> None:
    _require_real_identity(payload)
    _exact(
        payload,
        {
            "evidence_class",
            "runtime_run_id",
            "runtime_command_sha256",
            "partition",
            "budget_sha256",
            "sensor_profile_sha256",
            "public_contract_sha256",
            "evaluation_denominator_sha256",
            "primary_metric",
            "paired_independent_units",
            "rows",
            "selection_gain",
            "selection_regret",
            "qd_mechanism_evidence",
            "archive_build_time_s",
            "archive_bytes",
            "outcome_utilization",
            "negative_transfer_rate",
            "negative_transfer_limit",
            "calibration_error",
            "tuning_iterations",
            "adjustment_log",
            "selected_method_core_sha256",
            "mainline_components_removed",
            "recurrent_history_selector_exercised",
            "fragment_outcome_schema_sha256",
            "mechanism_pilot_passed",
        },
        "P08 payload",
    )
    if payload["partition"] != "validation":
        raise ValueError("P08 mechanism selection must use validation only")
    if payload["budget_sha256"] != context.get("budget_sha256"):
        raise ValueError("P08 budget differs from P07")
    if payload["sensor_profile_sha256"] != context.get("sensor_profile_sha256"):
        raise ValueError("P08 sensor profile differs from P06")
    if payload["public_contract_sha256"] != context.get("public_contract_sha256"):
        raise ValueError("P08 public observation contract differs from P04")
    if payload["evaluation_denominator_sha256"] != context.get("evaluation_denominator_sha256"):
        raise ValueError("P08 evaluator denominator differs from P04")
    if payload["primary_metric"] != protocol.primary_metric:
        raise ValueError("P08 primary metric mismatch")
    _integer(payload["paired_independent_units"], "paired_independent_units", minimum=12)
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != len(protocol.mechanism_variants):
        raise ValueError("P08 mechanism matrix is incomplete")
    by_variant: dict[str, dict[str, Any]] = {}
    for row in rows:
        _exact(
            row,
            {
                "variant_id",
                "configuration_sha256",
                "selector_backbone_sha256",
                "budget_sha256",
                "sensor_profile_sha256",
                "public_contract_sha256",
                "evaluation_denominator_sha256",
                "planned",
                "executed",
                "failed",
                "timeout",
                "oom",
                "other_failed",
                "explored_free_flight_volume_auc_time",
                "final_coverage_at_budget",
                "outcome_only_supervision",
                "fragment_outcome_count",
                "accepted_fragment_outcome_count",
                "outcome_gated_fragment_credit_count",
                "archive_effective_cells",
                "archive_coverage",
                "selector_history_mode",
            },
            "P08 variant row",
        )
        variant = require_identifier(row["variant_id"], "variant_id")
        variant = MECHANISM_VARIANT_ALIASES.get(variant, variant)
        if variant in by_variant:
            raise ValueError("P08 variant rows must be unique")
        require_sha256(row["configuration_sha256"], "configuration hash")
        require_sha256(row["selector_backbone_sha256"], "selector backbone hash")
        if row["budget_sha256"] != payload["budget_sha256"]:
            raise ValueError("P08 variants have unequal budgets")
        if row["sensor_profile_sha256"] != payload["sensor_profile_sha256"]:
            raise ValueError("P08 variants have unequal sensors")
        if row["public_contract_sha256"] != payload["public_contract_sha256"]:
            raise ValueError("P08 variants have unequal public observation contracts")
        if row["evaluation_denominator_sha256"] != payload["evaluation_denominator_sha256"]:
            raise ValueError("P08 variants have unequal evaluator denominators")
        _failure_denominator(row, f"P08 {variant}")
        auc = finite_number(row["explored_free_flight_volume_auc_time"], "mechanism AUC")
        if not 0.0 <= auc <= 1.0:
            raise ValueError("P08 AUC must be in [0, 1]")
        final_coverage = finite_number(row["final_coverage_at_budget"], "final coverage")
        if not 0.0 <= final_coverage <= 1.0:
            raise ValueError("P08 final coverage must be in [0, 1]")
        for name in (
            "fragment_outcome_count",
            "accepted_fragment_outcome_count",
            "outcome_gated_fragment_credit_count",
            "archive_effective_cells",
        ):
            _integer(row[name], f"P08 {variant} {name}")
        archive_coverage = finite_number(row["archive_coverage"], "archive coverage")
        if not 0.0 <= archive_coverage <= 1.0:
            raise ValueError("P08 archive coverage must be in [0, 1]")
        require_identifier(row["selector_history_mode"], "selector_history_mode")
        if variant in {"ogfr", "rb_sf_sac_selected"} and row["outcome_only_supervision"] is not True:
            raise ValueError("OGFR/RB-SF-SAC supervision must remain outcome grounded")
        by_variant[variant] = row
    if tuple(by_variant) != protocol.mechanism_variants:
        raise ValueError("P08 variants must use the frozen order and complete set")
    # P08 is a validation-side mechanism diagnostic, not a second formal
    # leaderboard.  It must execute every adjacent control under the common
    # contract, but it must not use an arbitrary validation gain as permission
    # to freeze the protocol.  Effect sizes remain in the immutable record and
    # are interpreted together with the frozen P10 holdout matrix.
    if payload["recurrent_history_selector_exercised"] is not True:
        raise ValueError("P08 recurrent history selector was not exercised")
    require_sha256(payload["fragment_outcome_schema_sha256"], "fragment outcome schema hash")
    if by_variant["rb_sf_sac_selected"]["selector_history_mode"] != "recurrent_public_outcomes":
        raise ValueError("selected mechanism did not use public outcome history")
    for variant in ("ogfr", "rb_sf_sac_selected"):
        row = by_variant[variant]
        if row["outcome_only_supervision"] is not True:
            raise ValueError("OGFR/RB-SF-SAC supervision must remain outcome grounded")
        if row["fragment_outcome_count"] < 1 or row["accepted_fragment_outcome_count"] < 1:
            raise ValueError("OGFR requires real FragmentOutcome evidence")
        if row["outcome_gated_fragment_credit_count"] < 1:
            raise ValueError("OGFR requires accepted outcome-gated credit transfer")
    if by_variant["realised_qd"]["archive_effective_cells"] < MINIMUM_REALISED_QD_JOINT_CELLS:
        raise ValueError("realised-QD archive has too few effective behaviour cells")
    if (
        by_variant["rb_sf_sac_selected"]["archive_effective_cells"]
        < MINIMUM_REALISED_QD_JOINT_CELLS
    ):
        raise ValueError("selected archive has too few effective behaviour cells")
    qd_evidence = payload["qd_mechanism_evidence"]
    if not isinstance(qd_evidence, dict):
        raise ValueError("P08 QD mechanism evidence must be an object")
    _exact(
        qd_evidence,
        {
            "descriptor_schema_version",
            "selector_backbone_sha256",
            "candidate_intent_admission",
            "train_descriptor_admission",
            "validation_outcome_schema",
            "paired_effect",
        },
        "P08 QD mechanism evidence",
    )
    if qd_evidence["descriptor_schema_version"] != HM3D_REALISED_QD_SCHEMA_VERSION:
        raise ValueError("P08 must use the current outcome-grounded QD descriptor schema")
    require_sha256(qd_evidence["selector_backbone_sha256"], "P08 QD selector backbone")
    for variant in ("no_qd", "planned_qd", "realised_qd"):
        if (
            by_variant[variant]["selector_backbone_sha256"]
            != qd_evidence["selector_backbone_sha256"]
        ):
            raise ValueError("P08 QD controls must use the identical candidate-value backbone")
    candidate_admission = qd_evidence["candidate_intent_admission"]
    if not isinstance(candidate_admission, dict):
        raise ValueError("P08 candidate-intent admission must be an object")
    _exact(
        candidate_admission,
        {
            "status",
            "assessed_pool_count",
            "admitted_pool_count",
            "minimum_feasible_candidates",
            "minimum_axis_bins",
            "minimum_joint_cells",
            "minimum_joint_shannon_effective_cells",
        },
        "P08 candidate-intent admission",
    )
    if candidate_admission["status"] != "QD_CANDIDATE_INTENT_ADMITTED":
        raise ValueError("P08 candidate pool has no admitted realised-QD intent repertoire")
    assessed_pools = _integer(
        candidate_admission["assessed_pool_count"], "P08 assessed pools", minimum=1
    )
    admitted_pools = _integer(
        candidate_admission["admitted_pool_count"], "P08 admitted pools", minimum=1
    )
    if admitted_pools != assessed_pools:
        raise ValueError("P08 may not omit unadmitted candidate pools from the QD comparison")
    if (
        _integer(
            candidate_admission["minimum_feasible_candidates"],
            "P08 candidate intent minimum_feasible_candidates",
            minimum=6,
        )
        < 6
    ):
        raise ValueError("P08 QD needs at least six legal candidate modes per decision")
    _integer(
        candidate_admission["minimum_axis_bins"],
        "P08 candidate intent minimum_axis_bins",
        minimum=2,
    )
    if (
        _integer(
            candidate_admission["minimum_joint_cells"],
            "P08 candidate intent minimum_joint_cells",
            minimum=6,
        )
        < 6
    ):
        raise ValueError("P08 QD candidate repertoire needs at least six joint intent cells")
    if (
        finite_number(
            candidate_admission["minimum_joint_shannon_effective_cells"],
            "P08 candidate intent Shannon-effective cell floor",
        )
        < 4.0
    ):
        raise ValueError("P08 candidate repertoire has too few effective intent modes")
    train_admission = qd_evidence["train_descriptor_admission"]
    if not isinstance(train_admission, dict):
        raise ValueError("P08 train descriptor admission must be an object")
    unsigned_train_admission = dict(train_admission)
    recorded_train_admission_hash = unsigned_train_admission.pop(
        "train_descriptor_admission_sha256", None
    )
    require_sha256(recorded_train_admission_hash, "P08 train descriptor admission hash")
    if canonical_sha256(unsigned_train_admission) != recorded_train_admission_hash:
        raise ValueError("P08 train descriptor admission hash is invalid")
    if train_admission.get("status") != "QD_TRAIN_DESCRIPTOR_ADMITTED":
        raise ValueError("P08 train descriptor is not admitted")
    if train_admission.get("descriptor_schema_version") != HM3D_REALISED_QD_SCHEMA_VERSION:
        raise ValueError("P08 train descriptor schema is stale")
    if train_admission.get("split_manifest_sha256") != context.get("split_manifest_sha256"):
        raise ValueError("P08 train descriptor split provenance drifted from P05")
    archive_spec_hash = train_admission.get("archive_spec_sha256")
    require_sha256(archive_spec_hash, "P08 train archive spec hash")
    if archive_spec_hash != HM3D_REALISED_QD_ARCHIVE_SPEC.digest:
        raise ValueError("P08 train archive specification drifted from the frozen runtime")
    _integer(
        train_admission.get("outcome_count"),
        "P08 train descriptor outcomes",
        minimum=MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION,
    )
    scene_ids = train_admission.get("scene_ids")
    if not isinstance(scene_ids, list) or len(set(scene_ids)) < 2:
        raise ValueError("P08 train descriptor admission lacks two train scenes")
    source_hashes = train_admission.get("source_runtime_record_sha256s")
    if not isinstance(source_hashes, list) or not source_hashes:
        raise ValueError("P08 train descriptor admission lacks raw outcome provenance")
    for source_hash in source_hashes:
        require_sha256(source_hash, "P08 train descriptor source outcome hash")
    for name, status in (
        ("richness_audit", "QD_DESCRIPTOR_ADMITTED"),
        ("footprint_separation_audit", "QD_FOOTPRINT_SEPARATION_ADMITTED"),
    ):
        audit = train_admission.get(name)
        if not isinstance(audit, dict) or audit.get("status") != status:
            raise ValueError(f"P08 train descriptor admission lacks {name}")
    richness = train_admission["richness_audit"]
    assert isinstance(richness, dict)
    _integer(
        richness.get("sample_count"),
        "P08 train descriptor richness sample count",
        minimum=MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION,
    )
    _integer(
        richness.get("joint_effective_cells"),
        "P08 train descriptor joint cells",
        minimum=MINIMUM_REALISED_QD_JOINT_CELLS,
    )
    richness_effective = finite_number(
        richness.get("joint_shannon_effective_cells"),
        "P08 train descriptor Shannon-effective cells",
    )
    richness_effective_floor = finite_number(
        richness.get("minimum_joint_shannon_effective_cells"),
        "P08 train descriptor Shannon-effective-cell floor",
    )
    if (
        richness_effective_floor + 1.0e-12 < MINIMUM_REALISED_QD_SHANNON_EFFECTIVE_CELLS
        or richness_effective + 1.0e-12 < richness_effective_floor
    ):
        raise ValueError("P08 train descriptor archive is not effectively populated")
    _integer(
        richness.get("minimum_joint_cells"),
        "P08 train descriptor joint-cell floor",
        minimum=MINIMUM_REALISED_QD_JOINT_CELLS,
    )
    observed_correlation = finite_number(
        richness.get("maximum_absolute_axis_correlation"),
        "P08 train descriptor observed axis correlation",
    )
    correlation_cap = finite_number(
        richness.get("maximum_absolute_axis_correlation_allowed"),
        "P08 train descriptor axis-correlation cap",
    )
    observed_determinant = finite_number(
        richness.get("axis_correlation_absolute_determinant"),
        "P08 train descriptor effective-dimension determinant",
    )
    determinant_floor = finite_number(
        richness.get("minimum_axis_correlation_absolute_determinant"),
        "P08 train descriptor effective-dimension floor",
    )
    if (
        correlation_cap - 1.0e-12 > MAXIMUM_REALISED_QD_AXIS_ABSOLUTE_CORRELATION
        or observed_correlation - 1.0e-12 > correlation_cap
        or determinant_floor + 1.0e-12 < MINIMUM_REALISED_QD_AXIS_CORRELATION_DETERMINANT
        or observed_determinant + 1.0e-12 < determinant_floor
    ):
        raise ValueError("P08 train descriptor axes are effectively collinear")
    validation_schema = qd_evidence["validation_outcome_schema"]
    if not isinstance(validation_schema, dict) or validation_schema != {
        "status": "QD_VALIDATION_OUTCOMES_SCHEMA_VALID",
        "policy": "validation_outcomes_do_not_tune_or_admit_descriptor_axes",
    }:
        raise ValueError("P08 validation outcomes may not tune QD descriptors")
    paired_effect = qd_evidence["paired_effect"]
    if not isinstance(paired_effect, dict):
        raise ValueError("P08 QD paired effect must be an object")
    _exact(
        paired_effect,
        {
            "unit_rows",
            "qd_active_decision_rate",
            "selection_change_rate",
            "minimum_selection_change_rate",
            "minimum_archive_entries_for_active_selection",
            "minimum_practical_relative_auc_gain",
            "minimum_effect_denominator_auc",
            "underpopulated_active_selection_count",
        },
        "P08 QD paired effect",
    )
    paired_rows = paired_effect["unit_rows"]
    if not isinstance(paired_rows, list) or len(paired_rows) != payload["paired_independent_units"]:
        raise ValueError("P08 QD paired records are incomplete")
    seen_unit_ids: set[str] = set()
    scenes: set[str] = set()
    fleet_size_values: set[int] = set()
    seeds: set[int] = set()
    for row in paired_rows:
        if not isinstance(row, dict):
            raise ValueError("P08 QD paired record must be an object")
        _exact(
            row,
            {
                "unit_id",
                "scene_id",
                "fleet_size",
                "seed",
                "initial_public_candidate_pool_sha256",
                "no_qd_auc",
                "planned_qd_auc",
                "realised_qd_auc",
            },
            "P08 QD paired record",
        )
        unit_id = require_identifier(row["unit_id"], "P08 QD unit_id")
        if unit_id in seen_unit_ids:
            raise ValueError("P08 QD paired unit_id must be unique")
        seen_unit_ids.add(unit_id)
        scenes.add(require_identifier(row["scene_id"], "P08 QD scene_id"))
        fleet_size = _integer(row["fleet_size"], "P08 QD fleet_size", minimum=1)
        if fleet_size != FORMAL_FLEET_SIZE:
            raise ValueError(f"P08 QD paired records must use N={FORMAL_FLEET_SIZE}")
        fleet_size_values.add(fleet_size)
        seeds.add(_integer(row["seed"], "P08 QD seed", minimum=0))
        require_sha256(
            row["initial_public_candidate_pool_sha256"],
            "P08 QD initial public candidate pool hash",
        )
        realised = finite_number(row["realised_qd_auc"], "P08 realised-QD paired AUC")
        if not 0.0 <= realised <= 1.0:
            raise ValueError("P08 realised-QD paired AUC must be in [0, 1]")
        for control in ("no_qd", "planned_qd"):
            value = finite_number(row[f"{control}_auc"], f"P08 {control} paired AUC")
            if not 0.0 <= value <= 1.0:
                raise ValueError("P08 QD control paired AUC must be in [0, 1]")
    if len(scenes) < 2 or fleet_size_values != {FORMAL_FLEET_SIZE} or len(seeds) < 2:
        raise ValueError(
            f"P08 QD paired evidence must cover two scenes, N={FORMAL_FLEET_SIZE}, and two seeds"
        )
    active_rate = finite_number(paired_effect["qd_active_decision_rate"], "P08 QD active rate")
    selection_change_rate = finite_number(
        paired_effect["selection_change_rate"], "P08 QD selection change rate"
    )
    minimum_selection_change_rate = finite_number(
        paired_effect["minimum_selection_change_rate"],
        "P08 QD selection change rate floor",
    )
    if (
        not 0.0 <= active_rate <= 1.0
        or not 0.0 <= selection_change_rate <= 1.0
        or not 0.0 <= minimum_selection_change_rate <= 1.0
    ):
        raise ValueError("P08 QD activity or selection-change rate is invalid")
    if (
        _integer(
            paired_effect["minimum_archive_entries_for_active_selection"],
            "P08 QD active archive entry floor",
            minimum=MINIMUM_REALISED_QD_JOINT_CELLS,
        )
        < MINIMUM_REALISED_QD_JOINT_CELLS
        or _integer(
            paired_effect["underpopulated_active_selection_count"],
            "P08 QD underpopulated active selection count",
        )
        != 0
    ):
        raise ValueError("P08 QD selected from an underpopulated archive")
    finite_number(payload["selection_gain"], "selection_gain")
    if finite_number(payload["selection_regret"], "selection_regret") < 0.0:
        raise ValueError("P08 selection regret is invalid")
    _nonnegative(payload["archive_build_time_s"], "archive_build_time_s")
    _integer(payload["archive_bytes"], "archive_bytes", minimum=1)
    outcome = finite_number(payload["outcome_utilization"], "outcome_utilization")
    transfer = finite_number(payload["negative_transfer_rate"], "negative_transfer_rate")
    transfer_limit = finite_number(payload["negative_transfer_limit"], "negative_transfer_limit")
    calibration = finite_number(payload["calibration_error"], "calibration_error")
    if not 0.0 < outcome <= 1.0:
        raise ValueError("P08 outcome utilization must be in (0, 1]")
    if not 0.0 <= transfer <= 1.0 or not 0.0 <= transfer_limit <= 1.0:
        raise ValueError("P08 OGFR negative-transfer report is invalid")
    if not 0.0 <= calibration <= 1.0:
        raise ValueError("P08 calibration error must be in [0, 1]")
    iterations = _integer(payload["tuning_iterations"], "tuning_iterations", minimum=1)
    log = payload["adjustment_log"]
    if not isinstance(log, list) or len(log) != iterations:
        raise ValueError("P08 adjustment log must retain every tuning iteration")
    for row in log:
        _exact(row, {"iteration", "change", "partition", "configuration_sha256"}, "P08 log")
        _integer(row["iteration"], "adjustment iteration", minimum=1)
        require_identifier(row["change"], "adjustment change")
        if row["partition"] not in {"train", "validation"}:
            raise ValueError("P08 adjustment log cannot use test data")
        require_sha256(row["configuration_sha256"], "adjustment configuration hash")
    if payload["mainline_components_removed"] != []:
        raise ValueError("P08 cannot delete QD, OGFR or RB-SF-SAC after a negative pilot")
    require_sha256(payload["selected_method_core_sha256"], "selected method core hash")
    if payload["mechanism_pilot_passed"] is not True:
        raise ValueError("P08 mechanism pilot is not ready")
    context["method_core_sha256"] = payload["selected_method_core_sha256"]


def _validate_p09(
    payload: dict[str, Any], protocol: HM3DFormalPreflightProtocol, context: dict[str, Any]
) -> None:
    expected = {
        "evidence_class",
        "frozen",
        "freeze_timestamp",
        "protocol_hash",
        "code_snapshot_sha256",
        "scene_manifest_sha256",
        "public_contract_sha256",
        "evaluation_denominator_sha256",
        "metric_registry_sha256",
        "sensor_profile_sha256",
        "dynamics_sha256",
        "controller_sha256",
        "method_core_sha256",
        "budget_sha256",
        "physical_time_s",
        "planner_calls",
        "candidate_count",
        "compute_cap_s",
        "memory_cap_mb",
        "seeds",
        "primary_metric",
        "coverage_role",
        "statistical_test",
        "alpha",
        "target_power",
        "test_access_count_before_freeze",
        "aerocity_bench_accesses",
        "freeze_sha256",
    }
    _exact(payload, expected, "P09 payload")
    if payload["evidence_class"] != "frozen_protocol":
        raise ValueError("P09 must be a frozen protocol artifact")
    if payload["frozen"] is not True:
        raise ValueError("P09 protocol is not frozen")
    require_identifier(payload["freeze_timestamp"], "freeze_timestamp")
    bindings = {
        "protocol_hash": protocol.protocol_hash,
        "scene_manifest_sha256": context.get("split_manifest_sha256"),
        "public_contract_sha256": context.get("public_contract_sha256"),
        "evaluation_denominator_sha256": context.get("evaluation_denominator_sha256"),
        "sensor_profile_sha256": context.get("sensor_profile_sha256"),
        "dynamics_sha256": context.get("dynamics_sha256"),
        "controller_sha256": context.get("controller_sha256"),
        "method_core_sha256": context.get("method_core_sha256"),
        "budget_sha256": context.get("budget_sha256"),
    }
    for name, expected_value in bindings.items():
        if payload[name] != expected_value:
            raise ValueError(f"P09 {name} drifted from admitted evidence")
    require_sha256(payload["code_snapshot_sha256"], "code snapshot hash")
    require_sha256(payload["metric_registry_sha256"], "metric registry hash")
    for name in ("physical_time_s", "planner_calls", "candidate_count", "compute_cap_s"):
        _positive(payload[name], name)
    _positive(payload["memory_cap_mb"], "memory_cap_mb")
    seeds = payload["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("P09 needs at least three frozen independent seeds")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("P09 seeds must be non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("P09 seeds must be unique")
    if payload["primary_metric"] != protocol.primary_metric:
        raise ValueError("P09 primary metric mismatch")
    if payload["coverage_role"] != "primary_task_quality":
        raise ValueError("exploration coverage must be the primary task quality")
    require_identifier(payload["statistical_test"], "statistical_test")
    alpha = finite_number(payload["alpha"], "alpha")
    power = finite_number(payload["target_power"], "target_power")
    if not 0.0 < alpha <= 0.1 or not 0.8 <= power <= 1.0:
        raise ValueError("P09 statistical alpha/power is not admissible")
    if payload["test_access_count_before_freeze"] != 0:
        raise ValueError("test scenes were accessed before P09 freeze")
    if payload["aerocity_bench_accesses"] != []:
        raise ValueError("P09 must not access AeroCityBench")
    require_sha256(payload["freeze_sha256"], "freeze_sha256")
    frozen_content = {key: value for key, value in payload.items() if key != "freeze_sha256"}
    if canonical_sha256(frozen_content) != payload["freeze_sha256"]:
        raise ValueError("P09 freeze hash was tampered with")
    context["formal_freeze"] = payload


def _validate_p10(
    payload: dict[str, Any],
    protocol: HM3DFormalPreflightProtocol,
    base: Path,
    context: dict[str, Any],
) -> None:
    _require_real_identity(payload)
    _exact(
        payload,
        {
            "evidence_class",
            "runtime_run_id",
            "runtime_command_sha256",
            "freeze_sha256",
            "run_partition",
            "scene_ids",
            "seeds",
            "fleet_size",
            "method_ids",
            "test_scene_admissions",
            "rows",
        },
        "P10 payload",
    )
    frozen = context.get("formal_freeze")
    if frozen is None or payload["freeze_sha256"] != frozen["freeze_sha256"]:
        raise ValueError("P10 does not match the admitted P09 freeze")
    if payload["run_partition"] != "test":
        raise ValueError("P10 may only run frozen test scenes")
    expected_scenes = tuple(
        sorted(
            scene_id
            for scene_id, scene in context.get("scenes", {}).items()
            if scene["split"] == "test"
        )
    )
    if tuple(payload["scene_ids"]) != expected_scenes:
        raise ValueError("P10 scene IDs do not exactly match the frozen test split")
    if tuple(payload["seeds"]) != tuple(frozen["seeds"]):
        raise ValueError("P10 seeds drifted after freeze")
    if _integer(payload["fleet_size"], "P10 fleet_size", minimum=1) != protocol.fleet_size:
        raise ValueError("P10 formal fleet-size contract is incomplete")
    if tuple(payload["method_ids"]) != protocol.formal_matrix_methods:
        raise ValueError("P10 formal method matrix is incomplete")
    admissions = payload["test_scene_admissions"]
    if not isinstance(admissions, list) or len(admissions) != len(expected_scenes):
        raise ValueError("P10 needs one runtime admission for every test scene")
    admitted_ids: set[str] = set()
    for admission in admissions:
        _exact(
            admission,
            {
                "scene_id",
                "source_geometry_sha256",
                "representation",
                "dimension",
                "free_flight_validated",
                "vertical_span_m",
                "free_flight_volume_m3",
                "connected_height_band_count",
                "flight_space_manifest_hash",
                "collision_geometry_sha256",
                "collision_derivative_sha256",
                "collision_replay_passed",
                "collision_replay_evidence_sha256",
            },
            "P10 test scene admission",
        )
        scene_id = require_identifier(admission["scene_id"], "P10 admitted scene_id")
        if scene_id in admitted_ids or scene_id not in expected_scenes:
            raise ValueError("P10 test scene admissions are duplicated or unexpected")
        admitted_ids.add(scene_id)
        if admission["source_geometry_sha256"] != context["scenes"][scene_id]["sha256"]:
            raise ValueError("P10 test geometry hash does not match P01")
        if admission["representation"] not in protocol.allowed_flight_representations:
            raise ValueError("P10 test flight space is not an allowed 3D representation")
        if admission["dimension"] != 3 or admission["free_flight_validated"] is not True:
            raise ValueError("P10 test free-flight representation is not validated 3D")
        if admission["vertical_span_m"] < protocol.three_d_thresholds["min_vertical_span_m"]:
            raise ValueError("P10 test scene lacks frozen vertical span")
        _positive(admission["free_flight_volume_m3"], "P10 free_flight_volume_m3")
        if (
            admission["connected_height_band_count"]
            < protocol.three_d_thresholds["min_connected_height_bands"]
        ):
            raise ValueError("P10 test scene lacks connected height bands")
        for name in (
            "flight_space_manifest_hash",
            "collision_geometry_sha256",
            "collision_derivative_sha256",
            "collision_replay_evidence_sha256",
        ):
            require_sha256(admission[name], f"P10 {name}")
        if admission["collision_replay_passed"] is not True:
            raise ValueError("P10 test collision replay failed")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("P10 rows must be a list")
    expected_keys = {
        (method, fleet, scene, seed)
        for method in protocol.formal_matrix_methods
        for fleet in (protocol.fleet_size,)
        for scene in expected_scenes
        for seed in frozen["seeds"]
    }
    actual_keys: set[tuple[str, int, str, int]] = set()
    raw_results: set[tuple[str, str]] = set()
    for row in rows:
        _exact(
            row,
            {
                "method_id",
                "fleet_size",
                "scene_id",
                "seed",
                "budget_sha256",
                "sensor_profile_sha256",
                "public_contract_sha256",
                "evaluation_denominator_sha256",
                "reads_private_truth",
                "ranked",
                "planned",
                "executed",
                "failed",
                "timeout",
                "oom",
                "other_failed",
                "explored_free_flight_volume_auc_time",
                "final_coverage_at_budget",
                "collision_count",
                "communication_failure_count",
                "energy_used_j",
                "raw_result_path",
                "raw_result_sha256",
            },
            "P10 result row",
        )
        key = (row["method_id"], row["fleet_size"], row["scene_id"], row["seed"])
        if key in actual_keys:
            raise ValueError("P10 contains duplicate method/fleet/scene/seed rows")
        actual_keys.add(key)
        if row["budget_sha256"] != frozen["budget_sha256"]:
            raise ValueError("P10 row budget drifted after freeze")
        if row["sensor_profile_sha256"] != frozen["sensor_profile_sha256"]:
            raise ValueError("P10 row sensor entitlement drifted after freeze")
        if row["public_contract_sha256"] != frozen["public_contract_sha256"]:
            raise ValueError("P10 public observation contract drifted after freeze")
        if row["evaluation_denominator_sha256"] != frozen["evaluation_denominator_sha256"]:
            raise ValueError("P10 evaluator denominator drifted after freeze")
        if row["reads_private_truth"] is not False or row["ranked"] is not True:
            raise ValueError("P10 ranked methods may only use public observations")
        _failure_denominator(row, f"P10 {key}")
        auc = finite_number(row["explored_free_flight_volume_auc_time"], "P10 exploration AUC")
        if not 0.0 <= auc <= 1.0:
            raise ValueError("P10 exploration AUC must be in [0, 1]")
        final_coverage = finite_number(row["final_coverage_at_budget"], "P10 final coverage")
        if not 0.0 <= final_coverage <= 1.0:
            raise ValueError("P10 final coverage must be in [0, 1]")
        for name in ("collision_count", "communication_failure_count"):
            _integer(row[name], f"P10 {name}")
        _nonnegative(row["energy_used_j"], "P10 energy_used_j")
        raw_path = _resolve_measured_file(
            base, row["raw_result_path"], row["raw_result_sha256"], f"P10 raw result {key}"
        )
        raw_identity = (str(raw_path), row["raw_result_sha256"])
        if raw_identity in raw_results:
            raise ValueError("P10 run cells must not reuse one raw result artifact")
        raw_results.add(raw_identity)
        raw_payload = read_json_object(raw_path)
        _exact(
            raw_payload,
            {
                "method_id",
                "fleet_size",
                "scene_id",
                "seed",
                "explored_free_flight_volume_auc_time",
                "final_coverage_at_budget",
                "status",
            },
            "P10 raw result",
        )
        if (
            raw_payload["method_id"],
            raw_payload["fleet_size"],
            raw_payload["scene_id"],
            raw_payload["seed"],
        ) != key:
            raise ValueError("P10 raw result identity does not match its matrix row")
        if (
            raw_payload["explored_free_flight_volume_auc_time"]
            != row["explored_free_flight_volume_auc_time"]
        ):
            raise ValueError("P10 raw result metric does not match its matrix row")
        if raw_payload["status"] not in {
            "SUCCESS",
            "FAILED",
            "TIMEOUT",
            "OOM",
            "OTHER_FAILED",
        }:
            raise ValueError("P10 raw result status is unsupported")
    if actual_keys != expected_keys:
        missing = len(expected_keys - actual_keys)
        extra = len(actual_keys - expected_keys)
        raise ValueError(f"P10 formal matrix mismatch; missing={missing}, extra={extra}")


def _load_artifact(
    reference: PreflightEvidenceArtifact,
    requirement: PreflightPhaseRequirement,
    evidence_root: Path,
) -> tuple[dict[str, Any], Path]:
    if reference.kind != requirement.kind or reference.origin != requirement.required_origin:
        raise ValueError("artifact kind/origin does not match the frozen phase requirement")
    path = Path(reference.path)
    if not path.is_absolute():
        path = (evidence_root / path).resolve()
    lowered = str(path).replace("\\", "/").casefold()
    if "aerocity-bench" in lowered:
        raise ValueError("artifact path accesses AeroCityBench")
    if not path.is_file():
        raise ValueError("artifact file is missing")
    if _file_sha256(path) != reference.sha256:
        raise ValueError("artifact file hash mismatch")
    envelope = read_json_object(path)
    _exact(
        envelope,
        {
            "schema_version",
            "phase_id",
            "kind",
            "origin",
            "measured",
            "synthetic",
            "denominator_complete",
            "payload",
        },
        "preflight artifact envelope",
    )
    if envelope["schema_version"] != PREFLIGHT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("preflight artifact schema mismatch")
    if envelope["phase_id"] != reference.phase_id or envelope["kind"] != reference.kind:
        raise ValueError("artifact identity does not match its manifest reference")
    if envelope["origin"] != reference.origin:
        raise ValueError("artifact origin does not match its manifest reference")
    if envelope["measured"] is not True or envelope["denominator_complete"] is not True:
        raise ValueError("artifact must be measured with a complete denominator")
    if envelope["synthetic"] is not False:
        raise ValueError("synthetic/mock evidence cannot close a runtime phase")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise ValueError("artifact payload must be an object")
    return payload, evidence_root


def audit_preflight_contract(protocol: HM3DFormalPreflightProtocol) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_PROTOCOL_SCHEMA_VERSION,
        "status": "CONTRACT_PASS",
        "protocol_hash": protocol.protocol_hash,
        "phases": [row.to_dict() for row in protocol.phases],
        "formal_experiment_start_requires": list(PHASE_IDS[:9]),
        "formal_results_require": list(PHASE_IDS),
        "claim_limit": "Code contracts only; no HM3D runtime or formal result is claimed.",
    }


def preflight_failure_attribution(rows: list[dict[str, Any]], global_reasons: list[str]) -> str:
    """Locate the first unresolved layer without blaming QD/OGFR prematurely."""

    if global_reasons:
        return "ATTRIBUTION_UNRESOLVED"
    unresolved = {row["phase_id"] for row in rows if row.get("status") != "READY"}
    if unresolved & {"P01", "P02", "P03"}:
        return "SCENE_OR_RUNTIME_INVALID"
    if "P04" in unresolved:
        return "METRIC_OR_PROTOCOL_INVALID"
    if unresolved & {"P05", "P06", "P07"}:
        return "TASK_INVALID_OR_UNCALIBRATED"
    if "P08" in unresolved:
        return "MECHANISM_ADJUSTMENT_REQUIRED"
    if "P09" in unresolved:
        return "METRIC_OR_PROTOCOL_INVALID"
    if "P10" in unresolved:
        return "FORMAL_RESULTS_INCOMPLETE"
    return "NO_UNRESOLVED_ATTRIBUTION"


def audit_hm3d_formal_preflight(
    protocol: HM3DFormalPreflightProtocol,
    evidence: HM3DFormalPreflightEvidence | None,
    *,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Audit P01--P10 in order without treating fixtures as runtime evidence."""

    global_reasons: list[str] = []
    if evidence is None:
        global_reasons.append("RUNTIME_EVIDENCE_MANIFEST_MISSING")
        artifacts: dict[str, PreflightEvidenceArtifact] = {}
    else:
        artifacts = {row.phase_id: row for row in evidence.artifacts}
        if evidence.protocol_hash != protocol.protocol_hash:
            global_reasons.append("PROTOCOL_HASH_MISMATCH")
        if evidence.method_core != protocol.method_core:
            global_reasons.append("METHOD_CORE_MISMATCH")
    requirements = {row.phase_id: row for row in protocol.phases}
    root = Path(evidence_root).resolve()
    context: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    validators = {
        "P01": lambda payload, base: _validate_p01(payload, base, context),
        "P02": lambda payload, base: _validate_p02(payload, context),
        "P03": lambda payload, base: _validate_p03(payload, protocol, context),
        "P04": lambda payload, base: _validate_p04(payload, context),
        "P05": lambda payload, base: _validate_p05(payload, context),
        "P06": lambda payload, base: _validate_p06(payload, protocol, context),
        "P07": lambda payload, base: _validate_p07(payload, protocol, context),
        "P08": lambda payload, base: _validate_p08(payload, protocol, context),
        "P09": lambda payload, base: _validate_p09(payload, protocol, context),
        "P10": lambda payload, base: _validate_p10(payload, protocol, base, context),
    }
    for phase_id in PHASE_IDS:
        reference = artifacts.get(phase_id)
        reasons: list[str] = []
        if reference is None:
            reasons.append("EVIDENCE_MISSING")
        else:
            try:
                payload, artifact_root = _load_artifact(reference, requirements[phase_id], root)
                validators[phase_id](payload, artifact_root)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                reasons.append(f"EVIDENCE_INVALID:{exc}")
        rows.append(
            {
                "phase_id": phase_id,
                "kind": requirements[phase_id].kind,
                "status": "READY" if not reasons else "RUNTIME_NOT_READY",
                "reasons": reasons,
            }
        )
    ready_by_phase = {row["phase_id"]: row["status"] == "READY" for row in rows}
    start_ready = not global_reasons and all(ready_by_phase[phase] for phase in PHASE_IDS[:9])
    results_ready = start_ready and ready_by_phase["P10"]
    requested_gate = None if evidence is None else evidence.requested_gate
    if results_ready:
        status = "FORMAL_RESULTS_READY"
    elif start_ready:
        status = "FORMAL_EXPERIMENT_READY"
    else:
        status = "RUNTIME_NOT_READY"
    if requested_gate == "formal_results" and not results_ready:
        status = "RUNTIME_NOT_READY"
    return {
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "protocol_hash": protocol.protocol_hash,
        "global_reasons": global_reasons,
        "phases": rows,
        "failure_attribution_status": preflight_failure_attribution(rows, global_reasons),
        "formal_experiment_start_authorized": start_ready,
        "formal_results_authorized": results_ready,
        "claim_limit": (
            "Frozen holdout matrix is complete and auditable."
            if results_ready
            else (
                "P01-P09 are ready; only the frozen formal experiment may start."
                if start_ready
                else "No long training, formal main table, or HM3D result claim is authorized."
            )
        ),
    }


def load_preflight_protocol(path: str | Path) -> HM3DFormalPreflightProtocol:
    return HM3DFormalPreflightProtocol.from_dict(read_json_object(path))


def load_preflight_evidence(path: str | Path) -> HM3DFormalPreflightEvidence:
    return HM3DFormalPreflightEvidence.from_dict(read_json_object(path))


__all__ = [
    "ALLOWED_FLIGHT_REPRESENTATIONS",
    "FORMAL_MATRIX_METHODS",
    "HM3DFormalPreflightEvidence",
    "HM3DFormalPreflightProtocol",
    "MECHANISM_VARIANTS",
    "METHOD_CORE",
    "PHASE_IDS",
    "PREFLIGHT_ARTIFACT_SCHEMA_VERSION",
    "PREFLIGHT_EVIDENCE_SCHEMA_VERSION",
    "PREFLIGHT_PROTOCOL_SCHEMA_VERSION",
    "PRIMARY_METRIC",
    "PreflightEvidenceArtifact",
    "PreflightPhaseRequirement",
    "TASK_VALIDITY_METHODS",
    "audit_hm3d_formal_preflight",
    "audit_preflight_contract",
    "load_preflight_evidence",
    "load_preflight_protocol",
    "preflight_failure_attribution",
]
