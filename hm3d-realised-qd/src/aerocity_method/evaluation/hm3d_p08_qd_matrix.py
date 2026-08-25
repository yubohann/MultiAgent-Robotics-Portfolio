"""Fail-closed assembly for the P08 realised-QD paired comparison.

This module consumes immutable, real P07 worker records.  It does not run a
simulator, create a score, or fill missing records.  Its job is to prevent two
common category errors: comparing different public candidate pools, and
declaring a QD gain from a hand-written aggregate rather than paired episodes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aerocity_method.contracts import FORMAL_FLEET_SIZE
from aerocity_method.contracts.io import canonical_sha256, finite_number, require_identifier
from aerocity_method.contracts.hm3d_public_schema import require_current_public_schema
from aerocity_method.runtime.hm3d_realised_qd import (
    HM3D_REALISED_QD_ARCHIVE_SPEC,
    HM3D_REALISED_QD_SCHEMA_VERSION,
    MAXIMUM_REALISED_QD_AXIS_ABSOLUTE_CORRELATION,
    MINIMUM_REALISED_QD_AXIS_CORRELATION_DETERMINANT,
    MINIMUM_REALISED_QD_JOINT_CELLS,
    MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION,
    MINIMUM_REALISED_QD_SHANNON_EFFECTIVE_CELLS,
    MINIMUM_OUTCOME_ARCHIVE_ENTRIES_FOR_SELECTION,
    RealisedQDDescriptor,
)

P08_QD_PAIRED_EVIDENCE_SCHEMA_VERSION = "hm3d-p08-qd-paired-evidence-v1"
# One changed choice in twelve can be a tie-break accident.  A QD mechanism
# claim needs multiple, value-protected interventions before its paired score
# is interpreted.
MINIMUM_QD_SELECTION_CHANGE_RATE = 0.10
MINIMUM_QD_VALUE_PROTECTED_DIVERSITY_OPPORTUNITY_RATE = 0.20
MINIMUM_QD_PRACTICAL_RELATIVE_AUC_GAIN = 0.05
# A target-free short-horizon AUC can be close to zero.  The floor makes the
# pre-registered relative effect stable while still requiring an absolute
# improvement of at least 0.0005 on the [0, 1] primary-metric scale.
MINIMUM_QD_EFFECT_DENOMINATOR_AUC = 0.01
MINIMUM_QD_ARCHIVE_ENTRIES_FOR_ACTIVE_SELECTION = MINIMUM_OUTCOME_ARCHIVE_ENTRIES_FOR_SELECTION
_REQUIRED_STRATEGIES = ("no_qd", "planned_qd", "realised_qd")
_PAIR_FIELDS = (
    "scene_id",
    "fleet_size",
    "random_key",
    "public_episode_id",
    "public_context_hash",
    "public_candidate_pool_hash",
    "candidate_pool_schema_version",
    "task_reservation_schema_version",
    "sensor_profile_sha256",
    "public_contract_sha256",
    "evaluation_denominator_sha256",
    "communication_contract_sha256",
    "action_budget_s",
    "candidate_limit",
    "physics_dt_s",
    "outcome_time_tolerance_s",
)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    int(value, 16)
    return value


def _number(value: object, label: str) -> float:
    return finite_number(value, label)


def _sign_test_p_value(wins: int, losses: int) -> float:
    """Exact one-sided sign test without a normal approximation."""

    compared = wins + losses
    if compared == 0:
        return 1.0
    return sum(math.comb(compared, count) for count in range(wins, compared + 1)) / 2**compared


def _verify_worker(payload: Mapping[str, Any], strategy: str) -> None:
    if payload.get("schema_version") != "hm3d-p07-exploration-execution-v1":
        raise ValueError("P08 QD evidence needs current P07 exploration worker records")
    if payload.get("status") != "P07_EXECUTION_SMOKE_COMPLETE":
        raise ValueError("P08 QD evidence cannot score a failed P07 worker")
    if payload.get("synthetic") is not False or payload.get("formal_result") is not False:
        raise ValueError("P08 QD evidence must use non-synthetic, non-formal P07 records")
    if payload.get("p07_task_validity_closed") is not False:
        raise ValueError("a worker may not claim P07 closure")
    if payload.get("selection_partition") != "validation":
        raise ValueError("P08 QD comparison may use validation records only")
    if payload.get("strategy") != strategy:
        raise ValueError("P08 QD input strategy does not match its declared branch")
    require_current_public_schema(payload, context="P08 QD worker record")
    unsigned = dict(payload)
    recorded_hash = _sha(unsigned.pop("runtime_record_sha256", None), "P07 runtime record hash")
    if canonical_sha256(unsigned) != recorded_hash:
        raise ValueError("P07 runtime record hash does not match its immutable content")
    for name in _PAIR_FIELDS:
        if name not in payload:
            raise ValueError(f"P07 record lacks pairing field: {name}")
    if not isinstance(payload["scene_id"], str) or not payload["scene_id"]:
        raise ValueError("P07 scene_id is invalid")
    if payload["fleet_size"] != FORMAL_FLEET_SIZE:
        raise ValueError(f"P08 QD comparison requires N={FORMAL_FLEET_SIZE}")
    if not isinstance(payload["random_key"], int) or isinstance(payload["random_key"], bool):
        raise ValueError("P07 random_key is invalid")
    for name in (
        "public_context_hash",
        "public_candidate_pool_hash",
        "sensor_profile_sha256",
        "public_contract_sha256",
        "evaluation_denominator_sha256",
        "communication_contract_sha256",
        "selector_backbone_sha256",
    ):
        _sha(payload.get(name), name)
    metric = payload.get("metric_report")
    if not isinstance(metric, Mapping):
        raise ValueError("P07 worker lacks exploration metric report")
    auc = _number(metric.get("explored_free_flight_volume_auc_time"), "P07 exploration AUC")
    if not 0.0 <= auc <= 1.0:
        raise ValueError("P07 exploration AUC must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class P08QDUnit:
    """One paired validation unit, built from three immutable worker records."""

    unit_id: str
    no_qd: Mapping[str, Any]
    planned_qd: Mapping[str, Any]
    realised_qd: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_identifier(self.unit_id, "P08 QD unit_id")
        records = {
            "no_qd": self.no_qd,
            "planned_qd": self.planned_qd,
            "realised_qd": self.realised_qd,
        }
        for strategy, payload in records.items():
            if not isinstance(payload, Mapping):
                raise TypeError(f"P08 QD {strategy} record must be an object")
            _verify_worker(payload, strategy)
        anchor = self.no_qd
        for strategy, payload in tuple(records.items())[1:]:
            drift = [name for name in _PAIR_FIELDS if payload[name] != anchor[name]]
            if drift:
                raise ValueError(f"P08 QD unit has pair drift in {strategy}: {drift}")
            if payload["selector_backbone_sha256"] != anchor["selector_backbone_sha256"]:
                raise ValueError("P08 QD controls must share one candidate-value backbone")

    @property
    def scene_id(self) -> str:
        return str(self.no_qd["scene_id"])

    @property
    def fleet_size(self) -> int:
        return int(self.no_qd["fleet_size"])

    @property
    def seed(self) -> int:
        return int(self.no_qd["random_key"])

    @property
    def initial_pool_hash(self) -> str:
        return str(self.no_qd["public_candidate_pool_hash"])

    @property
    def backbone_hash(self) -> str:
        return str(self.no_qd["selector_backbone_sha256"])

    def auc(self, strategy: str) -> float:
        record = getattr(self, strategy)
        metric = record["metric_report"]
        assert isinstance(metric, Mapping)
        return _number(metric["explored_free_flight_volume_auc_time"], f"{strategy} AUC")

    def to_pair_row(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "scene_id": self.scene_id,
            "fleet_size": self.fleet_size,
            "seed": self.seed,
            "initial_public_candidate_pool_sha256": self.initial_pool_hash,
            "no_qd_auc": self.auc("no_qd"),
            "planned_qd_auc": self.auc("planned_qd"),
            "realised_qd_auc": self.auc("realised_qd"),
        }


def _selection_records(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("P08 QD record has no online decision trace")
    rows: list[Mapping[str, Any]] = []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("P08 QD decision trace is malformed")
        selection = decision.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError("P08 QD decision lacks selector audit")
        rows.append(selection)
    return tuple(rows)


def _candidate_admissions(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int, int, int, float]:
    assessed = 0
    admitted = 0
    minimum_candidates: set[int] = set()
    minimum_axis_bins: set[int] = set()
    minimum_joint_cells: set[int] = set()
    minimum_joint_shannon_effective_cells: set[float] = set()
    for payload in payloads:
        qd = payload.get("realised_qd")
        if not isinstance(qd, Mapping):
            raise ValueError("P08 QD record lacks realised-QD diagnostics")
        audits = qd.get("candidate_intent_audits")
        if not isinstance(audits, list) or not audits:
            raise ValueError("P08 QD record lacks candidate-intent audits")
        for audit in audits:
            if not isinstance(audit, Mapping):
                raise ValueError("candidate-intent audit must be an object")
            assessed += 1
            for name in (
                "minimum_feasible_candidates",
                "minimum_axis_bins",
                "minimum_joint_cells",
            ):
                if not isinstance(audit.get(name), int):
                    raise ValueError(f"candidate-intent audit lacks {name}")
            minimum_candidates.add(int(audit["minimum_feasible_candidates"]))
            minimum_axis_bins.add(int(audit["minimum_axis_bins"]))
            minimum_joint_cells.add(int(audit["minimum_joint_cells"]))
            minimum_shannon = _number(
                audit.get("minimum_joint_shannon_effective_cells"),
                "minimum candidate intent Shannon-effective cells",
            )
            if minimum_shannon < 1.0:
                raise ValueError("candidate-intent Shannon-effective-cell floor is invalid")
            minimum_joint_shannon_effective_cells.add(minimum_shannon)
            if audit.get("status") == "QD_CANDIDATE_INTENT_ADMITTED":
                admitted += 1
    if (
        len(minimum_candidates) != 1
        or len(minimum_axis_bins) != 1
        or len(minimum_joint_cells) != 1
        or len(minimum_joint_shannon_effective_cells) != 1
    ):
        raise ValueError("P08 QD candidate admission thresholds drifted across workers")
    return (
        assessed,
        admitted,
        next(iter(minimum_candidates)),
        next(iter(minimum_axis_bins)),
        next(iter(minimum_joint_cells)),
        next(iter(minimum_joint_shannon_effective_cells)),
    )


def _value_protected_candidate_opportunities(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int, int, float]:
    """Count decisions where QD had diverse, near-value public alternatives.

    Candidate richness alone is not an opportunity to use QD: all diverse
    candidates may be outside the common value-protection band.  Keeping this
    diagnostic separate makes a no-effect result attributable instead of
    silently blaming the archive or selector.
    """

    assessed = 0
    admitted = 0
    minimum_candidates: set[int] = set()
    minimum_joint_cells: set[int] = set()
    utility_slacks: set[float] = set()
    for payload in payloads:
        qd = payload.get("realised_qd")
        if not isinstance(qd, Mapping):
            raise ValueError("P08 QD record lacks realised-QD diagnostics")
        audits = qd.get("value_protected_candidate_diversity_audits")
        if not isinstance(audits, list) or not audits:
            raise ValueError("P08 QD record lacks value-protected diversity audits")
        for audit in audits:
            if not isinstance(audit, Mapping):
                raise ValueError("value-protected diversity audit must be an object")
            assessed += 1
            for name in (
                "minimum_value_protected_candidates",
                "minimum_value_protected_joint_cells",
            ):
                if not isinstance(audit.get(name), int):
                    raise ValueError(f"value-protected diversity audit lacks {name}")
            minimum_candidates.add(int(audit["minimum_value_protected_candidates"]))
            minimum_joint_cells.add(int(audit["minimum_value_protected_joint_cells"]))
            utility_slack = _number(
                audit.get("utility_slack"), "value-protected diversity utility slack"
            )
            if not 0.0 <= utility_slack <= 1.0:
                raise ValueError("value-protected diversity utility slack is invalid")
            utility_slacks.add(utility_slack)
            if audit.get("status") == "QD_VALUE_PROTECTED_DIVERSITY_ADMITTED":
                admitted += 1
    if len(minimum_candidates) != 1 or len(minimum_joint_cells) != 1 or len(utility_slacks) != 1:
        raise ValueError("P08 QD value-protected opportunity thresholds drifted across workers")
    return (
        assessed,
        admitted,
        next(iter(minimum_candidates)),
        next(iter(minimum_joint_cells)),
        next(iter(utility_slacks)),
    )


def _realised_outcomes(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[RealisedQDDescriptor, ...],
    tuple[tuple[tuple[int, int, int], ...], ...],
]:
    intents: list[tuple[float, float, float]] = []
    descriptors: list[RealisedQDDescriptor] = []
    footprints: list[tuple[tuple[int, int, int], ...]] = []
    for payload in payloads:
        qd = payload.get("realised_qd")
        if not isinstance(qd, Mapping) or not isinstance(qd.get("admissions"), list):
            raise ValueError("P08 QD record lacks outcome admissions")
        for entry in qd["admissions"]:
            if not isinstance(entry, Mapping) or entry.get("feasible") is not True:
                continue
            intent = entry.get("public_candidate_intent")
            descriptor = entry.get("descriptor")
            if (
                not isinstance(intent, Sequence)
                or isinstance(intent, (str, bytes))
                or len(intent) != 3
                or not isinstance(descriptor, Mapping)
            ):
                raise ValueError("P08 QD outcome admission is malformed")
            if entry.get("executed") is not True:
                raise ValueError("P08 QD admission does not describe an executed candidate")
            if not isinstance(entry.get("candidate_id"), str) or not entry["candidate_id"]:
                raise ValueError("P08 QD admission lacks its executed candidate ID")
            _sha(entry.get("execution_outcome_sha256"), "P08 QD execution outcome hash")
            raw_footprint = entry.get("public_new_free_voxel_keys")
            if not isinstance(raw_footprint, list) or not raw_footprint:
                raise ValueError("P08 QD admission lacks its public execution footprint")
            footprint: list[tuple[int, int, int]] = []
            for key_index, raw_key in enumerate(raw_footprint):
                if (
                    not isinstance(raw_key, list)
                    or len(raw_key) != 3
                    or any(
                        not isinstance(value, int) or isinstance(value, bool) for value in raw_key
                    )
                ):
                    raise ValueError(
                        f"P08 QD public execution footprint key {key_index} is invalid"
                    )
                footprint.append((raw_key[0], raw_key[1], raw_key[2]))
            intents.append(tuple(_number(value, "P08 QD candidate intent") for value in intent))
            descriptors.append(
                RealisedQDDescriptor(
                    vertical_motion_ratio=_number(
                        descriptor.get("vertical_motion_ratio"), "outcome vertical motion"
                    ),
                    team_spatial_dispersion=_number(
                        descriptor.get("team_spatial_dispersion"), "outcome dispersion"
                    ),
                    public_observation_complementarity=_number(
                        descriptor.get("public_observation_complementarity"),
                        "outcome public observation complementarity",
                    ),
                    schema_version=str(descriptor.get("schema_version")),
                )
            )
            footprints.append(tuple(footprint))
    return tuple(intents), tuple(descriptors), tuple(footprints)


def _verified_train_descriptor_admission(
    payloads: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Verify the compact, train-only QD admission used by every P08 worker.

    Rich, non-collinear realised behaviour and semantically distinct footprints
    are necessary for a QD claim.  Forced intent campaigns, replay probes and
    descriptor-family tournaments remain recorded diagnostics, not preconditions
    for running the complete four-UAV paper matrix.
    """

    admissions: list[Mapping[str, Any]] = []
    for payload in payloads:
        qd = payload.get("realised_qd")
        if not isinstance(qd, Mapping):
            raise ValueError("P08 realised-QD record lacks QD diagnostics")
        history = qd.get("history")
        if not isinstance(history, Mapping):
            raise ValueError("P08 realised-QD record lacks train-history admission")
        admission = history.get("train_descriptor_admission")
        if not isinstance(admission, Mapping):
            raise ValueError("P08 realised-QD record lacks train descriptor admission")
        unsigned = dict(admission)
        recorded_hash = _sha(
            unsigned.pop("train_descriptor_admission_sha256", None),
            "train descriptor admission hash",
        )
        if canonical_sha256(unsigned) != recorded_hash:
            raise ValueError("P08 train descriptor admission hash is invalid")
        if admission.get("status") != "QD_TRAIN_DESCRIPTOR_ADMITTED":
            raise ValueError("P08 realised-QD record uses a non-admitted train descriptor")
        if admission.get("descriptor_schema_version") != HM3D_REALISED_QD_SCHEMA_VERSION:
            raise ValueError("P08 train descriptor schema does not match the frozen runtime")
        if admission.get("archive_spec_sha256") != HM3D_REALISED_QD_ARCHIVE_SPEC.digest:
            raise ValueError("P08 train archive spec does not match the frozen runtime")
        outcome_count = admission.get("outcome_count")
        scene_ids = admission.get("scene_ids")
        split_manifest_sha256 = admission.get("split_manifest_sha256")
        source_hashes = admission.get("source_runtime_record_sha256s")
        if (
            not isinstance(outcome_count, int)
            or outcome_count < MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION
        ):
            raise ValueError("P08 train descriptor admission has too few outcomes")
        if not isinstance(scene_ids, list) or len(set(scene_ids)) < 2:
            raise ValueError("P08 train descriptor admission lacks cross-scene evidence")
        _sha(split_manifest_sha256, "P08 train descriptor split manifest hash")
        if not isinstance(source_hashes, list) or not source_hashes:
            raise ValueError("P08 train descriptor admission lacks source record hashes")
        for value in source_hashes:
            _sha(value, "P08 train descriptor source record hash")
        richness = admission["richness_audit"]
        if not isinstance(richness, Mapping) or richness.get("status") != "QD_DESCRIPTOR_ADMITTED":
            raise ValueError("P08 train descriptor lacks realised-QD richness evidence")
        for name, minimum in (
            ("sample_count", MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION),
            ("joint_effective_cells", MINIMUM_REALISED_QD_JOINT_CELLS),
        ):
            value = richness.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"P08 train descriptor richness has invalid {name}")
        observed_effective_cells = _number(
            richness.get("joint_shannon_effective_cells"),
            "P08 train descriptor Shannon-effective cells",
        )
        declared_effective_floor = _number(
            richness.get("minimum_joint_shannon_effective_cells"),
            "P08 train descriptor Shannon-effective-cell floor",
        )
        if (
            declared_effective_floor + 1.0e-12 < MINIMUM_REALISED_QD_SHANNON_EFFECTIVE_CELLS
            or observed_effective_cells + 1.0e-12 < declared_effective_floor
        ):
            raise ValueError("P08 train descriptor archive is not effectively populated")
        declared_joint_floor = richness.get("minimum_joint_cells")
        if (
            not isinstance(declared_joint_floor, int)
            or isinstance(declared_joint_floor, bool)
            or declared_joint_floor < MINIMUM_REALISED_QD_JOINT_CELLS
        ):
            raise ValueError("P08 train descriptor joint-cell floor was weakened")
        observed_correlation = _number(
            richness.get("maximum_absolute_axis_correlation"),
            "P08 train descriptor observed axis correlation",
        )
        correlation_cap = _number(
            richness.get("maximum_absolute_axis_correlation_allowed"),
            "P08 train descriptor axis-correlation cap",
        )
        observed_determinant = _number(
            richness.get("axis_correlation_absolute_determinant"),
            "P08 train descriptor effective-dimension determinant",
        )
        determinant_floor = _number(
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
        footprint = admission.get("footprint_separation_audit")
        if (
            not isinstance(footprint, Mapping)
            or footprint.get("status") != "QD_FOOTPRINT_SEPARATION_ADMITTED"
        ):
            raise ValueError("P08 train descriptor lacks execution-footprint separation")
        admissions.append(admission)
    hashes = {str(admission["train_descriptor_admission_sha256"]) for admission in admissions}
    if len(hashes) != 1:
        raise ValueError("P08 realised-QD workers do not share one train descriptor admission")
    return admissions[0]


def _effect_status(
    unit_rows: Sequence[dict[str, object]],
) -> tuple[bool, dict[str, dict[str, float | int]]]:
    effects: dict[str, dict[str, float | int]] = {}
    passed = True
    for control in ("no_qd", "planned_qd"):
        deltas = [float(row["realised_qd_auc"]) - float(row[f"{control}_auc"]) for row in unit_rows]
        wins = sum(delta > 1.0e-12 for delta in deltas)
        losses = sum(delta < -1.0e-12 for delta in deltas)
        mean_delta = sum(deltas) / len(deltas)
        mean_control_auc = sum(float(row[f"{control}_auc"]) for row in unit_rows) / len(unit_rows)
        relative_gain = mean_delta / max(MINIMUM_QD_EFFECT_DENOMINATOR_AUC, abs(mean_control_auc))
        p_value = _sign_test_p_value(wins, losses)
        effects[control] = {
            "mean_auc_delta": mean_delta,
            "mean_control_auc": mean_control_auc,
            "relative_auc_gain": relative_gain,
            "wins": wins,
            "losses": losses,
            "one_sided_exact_sign_test_p": p_value,
        }
        passed = (
            passed
            and mean_delta > 0.0
            and relative_gain + 1.0e-12 >= MINIMUM_QD_PRACTICAL_RELATIVE_AUC_GAIN
            and wins > losses
            and p_value <= 0.05
        )
    return passed, effects


def assemble_p08_qd_paired_evidence(units: Sequence[P08QDUnit]) -> dict[str, object]:
    """Assemble immutable P08 mechanism diagnostics from paired workers.

    The function rejects malformed, incomparable, or non-outcome-grounded
    evidence.  It deliberately reports rather than gates on a validation
    effect-size threshold: P10 is the frozen holdout comparison.
    """

    rows = tuple(units)
    if len(rows) < 12:
        raise ValueError("P08 QD needs at least twelve paired validation units")
    # P08QDUnit validates at construction, but evidence mappings are mutable
    # JSON-like objects.  Revalidate here so a post-construction edit cannot
    # bypass the hash, pairing, or current-schema checks.
    for unit in rows:
        P08QDUnit(unit.unit_id, unit.no_qd, unit.planned_qd, unit.realised_qd)
    unit_ids = [row.unit_id for row in rows]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("P08 QD unit IDs must be unique")
    scene_ids = {row.scene_id for row in rows}
    fleet_size_values = {row.fleet_size for row in rows}
    seeds = {row.seed for row in rows}
    if len(scene_ids) < 2 or fleet_size_values != {FORMAL_FLEET_SIZE} or len(seeds) < 2:
        raise ValueError(f"P08 QD needs two scenes, N={FORMAL_FLEET_SIZE}, and two seeds")
    backbone_hashes = {row.backbone_hash for row in rows}
    if len(backbone_hashes) != 1:
        raise ValueError("P08 QD controls do not share one candidate-value backbone")
    pair_rows = tuple(row.to_pair_row() for row in rows)
    qd_records = tuple(record for unit in rows for record in (unit.planned_qd, unit.realised_qd))
    (
        assessed,
        admitted,
        minimum_candidates,
        minimum_axis_bins,
        minimum_joint_cells,
        minimum_joint_shannon_effective_cells,
    ) = _candidate_admissions(qd_records)
    (
        opportunity_assessed,
        opportunity_admitted,
        minimum_value_protected_candidates,
        minimum_value_protected_joint_cells,
        value_protected_utility_slack,
    ) = _value_protected_candidate_opportunities(qd_records)
    # Parse every validation outcome to reject missing/old descriptor fields,
    # but do not tune or admit a descriptor from validation observations.
    intents, descriptors, footprints = _realised_outcomes(tuple(unit.realised_qd for unit in rows))
    del intents, descriptors, footprints
    train_descriptor_admission = _verified_train_descriptor_admission(
        tuple(unit.realised_qd for unit in rows)
    )
    selection_rows = tuple(
        selection for unit in rows for selection in _selection_records(unit.realised_qd)
    )
    for selection in selection_rows:
        if not isinstance(selection.get("qd_abstained"), bool):
            raise ValueError("P08 QD selector audit lacks qd_abstained")
        if not isinstance(selection.get("archive_entry_count"), int):
            raise ValueError("P08 QD selector audit lacks archive_entry_count")
        if not isinstance(selection.get("archive_revision"), int):
            raise ValueError("P08 QD selector audit lacks archive_revision")
    qd_active_rows = tuple(
        selection for selection in selection_rows if not selection["qd_abstained"]
    )
    underpopulated_active_rows = tuple(
        selection
        for selection in qd_active_rows
        if selection["archive_entry_count"] < MINIMUM_QD_ARCHIVE_ENTRIES_FOR_ACTIVE_SELECTION
        or selection["archive_revision"] < 1
    )
    changed = sum(
        selection.get("diversity_changed_selection") is True for selection in qd_active_rows
    )
    qd_active_rate = len(qd_active_rows) / len(selection_rows)
    selection_change_rate = changed / len(selection_rows)
    effect_passed, effects = _effect_status(pair_rows)
    admissions_passed = admitted == assessed
    opportunity_rate = opportunity_admitted / opportunity_assessed
    opportunity_passed = (
        opportunity_rate + 1.0e-12 >= MINIMUM_QD_VALUE_PROTECTED_DIVERSITY_OPPORTUNITY_RATE
    )
    evidence = {
        "descriptor_schema_version": HM3D_REALISED_QD_SCHEMA_VERSION,
        "selector_backbone_sha256": next(iter(backbone_hashes)),
        "candidate_intent_admission": {
            "status": "QD_CANDIDATE_INTENT_ADMITTED"
            if admissions_passed
            else "QD_CANDIDATE_INTENT_NOT_ADMITTED",
            "assessed_pool_count": assessed,
            "admitted_pool_count": admitted,
            "minimum_feasible_candidates": minimum_candidates,
            "minimum_axis_bins": minimum_axis_bins,
            "minimum_joint_cells": minimum_joint_cells,
            "minimum_joint_shannon_effective_cells": minimum_joint_shannon_effective_cells,
        },
        "value_protected_diversity_opportunity": {
            "status": "QD_VALUE_PROTECTED_DIVERSITY_OPPORTUNITY_ADMITTED"
            if opportunity_passed
            else "QD_VALUE_PROTECTED_DIVERSITY_OPPORTUNITY_NOT_ADMITTED",
            "assessed_pool_count": opportunity_assessed,
            "admitted_pool_count": opportunity_admitted,
            "opportunity_rate": opportunity_rate,
            "minimum_opportunity_rate": (MINIMUM_QD_VALUE_PROTECTED_DIVERSITY_OPPORTUNITY_RATE),
            "minimum_value_protected_candidates": minimum_value_protected_candidates,
            "minimum_value_protected_joint_cells": minimum_value_protected_joint_cells,
            "utility_slack": value_protected_utility_slack,
        },
        "train_descriptor_admission": dict(train_descriptor_admission),
        "validation_outcome_schema": {
            "status": "QD_VALIDATION_OUTCOMES_SCHEMA_VALID",
            "policy": "validation_outcomes_do_not_tune_or_admit_descriptor_axes",
        },
        "paired_effect": {
            "unit_rows": list(pair_rows),
            "qd_active_decision_rate": qd_active_rate,
            "selection_change_rate": selection_change_rate,
            "minimum_selection_change_rate": MINIMUM_QD_SELECTION_CHANGE_RATE,
            "minimum_archive_entries_for_active_selection": (
                MINIMUM_QD_ARCHIVE_ENTRIES_FOR_ACTIVE_SELECTION
            ),
            "minimum_practical_relative_auc_gain": MINIMUM_QD_PRACTICAL_RELATIVE_AUC_GAIN,
            "minimum_effect_denominator_auc": MINIMUM_QD_EFFECT_DENOMINATOR_AUC,
            "underpopulated_active_selection_count": len(underpopulated_active_rows),
        },
        "selector_activity": {
            "status": "QD_SELECTION_TRACE_RECORDED",
            "assessed_decision_count": len(selection_rows),
            "active_decision_count": len(qd_active_rows),
            "changed_choice_count": changed,
            "policy": (
                "Intent-prediction, replay stability and public-need alignment remain "
                "train diagnostics; the formal mechanism claim rests on outcome-grounded "
                "archive use, changed choices, paired task AUC and safety."
            ),
        },
    }
    reasons: list[str] = []
    if not admissions_passed:
        reasons.append("QD_CANDIDATE_INTENT_NOT_ADMITTED")
    if not opportunity_passed:
        reasons.append("QD_VALUE_PROTECTED_DIVERSITY_OPPORTUNITY_INSUFFICIENT")
    if not qd_active_rows:
        reasons.append("QD_ALWAYS_ABSTAINED_FOR_UNCERTAIN_REALISATION")
    elif underpopulated_active_rows:
        reasons.append("QD_ACTIVE_WITHOUT_A_RICH_OUTCOME_GROUNDED_ARCHIVE")
    elif selection_change_rate + 1.0e-12 < MINIMUM_QD_SELECTION_CHANGE_RATE:
        reasons.append("QD_NEVER_CHANGED_THE_VALUE_PROTECTED_SELECTION")
    if not effect_passed:
        reasons.append("REALISED_QD_HAS_NO_SIGNIFICANT_PAIRED_ADVANTAGE")
    raw_hashes = {
        strategy: [
            _sha(getattr(unit, strategy).get("runtime_record_sha256"), "runtime record hash")
            for unit in rows
        ]
        for strategy in _REQUIRED_STRATEGIES
    }
    return {
        "schema_version": P08_QD_PAIRED_EVIDENCE_SCHEMA_VERSION,
        "status": "P08_QD_PILOT_COMPLETE",
        "synthetic": False,
        "formal_result": False,
        "claim_limit": (
            "QD-only P08 diagnostic. It does not validate OGFR, RB-SF-SAC, P08 as a whole, or P10."
        ),
        "paired_independent_units": len(rows),
        "scenes": sorted(scene_ids),
        "fleet_size": FORMAL_FLEET_SIZE,
        "seeds": sorted(seeds),
        "qd_mechanism_evidence": evidence,
        "paired_effect_summary": effects,
        "raw_worker_record_sha256s": raw_hashes,
        "reasons": reasons,
    }


__all__ = [
    "P08_QD_PAIRED_EVIDENCE_SCHEMA_VERSION",
    "MINIMUM_QD_ARCHIVE_ENTRIES_FOR_ACTIVE_SELECTION",
    "MINIMUM_QD_EFFECT_DENOMINATOR_AUC",
    "MINIMUM_QD_PRACTICAL_RELATIVE_AUC_GAIN",
    "MINIMUM_QD_SELECTION_CHANGE_RATE",
    "MINIMUM_QD_VALUE_PROTECTED_DIVERSITY_OPPORTUNITY_RATE",
    "P08QDUnit",
    "assemble_p08_qd_paired_evidence",
]
