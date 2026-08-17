"""Stream a bounded, explicit RLDS-shaped interchange projection.

The native capture stream records a command before each simulation step and a
state after that step.  RLDS defines an action as the action taken after the
current observation.  This module therefore emits the provable transitions
``state[i] -- command[i + 1] --> state[i + 1]`` and records the unrepresented
initial command in the episode metadata.  It never silently shifts an action,
fills a missing reward, or calls the JSONL interchange a TFDS dataset.

The output is intentionally dependency-free JSONL.  A future TFDS writer can
consume the same validated records once a public episode has a cleared reward
stream and an external reader agreement report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


RLDS_INTERCHANGE_SCHEMA = "org.rivermark.benchmark.rlds-jsonl.v1"
_CLAIM_BOUNDARY = "development-only RLDS-shaped interchange; no TFDS or formal-episode claim"
_RECORD_TYPES = frozenset({"episode_start", "step", "episode_end"})
_STATE_FIELDS = (
    "root_pos_w_m",
    "root_quat_wxyz",
    "root_lin_vel_w_mps",
    "root_ang_vel_b_radps",
    "applied_thrust_n",
)
_ACTION_FIELDS = (
    "desired_pos_w_m",
    "desired_vel_w_mps",
    "target_thrust_n",
)
_PUBLIC_PROVENANCE_FIELDS = frozenset(
    {
        "source_capture_receipt_sha256",
        "source_revision",
        "observation_abi_sha256",
        "collection_protocol_id",
        "collection_protocol_sha256",
        "collection_cell_id",
        "split",
        "episode_index",
        "episode_seed",
    }
)


class RldsProjectionError(ValueError):
    """Raised when an RLDS mapping cannot be proved from public data."""


@dataclass(frozen=True)
class RldsProjectionResult:
    output_root: Path
    episode_id: str
    step_count: int
    dropped_initial_command_count: int
    projection_manifest_sha256: str


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\\" in value or "/" in value:
        raise RldsProjectionError(f"{name} must be a non-empty path-free identifier")
    return value


def _validate_provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        raise RldsProjectionError("source_provenance is required for an RLDS projection")
    unknown = sorted(set(value) - _PUBLIC_PROVENANCE_FIELDS)
    if unknown:
        raise RldsProjectionError(f"source_provenance contains non-public fields: {unknown}")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"episode_index", "episode_seed"}:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise RldsProjectionError(f"source_provenance.{key} must be a non-negative integer")
            result[key] = item
        else:
            text = _require_id(item, f"source_provenance.{key}")
            if key.endswith("_sha256") and (len(text) != 64 or any(char not in "0123456789abcdef" for char in text)):
                raise RldsProjectionError(f"source_provenance.{key} must be a lowercase SHA-256")
            result[key] = text
    return result


def _array(values: Mapping[str, Any], name: str, *, steps: int, agents: int, width: int) -> np.ndarray:
    if name not in values:
        raise RldsProjectionError(f"missing source field: {name}")
    value = np.asarray(values[name])
    expected = (steps, agents, width)
    if value.shape != expected or not np.issubdtype(value.dtype, np.number):
        raise RldsProjectionError(f"{name} must have numeric shape {expected}, got {value.dtype} {value.shape}")
    if not np.isfinite(value).all():
        raise RldsProjectionError(f"{name} contains non-finite values")
    return value


def _validate_source(
    values: Mapping[str, Any],
    *,
    rewards: Sequence[float],
    discounts: Sequence[float] | None,
) -> tuple[
    int,
    int,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    for field in ("command_time_ns", "effective_time_ns"):
        if field not in values:
            raise RldsProjectionError(f"missing source field: {field}")
    command_time = np.asarray(values["command_time_ns"])
    effective_time = np.asarray(values["effective_time_ns"])
    if command_time.ndim != 1 or effective_time.shape != command_time.shape or command_time.size < 2:
        raise RldsProjectionError("command/effective timestamps must contain at least two aligned steps")
    if not np.issubdtype(command_time.dtype, np.integer) or not np.issubdtype(effective_time.dtype, np.integer):
        raise RldsProjectionError("command/effective timestamps must be integer nanoseconds")
    if np.any(command_time >= effective_time):
        raise RldsProjectionError("every command_time_ns must precede effective_time_ns")
    steps = int(command_time.size)
    candidates = [np.asarray(values[name]) for name in _STATE_FIELDS + _ACTION_FIELDS if name in values]
    if not candidates:
        raise RldsProjectionError("no state/action arrays were supplied")
    if candidates[0].ndim != 3 or candidates[0].shape[0] != steps:
        raise RldsProjectionError("state/action arrays must be [step, agent, component]")
    agents = int(candidates[0].shape[1])
    if agents <= 0:
        raise RldsProjectionError("agent count must be positive")
    state: dict[str, np.ndarray] = {}
    for field in _STATE_FIELDS:
        width = 4 if field == "root_quat_wxyz" else 4 if field == "applied_thrust_n" else 3
        state[field] = _array(values, field, steps=steps, agents=agents, width=width)
    action: dict[str, np.ndarray] = {}
    for field in _ACTION_FIELDS:
        width = 4 if field == "target_thrust_n" else 3
        action[field] = _array(values, field, steps=steps, agents=agents, width=width)
    try:
        reward_array = np.asarray(rewards, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RldsProjectionError("rewards must be numeric") from exc
    if reward_array.shape != (steps - 1,) or not np.isfinite(reward_array).all():
        raise RldsProjectionError(f"rewards must be finite with shape {(steps - 1,)}")
    if discounts is None:
        discount_array = np.ones(steps - 1, dtype=np.float64)
    else:
        try:
            discount_array = np.asarray(discounts, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RldsProjectionError("discounts must be numeric") from exc
        if discount_array.shape != (steps - 1,) or not np.isfinite(discount_array).all():
            raise RldsProjectionError(f"discounts must be finite with shape {(steps - 1,)}")
        if np.any((discount_array < 0.0) | (discount_array > 1.0)):
            raise RldsProjectionError("discounts must lie in [0, 1]")
    return steps, agents, state, action, command_time, effective_time, reward_array, discount_array


def _nested(values: Mapping[str, np.ndarray], index: int) -> dict[str, Any]:
    agent_count = int(next(iter(values.values())).shape[1])
    return {
        "agent_ids": list(range(agent_count)),
        **{key: value[index].tolist() for key, value in values.items()},
    }


def _write_line(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(_canonical_bytes(value))


def project_state_action_to_rlds(
    output_root: Path,
    *,
    episode_id: str,
    source_values: Mapping[str, Any],
    rewards: Sequence[float],
    discounts: Sequence[float] | None = None,
    source_provenance: Mapping[str, Any] | None = None,
    terminal: bool = False,
    truncated: bool = True,
    termination_reason: str = "fixed_horizon",
    allow_initial_command_drop: bool = False,
    invalid_episode: bool = False,
) -> RldsProjectionResult:
    """Write one development RLDS-shaped episode without unbounded buffering.

    The source has N post-step states and N pre-step commands.  The output has
    N steps: N-1 valid transitions and one final observation-only step.  The
    initial command is retained in metadata but is not presented as an RLDS
    action because its pre-command observation is not present in the source.
    """

    episode_id = _require_id(episode_id, "episode_id")
    provenance = _validate_provenance(source_provenance)
    if isinstance(terminal, bool) is False or isinstance(truncated, bool) is False:
        raise RldsProjectionError("terminal and truncated must be booleans")
    if terminal and truncated:
        raise RldsProjectionError("terminal and truncated cannot both be true")
    if not terminal and not truncated:
        raise RldsProjectionError("one of terminal or truncated must be true")
    if (
        not isinstance(termination_reason, str)
        or not termination_reason.strip()
        or "/" in termination_reason
        or "\\" in termination_reason
    ):
        raise RldsProjectionError("termination_reason must be a non-empty path-free value")
    if not allow_initial_command_drop:
        raise RldsProjectionError(
            "RLDS mapping needs an explicit allow_initial_command_drop=True "
            "because the source lacks the pre-command observation"
        )
    steps, agents, state, action, command_time, effective_time, reward_array, discount_array = _validate_source(
        source_values, rewards=rewards, discounts=discounts
    )
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise RldsProjectionError(f"projection output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        episode_path = staging / "episode.jsonl"
        metadata = {
            "episode_id": episode_id,
            "agent_count": agents,
            "source_provenance": provenance,
            "observation_timing": "state_after_step",
            "source_action_timing": "command_before_step",
            "rlds_action_timing": "action_after_current_observation",
            "transition_mapping": "observation[i] + command[i+1] -> observation[i+1]",
            "initial_command_dropped": True,
            "initial_command_index": 0,
            "initial_command_time_ns": int(command_time[0]),
            "initial_command_effective_time_ns": int(effective_time[0]),
            "final_action_valid": False,
            "final_reward_valid": False,
            "final_discount_valid": False,
            "terminal": terminal,
            "truncated": truncated,
            "termination_reason": termination_reason,
            "invalid_episode": bool(invalid_episode),
            "development_only": True,
            "claim_boundary": _CLAIM_BOUNDARY,
        }
        with episode_path.open("wb") as stream:
            _write_line(
                stream,
                {
                    "record_type": "episode_start",
                    "schema": RLDS_INTERCHANGE_SCHEMA,
                    "format": "jsonl",
                    "metadata": metadata,
                },
            )
            for index in range(steps):
                is_final = index == steps - 1
                source_action_index = min(index + 1, steps - 1)
                _write_line(
                    stream,
                    {
                        "record_type": "step",
                        "step_index": index,
                        "source_state_index": index,
                        "source_action_index": source_action_index,
                        "timestamp_ns": int(effective_time[index]),
                        "observation": _nested(state, index),
                        "action": _nested(action, source_action_index),
                        "reward": 0.0 if is_final else float(reward_array[index]),
                        "discount": 0.0 if is_final else float(discount_array[index]),
                        "is_first": index == 0,
                        "is_last": is_final,
                        "is_terminal": bool(is_final and terminal),
                        "truncated": bool(is_final and truncated),
                        "action_valid": not is_final,
                        "reward_valid": not is_final,
                        "discount_valid": not is_final,
                        "command_time_ns": int(command_time[source_action_index]),
                        "effective_time_ns": int(effective_time[source_action_index]),
                    },
                )
        prefix_sha256 = _sha256_file(episode_path)
        with episode_path.open("ab") as stream:
            _write_line(
                stream,
                {
                    "record_type": "episode_end",
                    "schema": RLDS_INTERCHANGE_SCHEMA,
                    "episode_id": episode_id,
                    "step_count": steps,
                    "prefix_sha256": prefix_sha256,
                },
            )
        manifest = {
            "schema": RLDS_INTERCHANGE_SCHEMA,
            "format": "jsonl",
            "status": "projected",
            "episode_id": episode_id,
            "step_count": steps,
            "agent_count": agents,
            "dropped_initial_command_count": 1,
            "source_provenance": provenance,
            "mapping": {
                "observation": "state_after_step[i]",
                "action": "command_before_step[i+1]",
                "reward": "caller-supplied public reward[i]",
                "discount": "caller-supplied public discount[i] or 1.0",
                "final_step_values": "numeric placeholders with *_valid=false",
            },
            "claim_boundary": _CLAIM_BOUNDARY,
            "files": [
                {
                    "path": "episode.jsonl",
                    "bytes": episode_path.stat().st_size,
                    "sha256": _sha256_file(episode_path),
                }
            ],
        }
        (staging / "projection_manifest.json").write_bytes(_canonical_bytes(manifest))
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest_path = destination / "projection_manifest.json"
    return RldsProjectionResult(
        output_root=destination,
        episode_id=episode_id,
        step_count=steps,
        dropped_initial_command_count=1,
        projection_manifest_sha256=_sha256_file(manifest_path),
    )


def iter_rlds_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL records after basic framing checks."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RldsProjectionError(f"missing RLDS interchange: {source}")
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RldsProjectionError(f"invalid JSONL at line {line_number}: {exc}") from exc
                if not isinstance(value, dict) or value.get("record_type") not in _RECORD_TYPES:
                    raise RldsProjectionError(f"invalid RLDS record at line {line_number}")
                yield value
    except OSError as exc:
        raise RldsProjectionError(f"cannot read RLDS interchange: {exc}") from exc


def verify_rlds_interchange(root: Path) -> dict[str, Any]:
    """Verify framing, ordering, core fields, and the prefix hash."""

    destination = Path(root).expanduser().resolve()
    episode_path = destination / "episode.jsonl"
    manifest_path = destination / "projection_manifest.json"
    if not manifest_path.is_file():
        raise RldsProjectionError("projection_manifest.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RldsProjectionError(f"invalid projection_manifest.json: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != RLDS_INTERCHANGE_SCHEMA:
        raise RldsProjectionError("projection manifest schema mismatch")
    episode_file = {
        "path": "episode.jsonl",
        "bytes": episode_path.stat().st_size,
        "sha256": _sha256_file(episode_path),
    }
    if manifest.get("files") != [episode_file]:
        raise RldsProjectionError("projection manifest does not bind episode.jsonl")
    start: dict[str, Any] | None = None
    end: dict[str, Any] | None = None
    step_count = 0
    previous_is_last = False
    final_step_is_last = False
    prefix_digest = hashlib.sha256()
    try:
        with episode_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RldsProjectionError(f"invalid JSONL at line {line_number}: {exc}") from exc
                if not isinstance(record, dict) or record.get("record_type") not in _RECORD_TYPES:
                    raise RldsProjectionError(f"invalid RLDS record at line {line_number}")
                record_type = record["record_type"]
                if record_type == "episode_start":
                    if start is not None or step_count:
                        raise RldsProjectionError("episode_start must be the first record")
                    start = record
                    prefix_digest.update(_canonical_bytes(record))
                    continue
                if record_type == "step":
                    if start is None or end is not None:
                        raise RldsProjectionError("step records must occur between start and end")
                    index = step_count
                    if previous_is_last:
                        raise RldsProjectionError("is_last may only be true on the final step")
                    if record.get("step_index") != index or record.get("is_first") != (index == 0):
                        raise RldsProjectionError("step indices or is_first flags are inconsistent")
                    if not isinstance(record.get("is_last"), bool):
                        raise RldsProjectionError("is_last must be a boolean")
                    for key in ("observation", "action", "reward", "discount", "is_terminal", "is_first", "is_last"):
                        if key not in record:
                            raise RldsProjectionError(f"RLDS core field is missing at step {index}: {key}")
                    prefix_digest.update(_canonical_bytes(record))
                    previous_is_last = bool(record["is_last"])
                    final_step_is_last = previous_is_last
                    step_count += 1
                    continue
                if end is not None:
                    raise RldsProjectionError("episode_end must be the final record")
                end = record
    except OSError as exc:
        raise RldsProjectionError(f"cannot read RLDS interchange: {exc}") from exc
    if start is None or end is None or step_count < 2:
        raise RldsProjectionError("RLDS interchange must have start, steps, and end records")
    if not final_step_is_last:
        raise RldsProjectionError("final RLDS step must set is_last=true")
    if start.get("schema") != RLDS_INTERCHANGE_SCHEMA or end.get("schema") != RLDS_INTERCHANGE_SCHEMA:
        raise RldsProjectionError("RLDS interchange schema mismatch")
    if end.get("prefix_sha256") != prefix_digest.hexdigest():
        raise RldsProjectionError("episode prefix hash mismatch")
    if end.get("step_count") != step_count:
        raise RldsProjectionError("episode step_count mismatch")
    if manifest.get("episode_id") != start.get("metadata", {}).get("episode_id"):
        raise RldsProjectionError("projection manifest episode_id mismatch")
    if manifest.get("step_count") != step_count:
        raise RldsProjectionError("projection manifest step_count mismatch")
    if start.get("metadata", {}).get("claim_boundary") != _CLAIM_BOUNDARY:
        raise RldsProjectionError("unexpected RLDS claim boundary")
    return {
        "schema": RLDS_INTERCHANGE_SCHEMA,
        "status": "valid",
        "episode_id": start.get("metadata", {}).get("episode_id"),
        "step_count": step_count,
        "bytes": episode_path.stat().st_size,
        "sha256": _sha256_file(episode_path),
        "claim_boundary": start.get("metadata", {}).get("claim_boundary"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify_rlds_interchange(args.projection), indent=2, sort_keys=True))
    except (OSError, RldsProjectionError) as exc:
        print(json.dumps({"schema": RLDS_INTERCHANGE_SCHEMA, "status": "invalid", "error": str(exc)}, indent=2))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
