"""City-Lite authority resolution, layer inventory, and scene receipts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    _OUTPUT_BINDINGS,
    AUTHORITY_SHA256,
    ENVIRONMENT_ID,
    EXPECTED_NATIVE_COLLISION_COUNTS,
    EXPECTED_UPSTREAM_PERMISSIONS,
    FINAL_SCENE_FILENAME,
    RIVERMARK_ASSET_ROOT_NAME,
    RIVERMARK_LAYER_INVENTORY_SCHEMA,
    SCENE_CONTRACT_FILENAME,
    SCENE_CONTRACT_GATE_STATUS,
    SCENE_CONTRACT_PAYLOAD_SHA256,
    SCENE_CONTRACT_SCHEMA,
    SCENE_CONTRACT_SHA256,
    SELECTIVE_REFERENCES,
)
from .materials import validate_city_task_obstacle_material_closure_receipt
from .scene import (
    CityLiteAuthorityError,
    canonical_payload_sha256,
    flight_contract_payload,
    sha256_file,
)


@dataclass(frozen=True)
class CityLiteAuthority:
    root: Path
    contract_path: Path
    final_scene_path: Path
    asset_paths: Mapping[str, Path]
    sha256: Mapping[str, str]
    contract_sha256: str
    contract_payload_sha256: str

    def provenance(self) -> dict[str, Any]:
        return {
            "environment_id": ENVIRONMENT_ID,
            "authority_root": str(self.root.resolve()),
            "scene_contract": {
                "path": str(self.contract_path.resolve()),
                "sha256": self.contract_sha256,
                "payload_sha256": self.contract_payload_sha256,
                "schema": SCENE_CONTRACT_SCHEMA,
                "gate_status": SCENE_CONTRACT_GATE_STATUS,
                "permissions": dict(EXPECTED_UPSTREAM_PERMISSIONS),
            },
            "authority_assets": {
                name: {
                    "path": str(self.asset_paths[name].resolve()),
                    "sha256": self.sha256[name],
                }
                for name in sorted(self.asset_paths)
            },
            "source_scene": str(self.final_scene_path.resolve()),
            "selective_references": [
                {"source_prim": source, "destination_prim": destination}
                for source, destination in SELECTIVE_REFERENCES
            ],
            "stage_units": {
                "meters_per_unit": 1.0,
                "up_axis": "Z",
                "time_codes_per_second": 60.0,
                "frames_per_second": 60.0,
            },
            "flight_contract": flight_contract_payload(),
            "expected_native_collision_counts": dict(
                EXPECTED_NATIVE_COLLISION_COUNTS
            ),
            "static_scene_authority_verified": True,
            "scene_runtime_admission": False,
            "formal_collection": False,
            "formal_benchmark_admission": False,
        }

def _resolved_file(path: str | Path, *, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as exc:
        raise CityLiteAuthorityError(f"{label} is missing or unresolved: {path}") from exc
    if not resolved.is_file():
        raise CityLiteAuthorityError(f"{label} is not a file: {resolved}")
    if resolved.suffix.casefold() not in {".usd", ".usda", ".usdc"}:
        raise CityLiteAuthorityError(f"{label} is not an OpenUSD layer: {resolved}")
    return resolved


def _path_key(path: Path) -> str:
    """Return a deterministic, Windows-safe identity for a resolved path."""

    return path.as_posix().casefold()


def _lexical_absolute_path(path: str | Path, *, label: str) -> Path:
    """Make a path absolute without dereferencing directory junctions."""

    try:
        return Path(path).expanduser().absolute()
    except (OSError, RuntimeError, TypeError) as exc:
        raise CityLiteAuthorityError(f"{label} is not a valid path: {path}") from exc


def _rivermark_root_ancestor(path: Path) -> Path | None:
    for candidate in path.parents:
        if candidate.name.casefold() == RIVERMARK_ASSET_ROOT_NAME.casefold():
            return candidate
    return None


def _layer_binding(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise CityLiteAuthorityError(f"OpenUSD layer changed while hashing: {path}")
    if after.st_size <= 0:
        raise CityLiteAuthorityError(f"OpenUSD layer is empty: {path}")
    return {
        "path": str(path),
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def make_rivermark_layer_inventory(
    authority: CityLiteAuthority,
    resolved_layer_paths: Sequence[str | Path],
    *,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Bind the layers used by a selective City-Lite OpenUSD composition.

    ``resolved_layer_paths`` is intended to be populated from
    ``Usd.Stage.GetUsedLayers()`` after composing only ``SELECTIVE_REFERENCES``.
    Anonymous root/session layers are ignored. The three local generated
    layers remain a separate authority class; every other admitted layer must
    be a real USD file below one unique ``RivermarkSrc51`` asset root.
    """

    if not isinstance(authority, CityLiteAuthority):
        raise CityLiteAuthorityError("authority must be a CityLiteAuthority")
    if isinstance(resolved_layer_paths, (str, bytes, Path)):
        raise CityLiteAuthorityError("resolved_layer_paths must be a sequence")
    try:
        supplied_layers = tuple(resolved_layer_paths)
    except TypeError as exc:
        raise CityLiteAuthorityError(
            "resolved_layer_paths must be a sequence"
        ) from exc
    if not supplied_layers:
        raise CityLiteAuthorityError("resolved layer inventory is empty")

    if set(authority.asset_paths) != set(AUTHORITY_SHA256):
        raise CityLiteAuthorityError("local City-Lite authority layer set is not exact")
    if set(authority.sha256) != set(AUTHORITY_SHA256):
        raise CityLiteAuthorityError("local City-Lite authority digest set is not exact")

    local_by_path: dict[str, str] = {}
    local_rows: list[dict[str, Any]] = []
    for filename in sorted(AUTHORITY_SHA256):
        path = _resolved_file(
            authority.asset_paths[filename],
            label=f"local City-Lite authority layer {filename}",
        )
        row = _layer_binding(path)
        if row["sha256"] != authority.sha256[filename]:
            raise CityLiteAuthorityError(
                f"local City-Lite authority hash mismatch for {filename}"
            )
        local_by_path[_path_key(path)] = filename
        local_rows.append(
            {
                "filename": filename,
                **row,
                "classification": "city_lite_local_authority",
            }
        )

    explicit_root: Path | None = None
    explicit_resolved_root: Path | None = None
    if asset_root is not None:
        explicit_root = _lexical_absolute_path(
            asset_root,
            label="Rivermark asset root",
        )
        if explicit_root.name.casefold() != RIVERMARK_ASSET_ROOT_NAME.casefold():
            raise CityLiteAuthorityError(
                f"Rivermark asset root must be named {RIVERMARK_ASSET_ROOT_NAME}"
            )
        try:
            explicit_resolved_root = explicit_root.resolve(strict=True)
        except (OSError, RuntimeError, TypeError) as exc:
            raise CityLiteAuthorityError(
                f"Rivermark asset root is missing or unresolved: {asset_root}"
            ) from exc
        if not explicit_resolved_root.is_dir():
            raise CityLiteAuthorityError(
                f"Rivermark asset root is not a directory: {explicit_root}"
            )

    anonymous_count = 0
    local_seen: set[str] = set()
    external_candidates: dict[str, Path] = {}
    discovered_roots: dict[str, tuple[Path, Path]] = {}
    for index, raw_path in enumerate(supplied_layers):
        identifier = str(raw_path).strip()
        if not identifier:
            raise CityLiteAuthorityError(
                f"resolved OpenUSD layer identifier {index} is empty"
            )
        if identifier.casefold().startswith("anon:"):
            anonymous_count += 1
            continue
        lexical_path = _lexical_absolute_path(
            identifier,
            label=f"resolved OpenUSD layer {index}",
        )
        lexical_root = _rivermark_root_ancestor(lexical_path)
        path = _resolved_file(
            identifier,
            label=f"resolved OpenUSD layer {index}",
        )
        key = _path_key(path)
        local_filename = local_by_path.get(key)
        if local_filename is not None:
            local_seen.add(local_filename)
            continue
        if lexical_root is not None:
            try:
                resolved_candidate_root = lexical_root.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise CityLiteAuthorityError(
                    f"RivermarkSrc51 asset root is unresolved: {lexical_root}"
                ) from exc
            if not resolved_candidate_root.is_dir():
                raise CityLiteAuthorityError(
                    f"RivermarkSrc51 asset root is not a directory: {lexical_root}"
                )
            discovered_roots[_path_key(lexical_root)] = (
                lexical_root,
                resolved_candidate_root,
            )
        external_candidates[key] = path

    missing_local = sorted(set(AUTHORITY_SHA256) - local_seen)
    if missing_local:
        raise CityLiteAuthorityError(
            "resolved layer inventory is missing local City-Lite authority layers: "
            + ", ".join(missing_local)
        )
    if not external_candidates:
        raise CityLiteAuthorityError(
            "resolved layer inventory has no RivermarkSrc51 external layers"
        )
    if len(discovered_roots) > 1:
        raise CityLiteAuthorityError(
            "resolved layer inventory spans multiple RivermarkSrc51 asset roots"
        )

    if explicit_root is None:
        if len(discovered_roots) != 1:
            raise CityLiteAuthorityError(
                "cannot resolve a unique RivermarkSrc51 asset root"
            )
        receipt_root, resolved_root = next(iter(discovered_roots.values()))
    else:
        assert explicit_resolved_root is not None
        receipt_root = explicit_root
        resolved_root = explicit_resolved_root
        if discovered_roots and next(iter(discovered_roots)) != _path_key(receipt_root):
            raise CityLiteAuthorityError(
                "resolved layer inventory does not match the explicit RivermarkSrc51 asset root"
            )

    external_rows: list[dict[str, Any]] = []
    for path in external_candidates.values():
        try:
            relative = path.relative_to(resolved_root)
        except ValueError as exc:
            raise CityLiteAuthorityError(
                f"resolved layer escapes the RivermarkSrc51 asset root: {path}"
            ) from exc
        external_rows.append(
            {
                "root_relative_path": relative.as_posix(),
                **_layer_binding(path),
                "classification": "rivermarksrc51_external_authority",
            }
        )
    external_rows.sort(key=lambda row: str(row["root_relative_path"]).casefold())

    local_portable = [
        {
            "filename": row["filename"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in local_rows
    ]
    external_portable = [
        {
            "root_relative_path": row["root_relative_path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in external_rows
    ]
    local_inventory_sha256 = canonical_payload_sha256(local_portable)
    external_inventory_sha256 = canonical_payload_sha256(external_portable)
    inventory_sha256 = canonical_payload_sha256(
        {
            "schema": RIVERMARK_LAYER_INVENTORY_SCHEMA,
            "composition_mode": "selective_references_only",
            "selective_references": [list(value) for value in SELECTIVE_REFERENCES],
            "local_authority_layers": local_portable,
            "rivermarksrc51_external_layers": external_portable,
        }
    )

    return {
        "schema": RIVERMARK_LAYER_INVENTORY_SCHEMA,
        "composition_scope": {
            "mode": "selective_references_only",
            "selective_references": [
                {"source_prim": source, "destination_prim": destination}
                for source, destination in SELECTIVE_REFERENCES
            ],
            "whole_final_stage_inventory": False,
        },
        "asset_root": {
            "name": RIVERMARK_ASSET_ROOT_NAME,
            "path": str(receipt_root),
            "resolved_path": str(resolved_root),
            "exists": True,
            "is_directory": True,
        },
        "input_resolved_layer_count": len(supplied_layers),
        "ignored_anonymous_layer_count": anonymous_count,
        "local_authority_layer_count": len(local_rows),
        "local_authority_layers": local_rows,
        "local_authority_inventory_sha256": local_inventory_sha256,
        "rivermarksrc51_external_layer_count": len(external_rows),
        "rivermarksrc51_external_layers": external_rows,
        "rivermarksrc51_external_inventory_sha256": external_inventory_sha256,
        "inventory_sha256": inventory_sha256,
    }

def _receipt_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CityLiteAuthorityError(f"{label} must be a lowercase SHA-256")
    return value


def _receipt_count(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CityLiteAuthorityError(f"{label} must be an integer >= {minimum}")
    return value


def _exact_receipt_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CityLiteAuthorityError(f"{label} fields are not exact")
    return value


def _receipt_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CityLiteAuthorityError(f"{label} must be a nonempty normalized path")
    if not Path(value).is_absolute():
        raise CityLiteAuthorityError(f"{label} must be absolute")
    return value


def _relative_usd_layer_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CityLiteAuthorityError(f"{label} must be a nonempty relative path")
    if "\\" in value or value.startswith("/"):
        raise CityLiteAuthorityError(f"{label} must be a normalized POSIX path")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise CityLiteAuthorityError(f"{label} contains an unsafe path component")
    if ":" in components[0]:
        raise CityLiteAuthorityError(f"{label} must not contain a drive or URI")
    if not value.casefold().endswith((".usd", ".usda", ".usdc")):
        raise CityLiteAuthorityError(f"{label} must identify an OpenUSD layer")
    return value


def validate_rivermark_layer_inventory_receipt(
    receipt: Mapping[str, Any],
) -> None:
    """Validate a serialized layer receipt without touching the filesystem."""

    top = _exact_receipt_keys(
        receipt,
        {
            "schema",
            "composition_scope",
            "asset_root",
            "input_resolved_layer_count",
            "ignored_anonymous_layer_count",
            "local_authority_layer_count",
            "local_authority_layers",
            "local_authority_inventory_sha256",
            "rivermarksrc51_external_layer_count",
            "rivermarksrc51_external_layers",
            "rivermarksrc51_external_inventory_sha256",
            "inventory_sha256",
        },
        label="Rivermark layer inventory receipt",
    )
    if top["schema"] != RIVERMARK_LAYER_INVENTORY_SCHEMA:
        raise CityLiteAuthorityError("Rivermark layer inventory schema is invalid")

    expected_scope = {
        "mode": "selective_references_only",
        "selective_references": [
            {"source_prim": source, "destination_prim": destination}
            for source, destination in SELECTIVE_REFERENCES
        ],
        "whole_final_stage_inventory": False,
    }
    if top["composition_scope"] != expected_scope:
        raise CityLiteAuthorityError(
            "Rivermark layer inventory must bind exact selective references"
        )

    asset_root = _exact_receipt_keys(
        top["asset_root"],
        {"name", "path", "resolved_path", "exists", "is_directory"},
        label="Rivermark asset root",
    )
    if asset_root["name"] != RIVERMARK_ASSET_ROOT_NAME:
        raise CityLiteAuthorityError("Rivermark asset root name is invalid")
    root_path = _receipt_path(asset_root["path"], label="Rivermark asset root path")
    resolved_root_path = _receipt_path(
        asset_root["resolved_path"],
        label="resolved Rivermark asset root path",
    )
    if _basename(root_path).casefold() != RIVERMARK_ASSET_ROOT_NAME.casefold():
        raise CityLiteAuthorityError(
            "Rivermark asset root path does not name RivermarkSrc51"
        )
    if asset_root["exists"] is not True or asset_root["is_directory"] is not True:
        raise CityLiteAuthorityError(
            "Rivermark asset root receipt must declare an existing directory"
        )

    local_count = _receipt_count(
        top["local_authority_layer_count"],
        label="local_authority_layer_count",
        minimum=3,
    )
    local_layers = top["local_authority_layers"]
    if not isinstance(local_layers, list) or local_count != 3 or len(local_layers) != 3:
        raise CityLiteAuthorityError(
            "Rivermark receipt must contain exactly three local authority layers"
        )
    local_portable: list[dict[str, Any]] = []
    local_paths: set[str] = set()
    for index, raw_row in enumerate(local_layers):
        row = _exact_receipt_keys(
            raw_row,
            {"filename", "path", "size_bytes", "sha256", "classification"},
            label=f"local authority layer {index}",
        )
        filename = row["filename"]
        if not isinstance(filename, str) or filename not in AUTHORITY_SHA256:
            raise CityLiteAuthorityError(
                f"local authority layer {index} filename is invalid"
            )
        path = _receipt_path(row["path"], label=f"local authority layer {index} path")
        if _basename(path) != filename:
            raise CityLiteAuthorityError(
                f"local authority layer {index} path does not match its filename"
            )
        path_key = path.replace("\\", "/").casefold()
        if path_key in local_paths:
            raise CityLiteAuthorityError("local authority layer paths must be unique")
        local_paths.add(path_key)
        size_bytes = _receipt_count(
            row["size_bytes"],
            label=f"local authority layer {index} size_bytes",
            minimum=1,
        )
        digest = _receipt_sha256(
            row["sha256"],
            label=f"local authority layer {index} sha256",
        )
        if row["classification"] != "city_lite_local_authority":
            raise CityLiteAuthorityError(
                f"local authority layer {index} classification is invalid"
            )
        local_portable.append(
            {"filename": filename, "size_bytes": size_bytes, "sha256": digest}
        )
    expected_local_filenames = sorted(AUTHORITY_SHA256)
    if [row["filename"] for row in local_portable] != expected_local_filenames:
        raise CityLiteAuthorityError(
            "local authority layers must be complete, unique, and sorted"
        )

    external_count = _receipt_count(
        top["rivermarksrc51_external_layer_count"],
        label="rivermarksrc51_external_layer_count",
        minimum=1,
    )
    external_layers = top["rivermarksrc51_external_layers"]
    if (
        not isinstance(external_layers, list)
        or not external_layers
        or len(external_layers) != external_count
    ):
        raise CityLiteAuthorityError(
            "Rivermark receipt external layer count is invalid"
        )
    external_portable: list[dict[str, Any]] = []
    external_paths: set[str] = set()
    relative_paths: set[str] = set()
    resolved_root = Path(resolved_root_path)
    for index, raw_row in enumerate(external_layers):
        row = _exact_receipt_keys(
            raw_row,
            {
                "root_relative_path",
                "path",
                "size_bytes",
                "sha256",
                "classification",
            },
            label=f"Rivermark external layer {index}",
        )
        relative = _relative_usd_layer_path(
            row["root_relative_path"],
            label=f"Rivermark external layer {index} root_relative_path",
        )
        relative_key = relative.casefold()
        if relative_key in relative_paths:
            raise CityLiteAuthorityError(
                "Rivermark external relative paths must be unique"
            )
        relative_paths.add(relative_key)
        path = _receipt_path(
            row["path"],
            label=f"Rivermark external layer {index} path",
        )
        path_key = path.replace("\\", "/").casefold()
        if path_key in external_paths or path_key in local_paths:
            raise CityLiteAuthorityError(
                "Rivermark external layer paths must be unique and non-local"
            )
        external_paths.add(path_key)
        try:
            actual_relative = Path(path).relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise CityLiteAuthorityError(
                f"Rivermark external layer {index} escapes the resolved asset root"
            ) from exc
        if actual_relative.casefold() != relative_key:
            raise CityLiteAuthorityError(
                f"Rivermark external layer {index} relative path is inconsistent"
            )
        size_bytes = _receipt_count(
            row["size_bytes"],
            label=f"Rivermark external layer {index} size_bytes",
            minimum=1,
        )
        digest = _receipt_sha256(
            row["sha256"],
            label=f"Rivermark external layer {index} sha256",
        )
        if row["classification"] != "rivermarksrc51_external_authority":
            raise CityLiteAuthorityError(
                f"Rivermark external layer {index} classification is invalid"
            )
        external_portable.append(
            {
                "root_relative_path": relative,
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )
    if [row["root_relative_path"].casefold() for row in external_portable] != sorted(
        relative_paths
    ):
        raise CityLiteAuthorityError("Rivermark external layers must be sorted")

    anonymous_count = _receipt_count(
        top["ignored_anonymous_layer_count"],
        label="ignored_anonymous_layer_count",
    )
    input_count = _receipt_count(
        top["input_resolved_layer_count"],
        label="input_resolved_layer_count",
        minimum=1,
    )
    if input_count < anonymous_count + local_count + external_count:
        raise CityLiteAuthorityError(
            "input_resolved_layer_count is smaller than the bound inventory"
        )

    declared_local_hash = _receipt_sha256(
        top["local_authority_inventory_sha256"],
        label="local_authority_inventory_sha256",
    )
    declared_external_hash = _receipt_sha256(
        top["rivermarksrc51_external_inventory_sha256"],
        label="rivermarksrc51_external_inventory_sha256",
    )
    declared_inventory_hash = _receipt_sha256(
        top["inventory_sha256"],
        label="inventory_sha256",
    )
    expected_local_hash = canonical_payload_sha256(local_portable)
    expected_external_hash = canonical_payload_sha256(external_portable)
    expected_inventory_hash = canonical_payload_sha256(
        {
            "schema": RIVERMARK_LAYER_INVENTORY_SCHEMA,
            "composition_mode": "selective_references_only",
            "selective_references": [list(value) for value in SELECTIVE_REFERENCES],
            "local_authority_layers": local_portable,
            "rivermarksrc51_external_layers": external_portable,
        }
    )
    if declared_local_hash != expected_local_hash:
        raise CityLiteAuthorityError("local authority inventory hash mismatch")
    if declared_external_hash != expected_external_hash:
        raise CityLiteAuthorityError("Rivermark external inventory hash mismatch")
    if declared_inventory_hash != expected_inventory_hash:
        raise CityLiteAuthorityError("Rivermark overall inventory hash mismatch")

def _load_contract(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CityLiteAuthorityError(
            f"cannot read City-Lite contract {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise CityLiteAuthorityError(
            f"City-Lite contract must be a JSON object: {path}"
        )
    return value


def _basename(value: Any) -> str:
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

def validate_upstream_scene_contract(
    contract: Mapping[str, Any],
    *,
    path: Path | None = None,
    expected_payload_sha256: str | None = None,
) -> str:
    """Validate the signed content and static-only boundary of the contract."""

    source = str(path) if path is not None else "<in-memory City-Lite contract>"
    if contract.get("schema") != SCENE_CONTRACT_SCHEMA:
        raise CityLiteAuthorityError(f"invalid City-Lite contract schema: {source}")
    if contract.get("scene_id") != ENVIRONMENT_ID:
        raise CityLiteAuthorityError(f"invalid City-Lite scene_id: {source}")
    if contract.get("gate_status") != SCENE_CONTRACT_GATE_STATUS:
        raise CityLiteAuthorityError(f"City-Lite static gate did not pass: {source}")

    permissions = contract.get("permissions")
    if permissions != dict(EXPECTED_UPSTREAM_PERMISSIONS):
        raise CityLiteAuthorityError(
            f"City-Lite permissions must exactly preserve the static-only boundary: {source}"
        )
    if contract.get("isaac_started") is not False:
        raise CityLiteAuthorityError(f"City-Lite static contract must have isaac_started=false: {source}")
    if contract.get("simulation_app_started") is not False:
        raise CityLiteAuthorityError(
            f"City-Lite static contract must have simulation_app_started=false: {source}"
        )

    checks = contract.get("checks")
    if not isinstance(checks, Mapping) or not checks or any(
        value is not True for value in checks.values()
    ):
        raise CityLiteAuthorityError(
            f"City-Lite construction checks are missing or failed: {source}"
        )

    outputs = contract.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(_OUTPUT_BINDINGS):
        raise CityLiteAuthorityError(f"City-Lite output bindings are not exact: {source}")
    for key, filename in _OUTPUT_BINDINGS.items():
        item = outputs.get(key)
        if not isinstance(item, Mapping):
            raise CityLiteAuthorityError(f"missing City-Lite output binding {key}: {source}")
        if _basename(item.get("path")) != filename:
            raise CityLiteAuthorityError(f"invalid City-Lite output path for {key}: {source}")
        if item.get("sha256") != AUTHORITY_SHA256[filename]:
            raise CityLiteAuthorityError(f"invalid City-Lite output digest for {key}: {source}")
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CityLiteAuthorityError(f"invalid City-Lite output size for {key}: {source}")

    declared_payload = contract.get("payload_sha256")
    if not isinstance(declared_payload, str) or len(declared_payload) != 64:
        raise CityLiteAuthorityError(f"City-Lite payload_sha256 is invalid: {source}")
    unsigned = {key: value for key, value in contract.items() if key != "payload_sha256"}
    try:
        computed_payload = canonical_payload_sha256(unsigned)
    except (TypeError, ValueError) as exc:
        raise CityLiteAuthorityError(
            f"City-Lite contract cannot be canonically hashed: {source}"
        ) from exc
    if declared_payload != computed_payload:
        raise CityLiteAuthorityError(f"City-Lite contract payload hash mismatch: {source}")
    if expected_payload_sha256 is not None and declared_payload != expected_payload_sha256:
        raise CityLiteAuthorityError(f"unexpected City-Lite authority payload: {source}")
    return declared_payload

def resolve_city_lite_authority(contract_or_root: str | Path) -> CityLiteAuthority:
    """Resolve and hash-check the exact md_qd_swarm v1_r2 scene authority."""

    supplied = Path(contract_or_root).expanduser()
    contract_path = (
        supplied
        if supplied.suffix.lower() == ".json"
        else supplied / SCENE_CONTRACT_FILENAME
    ).resolve()
    root = contract_path.parent
    if not contract_path.is_file():
        raise CityLiteAuthorityError(
            f"City-Lite scene contract not found: {contract_path}"
        )

    contract = _load_contract(contract_path)
    payload_sha256 = validate_upstream_scene_contract(
        contract,
        path=contract_path,
        expected_payload_sha256=SCENE_CONTRACT_PAYLOAD_SHA256,
    )
    contract_sha256 = sha256_file(contract_path)
    if contract_sha256 != SCENE_CONTRACT_SHA256:
        raise CityLiteAuthorityError(
            "City-Lite scene contract file hash does not match v1_r2: "
            f"expected {SCENE_CONTRACT_SHA256}, got {contract_sha256}"
        )

    outputs = contract["outputs"]
    paths: dict[str, Path] = {}
    actual_hashes: dict[str, str] = {}
    for filename, expected in AUTHORITY_SHA256.items():
        asset_path = root / filename
        if not asset_path.is_file():
            raise CityLiteAuthorityError(
                f"City-Lite authority asset not found: {asset_path}"
            )
        actual = sha256_file(asset_path)
        if actual != expected:
            raise CityLiteAuthorityError(
                f"City-Lite authority hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )
        output_key = next(
            key for key, bound_filename in _OUTPUT_BINDINGS.items()
            if bound_filename == filename
        )
        declared_size = outputs[output_key]["size_bytes"]
        if asset_path.stat().st_size != declared_size:
            raise CityLiteAuthorityError(
                f"City-Lite authority size mismatch for {filename}: "
                f"expected {declared_size}, got {asset_path.stat().st_size}"
            )
        paths[filename] = asset_path
        actual_hashes[filename] = actual

    return CityLiteAuthority(
        root=root,
        contract_path=contract_path,
        final_scene_path=paths[FINAL_SCENE_FILENAME],
        asset_paths=paths,
        sha256=actual_hashes,
        contract_sha256=contract_sha256,
        contract_payload_sha256=payload_sha256,
    )

def validate_static_scene_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate fields shared by capture, packing, and release gates."""

    if receipt.get("environment_id") != ENVIRONMENT_ID:
        raise CityLiteAuthorityError(
            f"expected environment_id={ENVIRONMENT_ID}, "
            f"got {receipt.get('environment_id')!r}"
        )
    if receipt.get("static_scene_authority_verified") is not True:
        raise CityLiteAuthorityError("static_scene_authority_verified must be true")
    if receipt.get("unresolved_reference_count") != 0:
        raise CityLiteAuthorityError(
            "City-Lite composition must have zero unresolved references"
        )
    if receipt.get("legacy_prim_count") != 0:
        raise CityLiteAuthorityError("legacy Mission/Drones prims must be absent")
    if receipt.get("forbidden_decoration_prim_count") != 0:
        raise CityLiteAuthorityError("removed decoration prims must be absent")

    try:
        validate_city_task_obstacle_material_closure_receipt(
            receipt.get("city_task_obstacle_material_closure", {})
        )
    except CityLiteAuthorityError as exc:
        raise CityLiteAuthorityError(
            "CityTaskObstacles material closure receipt is invalid"
        ) from exc

    scene_contract = receipt.get("scene_contract")
    expected_contract = {
        "sha256": SCENE_CONTRACT_SHA256,
        "payload_sha256": SCENE_CONTRACT_PAYLOAD_SHA256,
        "schema": SCENE_CONTRACT_SCHEMA,
        "gate_status": SCENE_CONTRACT_GATE_STATUS,
        "permissions": dict(EXPECTED_UPSTREAM_PERMISSIONS),
    }
    if not isinstance(scene_contract, Mapping) or any(
        scene_contract.get(key) != value for key, value in expected_contract.items()
    ):
        raise CityLiteAuthorityError("scene_contract does not bind the exact v1_r2 authority")

    assets = receipt.get("authority_assets")
    if not isinstance(assets, Mapping):
        raise CityLiteAuthorityError("authority_assets must be an object")
    for filename, expected in AUTHORITY_SHA256.items():
        item = assets.get(filename)
        if not isinstance(item, Mapping) or item.get("sha256") != expected:
            raise CityLiteAuthorityError(
                f"missing or invalid authority digest: {filename}"
            )

    references = receipt.get("selective_references")
    expected_references = [
        {"source_prim": source, "destination_prim": destination}
        for source, destination in SELECTIVE_REFERENCES
    ]
    if references != expected_references:
        raise CityLiteAuthorityError(
            "scene did not use the required selective references"
        )

    if receipt.get("stage_units") != {
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "time_codes_per_second": 60.0,
        "frames_per_second": 60.0,
    }:
        raise CityLiteAuthorityError("City-Lite stage units are not exact")
    if receipt.get("flight_contract") != flight_contract_payload():
        raise CityLiteAuthorityError("City-Lite flight contract is missing or stale")

    collision_counts = receipt.get("native_collision_counts")
    if collision_counts != dict(EXPECTED_NATIVE_COLLISION_COUNTS):
        raise CityLiteAuthorityError(
            "native collision counts do not match the selective City-Lite authority"
        )

    for key in (
        "scene_runtime_admission",
        "formal_collection",
        "formal_benchmark_admission",
    ):
        if receipt.get(key) is not False:
            raise CityLiteAuthorityError(
                f"the static scene receipt must preserve {key}=false"
            )
