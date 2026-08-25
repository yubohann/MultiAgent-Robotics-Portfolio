"""Load an AeroCityBench stage in Isaac Sim and capture RGB/depth health evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaacsim import SimulationApp

from aerocity_bench.canonical import content_hash
from aerocity_bench.geometry import review_camera_pose
from aerocity_bench.isaac_bridge import (
    REVIEW_BASE_FRAMES,
    VISUAL_REVIEW_EVIDENCE_SCOPE,
    aggregate_review_instance_visibility,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--view",
        choices=("overview", "street", "targets", "review"),
        default="overview",
    )
    parser.add_argument(
        "--episode",
        type=Path,
        help="evaluator-private episode JSON; required only for the targets view",
    )
    parser.add_argument(
        "--authority-record",
        type=Path,
        help="layout authority record; required for instance-bound review capture",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    return parser.parse_args()


ARGS = _arguments()
ARGS.output.mkdir(parents=True, exist_ok=True)


def _mark(message: str) -> None:
    with (ARGS.output / f"{ARGS.view}_progress.log").open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


_mark("before_simulation_app")
APP = SimulationApp(
    {
        "headless": True,
        "width": ARGS.width,
        "height": ARGS.height,
        "renderer": "RaytracedLighting",
    }
)
_mark("after_simulation_app")

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.semantics import add_labels  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_rgb(data: object, path: Path) -> dict[str, object]:
    array = np.asarray(data)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise RuntimeError(f"invalid RGB frame shape: {array.shape}")
    array = np.clip(array[..., :3], 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)
    grayscale = array.astype(np.float32).mean(axis=2)
    return {
        "shape": list(array.shape),
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std()), 6),
        "nonblack_fraction": round(float(np.mean(grayscale > 4.0)), 6),
        "nonwhite_fraction": round(float(np.mean(grayscale < 251.0)), 6),
        "review_marker_pixels": _review_marker_pixels(array),
        "sha256": _sha256(path),
    }


def _review_marker_pixels(array: object) -> dict[str, int]:
    """Count deliberately saturated review colors after Isaac rendering."""
    rgb = np.asarray(array, dtype=np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    masks = {
        "drone_start_yellow": (
            (red >= 180)
            & (green >= 150)
            & (blue <= 150)
            & ((red - blue) >= 50)
            & ((green - blue) >= 45)
        ),
        "roof_red": (
            (red >= 180)
            & (green <= 170)
            & (blue <= 170)
            & ((red - green) >= 45)
            & ((red - blue) >= 30)
        ),
        "facade_magenta": (
            (red >= 170)
            & (blue >= 125)
            & (green <= 175)
            & ((red - green) >= 35)
            & ((blue - green) >= 20)
        ),
        "entrance_green": (
            (green >= 130)
            & (red <= 180)
            & (blue <= 180)
            & ((green - red) >= 20)
            & ((green - blue) >= 15)
        ),
        "rubble_orange": (
            (red >= 175)
            & (green >= 80)
            & (green <= 195)
            & (blue <= 120)
            & ((red - green) >= 25)
            & ((green - blue) >= 35)
        ),
    }
    return {name: int(mask.sum()) for name, mask in masks.items()}


def _review_marker_visibility(
    frames: dict[str, dict[str, object]], private_target_audit: dict[str, object]
) -> dict[str, object]:
    keys = (
        "drone_start_yellow",
        "roof_red",
        "facade_magenta",
        "entrance_green",
        "rubble_orange",
    )
    totals = {key: 0 for key in keys}
    by_frame: dict[str, dict[str, int]] = {}
    for name, frame in frames.items():
        counts = dict(frame["rgb"]["review_marker_pixels"])
        by_frame[name] = {key: int(counts.get(key, 0)) for key in keys}
        for key in keys:
            totals[key] += by_frame[name][key]

    minimum_pixels = 24
    support_to_key = {
        "roof": "roof_red",
        "facade_marker_site": "facade_magenta",
        "entrance": "entrance_green",
        "rubble": "rubble_orange",
    }
    required = {"drone_start_yellow": minimum_pixels}
    for support_class in private_target_audit["support_classes"]:
        required[support_to_key[str(support_class)]] = minimum_pixels
    checks = {
        key: {
            "required_pixels": threshold,
            "observed_pixels": totals[key],
            "status": "PASS" if totals[key] >= threshold else "FAIL",
        }
        for key, threshold in required.items()
    }
    start_positions = private_target_audit.get("start_positions", [])
    if not isinstance(start_positions, list):
        start_positions = []
    expected_start_frames = [
        f"start_close_{index:03d}" for index in range(len(start_positions))
    ]
    observed_start_frames = sorted(
        name for name in by_frame if name.startswith("start_close_")
    )
    checks["start_close_frame_count"] = {
        "required_frames": len(expected_start_frames),
        "observed_frames": len(observed_start_frames),
        "status": (
            "PASS" if observed_start_frames == expected_start_frames else "FAIL"
        ),
    }
    for frame_name in expected_start_frames:
        yellow_pixels = by_frame.get(frame_name, {}).get("drone_start_yellow", 0)
        checks[f"{frame_name}_drone_start_yellow"] = {
            "required_pixels": minimum_pixels,
            "observed_pixels": yellow_pixels,
            "frame_present": frame_name in by_frame,
            "status": "PASS" if yellow_pixels >= minimum_pixels else "FAIL",
        }
    return {
        "status": (
            "PASS" if all(check["status"] == "PASS" for check in checks.values()) else "FAIL"
        ),
        "method": "saturated_review_color_pixel_audit",
        "scope": "review_overlay_visibility_only_not_per_target_occlusion_proof",
        "totals": totals,
        "by_frame": by_frame,
        "checks": checks,
    }


def _save_depth(data: object, path: Path) -> dict[str, object]:
    array = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(array) & (array > 0)
    if not finite.any():
        raise RuntimeError("depth frame contains no finite positive pixels")
    values = array[finite]
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if not high > low:
        high = low + 1.0
    normalized = np.zeros(array.shape, dtype=np.float32)
    normalized[finite] = 1.0 - np.clip((array[finite] - low) / (high - low), 0.0, 1.0)
    image = np.round(normalized * 65535.0).astype(np.uint16)
    Image.fromarray(image).save(path)
    return {
        "shape": list(array.shape),
        "finite_fraction": round(float(finite.mean()), 6),
        "minimum_m": round(float(values.min()), 6),
        "median_m": round(float(np.median(values)), 6),
        "maximum_m": round(float(values.max()), 6),
        "sha256": _sha256(path),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_instance_segmentation(data: object, name: str) -> dict[str, object]:
    if not isinstance(data, dict) or "data" not in data:
        raise RuntimeError("instance segmentation did not return data and metadata")
    mask = np.asarray(data["data"])
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise RuntimeError(f"invalid instance segmentation shape: {mask.shape}")
    if mask.dtype.kind not in {"u", "i"}:
        raise RuntimeError(f"invalid instance segmentation dtype: {mask.dtype}")
    mask = mask.astype(np.uint32, copy=False)
    info = data.get("info", {})
    if not isinstance(info, dict):
        raise RuntimeError("instance segmentation metadata is not a mapping")
    id_to_labels = _json_safe(info.get("idToLabels", {}))
    id_to_semantics = _json_safe(info.get("idToSemantics", {}))
    if not isinstance(id_to_labels, dict) or not id_to_labels:
        raise RuntimeError("instance segmentation returned no idToLabels mapping")
    ids, pixel_counts = np.unique(mask, return_counts=True)
    id_pixel_counts = {
        str(int(identifier)): int(count)
        for identifier, count in zip(ids.tolist(), pixel_counts.tolist(), strict=True)
    }
    mask_path = ARGS.output / f"{name}_instance_segmentation.npz"
    np.savez_compressed(mask_path, data=mask)
    labels_payload = {
        "schema": "org.aerocity.bench.review-instance-labels.v1",
        "frame": name,
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "idToLabels": id_to_labels,
        "idToSemantics": id_to_semantics,
    }
    labels_payload["mapping_hash"] = content_hash(labels_payload)
    labels_path = ARGS.output / f"{name}_instance_labels.json"
    labels_path.write_text(
        json.dumps(labels_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "mask_path": mask_path.name,
        "mask_sha256": _sha256(mask_path),
        "labels_path": labels_path.name,
        "labels_sha256": _sha256(labels_path),
        "mapping_hash": labels_payload["mapping_hash"],
        "id_pixel_counts": id_pixel_counts,
        "id_to_labels": id_to_labels,
        "id_to_semantics": id_to_semantics,
    }


def _capture(
    name: str,
    position: tuple[float, float, float],
    look_at: tuple[float, float, float],
    *,
    focal_length_mm: float = 24.0,
) -> dict[str, object]:
    _mark("before_camera")
    overlay_mode = _set_review_overlay_mode(name)
    camera = rep.create.camera(position=position, look_at=look_at, focal_length=focal_length_mm)
    render_product = rep.create.render_product(camera, resolution=(ARGS.width, ARGS.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu")
    instance = rep.AnnotatorRegistry.get_annotator(
        "instance_segmentation", init_params={"colorize": False}, device="cpu"
    )
    rgb.attach([render_product])
    depth.attach([render_product])
    instance.attach([render_product])
    _mark("after_annotator_attach")
    for _ in range(24):
        APP.update()
    rgb_data: object | None = None
    depth_data: object | None = None
    instance_data: object | None = None
    for render_attempt in range(20):
        _mark(f"before_orchestrator_step={render_attempt}")
        rep.orchestrator.step()
        for _ in range(4):
            APP.update()
        rgb_data = rgb.get_data()
        depth_data = depth.get_data()
        instance_data = instance.get_data()
        rgb_array = np.asarray(rgb_data)
        depth_array = np.asarray(depth_data)
        instance_array = np.asarray(
            instance_data.get("data", []) if isinstance(instance_data, dict) else []
        )
        _mark(
            f"render_attempt={render_attempt},rgb={rgb_array.shape},"
            f"depth={depth_array.shape},instance={instance_array.shape}"
        )
        if (
            rgb_array.ndim == 3
            and rgb_array.shape[-1] >= 3
            and depth_array.ndim == 2
            and instance_array.ndim in {2, 3}
            and isinstance(instance_data, dict)
            and bool(instance_data.get("info", {}).get("idToLabels", {}))
        ):
            break
    else:
        raise RuntimeError("Replicator did not produce ready RGB/depth buffers after 20 attempts")
    _mark("after_render_updates")
    rgb_path = ARGS.output / f"{name}_rgb.png"
    depth_path = ARGS.output / f"{name}_depth.png"
    result = {
        "review_overlay_mode": overlay_mode,
        "camera_position": list(position),
        "look_at": list(look_at),
        "focal_length_mm": focal_length_mm,
        "rgb": _save_rgb(rgb_data, rgb_path),
        "depth": _save_depth(depth_data, depth_path),
        "instance_segmentation": _save_instance_segmentation(instance_data, name),
    }
    _mark("after_frame_save")
    return result


def _material(stage: object, name: str, color: Gf.Vec3f) -> object:
    material = UsdShade.Material.Define(stage, Sdf.Path(f"/World/ReviewMaterials/{name}"))
    shader = UsdShade.Shader.Define(
        stage, Sdf.Path(f"/World/ReviewMaterials/{name}/PreviewSurface")
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(color * 0.65)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind(geometry: object, material: object, color: Gf.Vec3f) -> None:
    geometry.CreateDisplayColorAttr([color])
    UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)


def _set_review_overlay_mode(frame_name: str) -> str:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("cannot switch review overlay without a loaded stage")
    local_context = frame_name.startswith("target_close_")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/EvaluatorPrivateAudit/Target_"):
            continue
        if path.endswith(("/Marker", "/Beacon")):
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeInvisible() if local_context else imageable.MakeVisible()
        elif path.endswith("/LocalMarker"):
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeVisible() if local_context else imageable.MakeInvisible()
    return "local_context" if local_context else "overview_highlight"


def _add_review_overlay(stage: object, episode_path: Path) -> dict[str, object]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    colors = {
        "roof": Gf.Vec3f(0.95, 0.12, 0.10),
        "facade": Gf.Vec3f(0.95, 0.08, 0.75),
        "facade_marker_site": Gf.Vec3f(0.95, 0.08, 0.75),
        "entrance": Gf.Vec3f(0.10, 0.85, 0.30),
        "rubble": Gf.Vec3f(0.95, 0.70, 0.08),
    }
    materials = {
        name: _material(stage, f"Target_{name}", color) for name, color in colors.items()
    }
    drone_color = Gf.Vec3f(0.98, 0.88, 0.05)
    drone_material = _material(stage, "DroneStart", drone_color)
    for index, target in enumerate(episode["targets"]):
        support_class = str(target["support_class"])
        color = colors[support_class]
        root = Sdf.Path(f"/World/EvaluatorPrivateAudit/Target_{index:03d}")
        root_prim = stage.DefinePrim(root, "Xform")
        add_labels(root_prim, labels=[f"target_{index:03d}"], instance_name="class")
        sphere = UsdGeom.Sphere.Define(
            stage, root.AppendChild("Marker")
        )
        sphere.CreateRadiusAttr(1.15)
        _bind(sphere, materials[support_class], color)
        x, y, z = [float(value) for value in target["position"]]
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        local_marker = UsdGeom.Sphere.Define(stage, root.AppendChild("LocalMarker"))
        local_marker.CreateRadiusAttr(0.22)
        _bind(local_marker, materials[support_class], color)
        UsdGeom.Xformable(local_marker).AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        beacon = UsdGeom.Cylinder.Define(stage, root.AppendChild("Beacon"))
        beacon.CreateAxisAttr("Z")
        beacon.CreateRadiusAttr(0.13)
        beacon.CreateHeightAttr(4.5)
        _bind(beacon, materials[support_class], color)
        UsdGeom.Xformable(beacon).AddTranslateOp().Set(Gf.Vec3d(x, y, z + 2.25))
    start_marker_footprint_radius_m = 0.62
    for index, start in enumerate(episode["starts"]):
        root = Sdf.Path(f"/World/EvaluatorPrivateAudit/DroneStart_{index:03d}")
        root_prim = stage.DefinePrim(root, "Xform")
        add_labels(root_prim, labels=[f"drone_start_{index:03d}"], instance_name="class")
        x, y, z = [float(value) for value in start["position"]]
        body = UsdGeom.Sphere.Define(stage, root.AppendChild("Body"))
        body.CreateRadiusAttr(0.18)
        _bind(body, drone_material, drone_color)
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        for arm_name, scale in (
            ("ArmX", (0.36, 0.035, 0.025)),
            ("ArmY", (0.035, 0.36, 0.025)),
        ):
            arm = UsdGeom.Cube.Define(stage, root.AppendChild(arm_name))
            arm.CreateSizeAttr(2.0)
            _bind(arm, drone_material, drone_color)
            transform = UsdGeom.Xformable(arm)
            transform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            transform.AddScaleOp().Set(Gf.Vec3d(*scale))
        for rotor_index, (dx, dy) in enumerate(
            ((0.35, 0.35), (0.35, -0.35), (-0.35, 0.35), (-0.35, -0.35))
        ):
            rotor = UsdGeom.Cylinder.Define(stage, root.AppendChild(f"Rotor_{rotor_index}"))
            rotor.CreateAxisAttr("Z")
            rotor.CreateRadiusAttr(0.12)
            rotor.CreateHeightAttr(0.05)
            _bind(rotor, drone_material, drone_color)
            UsdGeom.Xformable(rotor).AddTranslateOp().Set(Gf.Vec3d(x + dx, y + dy, z))
        ground = UsdGeom.Cylinder.Define(stage, root.AppendChild("GroundMarker"))
        ground.CreateAxisAttr("Z")
        ground.CreateRadiusAttr(0.46)
        ground.CreateHeightAttr(0.06)
        _bind(ground, drone_material, drone_color)
        UsdGeom.Xformable(ground).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.12))
        beacon = UsdGeom.Cylinder.Define(stage, root.AppendChild("Beacon"))
        beacon.CreateAxisAttr("Z")
        beacon.CreateRadiusAttr(0.07)
        beacon.CreateHeightAttr(3.0)
        _bind(beacon, drone_material, drone_color)
        UsdGeom.Xformable(beacon).AddTranslateOp().Set(Gf.Vec3d(x, y, z + 1.5))
    start_positions = [
        tuple(float(value) for value in start["position"]) for start in episode["starts"]
    ]
    start_pair_distances = [
        float(np.linalg.norm(np.asarray(first) - np.asarray(second)))
        for first_index, first in enumerate(start_positions)
        for second in start_positions[first_index + 1 :]
    ]
    minimum_start_separation = min(start_pair_distances, default=float("inf"))
    return {
        "episode_hash": episode["episode_hash"],
        "target_count": len(episode["targets"]),
        "start_count": len(episode["starts"]),
        "formal_score_eligible": bool(episode.get("formal_score_eligible", False)),
        "support_classes": sorted({target["support_class"] for target in episode["targets"]}),
        "altitude_bands": sorted({target["altitude_band"] for target in episode["targets"]}),
        "target_positions": [target["position"] for target in episode["targets"]],
        "target_legal_observation_review_poses": [
            target["local_review_pose"] for target in episode["targets"]
        ],
        "target_local_context_review_poses": [
            target["local_context_review_pose"] for target in episode["targets"]
        ],
        "target_local_context_review_look_ats": [
            target["local_context_review_look_at"] for target in episode["targets"]
        ],
        "target_local_context_review_metadata": [
            {
                "target_id": target["target_id"],
                "distance_m": target["local_context_review_distance_m"],
                "clearance_m": target["local_context_review_clearance_m"],
                "oblique_lateral_ratio": target["local_context_review_oblique_lateral_ratio"],
                "visible_context_collider_ids": target[
                    "local_context_visible_collider_ids"
                ],
            }
            for target in episode["targets"]
        ],
        "start_positions": [start["position"] for start in episode["starts"]],
        "start_marker_footprint_radius_m": start_marker_footprint_radius_m,
        "minimum_start_separation_m": round(minimum_start_separation, 6),
        "start_markers_overlap_free": (
            minimum_start_separation >= 2.0 * start_marker_footprint_radius_m
        ),
        "marker_semantics": {
            "drone_start": "yellow compact programmatic quadrotor plus vertical beacon",
            "roof_target": "red sphere plus vertical beacon",
            "facade_target": "magenta sphere plus vertical beacon",
            "entrance_target": "green sphere plus vertical beacon",
            "rubble_target": "orange sphere plus vertical beacon",
        },
        "expected_instance_labels": [
            *[f"target_{index:03d}" for index in range(len(episode["targets"]))],
            *[f"drone_start_{index:03d}" for index in range(len(episode["starts"]))],
        ],
    }


def _contact_sheet(
    frame_names: list[str],
    output: Path,
    *,
    layout_id: str,
    target_count: int,
    start_count: int,
    filename: str = "review_contact_sheet.png",
    thumbnail_size: tuple[int, int] = (480, 320),
    columns: int = 2,
) -> dict[str, object]:
    rows = (len(frame_names) + columns - 1) // columns
    header = 106
    sheet = Image.new("RGB", (thumbnail_size[0] * columns, header + thumbnail_size[1] * rows))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, header), fill=(20, 22, 24))
    draw.text((14, 10), f"AeroCityBench Isaac review | layout={layout_id}", fill=(245, 245, 245))
    draw.text(
        (14, 36),
        f"YELLOW={start_count} UAV starts | highlighted targets={target_count}",
        fill=(255, 238, 65),
    )
    draw.text(
        (14, 62),
        "Targets: RED=roof  MAGENTA=facade  GREEN=entrance  ORANGE=rubble",
        fill=(245, 245, 245),
    )
    for index, name in enumerate(frame_names):
        image = Image.open(output / f"{name}_rgb.png").convert("RGB")
        image.thumbnail(thumbnail_size)
        x = (index % columns) * thumbnail_size[0]
        y = header + (index // columns) * thumbnail_size[1]
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + 180, y + 24), fill=(0, 0, 0))
        draw.text((x + 6, y + 5), name, fill=(255, 255, 255))
    path = output / filename
    sheet.save(path)
    return {"path": path.name, "shape": [sheet.height, sheet.width, 3], "sha256": _sha256(path)}


def _frame_diversity(frames: dict[str, dict[str, object]]) -> dict[str, object]:
    pose_groups: dict[str, list[str]] = {}
    image_groups: dict[str, list[str]] = {}
    for name, frame in frames.items():
        pose_key = json.dumps(
            {
                "camera_position": frame["camera_position"],
                "look_at": frame["look_at"],
            },
            sort_keys=True,
        )
        pose_groups.setdefault(pose_key, []).append(name)
        image_groups.setdefault(str(frame["rgb"]["sha256"]), []).append(name)
    duplicate_poses = [names for names in pose_groups.values() if len(names) > 1]
    duplicate_images = [names for names in image_groups.values() if len(names) > 1]
    return {
        "status": "PASS" if not duplicate_poses and not duplicate_images else "FAIL",
        "method": "exact_camera_pose_and_rgb_sha256_duplicate_gate",
        "frame_count": len(frames),
        "duplicate_pose_groups": duplicate_poses,
        "duplicate_rgb_groups": duplicate_images,
    }


def main() -> int:
    stage_path = ARGS.stage.resolve()
    ARGS.output.mkdir(parents=True, exist_ok=True)
    context = omni.usd.get_context()
    _mark("before_open_stage")
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"Isaac could not open stage: {stage_path}")
    for _ in range(20):
        APP.update()
    _mark("after_open_stage")
    stage = context.get_stage()
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError("stage has no valid default prim")

    city_path = stage_path.parent / "cityspec.json"
    city = json.loads(city_path.read_text(encoding="utf-8"))
    private_target_audit = None
    authority_record = None
    if ARGS.view in {"targets", "review"}:
        if ARGS.episode is None:
            raise RuntimeError("--episode is required for targets/review views")
        private_target_audit = _add_review_overlay(stage, ARGS.episode.resolve())
    if ARGS.view == "review":
        if ARGS.authority_record is None:
            raise RuntimeError("--authority-record is required for review capture")
        authority_record = json.loads(
            ARGS.authority_record.resolve().read_text(encoding="utf-8")
        )
        authority_payload = dict(authority_record)
        authority_hash = str(authority_payload.pop("authority_record_hash", ""))
        if content_hash(authority_payload) != authority_hash:
            raise RuntimeError("review authority record hash mismatch")
        authority_checks = {
            "layout_id": city["layout_id"],
            "layout_hash": city["layout_hash"],
            "cityspec_sha256": _sha256(city_path),
            "stage_sha256": _sha256(stage_path),
            "scene_sha256": _sha256(stage_path.parent / "scene.usda"),
            "collision_sha256": _sha256(stage_path.parent / "collision.usda"),
        }
        if any(authority_record.get(key) != value for key, value in authority_checks.items()):
            raise RuntimeError("review authority record does not match the loaded layout bytes")
    expected_collision_count = (
        1
        + sum(len(building["components"]) for building in city["buildings"])
        + len(city["obstacles"])
    )
    collision_paths = []
    visual_collision_paths = []
    review_overlay_collision_paths = []
    review_overlay_rigid_body_paths = []
    prim_count = 0
    for prim in stage.Traverse():
        prim_count += 1
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(path)
            if path.startswith("/World/VisualDecorations/") or path.startswith(
                "/World/UrbanGroundDetail/"
            ):
                visual_collision_paths.append(path)
            if path.startswith("/World/EvaluatorPrivateAudit/"):
                review_overlay_collision_paths.append(path)
        if path.startswith("/World/EvaluatorPrivateAudit/") and prim.HasAPI(
            UsdPhysics.RigidBodyAPI
        ):
            review_overlay_rigid_body_paths.append(path)
    if len(collision_paths) != expected_collision_count:
        raise RuntimeError(
            f"collision prim count {len(collision_paths)} != expected {expected_collision_count}"
        )
    if visual_collision_paths:
        raise RuntimeError(f"visual references carry collision APIs: {visual_collision_paths[:10]}")
    if review_overlay_collision_paths or review_overlay_rigid_body_paths:
        raise RuntimeError(
            "private review overlays changed physical execution: "
            f"collision={review_overlay_collision_paths[:10]}, "
            f"rigid_body={review_overlay_rigid_body_paths[:10]}"
        )
    _mark("after_collision_audit")

    sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/AuditSun"))
    sun.CreateIntensityAttr(3500.0)
    sun.CreateAngleAttr(0.8)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 35.0))
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/AuditDome"))
    dome.CreateIntensityAttr(650.0)
    size = float(city["size_m"])
    maximum_height = float(city["metrics"]["height_max_m"])
    vertical_roads = []
    for road in city["roads"]:
        if road.get("axis") == "x":
            vertical_roads.append((float(road["x"]), road))
        elif "start" in road and "end" in road:
            delta_x = float(road["end"][0]) - float(road["start"][0])
            delta_y = float(road["end"][1]) - float(road["start"][1])
            if abs(delta_y) >= abs(delta_x):
                center_x = (float(road["start"][0]) + float(road["end"][0])) / 2.0
                vertical_roads.append((center_x, road))
    street_x = vertical_roads[0][0] if vertical_roads else 0.0
    views = {
        "overview": (
            (size * 1.05, -size * 1.05, max(size * 1.25, maximum_height + 55.0)),
            (0.0, 0.0, maximum_height * 0.28),
        ),
        "street": (
            (street_x, -size * 0.43, 7.5),
            (street_x, size * 0.32, 11.0),
        ),
        "targets": (
            (size * 0.92, -size * 0.92, max(size * 1.02, maximum_height + 45.0)),
            (0.0, 0.0, maximum_height * 0.32),
        ),
    }
    if ARGS.view == "review":
        review_height = max(size * 1.22, maximum_height + 58.0)
        middle_height = max(28.0, maximum_height * 0.58)
        start_positions = private_target_audit["start_positions"]
        start_camera, start_center = review_camera_pose(start_positions, city)
        review_views = {
            "overview_ne": ((size * 1.05, -size * 1.05, review_height), (0.0, 0.0, 15.0)),
            "overview_nw": ((-size * 1.05, -size * 1.05, review_height), (0.0, 0.0, 15.0)),
            "overview_se": ((size * 1.05, size * 1.05, review_height), (0.0, 0.0, 15.0)),
            "overview_sw": ((-size * 1.05, size * 1.05, review_height), (0.0, 0.0, 15.0)),
            "top": ((0.2, -0.2, size * 1.85), (0.0, 0.0, 0.0)),
            "north_low": ((0.0, -size * 1.08, middle_height), (0.0, 0.0, 15.0)),
            "south_low": ((0.0, size * 1.08, middle_height), (0.0, 0.0, 15.0)),
            "east_low": ((size * 1.08, 0.0, middle_height), (0.0, 0.0, 15.0)),
            "west_low": ((-size * 1.08, 0.0, middle_height), (0.0, 0.0, 15.0)),
            "starts_close": (
                start_camera,
                start_center,
            ),
        }
        if tuple(review_views) != REVIEW_BASE_FRAMES:
            raise RuntimeError("review base-frame contract drifted from the package contract")
        local_poses = private_target_audit["target_local_context_review_poses"]
        local_look_ats = private_target_audit["target_local_context_review_look_ats"]
        if (
            len(local_poses) != int(private_target_audit["target_count"])
            or len(local_poses) != len(local_look_ats)
        ):
            raise RuntimeError("target local-review pose count mismatch")
        local_views = {
            f"target_close_{index:03d}": (
                tuple(float(value) for value in pose["position"]),
                tuple(float(value) for value in look_at),
            )
            for index, (pose, look_at) in enumerate(
                zip(local_poses, local_look_ats, strict=True)
            )
        }
        review_views.update(local_views)
        start_local_views = {}
        for index, start_position in enumerate(private_target_audit["start_positions"]):
            start_camera, start_center = review_camera_pose([start_position], city)
            start_local_views[f"start_close_{index:03d}"] = (start_camera, start_center)
        review_views.update(start_local_views)
        frames = {
            name: _capture(
                name,
                *camera,
                focal_length_mm=18.0 if name.startswith("target_close_") else 24.0,
            )
            for name, camera in review_views.items()
        }
        instance_visibility = aggregate_review_instance_visibility(
            {
                name: {
                    "id_pixel_counts": frame["instance_segmentation"]["id_pixel_counts"],
                    "id_to_labels": frame["instance_segmentation"]["id_to_labels"],
                    "id_to_semantics": frame["instance_segmentation"]["id_to_semantics"],
                }
                for name, frame in frames.items()
            },
            target_count=int(private_target_audit["target_count"]),
            start_count=int(private_target_audit["start_count"]),
            frame_pixel_count=ARGS.width * ARGS.height,
        )
        if instance_visibility["status"] != "PASS":
            raise RuntimeError(
                "per-instance review visibility failed: "
                f"missing={instance_visibility['missing_instances']}, "
                f"scene_overview_diagnostic={instance_visibility['unseen_in_scene_overviews']}, "
                f"local={instance_visibility['missing_local_targets']}, "
                f"oversized_local={instance_visibility['oversized_local_targets']}, "
                f"starts_close={instance_visibility['missing_starts_close']}, "
                f"start_local={instance_visibility['missing_start_local']}, "
                f"ambiguous={instance_visibility['ambiguous_instance_ids']}"
            )
        if not private_target_audit["start_markers_overlap_free"]:
            raise RuntimeError("review start-marker footprints overlap")
        frame_diversity = _frame_diversity(frames)
        if frame_diversity["status"] != "PASS":
            raise RuntimeError(f"review frames are duplicated: {frame_diversity}")
        marker_visibility = _review_marker_visibility(frames, private_target_audit)
        if marker_visibility["status"] != "PASS":
            raise RuntimeError(
                "per-start review-marker visibility failed: "
                f"checks={marker_visibility['checks']}"
            )
        contact_sheet = _contact_sheet(
            list(REVIEW_BASE_FRAMES),
            ARGS.output,
            layout_id=str(city["layout_id"]),
            target_count=int(private_target_audit["target_count"]),
            start_count=int(private_target_audit["start_count"]),
        )
        target_contact_sheet = _contact_sheet(
            [*local_views, *start_local_views],
            ARGS.output,
            layout_id=str(city["layout_id"]),
            target_count=int(private_target_audit["target_count"]),
            start_count=int(private_target_audit["start_count"]),
            filename="target_review_contact_sheet.png",
            thumbnail_size=(320, 240),
            columns=4,
        )
    else:
        position, look_at = views[ARGS.view]
        frames = {ARGS.view: _capture(ARGS.view, position, look_at)}
        contact_sheet = None
        target_contact_sheet = None
        marker_visibility = None
        instance_visibility = None
        frame_diversity = None
    for frame in frames.values():
        if float(frame["rgb"]["nonblack_fraction"]) < 0.10:
            raise RuntimeError("RGB frame is substantially blank")
        if float(frame["rgb"]["std"]) < 3.0:
            raise RuntimeError("RGB frame has degenerate pixel variance")
    report = {
        "schema": "org.aerocity.bench.isaac-scene-health.v6",
        "status": (
            "passed"
            if (
                marker_visibility is None
                or (
                    marker_visibility["status"] == "PASS"
                    and instance_visibility is not None
                    and instance_visibility["status"] == "PASS"
                    and frame_diversity is not None
                    and frame_diversity["status"] == "PASS"
                )
            )
            else "failed"
        ),
        "stage": str(stage_path),
        "stage_sha256": _sha256(stage_path),
        "scene_sha256": _sha256(stage_path.parent / "scene.usda"),
        "collision_sha256": _sha256(stage_path.parent / "collision.usda"),
        "layout_id": city["layout_id"],
        "layout_hash": city["layout_hash"],
        "prim_count": prim_count,
        "collision_prim_count": len(collision_paths),
        "expected_collision_prim_count": expected_collision_count,
        "visual_collision_prim_count": len(visual_collision_paths),
        "review_overlay_collision_prim_count": len(review_overlay_collision_paths),
        "review_overlay_rigid_body_prim_count": len(review_overlay_rigid_body_paths),
        "private_target_audit": private_target_audit,
        "view": ARGS.view,
        "frames": frames,
        "contact_sheet": contact_sheet,
        "target_contact_sheet": target_contact_sheet,
        "review_marker_visibility": marker_visibility,
        "instance_visibility": instance_visibility,
        "frame_diversity": frame_diversity,
        "authority_record": authority_record,
        "evidence_scope": (
            VISUAL_REVIEW_EVIDENCE_SCOPE
            if ARGS.view == "review"
            else "static_scene_health_only_not_N1_N2_or_N3_N5"
        ),
    }
    report["health_report_hash"] = content_hash(report)
    report_path = ARGS.output / f"isaac_scene_health_{ARGS.view}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _mark("after_report_write")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


exit_code = 0
try:
    exit_code = main()
except BaseException as exc:
    import traceback

    _mark(f"exception={type(exc).__name__}: {exc}")
    _mark(traceback.format_exc())
    exit_code = 1
finally:
    APP.close()
raise SystemExit(exit_code)
