#!/usr/bin/env python3
"""Render a compact H.264 video from a real headless localization run."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def load_xy(path: Path, limit: int = 12000) -> list[tuple[float, float]]:
    if not path.is_file():
        return []
    points: list[tuple[float, float]] = []
    data = path.read_text(encoding="ascii", errors="ignore").split("DATA ascii", 1)
    if len(data) != 2:
        return points
    for line in data[1].splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            x, y = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
            if len(points) >= limit:
                break
    return points


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/msyh.ttc")):
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_frame(path: Path, points: list[tuple[float, float]], event: dict, index: int, total: int, map_name: str) -> None:
    image = Image.new("RGB", (1280, 720), (247, 249, 251))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = font(30), font(22), font(17)
    draw.text((48, 30), "MID360 SIMULATION", fill=(10, 45, 75), font=title_font)
    draw.text((50, 72), "HEADLESS FIXED-MAP LOCALIZATION", fill=(20, 90, 100), font=body_font)
    draw.text((50, 108), "Evidence: gazebo_simulation | source: ROS JSONL + frozen PCD", fill=(85, 95, 105), font=small_font)
    plot = (50, 170, 780, 665)
    draw.rectangle(plot, fill=(255, 255, 255), outline=(155, 165, 175), width=2)
    if points:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        sx, sy = max(xmax - xmin, 1e-3), max(ymax - ymin, 1e-3)
        for x, y in points:
            px = int(plot[0] + 10 + (x - xmin) / sx * (plot[2] - plot[0] - 20))
            py = int(plot[3] - 10 - (y - ymin) / sy * (plot[3] - plot[1] - 20))
            draw.point((px, py), fill=(0, 114, 178))
    draw.text((70, 185), f"Frozen map XY projection: {map_name}", fill=(35, 50, 65), font=small_font)
    payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
    state = str(payload.get("tracking_state", payload.get("status", "RUNNING")))
    lock = event.get("value", payload.get("map_locked", "unknown"))
    quality = payload.get("quality", {}) if isinstance(payload.get("quality"), dict) else {}
    fitness = payload.get("last_fitness", quality.get("last_fitness", "unknown"))
    scan_points = payload.get("last_scan_points", quality.get("last_scan_points", "unknown"))
    pose_age = payload.get("pose_age_sec", "unknown")
    lines = [f"event {index + 1} / {total}", f"tracking state: {state}", f"map locked: {lock}", f"ICP fitness: {fitness}", f"scan points: {scan_points}", f"pose age (s): {pose_age}", "", "Rendered from the headless run's", "recorded diagnostics."]
    x, y = 850, 190
    for line in lines:
        draw.text((x, y), line, fill=(15, 60, 90) if line else (247, 249, 251), font=body_font if line.startswith("tracking") else small_font)
        y += 42 if line else 24
    image.save(path, format="PPM")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--map-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    rows = load_jsonl(args.run_dir / "localization_telemetry.jsonl")
    events = [row for row in rows if row.get("kind") in {"map_diagnostic", "pose_status", "map_locked", "map_to_odom_correction"}]
    if not events:
        raise SystemExit("no localization events found")
    if len(events) > 240:
        stride = max(1, len(events) // 240)
        events = events[::stride]
    summary = next((row for row in reversed(rows) if row.get("kind") == "summary"), None)
    if summary:
        final_event = {
            "kind": "map_diagnostic",
            "payload": {
                "status": summary.get("map_status", "unknown"),
                "tracking_state": summary.get("map_status", "unknown"),
                "map_locked": summary.get("map_locked_seen", False),
                "last_fitness": summary.get("last_fitness", "unknown"),
                "last_scan_points": summary.get("last_scan_points", "unknown"),
            },
            "value": summary.get("map_locked_seen", False),
        }
        events.extend([final_event] * 20)
    points = load_xy(args.map_file)
    frame_dir = args.run_dir / "video_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, event in enumerate(events):
        draw_frame(frame_dir / f"frame_{index:06d}.ppm", points, event, index, len(events), args.map_file.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def win_path(path: Path) -> str:
        raw = str(path)
        if raw.startswith("/mnt/") and len(raw) > 6:
            return raw[5].upper() + ":" + raw[6:].replace("/", "\\")
        return raw

    subprocess.run([args.ffmpeg, "-y", "-framerate", str(args.fps), "-i", win_path(frame_dir / "frame_%06d.ppm"), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", win_path(args.output)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
