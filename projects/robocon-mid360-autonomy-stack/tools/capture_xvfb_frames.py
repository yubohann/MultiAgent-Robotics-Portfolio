#!/usr/bin/env python3
"""Capture raw Gazebo GUI frames from an isolated Xvfb display."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image
from Xlib import X, display


def window_titles(root) -> list[str]:
    titles: list[str] = []
    for child in root.query_tree().children:
        try:
            name = child.get_wm_name()
        except Exception:  # X clients may disappear during enumeration.
            continue
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        if name:
            titles.append(str(name))
    return titles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--wait-sec", type=float, default=60.0)
    args = parser.parse_args()
    if args.duration_sec <= 0 or args.fps <= 0:
        raise SystemExit("duration and fps must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = display.Display(args.display)
    screen = connection.screen()
    root = screen.root
    width, height = screen.width_in_pixels, screen.height_in_pixels
    deadline = time.monotonic() + args.wait_sec
    titles: list[str] = []
    while time.monotonic() < deadline:
        titles = window_titles(root)
        if any("gazebo" in title.lower() for title in titles):
            break
        time.sleep(0.25)
    if not any("gazebo" in title.lower() for title in titles):
        raise SystemExit(f"Gazebo window not found on {args.display}; titles={titles}")

    interval = 1.0 / args.fps
    frames = max(1, int(round(args.duration_sec * args.fps)))
    frame_hashes: list[str] = []
    started = time.monotonic()
    for index in range(frames):
        image = root.get_image(0, 0, width, height, X.ZPixmap, 0xFFFFFFFF)
        rgb = Image.frombytes("RGB", (width, height), image.data, "raw", "BGRX")
        payload = rgb.tobytes()
        frame_hashes.append(hashlib.sha256(payload).hexdigest())
        rgb.save(args.output_dir / f"gazebo_{index:06d}.jpg", quality=95, subsampling=0)
        due = started + (index + 1) * interval
        time.sleep(max(0.0, due - time.monotonic()))
    connection.close()
    metadata = {
        "display": args.display,
        "width": width,
        "height": height,
        "fps": args.fps,
        "frames": frames,
        "titles_at_start": titles,
        "unique_frame_hashes": len(set(frame_hashes)),
        "capture_kind": "raw_gazebo_gui_from_isolated_xvfb",
    }
    (args.output_dir / "capture_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if metadata["unique_frame_hashes"] < 3:
        raise SystemExit("capture contained fewer than three distinct frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
