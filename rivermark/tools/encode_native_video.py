#!/usr/bin/env python3
"""Encode an MP4 directly from a native Rivermark Isaac frame archive.

No geometry, trajectory, label, or synthetic background is rendered here. The
input archive must have been produced by the native Isaac capture path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from rivermark_benchmark.frame_archive import ChunkedFrameArchive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_path(capture_dir: Path, view: str) -> Path:
    relative = {
        "overview": Path("sensors") / "overview_rgb.npz",
        "onboard": Path("sensors") / "onboard_rgbd.npz",
    }[view]
    path = capture_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"native RGB archive is missing: {path}")
    return path


def encode(capture_dir: Path, output: Path, *, view: str, fps: int, overwrite: bool) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required; no fallback renderer is provided")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    archive_path = _archive_path(capture_dir, view)
    with ChunkedFrameArchive(archive_path) as archive:
        if "rgb" not in archive.frame_fields:
            raise ValueError(f"native archive has no RGB frame field: {archive_path}")
        descriptor = archive.descriptor("rgb")
        if len(descriptor.shape) != 4 or descriptor.shape[-1] != 3 or descriptor.dtype.name != "uint8":
            raise ValueError(f"expected uint8 [T,H,W,3] RGB archive, got {descriptor}")
        _, height, width, _ = descriptor.shape
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            assert process.stdin is not None
            for index in range(archive.frame_count):
                frame = archive.frame("rgb", index)
                process.stdin.write(frame.tobytes(order="C"))
            process.stdin.close()
            return_code = process.wait()
        except BaseException:
            if process.stdin is not None:
                process.stdin.close()
            process.kill()
            process.wait()
            output.unlink(missing_ok=True)
            raise
        if return_code != 0 or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg failed to encode {output} (exit {return_code})")
        timestamps = archive.array("timestamps_ns")
        manifest = {
            "schema": "org.rivermark.native-isaac-video.v1",
            "capture_dir": str(capture_dir),
            "view": view,
            "source_archive": str(archive_path),
            "source_archive_sha256": _sha256(archive_path),
            "frame_count": archive.frame_count,
            "image_shape_hwc": [height, width, 3],
            "fps": fps,
            "simulation_time_start_ns": int(timestamps[0]),
            "simulation_time_end_ns": int(timestamps[-1]),
            "video_path": str(output),
            "video_sha256": _sha256(output),
            "rendering": "native Isaac RGB frames encoded without drawing or interpolation",
        }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--view", choices=("overview", "onboard"), default="overview")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    manifest = encode(args.capture_dir.resolve(), args.output.resolve(), view=args.view, fps=args.fps, overwrite=args.overwrite)
    print(f"encoded native Isaac video: {args.output}")
    print(f"wrote video manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
