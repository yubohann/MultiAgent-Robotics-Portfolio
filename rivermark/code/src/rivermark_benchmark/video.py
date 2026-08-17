"""Compatibility checks and deterministic H.264 transcoding for release videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .frame_archive import (
    LEGACY_FRAME_MEMBER_MAX_UNCOMPRESSED_BYTES,
    ChunkedFrameArchive,
    FrameArchiveError,
    is_chunked_frame_archive,
    oversized_legacy_frame_members,
)


FIXED_ROUTE_INDEPENDENT_VALIDATION_SCHEMA = "org.rivermark.isaac-independent-validation.v1"
STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA = (
    "org.rivermark.isaac-state-only-transfer-independent-validation.v1"
)
SUPPORTED_INDEPENDENT_VALIDATION_SCHEMAS = frozenset(
    {
        FIXED_ROUTE_INDEPENDENT_VALIDATION_SCHEMA,
        STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA,
    }
)
STATE_ONLY_TRANSFER_CONTROL_MODE = "sb3_state_only_transfer"
STATE_ONLY_TRANSFER_TASK_KIND = "state_only_control_transfer_smoke"


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("video tooling requires imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _opencv() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("video verification requires OpenCV") from exc
    return cv2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path) -> dict[str, Any]:
    """Fully decode the first video stream and return ffmpeg's frame count."""

    command = [
        _ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-progress",
        "pipe:1",
        "-nostats",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"ffmpeg cannot fully decode {path}: {result.stderr.strip()}")
    frame_values = [
        int(match.group(1))
        for line in result.stdout.splitlines()
        if (match := re.fullmatch(r"frame=\s*(\d+)", line.strip()))
    ]
    if not frame_values or frame_values[-1] <= 0:
        raise RuntimeError(f"ffmpeg did not report a decoded frame count for {path}")
    return {"ffmpeg_full_decode": True, "frame_count": frame_values[-1]}


def _stream_metadata(path: Path) -> tuple[str, str]:
    """Read the encoded codec and pixel format from ffmpeg's input metadata."""

    command = [
        _ffmpeg(),
        "-hide_banner",
        "-i",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    stream_lines = [line for line in result.stderr.splitlines() if "Video:" in line]
    if not stream_lines:
        raise RuntimeError(f"ffmpeg did not report a video stream for {path}")
    stream_line = stream_lines[0]
    codec_match = re.search(r"Video:\s*([^\s,(]+)", stream_line)
    pixel_match = re.search(r"\b(yuv(?:j)?\d{3}p(?:\d{2}(?:le|be))?)\b", stream_line)
    if codec_match is None or pixel_match is None:
        raise RuntimeError(f"ffmpeg did not report codec/pixel format for {path}: {stream_line.strip()}")
    return codec_match.group(1).lower(), pixel_match.group(1).lower()


def _top_level_mp4_boxes(path: Path) -> list[tuple[bytes, int]]:
    """Parse top-level ISO-BMFF boxes without loading the video into memory."""

    boxes: list[tuple[bytes, int]] = []
    file_size = path.stat().st_size
    offset = 0
    with path.open("rb") as stream:
        while offset + 8 <= file_size:
            stream.seek(offset)
            header = stream.read(8)
            if len(header) != 8:
                break
            size = int.from_bytes(header[:4], "big")
            box_type = header[4:8]
            header_size = 8
            if size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    break
                size = int.from_bytes(extended, "big")
                header_size = 16
            elif size == 0:
                size = file_size - offset
            if size < header_size or offset + size > file_size:
                break
            boxes.append((box_type, offset))
            offset += size
    return boxes


def _contains_any(path: Path, signatures: tuple[bytes, ...]) -> bool:
    overlap = max(len(signature) for signature in signatures) - 1
    previous = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            payload = previous + chunk
            if any(signature in payload for signature in signatures):
                return True
            previous = payload[-overlap:] if overlap else b""
    return False


@dataclass(frozen=True)
class VideoAudit:
    path: str
    sha256: str
    bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    first_frame_nonconstant: bool
    ffmpeg_full_decode: bool
    h264_signature_present: bool
    pixel_format: str
    faststart: bool
    codec_name: str = "h264"
    moov_before_mdat: bool = True
    opencv_full_decode: bool = True
    ffmpeg_frame_count: int = 0
    opencv_reported_frame_count: int = 0


def audit_video(path: Path) -> VideoAudit:
    path = path.resolve()
    cv2 = _opencv()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
    frame_count = 0
    first_nonconstant = False
    invalid_decoded_frame = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                invalid_decoded_frame = True
                break
            frame_count += 1
            if frame_count == 1:
                first_nonconstant = bool(frame.max() > frame.min())
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid decoded video properties for {path}")
    probe = _probe(path)
    ffmpeg_frame_count = int(probe["frame_count"])
    codec_name, pixel_format = _stream_metadata(path)
    boxes = _top_level_mp4_boxes(path)
    box_offsets = {
        name: [offset for box_name, offset in boxes if box_name == name]
        for name in (b"ftyp", b"moov", b"mdat")
    }
    moov_before_mdat = bool(
        box_offsets[b"ftyp"]
        and box_offsets[b"moov"]
        and box_offsets[b"mdat"]
        and box_offsets[b"ftyp"][0] < box_offsets[b"moov"][0] < box_offsets[b"mdat"][0]
    )
    h264 = codec_name == "h264" and _contains_any(path, (b"avc1", b"avc3"))
    opencv_full_decode = bool(
        not invalid_decoded_frame
        and frame_count == ffmpeg_frame_count
        and (reported_frame_count <= 0 or reported_frame_count == frame_count)
    )
    return VideoAudit(
        path=str(path),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        first_frame_nonconstant=first_nonconstant,
        ffmpeg_full_decode=True,
        h264_signature_present=h264,
        pixel_format=pixel_format,
        faststart=moov_before_mdat,
        codec_name=codec_name,
        moov_before_mdat=moov_before_mdat,
        opencv_full_decode=opencv_full_decode,
        ffmpeg_frame_count=ffmpeg_frame_count,
        opencv_reported_frame_count=reported_frame_count,
    )


def require_release_video(audit: VideoAudit, *, expected_frames: int | None = None) -> None:
    """Fail unless a decoded MP4 satisfies the portable release contract."""

    failures: list[str] = []
    if audit.codec_name != "h264" or not audit.h264_signature_present:
        failures.append(f"codec is {audit.codec_name!r}, not H.264/AVC")
    if audit.pixel_format != "yuv420p":
        failures.append(f"pixel format is {audit.pixel_format!r}, expected 'yuv420p'")
    if not audit.faststart or not audit.moov_before_mdat:
        failures.append("moov box does not precede mdat")
    if not audit.ffmpeg_full_decode:
        failures.append("ffmpeg did not fully decode the video")
    if not audit.opencv_full_decode:
        failures.append("OpenCV did not decode every frame")
    if audit.ffmpeg_frame_count > 0 and audit.frame_count != audit.ffmpeg_frame_count:
        failures.append(
            f"OpenCV decoded {audit.frame_count} frames but ffmpeg decoded {audit.ffmpeg_frame_count}"
        )
    if not audit.first_frame_nonconstant:
        failures.append("first frame is constant")
    if expected_frames is not None and audit.frame_count != expected_frames:
        failures.append(f"decoded {audit.frame_count} frames, expected {expected_frames}")
    if failures:
        raise RuntimeError("release video gate failed: " + "; ".join(failures))


def transcode_h264(source: Path, destination: Path) -> VideoAudit:
    source = source.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(destination.parent)) as temporary:
        staged = Path(temporary) / destination.name
        command = [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(staged),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"ffmpeg transcode failed for {source}: {result.stderr.strip()}")
        audit = audit_video(staged)
        require_release_video(audit)
        staged.replace(destination)
    return audit_video(destination)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(path.parent)) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staged.replace(path)


def _portable_audit(audit: VideoAudit, destination: Path) -> dict[str, Any]:
    payload = asdict(audit)
    payload["path"] = destination.name
    return payload


def _timestamp_sha256(timestamps: Any) -> str:
    import numpy as np

    canonical = np.ascontiguousarray(timestamps.astype("<i8", copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _validate_timestamps(timestamps: Any, *, label: str) -> None:
    import numpy as np

    if timestamps.dtype != np.int64 or timestamps.ndim != 1 or len(timestamps) < 2:
        raise RuntimeError(f"{label} timestamps must be int64 [T] with at least two samples")
    if not np.all(np.diff(timestamps) > 0):
        raise RuntimeError(f"{label} timestamps must be strictly increasing")


def _output_fps(timestamps: Any, fps: float | None) -> float:
    import numpy as np

    inferred = 1_000_000_000.0 / float(np.median(np.diff(timestamps)))
    value = inferred if fps is None else float(fps)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("video fps must be finite and positive")
    return value


def _load_rgb_npz(path: Path, *, label: str) -> tuple[Any, Any]:
    import numpy as np

    oversized = oversized_legacy_frame_members(path, ("rgb",))
    if oversized:
        limit_mib = LEGACY_FRAME_MEMBER_MAX_UNCOMPRESSED_BYTES // (1024 * 1024)
        raise RuntimeError(
            f"{label} legacy RGB archive exceeds the {limit_mib} MiB bounded-memory limit; "
            "re-encoding it is refused because it could exhaust Windows system commit"
        )
    with np.load(path, allow_pickle=False) as archive:
        fields = set(archive.files)
        missing = {"timestamps_ns", "rgb"} - fields
        if missing:
            raise RuntimeError(f"{label} NPZ is missing fields: {sorted(missing)}")
        return archive["timestamps_ns"].copy(), archive["rgb"].copy()


class _ChunkedRgbFrames:
    """Sequence view that decompresses exactly one RGB frame per access."""

    def __init__(self, archive: ChunkedFrameArchive, *, label: str):
        self._archive = archive
        try:
            descriptor = archive.descriptor("rgb")
        except FrameArchiveError as exc:
            raise RuntimeError(f"{label} chunked NPZ is missing RGB frames") from exc
        self.dtype = descriptor.dtype
        self.shape = descriptor.shape
        self.ndim = len(descriptor.shape)

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, frame_index: int) -> Any:
        if not isinstance(frame_index, int):
            raise TypeError("chunked RGB frames support integer frame access only")
        try:
            return self._archive.frame("rgb", frame_index)
        except FrameArchiveError as exc:
            raise RuntimeError(f"chunked RGB frame {frame_index} is invalid") from exc

    def __iter__(self) -> Iterator[Any]:
        for frame_index in range(len(self)):
            yield self[frame_index]


@contextmanager
def _open_rgb_frames(path: Path, *, label: str) -> Iterator[tuple[Any, Any]]:
    """Open legacy RGB NPZs or bounded-memory v1 frame archives."""

    if not is_chunked_frame_archive(path):
        yield _load_rgb_npz(path, label=label)
        return
    try:
        archive = ChunkedFrameArchive(path)
    except FrameArchiveError as exc:
        raise RuntimeError(f"{label} chunked NPZ is invalid: {exc}") from exc
    try:
        yield archive.timestamps_ns.copy(), _ChunkedRgbFrames(archive, label=label)
    finally:
        archive.close()


def _bound_capture_sources(capture_root: Path, relative_paths: Sequence[str]) -> tuple[Path, dict[str, dict[str, Any]]]:
    receipt_path = capture_root / "capture_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"capture receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise RuntimeError("capture receipt must be a JSON object")
    if receipt.get("schema") != "org.rivermark.isaac-swarm-capture.v1" or receipt.get("ok") is not True:
        raise RuntimeError("refusing video encoding from an unsuccessful or unknown capture")
    artifact_hashes = receipt.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise RuntimeError("capture receipt artifact inventory must be an object")

    verified: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        source = capture_root / Path(relative)
        if not source.is_file():
            raise FileNotFoundError(f"capture artifact is missing: {relative}")
        bound = artifact_hashes.get(relative)
        if not isinstance(bound, dict):
            raise RuntimeError(f"{relative} is not bound by the capture receipt")
        actual_hash = sha256_file(source)
        if bound.get("sha256") != actual_hash:
            raise RuntimeError(f"{relative} is not bound by the capture receipt")
        actual_bytes = source.stat().st_size
        if "bytes" in bound and bound.get("bytes") != actual_bytes:
            raise RuntimeError(f"{relative} byte count disagrees with the capture receipt")
        verified[relative] = {"sha256": actual_hash, "bytes": actual_bytes}
    return receipt_path, verified


def independent_validation_schema_for_capture(
    validation: Mapping[str, Any], capture_receipt: Mapping[str, Any]
) -> str:
    """Fail closed if a validation schema is not compatible with its capture mode.

    The legacy independent-validation schema remains the sole gate for normal
    fixed-public-route captures.  The state-only schema is deliberately bound
    to the development transfer capture contract so it cannot be used to
    relabel a Search3D result or a formal benchmark episode.
    """

    schema = validation.get("schema")
    if not isinstance(schema, str) or schema not in SUPPORTED_INDEPENDENT_VALIDATION_SCHEMAS:
        raise ValueError("unsupported independent validation receipt schema")

    command = capture_receipt.get("command")
    control_mode = command.get("control_mode") if isinstance(command, Mapping) else None
    task_kind = capture_receipt.get("task_kind")
    is_transfer_capture = (
        control_mode == STATE_ONLY_TRANSFER_CONTROL_MODE
        or task_kind == STATE_ONLY_TRANSFER_TASK_KIND
    )
    if schema == STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA:
        claim_boundary = capture_receipt.get("claim_boundary")
        if (
            control_mode != STATE_ONLY_TRANSFER_CONTROL_MODE
            or task_kind != STATE_ONLY_TRANSFER_TASK_KIND
            or not isinstance(claim_boundary, Mapping)
            or claim_boundary.get("development_control_transfer") is not True
            or claim_boundary.get("formal_benchmark_admission") is not False
            or validation.get("formal_benchmark_admission") is not False
        ):
            raise ValueError(
                "state-only transfer validation requires an explicitly development-only "
                "sb3_state_only_transfer capture"
            )
    elif is_transfer_capture:
        raise ValueError(
            "state-only transfer capture requires the state-only transfer independent validation schema"
        )
    return str(schema)


def _bound_independent_validation_sha256(capture_root: Path, capture_receipt_path: Path) -> str:
    """Return the hash of the passing validation receipt bound to this capture."""

    validation_path = capture_root / "independent_validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(f"independent validation receipt is missing: {validation_path}")
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("independent validation receipt is not valid JSON") from exc
    if not isinstance(validation, dict):
        raise RuntimeError("independent validation receipt must be a JSON object")
    if (
        not isinstance(validation.get("schema"), str)
        or validation.get("schema") not in SUPPORTED_INDEPENDENT_VALIDATION_SCHEMAS
        or validation.get("status") != "passed"
        or validation.get("issues") != []
    ):
        raise RuntimeError("refusing Isaac video encoding without a passing independent validation receipt")
    if validation.get("capture_receipt_sha256") != sha256_file(capture_receipt_path):
        raise RuntimeError("independent validation receipt does not bind this capture receipt")
    try:
        capture_receipt = json.loads(capture_receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("capture receipt is not valid JSON") from exc
    if not isinstance(capture_receipt, dict):
        raise RuntimeError("capture receipt must be a JSON object")
    try:
        independent_validation_schema_for_capture(validation, capture_receipt)
    except ValueError as exc:
        raise RuntimeError(
            "refusing Isaac video encoding without a compatible passing independent validation receipt: "
            f"{exc}"
        ) from exc
    return sha256_file(validation_path)


def _encode_rgb_frames(
    frames: Iterable[Any],
    destination: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    label: str,
) -> VideoAudit:
    import numpy as np

    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("H.264 yuv420p output dimensions must be positive and even")
    destination = destination.resolve()
    if destination.suffix.lower() != ".mp4":
        raise ValueError("release video destination must use the .mp4 extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(destination.parent)) as temporary:
        staged = Path(temporary) / destination.name
        command = [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.12g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(staged),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        written = 0
        stderr = b""
        try:
            if process.stdin is None or process.stderr is None:
                raise RuntimeError("ffmpeg pipes were not created")
            for frame in frames:
                array = np.asarray(frame)
                if array.dtype != np.uint8 or array.shape != (height, width, 3):
                    raise RuntimeError(
                        f"{label} frame {written} must be uint8 [{height},{width},3], got {array.dtype} {array.shape}"
                    )
                process.stdin.write(np.ascontiguousarray(array).tobytes())
                written += 1
            process.stdin.close()
            stderr = process.stderr.read()
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        finally:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.stderr is not None:
                process.stderr.close()
        if written != frame_count:
            raise RuntimeError(f"{label} supplied {written} frames, expected {frame_count}")
        if returncode:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg {label} encode failed: {detail}")
        staged_audit = audit_video(staged)
        require_release_video(staged_audit, expected_frames=frame_count)
        staged.replace(destination)
    audit = audit_video(destination)
    require_release_video(audit, expected_frames=frame_count)
    return audit


def _single_view_frames(frames: Any, *, output_height: int, output_width: int) -> Iterator[Any]:
    import numpy as np

    height, width = (int(value) for value in frames.shape[1:3])
    for frame in frames:
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        canvas[:height, :width] = frame[..., :3]
        yield canvas


def encode_isaac_overview(capture_root: Path, destination: Path, *, fps: float | None = None) -> dict[str, Any]:
    """Encode the overview frames from one hash-bound raw Isaac capture."""

    import numpy as np

    capture_root = capture_root.resolve()
    relative = "sensors/overview_rgb.npz"
    receipt_path, artifacts = _bound_capture_sources(capture_root, [relative])
    validation_sha256 = _bound_independent_validation_sha256(capture_root, receipt_path)
    with _open_rgb_frames(capture_root / relative, label="overview") as (timestamps, frames):
        _validate_timestamps(timestamps, label="overview")
        if (
            frames.dtype != np.uint8
            or frames.ndim != 4
            or frames.shape[0] != len(timestamps)
            or frames.shape[1] <= 0
            or frames.shape[2] <= 0
            or frames.shape[-1] not in (3, 4)
        ):
            raise RuntimeError("overview RGB must be uint8 [T,H,W,3|4] and match timestamps")
        output_fps = _output_fps(timestamps, fps)
        height, width = (int(value) for value in frames.shape[1:3])
        output_height = height + height % 2
        output_width = width + width % 2
        destination = destination.resolve()
        audit = _encode_rgb_frames(
            _single_view_frames(frames, output_height=output_height, output_width=output_width),
            destination,
            frame_count=len(frames),
            width=output_width,
            height=output_height,
            fps=output_fps,
            label="Isaac overview",
        )
    timestamp_hash = _timestamp_sha256(timestamps)
    result = {
        "schema": "org.rivermark.isaac-demo-video.v1",
        "ok": True,
        "capture_receipt_sha256": sha256_file(receipt_path),
        "independent_validation_sha256": validation_sha256,
        "overview_npz_sha256": artifacts[relative]["sha256"],
        "timestamps_sha256": timestamp_hash,
        "input_artifacts": {relative: artifacts[relative]},
        "timestamps": {
            "dtype": "int64",
            "count": int(len(timestamps)),
            "first_ns": int(timestamps[0]),
            "last_ns": int(timestamps[-1]),
            "sha256": timestamp_hash,
        },
        "fps": output_fps,
        "video_sha256": audit.sha256,
        "audit": _portable_audit(audit, destination),
    }
    _write_json_atomic(destination.with_suffix(destination.suffix + ".receipt.json"), result)
    return result


def _fit_rgb(view: Any, *, target_height: int, target_width: int) -> Any:
    """Resize one RGB view to fit without cropping, then center-letterbox it."""

    import numpy as np

    cv2 = _opencv()
    source = np.ascontiguousarray(view[..., :3])
    height, width = (int(value) for value in source.shape[:2])
    scale = min(target_height / height, target_width / width)
    resized_height = max(1, min(target_height, int(round(height * scale))))
    resized_width = max(1, min(target_width, int(round(width * scale))))
    if (resized_height, resized_width) == (height, width):
        resized = source
    else:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(source, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _composite_frames(
    overview: Any,
    onboard: Any,
    onboard_frame_indices: Any,
    *,
    overview_height: int,
    overview_width: int,
    cell_height: int,
    cell_width: int,
) -> Iterator[Any]:
    import numpy as np

    for frame_index in range(len(overview)):
        onboard_frame = onboard[int(onboard_frame_indices[frame_index])]
        canvas = np.zeros((overview_height, overview_width + 2 * cell_width, 3), dtype=np.uint8)
        canvas[:, :overview_width] = _fit_rgb(
            overview[frame_index],
            target_height=overview_height,
            target_width=overview_width,
        )
        for agent in range(8):
            row = agent // 2
            column = agent % 2
            top = row * cell_height
            left = overview_width + column * cell_width
            canvas[top : top + cell_height, left : left + cell_width] = _fit_rgb(
                onboard_frame[agent],
                target_height=cell_height,
                target_width=cell_width,
            )
        yield canvas


def _exact_timestamp_subset_indices(
    overview_timestamps: Any,
    onboard_timestamps: Any,
) -> Any:
    """Map every overview frame to one native onboard frame without interpolation."""

    import numpy as np

    indices = np.searchsorted(onboard_timestamps, overview_timestamps)
    if (
        indices.shape != overview_timestamps.shape
        or np.any(indices >= len(onboard_timestamps))
        or not np.array_equal(onboard_timestamps[indices], overview_timestamps)
    ):
        raise RuntimeError(
            "overview timestamps are not an exact ordered subset of onboard timestamps"
        )
    return indices.astype(np.int64, copy=False)


def encode_isaac_composite(capture_root: Path, destination: Path, *, fps: float | None = None) -> dict[str, Any]:
    """Encode a dominant overview plus eight onboard Isaac cameras as a 4x4 dashboard."""

    import numpy as np

    capture_root = capture_root.resolve()
    overview_relative = "sensors/overview_rgb.npz"
    onboard_relative = "sensors/onboard_rgbd.npz"
    receipt_path, artifacts = _bound_capture_sources(capture_root, [overview_relative, onboard_relative])
    validation_sha256 = _bound_independent_validation_sha256(capture_root, receipt_path)
    with (
        _open_rgb_frames(capture_root / overview_relative, label="overview") as (overview_timestamps, overview),
        _open_rgb_frames(capture_root / onboard_relative, label="onboard") as (onboard_timestamps, onboard),
    ):
        _validate_timestamps(overview_timestamps, label="overview")
        _validate_timestamps(onboard_timestamps, label="onboard")
        onboard_frame_indices = _exact_timestamp_subset_indices(
            overview_timestamps, onboard_timestamps
        )
        if (
            overview.dtype != np.uint8
            or overview.ndim != 4
            or overview.shape[0] != len(overview_timestamps)
            or overview.shape[1] <= 0
            or overview.shape[2] <= 0
            or overview.shape[-1] not in (3, 4)
        ):
            raise RuntimeError("overview RGB must be uint8 [T,H,W,3|4] and match timestamps")
        if (
            onboard.dtype != np.uint8
            or onboard.ndim != 5
            or onboard.shape[:2] != (len(onboard_timestamps), 8)
            or onboard.shape[2] <= 0
            or onboard.shape[3] <= 0
            or onboard.shape[-1] not in (3, 4)
        ):
            raise RuntimeError("onboard RGB must be uint8 [T,8,H,W,3|4] and match timestamps")

        overview_height = ((int(overview.shape[1]) + 3) // 4) * 4
        overview_width = int(overview.shape[2]) + int(overview.shape[2]) % 2
        cell_height = overview_height // 4
        cell_width = max(2, int(round(cell_height * onboard.shape[3] / onboard.shape[2])))
        cell_width += cell_width % 2
        output_fps = _output_fps(overview_timestamps, fps)
        destination = destination.resolve()
        audit = _encode_rgb_frames(
            _composite_frames(
                overview,
                onboard,
                onboard_frame_indices,
                overview_height=overview_height,
                overview_width=overview_width,
                cell_height=cell_height,
                cell_width=cell_width,
            ),
            destination,
            frame_count=len(overview_timestamps),
            width=overview_width + 2 * cell_width,
            height=overview_height,
            fps=output_fps,
            label="Isaac native-overview plus eight-onboard composite",
        )
    timestamp_hash = _timestamp_sha256(overview_timestamps)
    mapping = [
        {
            "overview_frame_index": int(overview_frame_index),
            "onboard_frame_index": int(onboard_frame_index),
            "timestamp_ns": int(overview_timestamps[overview_frame_index]),
        }
        for overview_frame_index, onboard_frame_index in enumerate(onboard_frame_indices)
    ]
    layout = [
        {
            "slot": 0,
            "source": "overview",
            "x": 0,
            "y": 0,
            "width": overview_width,
            "height": overview_height,
        },
        *[
            {
                "slot": agent + 1,
                "source": "onboard",
                "agent_id": agent,
                "x": overview_width + (agent % 2) * cell_width,
                "y": (agent // 2) * cell_height,
                "width": cell_width,
                "height": cell_height,
            }
            for agent in range(8)
        ],
    ]
    result = {
        "schema": "org.rivermark.isaac-swarm-composite-video.v1",
        "ok": True,
        "capture_receipt_sha256": sha256_file(receipt_path),
        "independent_validation_sha256": validation_sha256,
        "overview_npz_sha256": artifacts[overview_relative]["sha256"],
        "onboard_npz_sha256": artifacts[onboard_relative]["sha256"],
        "timestamps_sha256": timestamp_hash,
        "input_artifacts": {
            overview_relative: artifacts[overview_relative],
            onboard_relative: artifacts[onboard_relative],
        },
        "timestamp_bindings": {
            overview_relative: _timestamp_sha256(overview_timestamps),
            onboard_relative: _timestamp_sha256(onboard_timestamps),
        },
        "onboard_frame_mapping": {
            "schema": "org.rivermark.exact-timestamp-subset-mapping.v1",
            "interpolation": "forbidden",
            "entries": mapping,
        },
        "timestamps": {
            "dtype": "int64",
            "count": int(len(overview_timestamps)),
            "first_ns": int(overview_timestamps[0]),
            "last_ns": int(overview_timestamps[-1]),
            "sha256": timestamp_hash,
        },
        "layout": {
            "kind": "native-overview-left-plus-eight-onboard",
            "output_width": overview_width + 2 * cell_width,
            "output_height": overview_height,
            "overview_width": overview_width,
            "overview_height": overview_height,
            "onboard_rows": 4,
            "onboard_columns": 2,
            "cell_height": cell_height,
            "cell_width": cell_width,
            "slots": layout,
        },
        "fps": output_fps,
        "video_sha256": audit.sha256,
        "audit": _portable_audit(audit, destination),
    }
    _write_json_atomic(destination.with_suffix(destination.suffix + ".receipt.json"), result)
    return result


# Explicit alias for callers that name the artifact by its swarm role.
encode_isaac_swarm_composite = encode_isaac_composite


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("video", type=Path)
    transcode = subparsers.add_parser("transcode")
    transcode.add_argument("source", type=Path)
    transcode.add_argument("destination", type=Path)
    encode = subparsers.add_parser("encode-isaac-overview")
    encode.add_argument("capture_root", type=Path)
    encode.add_argument("destination", type=Path)
    encode.add_argument("--fps", type=float)
    composite = subparsers.add_parser("encode-isaac-composite")
    composite.add_argument("capture_root", type=Path)
    composite.add_argument("destination", type=Path)
    composite.add_argument("--fps", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "audit":
        result: Any = audit_video(args.video)
        require_release_video(result)
        payload = asdict(result)
    elif args.command == "transcode":
        payload = asdict(transcode_h264(args.source, args.destination))
    elif args.command == "encode-isaac-overview":
        payload = encode_isaac_overview(args.capture_root, args.destination, fps=args.fps)
    else:
        payload = encode_isaac_composite(args.capture_root, args.destination, fps=args.fps)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
