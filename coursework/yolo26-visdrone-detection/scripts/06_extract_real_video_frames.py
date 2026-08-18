from argparse import ArgumentParser
from pathlib import Path
import shutil

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "external" / "real_videos"
OUTPUT_DIR = ROOT / "data_raw" / "images"
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def parse_args():
    parser = ArgumentParser(description="Extract real camera frames for sim2real training.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Directory containing real camera videos.")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output image directory.")
    parser.add_argument("--stride", type=int, default=15, help="Save one frame every N frames.")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum total frames to save.")
    parser.add_argument("--prefix", default="real", help="Output filename prefix.")
    parser.add_argument("--clean", action="store_true", help="Clean output directory first.")
    return parser.parse_args()


def clean_dir(path: Path) -> None:
    root = ROOT.resolve()
    target = path.resolve()
    if not str(target).startswith(str(root)):
        raise RuntimeError(f"Refusing to clean outside project root: {target}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def iter_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    videos = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        raise SystemExit(f"No videos found in {input_dir}")
    return videos


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if args.stride <= 0:
        raise SystemExit("--stride must be greater than 0")
    if args.max_frames <= 0:
        raise SystemExit("--max-frames must be greater than 0")

    if args.clean:
        clean_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    videos = iter_videos(input_dir)
    saved = 0

    for video_index, video_path in enumerate(videos):
        if saved >= args.max_frames:
            break

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skip unreadable video: {video_path}")
            continue

        frame_index = 0
        video_saved = 0
        while saved < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % args.stride == 0:
                out_name = f"{args.prefix}_{video_index:03d}_{video_path.stem}_{frame_index:06d}.jpg"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), frame)
                saved += 1
                video_saved += 1

            frame_index += 1

        cap.release()
        print(f"{video_path.name}: saved {video_saved} frames")

    print(f"Total saved frames: {saved}")
    print(f"Output directory: {output_dir}")
    print("Next step: label these images with LabelImg and save txt labels to data_raw/labels.")


if __name__ == "__main__":
    main()
