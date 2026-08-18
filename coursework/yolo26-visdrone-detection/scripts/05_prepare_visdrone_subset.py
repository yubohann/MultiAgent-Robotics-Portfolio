from argparse import ArgumentParser
from pathlib import Path
import shutil
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "VisDrone"
DATA_RAW_IMAGES = ROOT / "data_raw" / "images"
DATA_RAW_LABELS = ROOT / "data_raw" / "labels"
CLASSES_FILE = ROOT / "configs" / "classes.txt"
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
VISDRONE_SPLITS = ("train", "val")

VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = ArgumentParser(description="Download VisDrone and prepare raw images for YOLO26 lab.")
    parser.add_argument(
        "--mode",
        choices=["obstacle", "visdrone10"],
        default="obstacle",
        help="obstacle merges all VisDrone classes into one class. visdrone10 keeps original classes.",
    )
    parser.add_argument("--max-images", type=int, default=300, help="Maximum images copied to data_raw.")
    parser.add_argument("--clean", action="store_true", help="Clean data_raw/images and data_raw/labels first.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VISDRONE_SPLITS,
        default=["train", "val"],
        help="VisDrone split(s) to download and sample from. Use --splits val for a small course subset.",
    )
    return parser.parse_args()


def ensure_empty_dir(path: Path) -> None:
    root = ROOT.resolve()
    target = path.resolve()
    if not str(target).startswith(str(root)):
        raise RuntimeError(f"Refusing to clean outside project root: {target}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, target: Path) -> None:
    if target.exists():
        print(f"Using existing archive: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as file:
        shutil.copyfileobj(response, file)
    print(f"Downloaded: {target}")


def extract_archive(archive: Path, destination: Path) -> None:
    marker = destination / archive.stem
    if marker.exists():
        print(f"Using existing extracted dataset: {marker}")
        return

    print(f"Extracting {archive}")
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(destination)
    print(f"Extracted to: {destination}")


def download_visdrone(splits: list[str]) -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for split in splits:
        archive = EXTERNAL / f"VisDrone2019-DET-{split}.zip"
        url = f"{ASSETS_URL}/{archive.name}"
        download_file(url, archive)
        extract_archive(archive, EXTERNAL)


def visdrone_image_dirs(splits: list[str]) -> list[Path]:
    return [
        EXTERNAL / f"VisDrone2019-DET-{split}" / "images"
        for split in splits
    ]


def convert_annotation(annotation_file: Path, image_w: int, image_h: int, mode: str) -> list[str]:
    lines: list[str] = []
    if not annotation_file.exists():
        return lines

    for raw_line in annotation_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue

        parts = raw_line.split(",")
        if len(parts) < 8:
            continue

        x, y, w, h = (float(parts[i]) for i in range(4))
        score = int(float(parts[4]))
        category = int(float(parts[5]))

        # VisDrone category 0 marks ignored regions.
        if score == 0 or category <= 0 or category > len(VISDRONE_CLASSES):
            continue
        if w <= 1 or h <= 1:
            continue

        cls = 0 if mode == "obstacle" else category - 1
        x_center = (x + w / 2) / image_w
        y_center = (y + h / 2) / image_h
        width = w / image_w
        height = h / image_h

        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            continue

        lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def prepare_classes(mode: str) -> None:
    CLASSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if mode == "obstacle":
        classes = ["obstacle"]
    else:
        classes = VISDRONE_CLASSES
    CLASSES_FILE.write_text("\n".join(classes) + "\n", encoding="utf-8")
    print(f"Wrote classes to {CLASSES_FILE}")


def prepare_subset(mode: str, max_images: int, clean: bool, splits: list[str]) -> None:
    if clean:
        ensure_empty_dir(DATA_RAW_IMAGES)
        ensure_empty_dir(DATA_RAW_LABELS)
    else:
        DATA_RAW_IMAGES.mkdir(parents=True, exist_ok=True)
        DATA_RAW_LABELS.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0

    for images_dir in visdrone_image_dirs(splits):
        annotations_dir = images_dir.parent / "annotations"
        split_name = images_dir.parent.name.replace("VisDrone2019-DET-", "")
        if not images_dir.exists():
            raise SystemExit(f"Missing {images_dir}. Download failed or directory layout changed.")

        for image_path in sorted(images_dir.iterdir()):
            if copied_images >= max_images:
                break
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            annotation_file = annotations_dir / f"{image_path.stem}.txt"
            width, height = image_size(image_path)
            yolo_lines = convert_annotation(annotation_file, width, height, mode)
            if not yolo_lines:
                continue

            dst_stem = f"visdrone_{split_name}_{image_path.stem}"
            image_dst = DATA_RAW_IMAGES / f"{dst_stem}{image_path.suffix.lower()}"
            label_dst = DATA_RAW_LABELS / f"{dst_stem}.txt"
            shutil.copy2(image_path, image_dst)
            label_dst.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

            copied_images += 1
            copied_labels += 1

        if copied_images >= max_images:
            break

    prepare_classes(mode)
    print(f"Copied images: {copied_images}")
    print(f"Copied labels: {copied_labels}")
    print(f"Raw image dir: {DATA_RAW_IMAGES}")
    print(f"Raw label dir: {DATA_RAW_LABELS}")

    if copied_images == 0:
        raise SystemExit("No usable VisDrone images were copied.")


def main() -> None:
    args = parse_args()
    download_visdrone(args.splits)
    prepare_subset(args.mode, args.max_images, args.clean, args.splits)


if __name__ == "__main__":
    main()
