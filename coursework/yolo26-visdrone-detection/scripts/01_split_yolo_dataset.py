from pathlib import Path
import random
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_IMAGES = ROOT / "data_raw" / "images"
SOURCE_LABELS = ROOT / "data_raw" / "labels"
OUTPUT = ROOT / "my_data" / "detection"
TRAIN_RATIO = 0.8
SEED = 2026
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def copy_pair(image_path: Path, label_path: Path, split: str) -> None:
    image_dst = OUTPUT / "images" / split / image_path.name
    label_dst = OUTPUT / "labels" / split / label_path.name
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_dst)
    shutil.copy2(label_path, label_dst)


def main() -> None:
    image_files = sorted(
        p for p in SOURCE_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        raise SystemExit(f"No images found in {SOURCE_IMAGES}")

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []
    for image_path in image_files:
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path)

    if not pairs:
        raise SystemExit(f"No image/label pairs found. Check {SOURCE_LABELS}")

    rng = random.Random(SEED)
    rng.shuffle(pairs)

    if len(pairs) == 1:
        train_pairs = pairs
        val_pairs = pairs
        print("Only one valid pair found. Reusing it for val so the lab pipeline can run.")
    else:
        split_index = max(1, int(len(pairs) * TRAIN_RATIO))
        if split_index >= len(pairs):
            split_index = len(pairs) - 1
        train_pairs = pairs[:split_index]
        val_pairs = pairs[split_index:]

    for image_path, label_path in train_pairs:
        copy_pair(image_path, label_path, "train")
    for image_path, label_path in val_pairs:
        copy_pair(image_path, label_path, "val")

    print(f"Total images: {len(image_files)}")
    print(f"Valid pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Dataset written to: {OUTPUT}")

    if missing_labels:
        print("Images skipped because label txt is missing:")
        for path in missing_labels[:20]:
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
