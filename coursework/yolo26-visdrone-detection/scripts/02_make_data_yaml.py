from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASSES_FILE = ROOT / "configs" / "classes.txt"
OUTPUT_FILE = ROOT / "configs" / "my_detect.local.yaml"
DATASET_ROOT = ROOT / "my_data" / "detection"


def yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    classes = [
        line.strip()
        for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not classes:
        raise SystemExit(f"No classes found in {CLASSES_FILE}")

    dataset_path = DATASET_ROOT.resolve().as_posix()
    lines = [
        f"path: {yaml_quote(dataset_path)}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for index, name in enumerate(classes):
        lines.append(f"  {index}: {yaml_quote(name)}")

    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE}")
    print(OUTPUT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
