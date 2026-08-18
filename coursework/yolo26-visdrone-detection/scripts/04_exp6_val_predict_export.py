from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "configs" / "my_detect.local.yaml"
DEFAULT_WEIGHTS = ROOT / "runs" / "yolo26" / "train" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "detect_test"


def parse_args():
    parser = ArgumentParser(description="Validate, predict, and export YOLO26 for experiment 6.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="Use 0 for first GPU or cpu.")
    parser.add_argument(
        "--compat-yolov8-output",
        action="store_true",
        help="Export with end2end=False for old YOLOv8-style post-processing code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    source = Path(args.source)

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found. Run scripts/02_make_data_yaml.py first.")
    if not weights.exists():
        raise SystemExit(f"{weights} not found. Finish experiment 5 training first.")
    if not source.exists() or not any(p.is_file() for p in source.iterdir()):
        raise SystemExit(f"No test images found in {source}")

    model = YOLO(str(weights))
    common_kwargs = {"imgsz": args.imgsz}
    if args.device is not None:
        common_kwargs["device"] = args.device

    metrics = model.val(
        data=str(DATA_YAML),
        project=str(ROOT / "runs" / "yolo26"),
        name="val",
        exist_ok=True,
        **common_kwargs,
    )
    print(metrics)

    predictions = model.predict(
        source=str(source),
        save=True,
        conf=args.conf,
        project=str(ROOT / "runs" / "yolo26"),
        name="predict",
        exist_ok=True,
        **common_kwargs,
    )
    print(f"Predict result dir: {predictions[0].save_dir}")

    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "dynamic": True,
    }
    if args.compat_yolov8_output:
        export_kwargs["end2end"] = False

    onnx_path = model.export(**export_kwargs)
    print(f"ONNX exported to: {onnx_path}")


if __name__ == "__main__":
    main()
