from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "configs" / "my_detect.local.yaml"


def parse_args():
    parser = ArgumentParser(description="Train YOLO26 for experiment 5.")
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="Use 0 for first GPU or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found. Run scripts/02_make_data_yaml.py first.")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(ROOT / "runs" / "yolo26"),
        "name": "train",
        "exist_ok": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    print(results)
    print(f"Best weights: {ROOT / 'runs' / 'yolo26' / 'train' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
