from pathlib import Path
import urllib.request

import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
IMAGE = ASSETS / "bus.jpg"
LOCAL_MODEL = ROOT / "yolo26m.pt"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    if not IMAGE.exists():
        print(f"Downloading sample image to {IMAGE}")
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", IMAGE)

    print(f"ultralytics={ultralytics.__version__}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")

    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else "yolo26m.pt"
    print(f"model={model_path}")
    model = YOLO(model_path)
    results = model.predict(
        source=str(IMAGE),
        save=True,
        conf=0.25,
        imgsz=640,
        project=str(ROOT / "runs" / "smoke_yolo26"),
        name="predict",
        exist_ok=True,
    )

    print(f"Smoke test done. Result dir: {results[0].save_dir}")


if __name__ == "__main__":
    main()
