# YOLO26 + VisDrone Detection — Experiments 5 & 6



Object detection with **YOLO26 (Ultralytics)** on the **VisDrone** aerial dataset, as part of my *Embedded Systems* course design at [REDACTED].



## About this work



- **Author**: Bohan Yu (Bohan Yu, student ID [REDACTED])

- **Course**: Embedded Systems course design — experiments 5 & 6 (嵌入式系统课程设计·实验五六)

- **What it covers**: the full detection workflow — dataset preparation (VisDrone subset), YOLO26 training, validation, prediction, and ONNX export — with step-by-step lab instructions (see `README.md`, Chinese) and screenshot evidence for every stage.



## What's in this repository



```text

scripts/         00..06 pipeline scripts (smoke test, dataset split, data.yaml, train, val/predict/export, VisDrone subset prep, video frame extraction)

configs/         class list and dataset yamls

detect_test/     test images for prediction demos

screenshots/     evidence for experiments 5 & 6 (per-stage screenshots)

docs/            sim-to-real notes (real camera data suggestions, raw drone images)

yolo26n.pt       base YOLO26n weights (used as the training starting point)

requirements_yolo26.txt

```



## Pipeline



```text

VisDrone subset prep → train/val split → data.yaml → YOLO26 training

→ validation & prediction → ONNX export → (optional) real video frame extraction

```



Key scripts:



| Script | Step |

| --- | --- |

| `00_smoke_test_yolo26.py` | environment smoke test |

| `05_prepare_visdrone_subset.py` | build the VisDrone subset |

| `01_split_yolo_dataset.py` | train/val split |

| `02_make_data_yaml.py` | generate dataset yaml |

| `03_train_yolo26.py` | train YOLO26 |

| `04_exp6_val_predict_export.py` | validate, predict, export ONNX |

| `06_extract_real_video_frames.py` | extract frames from real drone video |



## Getting started



```bash

pip install -r requirements_yolo26.txt

python scripts/00_smoke_test_yolo26.py

```



Detailed, command-by-command instructions for both experiments are in

[README.md](README.md) (Chinese), written as a lab manual.



*Bohan Yu — Embedded Systems course design.*