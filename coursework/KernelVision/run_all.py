# -*- coding: utf-8 -*-
"""一键运行四个实验，产物写入各实验目录的 figures 与 results 与 output。

用法: python run_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "exp1_filtering/code/experiment1_filtering.py",
    "exp2_features/code/experiment2_features.py",
    "exp3_hog/code/experiment3_hog.py",
    "exp4_cnn/code/experiment4_cnn.py",
]


def main():
    for rel in SCRIPTS:
        print("=" * 60)
        print("运行", rel)
        subprocess.run([sys.executable, str(ROOT / rel)], check=True)


if __name__ == "__main__":
    main()
