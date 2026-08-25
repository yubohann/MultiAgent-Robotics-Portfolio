from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from a PDF with pypdf.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    reader = PdfReader(str(args.source))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages).replace("\x00", "")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(text, encoding="utf-8")
    print(f"pages={len(reader.pages)} bytes={args.destination.stat().st_size}")


if __name__ == "__main__":
    main()
