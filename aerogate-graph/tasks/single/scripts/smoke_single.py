"""Smoke entry for the single-agent Graph-FlashSAC experiment."""

from __future__ import annotations


def main() -> None:
    from tasks.single.smoke import run_single_smoke_test

    summary = run_single_smoke_test()
    print("single-agent smoke test complete")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

