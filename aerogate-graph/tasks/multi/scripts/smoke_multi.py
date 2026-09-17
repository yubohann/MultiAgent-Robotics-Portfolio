"""Smoke entry for the multi-agent Graph-FlashSAC experiment."""

from __future__ import annotations


def main() -> None:
    from tasks.multi.smoke import run_multi_smoke_test

    summary = run_multi_smoke_test()
    print("multi-agent smoke test complete")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

