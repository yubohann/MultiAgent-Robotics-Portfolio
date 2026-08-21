"""Download local VLM weights through Ollama.

本脚本封装跨平台最佳实践：先检查 ollama 命令、检查服务是否启动、检查版本，
再按 16GB/32GB 档位下载模型。这样学生不用记住每个模型的 pull 命令。
"""

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import OLLAMA_BASE_URL  # noqa: E402
from app.model_registry import MODEL_CANDIDATES  # noqa: E402


def _ollama_command() -> str:
    """Find the Ollama CLI and raise OS-specific installation guidance if missing."""
    command = shutil.which("ollama")
    if command:
        return command
    system = platform.system()
    if system == "Windows":
        raise SystemExit(
            "Ollama is not in PATH. Install it from https://ollama.com/download/windows "
            "and reopen PowerShell."
        )
    if system == "Darwin":
        raise SystemExit(
            "Ollama is not in PATH. Install it from https://ollama.com/download/mac "
            "and start the Ollama app."
        )
    raise SystemExit(
        "Ollama is not in PATH. On Linux, install with: curl -fsSL https://ollama.com/install.sh | sh"
    )


def _parse_version(value: str) -> tuple[int, int, int]:
    """Parse a semantic-ish version string into a comparable tuple."""
    parts = []
    for token in value.split(".")[:3]:
        digits = "".join(char for char in token if char.isdigit())
        parts.append(int(digits or "0"))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _check_daemon() -> str:
    """Verify the Ollama HTTP daemon is reachable and return its version."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=3)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Ollama is installed but not reachable at {OLLAMA_BASE_URL}. "
            "Start it with `ollama serve` or open the Ollama desktop app."
        ) from exc
    return str(response.json().get("version", "0.0.0"))


def _check_model_runtime(candidate_id: str, ollama_version: str) -> None:
    """Reject model/runtime combinations known to be incompatible."""
    candidate = MODEL_CANDIDATES[candidate_id]
    if candidate.ollama_model.startswith("qwen3-vl") and _parse_version(ollama_version) < (0, 12, 7):
        raise SystemExit(
            f"{candidate.ollama_model} requires Ollama >= 0.12.7. "
            f"Current version is {ollama_version}. Upgrade Ollama first."
        )


def _local_models() -> set[str]:
    """Return local Ollama model names, including both full and base names."""
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    response.raise_for_status()
    names: set[str] = set()
    for model in response.json().get("models", []):
        name = model.get("name")
        if name:
            names.add(name)
            names.add(name.split(":")[0])
    return names


def _candidate_ids_for_tier(tier: str) -> list[str]:
    """Map memory tiers to recommended model ids from the shared registry."""
    if tier == "local":
        return [
            "ministral-3-8b-ollama",
            "ministral-3-3b-ollama",
        ]
    if tier == "baselines":
        return [
            "ministral-3-8b-ollama",
            "qwen3-vl-4b-ollama",
            "gemma3-4b-ollama",
        ]
    if tier == "16gb":
        return [
            "ministral-3-3b-ollama",
            "qwen3-vl-4b-ollama",
            "qwen3-vl-2b-ollama",
            "qwen2_5-vl-3b-ollama",
            "gemma3-4b-ollama",
        ]
    if tier == "32gb":
        return [
            "ministral-3-8b-ollama",
            "ministral-3-3b-ollama",
            "qwen3-vl-4b-ollama",
            "qwen3-vl-8b-ollama",
            "qwen2_5-vl-7b-ollama",
            "gemma3-12b-ollama",
            "minicpm-v-ollama",
        ]
    return [
        model_id
        for model_id, candidate in MODEL_CANDIDATES.items()
        if candidate.mode == "local_ollama_vlm"
    ]


def pull_model(model_id: str) -> None:
    """Pull one model unless it is already present locally."""
    candidate = MODEL_CANDIDATES[model_id]
    if candidate.mode != "local_ollama_vlm":
        print(f"skip: {candidate.id} does not need Ollama weights")
        return
    before = _local_models()
    if candidate.ollama_model in before:
        print(f"exists: {candidate.ollama_model}")
        return

    command = [_ollama_command(), "pull", candidate.ollama_model]
    print(f"pulling: {candidate.ollama_model} (~{candidate.estimated_disk_gb}GB)")
    # 使用官方 CLI 下载，避免自己实现断点续传、缓存和平台差异处理。
    subprocess.run(command, check=True)
    time.sleep(0.5)
    after = _local_models()
    if candidate.ollama_model not in after:
        raise SystemExit(f"pull finished but model is not listed by Ollama: {candidate.ollama_model}")
    print(f"ready: {candidate.ollama_model}")


def main() -> None:
    """Parse CLI options and download the requested model set."""
    parser = argparse.ArgumentParser(description="Download cross-platform local VLMs through Ollama.")
    parser.add_argument("--model", choices=sorted(MODEL_CANDIDATES.keys()))
    parser.add_argument("--tier", choices=["local", "baselines", "16gb", "32gb", "all"], default="local")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        # 列表模式不要求 Ollama 已启动，便于学生先查看可选模型和磁盘占用。
        for model_id, candidate in MODEL_CANDIDATES.items():
            if candidate.mode != "local_ollama_vlm":
                continue
            print(
                f"{candidate.id}\t{candidate.ollama_model}\t{candidate.memory_tier}\t"
                f"~{candidate.estimated_disk_gb}GB\t{candidate.pull_command}"
            )
        return

    _ollama_command()
    ollama_version = _check_daemon()
    model_ids = [args.model] if args.model else _candidate_ids_for_tier(args.tier)
    for model_id in model_ids:
        # 每个模型单独检查运行时版本，方便未来不同模型有不同最低版本要求。
        _check_model_runtime(model_id, ollama_version)
        pull_model(model_id)


if __name__ == "__main__":
    main()
