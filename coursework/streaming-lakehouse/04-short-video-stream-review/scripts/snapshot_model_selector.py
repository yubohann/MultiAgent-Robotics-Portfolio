"""Print a screenshot-friendly model selector evidence table."""



import json
import os
import sys

import _bootstrap  # noqa: F401 -- side-effect import: allow direct execution from scripts/
from app.model_registry import get_active_model, list_model_candidates
from app.ollama_vlm import OllamaModelError, OllamaVLMClient

PRIORITY = [

    "ministral-3-8b-ollama",

    "qwen3-vl-4b-ollama",

    "gemma3-4b-ollama",

    "local-baseline",

    "ministral-3-3b-ollama",

]





def main() -> None:

    """Show all configured models with active/downloaded state."""

    if hasattr(sys.stdout, "reconfigure"):

        sys.stdout.reconfigure(encoding="utf-8")

    try:

        local_names = OllamaVLMClient().list_local_models()

        ollama_ready = True

    except OllamaModelError:

        local_names = set()

        ollama_ready = False



    active = get_active_model()

    print(f"STUDENT_ID={os.getenv('STUDENT_ID', '[REDACTED]')}")

    print(f"active_model={active.id} | {active.name} | {active.ollama_model} | {active.mode}")

    print(f"ollama_ready={str(ollama_ready).lower()}")



    candidates = list_model_candidates()

    rank = {model_id: index for index, model_id in enumerate(PRIORITY)}

    original_order = {item["id"]: index for index, item in enumerate(candidates)}

    candidates.sort(key=lambda item: rank.get(item["id"], len(PRIORITY) + original_order[item["id"]]))

    for item in candidates:

        ollama_model = item.get("ollama_model") or ""

        downloaded = True if not ollama_model else ollama_model in local_names

        compact = {

            "id": item["id"],

            "name": item["name"],

            "backend": item["mode"],

            "ollama_model": ollama_model or "no-weight",

            "memory_tier": item["memory_tier"],

            "downloaded": downloaded,

            "active": item["id"] == active.id,

        }

        print(json.dumps(compact, ensure_ascii=False))





if __name__ == "__main__":

    main()

