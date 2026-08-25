from __future__ import annotations

import re
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = PROJECT_ROOT.parent
DEFAULT_KIT_RUNTIME_ROOT = Path(
    os.environ.get(
        "ISAAC_DRONE_KIT_RUNTIME_ROOT",
        PROJECT_ROOT / "runtime" / "kit_runtime",
    )
)

KIT_STABILITY_ARGS = (
    "--/renderer/multiGpu/enabled=false",
    "--/renderer/multiGpu/autoEnable=false",
    "--/renderer/multiGpu/maxGpuCount=1",
    "--/app/renderer/resolution/width=64",
    "--/app/renderer/resolution/height=64",
    "--/ngx/enabled=false",
    "--/telemetry/enableAnonymousData=false",
    "--/telemetry/enableNVDF=false",
    "--/telemetry/enableSentry=false",
    "--/telemetry/useOpenEndpoint=false",
    "--/privacy/performance=false",
    "--/privacy/usage=false",
)


def _portable_runtime_disabled() -> bool:
    value = os.environ.get("ISAAC_DRONE_DISABLE_PORTABLE_KIT", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _slugify_app_name(app_name: str | None) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(app_name or "").strip()).strip("._-")
    if len(slug) > 96:
        slug = slug[:96].rstrip("._-")
    return slug or "isaac_app"


def build_writable_kit_args(
    existing_kit_args: str | None,
    *,
    app_name: str,
    base_root: Path | None = None,
) -> tuple[str, Path | None]:
    """Inject a writable portable-root unless the caller already pinned one."""

    resolved_kit_args = str(existing_kit_args or "").strip()
    if "--portable-root" in resolved_kit_args:
        return resolved_kit_args, None

    if _portable_runtime_disabled():
        additions = [arg for arg in KIT_STABILITY_ARGS if arg not in resolved_kit_args]
        updated_kit_args = " ".join(part for part in [resolved_kit_args, *additions] if part).strip()
        return updated_kit_args, None

    portable_root = Path(base_root or DEFAULT_KIT_RUNTIME_ROOT) / _slugify_app_name(app_name)
    additions: list[str] = []
    if "--portable" not in resolved_kit_args:
        additions.append("--portable")
    additions.extend(["--portable-root", portable_root.as_posix()])
    additions.extend(arg for arg in KIT_STABILITY_ARGS if arg not in resolved_kit_args)

    updated_kit_args = " ".join(part for part in [resolved_kit_args, *additions] if part).strip()
    return updated_kit_args, portable_root


def ensure_writable_kit_runtime(args_cli, *, app_name: str, base_root: Path | None = None) -> Path | None:
    """Route Isaac Sim Kit runtime writes away from read-only install directories."""

    if not hasattr(args_cli, "kit_args"):
        return None

    updated_kit_args, portable_root = build_writable_kit_args(
        getattr(args_cli, "kit_args", ""),
        app_name=app_name,
        base_root=base_root,
    )
    setattr(args_cli, "kit_args", updated_kit_args)

    if portable_root is None:
        if _portable_runtime_disabled():
            print(
                f"[kit runtime] {app_name}: portable root disabled by "
                "ISAAC_DRONE_DISABLE_PORTABLE_KIT; using stability args only",
                flush=True,
            )
        return None

    portable_root.mkdir(parents=True, exist_ok=True)
    kit_version_dir = Path("Kit") / "Isaac-Sim" / "5.1"
    for child in (
        Path("data") / kit_version_dir,
        Path("logs") / kit_version_dir,
        Path("cache"),
        Path("omni_configs"),
    ):
        (portable_root / child).mkdir(parents=True, exist_ok=True)
    print(f"[kit runtime] {app_name}: using writable portable root {portable_root}", flush=True)
    return portable_root
