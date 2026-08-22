"""Build a temporary venv, install a wheel, and preserve only its evidence."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json  # noqa: E402


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _run(command: list[str], *, cwd: Path, timeout_s: int = 120) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return {
        "returncode": completed.returncode,
        "stdout_sha256": content_hash(completed.stdout),
        "stderr_sha256": content_hash(completed.stderr),
    }


def _venv_python(venv: Path) -> Path:
    candidate = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not candidate.is_file():
        raise FileNotFoundError(f"venv interpreter is absent: {candidate}")
    return candidate


def _environment_binding(path: Path) -> dict[str, object]:
    manifest = read_json(path.resolve())
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    source_tree = manifest.get("source_tree")
    if (
        manifest.get("schema") != "org.aerocity.bench.release-environment-manifest.v1"
        or manifest.get("manifest_hash") != content_hash(payload)
        or not isinstance(source_tree, dict)
        or source_tree.get("state") not in {"CLEAN", "DIRTY", "UNVERIFIABLE"}
        or source_tree.get("official_release_binding") not in {"VALID", "REJECTED"}
    ):
        raise ValueError("environment manifest is invalid or unbound")
    return {
        "environment_manifest_hash": manifest["manifest_hash"],
        "source_tree": {
            "state": source_tree["state"],
            "source_commit": source_tree.get("source_commit"),
            "official_release_binding": source_tree["official_release_binding"],
        },
    }


def verify_clean_wheel_install(
    base_python: Path, wheel: Path, environment_manifest: Path
) -> dict[str, object]:
    """Prove the installed wheel imports outside the source tree.

    The temporary virtual environment is deleted after the checks.  This keeps
    a large, disposable interpreter tree out of the evidence directory while
    retaining wheel and command hashes required to repeat the verification.
    """

    base = base_python.resolve()
    artifact = wheel.resolve()
    if not base.is_file():
        raise FileNotFoundError(f"base interpreter does not exist: {base}")
    if not artifact.is_file() or artifact.suffix != ".whl":
        raise ValueError("wheel must be an existing .whl file")
    binding = _environment_binding(environment_manifest)

    with tempfile.TemporaryDirectory(prefix="aerocity-clean-wheel-") as temporary:
        root = Path(temporary)
        venv = root / "venv"
        smoke_cwd = root / "smoke"
        smoke_cwd.mkdir()
        create = _run([str(base), "-m", "venv", str(venv)], cwd=root)
        if create["returncode"] != 0:
            raise RuntimeError("clean virtualenv creation failed")
        clean_python = _venv_python(venv)
        before = _run(
            [
                str(clean_python),
                "-I",
                "-c",
                "import importlib.metadata as m; print(m.version('pip'))",
            ],
            cwd=smoke_cwd,
        )
        install = _run(
            [
                str(clean_python),
                "-I",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--force-reinstall",
                str(artifact),
            ],
            cwd=smoke_cwd,
        )
        imported = _run(
            [
                str(clean_python),
                "-I",
                "-c",
                "import aerocity_bench, pathlib; "
                "p=pathlib.Path(aerocity_bench.__file__).resolve(); "
                "print(aerocity_bench.__version__); "
                "raise SystemExit(0 if 'site-packages' in p.parts else 3)",
            ],
            cwd=smoke_cwd,
        )
        presets = _run(
            [str(clean_python), "-I", "-m", "aerocity_bench", "list-presets", "--json"],
            cwd=smoke_cwd,
        )

    checks = {
        "create_venv": create,
        "clean_interpreter": before,
        "install_wheel_no_deps": install,
        "import_from_site_packages": imported,
        "wheel_bundled_presets": presets,
    }
    status = "PASS_DEVELOPMENT_ONLY" if all(
        item["returncode"] == 0 for item in checks.values()
    ) else "FAIL"
    report: dict[str, object] = {
        "schema": "org.aerocity.bench.clean-wheel-install.v1",
        "status": status,
        "formal_score_eligible": False,
        "scope": "development_reproducibility_only",
        "wheel": {"filename": artifact.name, "sha256": file_hash(artifact)},
        "base_python_sha256": file_hash(base),
        "environment_binding": binding,
        "temporary_venv_retained": False,
        "checks": checks,
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite clean-install evidence: {args.output}")
    report = verify_clean_wheel_install(
        args.base_python, args.wheel, args.environment_manifest
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"CLEAN_WHEEL_INSTALL={report['status']}")
    return 0 if report["status"] == "PASS_DEVELOPMENT_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
