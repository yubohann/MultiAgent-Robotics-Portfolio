"""Run reproducible source and model probes for external baseline candidates.

This audit deliberately separates three claims:

1. an upstream repository is present and auditable;
2. its original neural core can execute on this machine;
3. it is suitable for the four-CF2X HM3D exploration protocol.

Passing (1) or (2) never implies (3).  The JSON report records the evidence so
the paper-facing baseline decision does not depend on repository titles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "external-baseline-candidate-audit-current.json"

RL_LITERATURE_ROOT = Path(
    r"E:\HM3D_2026_papers_and_repos\hm3d_rl_literature_2026-08-03\repos"
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    root: Path
    task_fit: str
    required_modules: tuple[str, ...] = ()
    checkpoints: tuple[str, ...] = ()
    evidence_patterns: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class ProbeResult:
    status: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


SPECS = (
    CandidateSpec(
        "marl_ipp",
        "rl",
        Path(r"E:\github_repos\marl_ipp-main"),
        "native_3d_multi_robot_graph_rl_but_target_mapping_reward",
        ("torch", "numpy", "scipy", "sklearn", "matplotlib", "imageio", "ray"),
        (
            "model/our_approach/best_model_checkpoint.pth",
            "model/our_approach/checkpoint.pth",
        ),
        {
            "three_dimensional_state": ("occupancy grid of 50 cells in each of 3 directions",),
            "multi_robot": ("num_agents", "closest agent position features"),
            "communication": ("comms_dist", "neib_info"),
            "learning": ("attentionnet", "lstm", "optimizer"),
        },
    ),
    CandidateSpec(
        "marvel",
        "rl",
        Path(r"E:\github_repos\MARVEL-main"),
        "multi_robot_exploration_sac_but_2d_nodes_and_heading_bins",
        ("torch", "numpy", "ray", "wandb"),
        ("load_model/MARVEL/checkpoint.pth",),
        {
            "two_dimensional_state": ("node_coords[:, 0]", "node_coords[:, 1]"),
            "fixed_height": ("drone_height",),
            "multi_robot": ("n_agents", "all_agent_indices"),
            "learning": ("policynet", "qnet", "soft actor-critic"),
        },
    ),
    CandidateSpec(
        "ir2",
        "rl",
        Path(r"E:\github_repos\IR2-Multi-Robot-RL-Exploration-master"),
        "multi_robot_exploration_and_sparse_communication_but_2d_maps",
        ("torch", "numpy", "scipy", "sklearn", "skimage", "ray"),
        ("model/stage2/checkpoint.pth",),
        {
            "two_dimensional_state": ("dungeonmaps", "node_coords"),
            "multi_robot": ("num_robots_min", "num_robots_max"),
            "communication": ("signal strength", "rendezvous"),
            "learning": ("policynet", "qnet", "soft actor"),
        },
    ),
    CandidateSpec(
        "recuriosity",
        "rl",
        RL_LITERATURE_ROOT / "recuriosity__recuriosity",
        "native_hm3d_target_free_exploration_but_single_camera_discrete_actions",
        ("torch", "numpy", "habitat_sim", "einops", "gsplat"),
        ("checkpoints/explorer.pt", "checkpoints/explore.pt"),
        {
            "hm3d": ("habitat-matterport 3d", "hm3d_data_root"),
            "target_free_exploration": ("intrinsic reward", "surface coverage completeness"),
            "single_agent_camera": ("rgb camera", "turn_left", "move_forward"),
            "learning": ("ppo", "transformer", "dino"),
        },
    ),
    CandidateSpec(
        "visfly",
        "rl_platform",
        RL_LITERATURE_ROOT / "visfly__SJTU_ViSYS_team",
        "native_hm3d_continuous_quadrotor_platform_but_no_matching_exploration_policy",
        ("torch", "numpy", "habitat_sim", "gymnasium", "stable_baselines3"),
        (),
        {
            "hm3d": ("hm3d", "habitat"),
            "quadrotor": ("drone", "dynamics"),
            "multi_agent": ("num_agent", "multi"),
            "learning": ("ppo", "sac"),
        },
    ),
    CandidateSpec(
        "rvn_bench",
        "rl_benchmark",
        RL_LITERATURE_ROOT / "rvn_bench__Sequor_Robotics_Research",
        "hm3d_rl_benchmark_but_single_ground_robot_sequential_point_goal",
        ("torch", "numpy", "habitat_sim", "gymnasium"),
        (),
        {
            "hm3d": ("hm3d",),
            "point_goal": ("pointgoal", "sequential"),
            "ground_robot": ("jackal",),
            "learning": ("ppo",),
        },
    ),
    CandidateSpec(
        "ovon",
        "rl",
        RL_LITERATURE_ROOT / "ovon__naokiyokoyama",
        "hm3d_object_navigation_but_single_ground_agent_and_target_driven",
        ("torch", "numpy", "habitat_sim"),
        (),
        {
            "hm3d": ("hm3d",),
            "object_navigation": ("objectnav", "object goal"),
            "learning": ("ppo",),
        },
    ),
    CandidateSpec(
        "falcon",
        "rl",
        RL_LITERATURE_ROOT / "falcon__Zeying_Gong",
        "hm3d_navigation_training_reference_but_not_four_uav_target_free_exploration",
        ("torch", "numpy", "habitat_sim"),
        (),
        {
            "hm3d": ("hm3d",),
            "navigation": ("navigation", "habitat"),
            "learning": ("ppo",),
        },
    ),
    CandidateSpec(
        "ovrl",
        "rl",
        RL_LITERATURE_ROOT / "ovrl__ykarmesh",
        "paper_pointer_without_reusable_implementation",
        (),
        (),
        {"hm3d": ("hm3d",), "learning": ("reinforcement learning",)},
    ),
    CandidateSpec(
        "gvp_mrep",
        "planner",
        Path(
            r"E:\Outcome_Grounded_Repertoire_Literature_2026"
            r"\official_repositories\GVP-MREP-main"
        ),
        "native_multi_uav_unknown_3d_target_free_exploration_with_allocation",
        ("roscpp", "pcl_ros"),
        (),
        {
            "three_dimensional": ("vector3d", "point3d", ".z"),
            "frontier": ("frontier",),
            "multi_uav": ("drone_num", "robot_num", "voronoi"),
            "communication": ("communication", "neighbor"),
        },
    ),
    CandidateSpec(
        "c2_explorer",
        "planner",
        Path(
            r"E:\Outcome_Grounded_Repertoire_Literature_2026"
            r"\official_repositories\C2-Explorer-main"
        ),
        "multi_uav_3d_trajectory_exploration_but_high_level_single_layer_partition",
        ("roscpp", "pcl_ros"),
        (),
        {
            "three_dimensional_trajectory": ("vector3d", "esdf"),
            "single_layer_manager": ("single-layer", "only x and y since grid is 2d"),
            "multi_uav": ("drone_num", "connectivity"),
            "frontier": ("frontier",),
        },
    ),
    CandidateSpec(
        "racer",
        "planner",
        Path(r"E:\github_repos\RACER-main"),
        "native_multi_uav_3d_exploration_but_2023_ros_stack_and_high_port_cost",
        ("roscpp", "pcl_ros"),
        (),
        {
            "three_dimensional": ("vector3d", "voxel", "esdf"),
            "frontier": ("frontier",),
            "multi_uav": ("drone_num", "swarm"),
            "communication": ("communication", "topological"),
        },
    ),
)


SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".md", ".yaml", ".yml"}
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", "build", "devel", "logs"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in SOURCE_SUFFIXES
        and not any(part in IGNORED_PARTS for part in path.parts)
        and path.stat().st_size <= 8 * 1024 * 1024
    )


def _tree_hash(root: Path, files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _license_evidence(root: Path) -> list[dict[str, Any]]:
    rows = []
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"):
        path = root / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")[:4000]
            rows.append(
                {
                    "path": name,
                    "sha256": _sha256(path),
                    "mentions_mit": "mit license" in text.casefold(),
                    "mentions_apache": "apache license" in text.casefold(),
                }
            )
    return rows


def _syntax_probe(files: Sequence[Path]) -> ProbeResult:
    python_files = [path for path in files if path.suffix.casefold() == ".py"]
    failures = []
    for path in python_files:
        try:
            source = path.read_text(encoding="utf-8-sig", errors="strict")
            compile(source, str(path), "exec")
        except (OSError, UnicodeError, SyntaxError) as exc:
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return ProbeResult(
        "pass" if not failures else "partial",
        {
            "python_file_count": len(python_files),
            "compiled_count": len(python_files) - len(failures),
            "failure_count": len(failures),
            "failures": failures[:30],
        },
    )


def _dependency_probe(modules: Sequence[str]) -> ProbeResult:
    details = {module: importlib.util.find_spec(module) is not None for module in modules}
    return ProbeResult("pass" if all(details.values()) else "blocked", details)


def _pattern_probe(files: Sequence[Path], patterns: dict[str, tuple[str, ...]]) -> ProbeResult:
    counts = {label: {pattern: 0 for pattern in values} for label, values in patterns.items()}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").casefold()
        except OSError:
            continue
        for label, values in patterns.items():
            for pattern in values:
                counts[label][pattern] += text.count(pattern.casefold())
    matched = {
        label: any(value > 0 for value in pattern_counts.values())
        for label, pattern_counts in counts.items()
    }
    return ProbeResult(
        "pass" if all(matched.values()) else "partial",
        {"matched": matched, "counts": counts},
    )


def _checkpoint_probe(spec: CandidateSpec) -> ProbeResult:
    rows = []
    for relative in spec.checkpoints:
        path = spec.root / relative
        rows.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path) if path.is_file() else None,
            }
        )
    if not rows:
        return ProbeResult("not_applicable", {"checkpoints": []})
    status = "pass" if any(row["exists"] for row in rows) else "blocked"
    return ProbeResult(status, {"checkpoints": rows})


@contextmanager
def _repo_import(root: Path) -> Iterator[None]:
    old_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path[:] = old_path


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _torch_load(path: Path, *, trust_official_checkpoints: bool) -> tuple[Any, str]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only"
    except Exception:
        if not trust_official_checkpoints:
            raise
        return torch.load(path, map_location="cpu", weights_only=False), "trusted_pickle_fallback"


def _timed_forward(model: Any, args: tuple[Any, ...], repeats: int = 20) -> float:
    import torch

    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(*args)
        started = time.perf_counter()
        for _ in range(repeats):
            model(*args)
    return (time.perf_counter() - started) * 1000.0 / repeats


def _parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _gradient_probe(model: Any, output: Any) -> dict[str, Any]:
    import torch

    model.zero_grad(set_to_none=True)
    tensors = (
        [row for row in output if torch.is_tensor(row)]
        if isinstance(output, tuple)
        else [output]
    )
    loss = sum(row.float().mean() for row in tensors if row.dtype.is_floating_point)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return {
        "loss_finite": bool(torch.isfinite(loss).item()),
        "gradient_tensor_count": len(gradients),
        "all_gradients_finite": bool(
            gradients and all(torch.isfinite(gradient).all().item() for gradient in gradients)
        ),
    }


def _probe_marl_ipp(spec: CandidateSpec, trust: bool) -> ProbeResult:
    import torch

    with _repo_import(spec.root):
        sys.modules.pop("parameters", None)
        module = _load_module("baseline_audit_marl_ipp", spec.root / "attention_net.py")
        model = module.AttentionNet(8, 128)
        checkpoint, load_mode = _torch_load(
            spec.root / "model/our_approach/best_model_checkpoint.pth",
            trust_official_checkpoints=trust,
        )
        incompat = model.load_state_dict(checkpoint["model"], strict=True)

        batch, nodes, neighbors = 4, 80, 20
        torch.manual_seed(19)
        node_inputs = torch.rand(batch, nodes, 8)
        edge_inputs = torch.stack(
            [
                torch.stack(
                    [(torch.arange(neighbors) + node) % nodes for node in range(nodes)]
                )
                for _ in range(batch)
            ]
        ).long()
        budget_inputs = torch.ones(batch, nodes, 1)
        current_index = torch.zeros(batch, 1, 1, dtype=torch.long)
        lstm_h = torch.zeros(batch, 1, 128)
        lstm_c = torch.zeros(batch, 1, 128)
        pos_encoding = torch.rand(batch, nodes, 32)
        legal_mask = torch.zeros(batch, nodes, neighbors, dtype=torch.long)
        args = (
            node_inputs,
            edge_inputs,
            budget_inputs,
            current_index,
            lstm_h,
            lstm_c,
            pos_encoding,
            legal_mask,
        )
        output = model(*args)
        logp, value = output[:2]

        masked = legal_mask.clone()
        masked[:, 0, 1] = 1
        masked_output = model(*args[:-1], masked)[0]
        mask_delta = masked_output[:, 1]

        height_inputs = node_inputs.clone()
        height_inputs[:, :, 2] += torch.linspace(0.0, 1.0, nodes)
        height_output = model(height_inputs, *args[1:])[0]
        height_delta = (height_output - logp).abs().max().item()

        small_args = (
            node_inputs[:, :40],
            edge_inputs[:, :40, :12] % 40,
            budget_inputs[:, :40],
            current_index,
            lstm_h,
            lstm_c,
            pos_encoding[:, :40],
            legal_mask[:, :40, :12],
        )
        small_output = model(*small_args)
        gradient = _gradient_probe(model, output)
        return ProbeResult(
            "pass",
            {
                "checkpoint_load_mode": load_mode,
                "strict_checkpoint_load": (
                    not incompat.missing_keys and not incompat.unexpected_keys
                ),
                "parameter_count": _parameter_count(model),
                "four_agent_output_shapes": [list(logp.shape), list(value.shape)],
                "variable_graph_output_shape": list(small_output[0].shape),
                "masked_action_logp_max": float(mask_delta.max().item()),
                "masked_action_is_rejected": bool((mask_delta < -1.0e3).all().item()),
                "z_channel_perturbation_max_abs_delta": height_delta,
                "z_sensitive": height_delta > 1.0e-6,
                "backward": gradient,
                "cpu_forward_ms_mean": _timed_forward(model, args),
            },
        )


def _probe_marvel(spec: CandidateSpec, trust: bool) -> ProbeResult:
    import torch

    module = _load_module("baseline_audit_marvel", spec.root / "utils/model.py")
    model = module.PolicyNet(6, 128, 36)
    checkpoint, load_mode = _torch_load(
        spec.root / "load_model/MARVEL/checkpoint.pth",
        trust_official_checkpoints=trust,
    )
    incompat = model.load_state_dict(checkpoint["policy_model"], strict=True)

    batch, nodes, neighbors, headings = 4, 64, 25, 3
    torch.manual_seed(23)
    node_inputs = torch.rand(batch, nodes, 6)
    node_padding = torch.zeros(batch, 1, nodes, dtype=torch.long)
    edge_mask = torch.zeros(batch, nodes, nodes, dtype=torch.long)
    current_index = torch.zeros(batch, 1, 1, dtype=torch.long)
    current_edge = torch.arange(neighbors).view(1, neighbors, 1).repeat(batch, 1, 1)
    edge_padding = torch.zeros(batch, 1, neighbors, dtype=torch.long)
    frontier_distribution = torch.rand(batch, nodes, 36)
    headings_visited = torch.rand(batch, nodes, 36)
    neighbor_headings = torch.rand(batch, neighbors, headings, 36)
    args = (
        node_inputs,
        node_padding,
        edge_mask,
        current_index,
        current_edge,
        edge_padding,
        frontier_distribution,
        headings_visited,
        neighbor_headings,
    )
    output = model(*args)
    masked = edge_padding.clone()
    masked[:, :, 1] = 1
    masked_output = model(*args[:5], masked, *args[6:])
    masked_action_indices = (1 * headings) + torch.arange(headings)
    gradient = _gradient_probe(model, output)
    small_args = (
        node_inputs[:, :32],
        node_padding[:, :, :32],
        edge_mask[:, :32, :32],
        current_index,
        current_edge[:, :12] % 32,
        edge_padding[:, :, :12],
        frontier_distribution[:, :32],
        headings_visited[:, :32],
        neighbor_headings[:, :12],
    )
    return ProbeResult(
        "pass",
        {
            "checkpoint_load_mode": load_mode,
            "strict_checkpoint_load": not incompat.missing_keys and not incompat.unexpected_keys,
            "parameter_count": _parameter_count(model),
            "four_agent_output_shape": list(output.shape),
            "variable_graph_output_shape": list(model(*small_args).shape),
            "masked_action_logp_max": float(masked_output[:, masked_action_indices].max().item()),
            "masked_action_is_rejected": bool(
                (masked_output[:, masked_action_indices] < -1.0e7).all().item()
            ),
            "author_state_has_z_coordinate": False,
            "author_state_evidence": (
                "agent.py forms node inputs from relative x/y plus four scalar map features"
            ),
            "backward": gradient,
            "cpu_forward_ms_mean": _timed_forward(model, args),
        },
    )


def _probe_ir2(spec: CandidateSpec, trust: bool) -> ProbeResult:
    import torch

    module = _load_module("baseline_audit_ir2", spec.root / "model.py")
    model = module.PolicyNet(6, 128)
    checkpoint, load_mode = _torch_load(
        spec.root / "model/stage2/checkpoint.pth",
        trust_official_checkpoints=trust,
    )
    incompat = model.load_state_dict(checkpoint["policy_model"], strict=True)

    batch, nodes, neighbors = 4, 100, 30
    torch.manual_seed(29)
    node_inputs = torch.rand(batch, nodes, 6)
    edge_inputs = torch.arange(neighbors).view(1, 1, neighbors).repeat(batch, 1, 1)
    current_index = torch.zeros(batch, 1, 1, dtype=torch.long)
    node_padding = torch.zeros(batch, 1, nodes, dtype=torch.long)
    edge_padding = torch.zeros(batch, 1, neighbors, dtype=torch.long)
    edge_mask = torch.zeros(batch, nodes, nodes, dtype=torch.long)
    args = (node_inputs, edge_inputs, current_index, node_padding, edge_padding, edge_mask)
    output = model(*args)
    masked = edge_padding.clone()
    masked[:, :, 1] = 1
    masked_output = model(node_inputs, edge_inputs, current_index, node_padding, masked, edge_mask)
    gradient = _gradient_probe(model, output)
    small_args = (
        node_inputs[:, :48],
        edge_inputs[:, :, :12] % 48,
        current_index,
        node_padding[:, :, :48],
        edge_padding[:, :, :12],
        edge_mask[:, :48, :48],
    )
    return ProbeResult(
        "pass",
        {
            "checkpoint_load_mode": load_mode,
            "strict_checkpoint_load": not incompat.missing_keys and not incompat.unexpected_keys,
            "parameter_count": _parameter_count(model),
            "four_agent_output_shape": list(output.shape),
            "variable_graph_output_shape": list(model(*small_args).shape),
            "masked_action_logp_max": float(masked_output[:, 1].max().item()),
            "masked_action_is_rejected": bool((masked_output[:, 1] < -1.0e7).all().item()),
            "author_state_has_z_coordinate": False,
            "author_state_evidence": (
                "multi_robot_worker.py concatenates 2D node_coords with four scalars"
            ),
            "backward": gradient,
            "cpu_forward_ms_mean": _timed_forward(model, args),
        },
    )


MODEL_PROBES = {"marl_ipp": _probe_marl_ipp, "marvel": _probe_marvel, "ir2": _probe_ir2}


def _probe_candidate(spec: CandidateSpec, *, trust: bool) -> dict[str, Any]:
    started = time.perf_counter()
    if not spec.root.is_dir():
        return {
            "candidate_id": spec.candidate_id,
            "family": spec.family,
            "root": str(spec.root),
            "task_fit": spec.task_fit,
            "repository_status": "missing",
        }
    files = _source_files(spec.root)
    model_probe = ProbeResult("not_applicable")
    if spec.candidate_id in MODEL_PROBES:
        try:
            model_probe = MODEL_PROBES[spec.candidate_id](spec, trust)
        except Exception as exc:
            model_probe = ProbeResult("blocked", error=f"{type(exc).__name__}: {exc}")
    return {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "root": str(spec.root),
        "task_fit": spec.task_fit,
        "repository_status": "present",
        "source_file_count": len(files),
        "source_tree_sha256": _tree_hash(spec.root, files),
        "licenses": _license_evidence(spec.root),
        "syntax": asdict(_syntax_probe(files)),
        "dependencies": asdict(_dependency_probe(spec.required_modules)),
        "source_evidence": asdict(_pattern_probe(files, spec.evidence_patterns)),
        "checkpoints": asdict(_checkpoint_probe(spec)),
        "original_model_probe": asdict(model_probe),
        "elapsed_s": time.perf_counter() - started,
    }


def _environment() -> dict[str, Any]:
    torch_version = None
    cuda_available = False
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ModuleNotFoundError:
        pass
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--trust-official-checkpoints",
        action="store_true",
        help="Allow pickle fallback for checkpoints that PyTorch cannot read in weights-only mode.",
    )
    args = parser.parse_args()

    report = {
        "schema_version": "external-baseline-candidate-audit-v1",
        "generated_at_unix_s": time.time(),
        "claim_limit": (
            "Source/model probes are migration evidence, not HM3D evaluation results. "
            "A candidate becomes a formal baseline only after common-protocol training "
            "and evaluation."
        ),
        "environment": _environment(),
        "candidates": [
            _probe_candidate(spec, trust=args.trust_official_checkpoints) for spec in SPECS
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
