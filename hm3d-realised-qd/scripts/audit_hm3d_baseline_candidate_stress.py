"""Stress-audit HM3D baseline candidates without fabricating performance claims.

The synthetic states in this script test interface compatibility, legality
masking, numerical stability, vertical sensitivity, and selector separation.
They are not simulator rollouts and therefore cannot qualify a method for a
paper result table.  Real HM3D/CF2X/PhysX episodes remain mandatory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.adapters.hm3d_baselines import (  # noqa: E402
    ConservativeTransitTimingModel,
    PublicAgentPose,
    PublicFrontier,
    PublicSearchState,
    build_public_candidate_pool,
    identity_path_guard,
    select_public_baseline,
)
from aerocity_method.adapters.hm3d_external_baselines import (  # noqa: E402
    GVP_MREP_AUTHOR_COMMIT,
    GVP_MREP_GRAPH_PARTITION_SHA256,
    select_gvp_mrep_port,
)
from aerocity_method.adapters.hm3d_marl_ipp import (  # noqa: E402
    MarlIPPPortConfig,
    MarlIPPPortPolicy,
    public_marl_ipp_graph_input,
)
from aerocity_method.adapters.hm3d_marvel import (  # noqa: E402
    MarvelSupplementaryReferenceConfig,
    MarvelSupplementaryReferencePolicy,
    public_marvel_adjacency,
    public_marvel_agent_features,
    public_marvel_candidate_features,
)
from aerocity_method.adapters.hm3d_single_rl import (  # noqa: E402
    public_candidate_features,
    public_context_features,
)
from aerocity_method.contracts.io import canonical_sha256  # noqa: E402
from aerocity_method.contracts.models import (  # noqa: E402
    CandidateFragmentManifest,
    PublicMethodContext,
)
from aerocity_method.learning.rb_sf_sac import RBSFSAC, RBSFSACConfig  # noqa: E402

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - dependency failure path
    raise RuntimeError("baseline stress audit requires PyTorch") from error


AUDIT_SCHEMA_VERSION = "hm3d-four-uav-baseline-candidate-stress-v4"
SPLIT_HASH = "b" * 64
DEFAULT_MARL_IPP_ROOT = Path(r"E:\github_repos\marl_ipp-main")
DEFAULT_MARL_IPP_CHECKPOINT = (
    DEFAULT_MARL_IPP_ROOT / "model" / "our_approach" / "best_model_checkpoint.pth"
)
LITERATURE_ROOT = Path(r"E:\HM3D_2026_papers_and_repos\hm3d_rl_literature_2026-08-03")
RECURIOSITY_ROOT = LITERATURE_ROOT / "repos" / "recuriosity__recuriosity"
VISFLY_ROOT = LITERATURE_ROOT / "repos" / "visfly__SJTU_ViSYS_team"
FALCON_ROOT = LITERATURE_ROOT / "repos" / "falcon__Zeying_Gong"
OVON_ROOT = LITERATURE_ROOT / "repos" / "ovon__naokiyokoyama"
RVN_ROOT = LITERATURE_ROOT / "repos" / "rvn_bench__Sequor_Robotics_Research"
MARVEL_ROOT = Path(r"E:\github_repos\MARVEL-main")
MARVEL_CHECKPOINT = MARVEL_ROOT / "load_model" / "MARVEL" / "checkpoint.pth"
GVP_ROOT = Path(
    r"E:\Outcome_Grounded_Repertoire_Literature_2026\official_repositories\GVP-MREP-main"
)
C2_ROOT = Path(
    r"E:\Outcome_Grounded_Repertoire_Literature_2026\official_repositories\C2-Explorer-main"
)
MARL_IPP_PORT_SOURCE = (
    ROOT / "src" / "aerocity_method" / "adapters" / "hm3d_marl_ipp.py"
)
MARL_IPP_TRAINING_SOURCE = (
    ROOT / "src" / "aerocity_method" / "evaluation" / "hm3d_marl_ipp_training.py"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_author_classes(
    path: Path,
    class_names: Sequence[str],
    namespace: dict[str, object],
) -> dict[str, object]:
    """Execute exact author class definitions without importing heavy runtimes."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    wanted = set(class_names)
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            node.decorator_list = []
            body.append(node)
    found = {node.name for node in body if isinstance(node, ast.ClassDef)}
    missing = wanted - found
    if missing:
        raise RuntimeError(f"author classes missing from {path}: {sorted(missing)}")
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    loaded = dict(namespace)
    exec(compile(module, str(path), "exec"), loaded)
    return {name: loaded[name] for name in class_names}


def _load_module_from_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load author module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _python_source_audit(root: Path) -> dict[str, object]:
    files = tuple(sorted(root.rglob("*.py"))) if root.is_dir() else ()
    failures: list[dict[str, object]] = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as error:
            failures.append(
                {
                    "path": str(path.relative_to(root)),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    return {
        "python_file_count": len(files),
        "syntax_success_count": len(files) - len(failures),
        "syntax_failure_count": len(failures),
        "first_failures": failures[:20],
        "qualification_scope": "source_parse_only_not_runtime_or_performance",
    }


def _state(case: int, *, vertical_scale: float) -> PublicSearchState:
    time_remaining_s = 8.0 + float(case % 33)
    context = PublicMethodContext(
        context_id=f"stress-context-{case}-{int(vertical_scale * 1000)}",
        episode_id=f"stress-episode-{case}",
        decision_id=f"decision-{case % 11}",
        agent_features=tuple(
            (f"uav{index}", (1.0 - 0.02 * index, float((case + index) % 4)))
            for index in range(4)
        ),
        public_features=(("sparse_range_schedule_hz", 10.0),),
        budget=(("time_remaining_s", time_remaining_s),),
    )
    agents = tuple(
        PublicAgentPose(
            f"uav{index}",
            (
                3.5 * index + 0.15 * math.sin(case + index),
                0.8 * ((index + case) % 3),
                1.0 + 0.20 * index,
            ),
            1.0 - 0.05 * index,
            (case + index) % 4,
        )
        for index in range(4)
    )
    frontiers: list[PublicFrontier] = []
    for index, agent in enumerate(agents):
        phase = 0.17 * case + index
        base_x, base_y, base_z = agent.position_m
        choices = (
            ("level", 1.0, 0.3 * math.sin(phase), 0.0, 0.45),
            ("up", 0.9, 0.5, vertical_scale * (1.3 + 0.2 * index), 0.95),
            ("down", 0.8, -0.5, -vertical_scale * (0.45 + 0.05 * index), 0.68),
        )
        for offset, (kind, dx, dy, dz, gain) in enumerate(choices):
            frontiers.append(
                PublicFrontier(
                    f"uav{index}-{kind}-{case}",
                    (base_x + dx, base_y + dy, max(0.35, base_z + dz)),
                    gain + 0.03 * ((case + offset + index) % 5),
                    0.03 + 0.025 * ((case + 2 * offset + index) % 6),
                    f"uav{index}",
                )
            )
    return PublicSearchState(
        context=context,
        agents=agents,
        frontiers=tuple(frontiers),
        decision_start_s=float(case % 7),
        decision_duration_s=40.0,
        transit_timing_model=ConservativeTransitTimingModel(
            "four-uav-stress", 2.0, 2.0, 0.05
        ),
        observe_dwell_s=0.5,
        communication_range_m=6.0 + float(case % 10),
    )


def _pool(state: PublicSearchState, case: int) -> tuple[CandidateFragmentManifest, ...]:
    count = 4 + case % 5
    rows = list(
        build_public_candidate_pool(
            state,
            identity_path_guard,
            candidate_limit=count,
            minimum_feasible_candidates=2,
        )
    )
    rejected = {case % len(rows)}
    if len(rows) >= 6 and case % 3 == 0:
        rejected.add((case * 3 + 1) % len(rows))
    if len(rejected) >= len(rows) - 1:
        rejected = {case % len(rows)}
    for index in rejected:
        rows[index] = replace(
            rows[index],
            feasible=False,
            admission_reasons=("stress_guard_rejected",),
        )
    return tuple(rows)


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(max(value, 1.0e-12)) for value in probabilities)


def _score_variance(selection: Any) -> float:
    values = [float(value) for _, value in selection.scores]
    return statistics.pvariance(values) if len(values) > 1 else 0.0


def _vertical_ratio(manifest: CandidateFragmentManifest) -> float:
    return float(manifest.planned_descriptor[0])


@dataclass
class LearningAccumulator:
    inference_ns: list[int]
    selected: Counter[str]
    probability_sum_error_max: float = 0.0
    illegal_probability_mass_max: float = 0.0
    illegal_selection_count: int = 0
    entropy_sum: float = 0.0
    vertical_selected_count: int = 0
    height_response_l1_sum: float = 0.0

    @classmethod
    def create(cls) -> LearningAccumulator:
        return cls([], Counter())

    def add(
        self,
        pool: Sequence[CandidateFragmentManifest],
        probabilities: Sequence[float],
        elapsed_ns: int,
        high_probabilities: Sequence[float],
    ) -> None:
        if len(probabilities) != len(pool) or len(high_probabilities) != len(pool):
            raise RuntimeError("learning selector probability width changed under height audit")
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise RuntimeError("learning selector emitted an invalid probability")
        probability_sum = sum(probabilities)
        self.probability_sum_error_max = max(
            self.probability_sum_error_max, abs(probability_sum - 1.0)
        )
        illegal_mass = sum(
            probability for row, probability in zip(pool, probabilities, strict=True)
            if not row.feasible
        )
        self.illegal_probability_mass_max = max(
            self.illegal_probability_mass_max, illegal_mass
        )
        selected_index = max(range(len(pool)), key=lambda index: probabilities[index])
        selected = pool[selected_index]
        self.illegal_selection_count += int(not selected.feasible)
        self.selected[selected.candidate_id] += 1
        self.inference_ns.append(elapsed_ns)
        self.entropy_sum += _entropy(probabilities)
        self.vertical_selected_count += int(_vertical_ratio(selected) > 0.05)
        self.height_response_l1_sum += sum(
            abs(left - right)
            for left, right in zip(probabilities, high_probabilities, strict=True)
        )

    def summary(self, cases: int) -> dict[str, object]:
        ordered = sorted(self.inference_ns)
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
        return {
            "case_count": cases,
            "illegal_selection_count": self.illegal_selection_count,
            "illegal_probability_mass_max": self.illegal_probability_mass_max,
            "probability_sum_error_max": self.probability_sum_error_max,
            "mean_entropy": self.entropy_sum / cases,
            "mean_inference_ms": statistics.fmean(self.inference_ns) / 1.0e6,
            "p95_inference_ms": p95 / 1.0e6,
            "vertical_selected_fraction": self.vertical_selected_count / cases,
            "mean_height_perturbation_probability_l1": self.height_response_l1_sum / cases,
            "distinct_argmax_candidate_ids": len(self.selected),
            "qualification_scope": "structural_stability_only_not_hm3d_performance",
        }


def _single_probabilities(
    model: RBSFSAC,
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
) -> tuple[float, ...]:
    return model.action_probabilities(
        public_context_features(state),
        tuple(public_candidate_features(row) for row in pool),
        tuple(row.feasible for row in pool),
    )


def _marvel_probabilities(
    model: MarvelSupplementaryReferencePolicy,
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
) -> tuple[float, ...]:
    return model.action_probabilities(
        public_context_features(state),
        public_marvel_agent_features(state),
        public_marvel_adjacency(state),
        public_marvel_candidate_features(state, pool),
        tuple(row.feasible for row in pool),
    )


def _marl_ipp_probabilities(
    model: MarlIPPPortPolicy,
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
) -> tuple[float, ...]:
    return model.action_probabilities(public_marl_ipp_graph_input(state, pool))


def _learning_stress(
    cases: int,
    *,
    marl_ipp_root: Path,
    marl_ipp_checkpoint: Path,
) -> dict[str, object]:
    torch.set_num_threads(1)
    models: dict[str, tuple[Any, Callable[..., tuple[float, ...]], str]] = {
        "single_rl": (
            RBSFSAC(
                RBSFSACConfig(
                    context_dim=4,
                    candidate_dim=5,
                    preference_dim=0,
                    sf_dim=0,
                    hidden_dim=64,
                ),
                seed=20260804,
            ),
            _single_probabilities,
            "random_initialization_for_interface_stress_only",
        ),
        # MARVEL remains a two-dimensional architecture reference.  Keep this
        # synthetic interface audit separate from runnable HM3D baselines so
        # its label cannot be mistaken for a formal P07 strategy.
        "marvel_supplementary_reference": (
            MarvelSupplementaryReferencePolicy(MarvelSupplementaryReferenceConfig(), seed=20260804),
            _marvel_probabilities,
            "local_reimplementation_random_initialization_for_interface_stress_only",
        ),
        "marl_ipp_port": (
            MarlIPPPortPolicy(
                marl_ipp_root,
                MarlIPPPortConfig(),
                source_checkpoint=marl_ipp_checkpoint,
                seed=20260804,
            ),
            _marl_ipp_probabilities,
            "author_attention_net_and_author_checkpoint_controlled_transfer",
        ),
    }
    accumulators = {name: LearningAccumulator.create() for name in models}
    for case in range(cases):
        state = _state(case, vertical_scale=0.65)
        high_state = _state(case, vertical_scale=1.65)
        pool = _pool(state, case)
        high_pool = _pool(high_state, case)
        for name, (model, probability_fn, _) in models.items():
            start = time.perf_counter_ns()
            probabilities = probability_fn(model, state, pool)
            elapsed_ns = time.perf_counter_ns() - start
            high_probabilities = probability_fn(model, high_state, high_pool)
            accumulators[name].add(pool, probabilities, elapsed_ns, high_probabilities)
    results: dict[str, object] = {}
    for name, (_, _, provenance) in models.items():
        row = accumulators[name].summary(cases)
        row["implementation_provenance"] = provenance
        results[name] = row
    return results


def _planning_stress(cases: int) -> dict[str, object]:
    selectors = ("frontier_3d", "auction", "gvp_mrep_port")
    selected_counts = {name: Counter() for name in selectors}
    vertical = Counter()
    score_variance_sum = Counter()
    disagreements = Counter()
    zero_variance = Counter()
    for case in range(cases):
        state = _state(case, vertical_scale=1.0)
        pool = _pool(state, case)
        selected: dict[str, CandidateFragmentManifest] = {}
        frontier, frontier_selection = select_public_baseline("frontier_3d", pool)
        auction, auction_selection = select_public_baseline("auction", pool)
        gvp, gvp_selection = select_gvp_mrep_port(state, pool)
        rows = {
            "frontier_3d": (frontier, frontier_selection),
            "auction": (auction, auction_selection),
            "gvp_mrep_port": (gvp, gvp_selection),
        }
        for name, (manifest, selection) in rows.items():
            if not manifest.feasible:
                raise RuntimeError(f"{name} selected an illegal candidate")
            selected[name] = manifest
            selected_counts[name][manifest.candidate_id] += 1
            vertical[name] += int(_vertical_ratio(manifest) > 0.05)
            variance = _score_variance(selection)
            score_variance_sum[name] += variance
            zero_variance[name] += int(variance <= 1.0e-15)
        for left_index, left in enumerate(selectors):
            for right in selectors[left_index + 1 :]:
                key = f"{left}_vs_{right}"
                disagreements[key] += int(
                    selected[left].manifest_hash != selected[right].manifest_hash
                )
    return {
        "methods": {
            name: {
                "case_count": cases,
                "illegal_selection_count": 0,
                "vertical_selected_fraction": vertical[name] / cases,
                "mean_score_variance": score_variance_sum[name] / cases,
                "zero_score_variance_fraction": zero_variance[name] / cases,
                "distinct_argmax_candidate_ids": len(selected_counts[name]),
            }
            for name in selectors
        },
        "pairwise_selection_disagreement_fraction": {
            key: value / cases for key, value in sorted(disagreements.items())
        },
        "qualification_scope": "selector_separation_only_not_hm3d_performance",
    }


def _timing_summary(samples_ns: Sequence[int]) -> dict[str, float]:
    ordered = sorted(samples_ns)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "mean_ms": statistics.fmean(samples_ns) / 1.0e6,
        "p95_ms": p95 / 1.0e6,
    }


def _recuriosity_author_stress(cases: int) -> dict[str, object]:
    source = RECURIOSITY_ROOT / "modules" / "agent" / "agent.py"
    classes = _load_author_classes(
        source,
        ("LinearMemState", "LinearAttentionMemory"),
        {
            "torch": torch,
            "nn": torch.nn,
            "List": list,
            "Optional": Optional,
        },
    )
    state_cls = classes["LinearMemState"]
    memory_cls = classes["LinearAttentionMemory"]
    model = memory_cls(d_model=32, d_k=8, d_v=8)
    torch.nn.init.normal_(model.readout.weight, mean=0.0, std=0.05)
    generator = torch.Generator(device="cpu").manual_seed(20260804)
    max_parallel_incremental_error = 0.0
    first_token_nonzero_count = 0
    nonfinite_gradient_count = 0
    clear_failure_count = 0
    timings: list[int] = []
    for case in range(cases):
        batch_size = 1 + case % 4
        sequence_length = 2 + case % 15
        x = torch.randn(
            batch_size,
            sequence_length,
            32,
            generator=generator,
            requires_grad=True,
        )
        started = time.perf_counter_ns()
        parallel = model.forward_parallel(x)
        state = state_cls(1, batch_size, 8, 8, device=x.device, dtype=x.dtype)
        incremental = torch.stack(
            [model.forward_step(x[:, step], state, 0) for step in range(sequence_length)],
            dim=1,
        )
        timings.append(time.perf_counter_ns() - started)
        max_parallel_incremental_error = max(
            max_parallel_incremental_error,
            float((parallel - incremental).abs().max().item()),
        )
        first_token_nonzero_count += int(
            float(parallel[:, 0].detach().abs().max().item()) > 1.0e-7
        )
        model.zero_grad(set_to_none=True)
        parallel.square().mean().backward()
        nonfinite_gradient_count += int(
            any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )
        state.clear()
        clear_failure_count += int(
            any(value is not None for value in state.S)
            or any(value is not None for value in state.z)
        )
    return {
        "source_file": str(source),
        "source_sha256": _sha256_file(source),
        "case_count": cases,
        "max_parallel_incremental_abs_error": max_parallel_incremental_error,
        "first_token_read_before_update_failure_count": first_token_nonzero_count,
        "nonfinite_gradient_case_count": nonfinite_gradient_count,
        "state_clear_failure_count": clear_failure_count,
        "timing": _timing_summary(timings),
        "author_contract": "single_agent_habitat_rgb_n_act_4",
        "qualification_scope": (
            "author_memory_component_stability_only; not a four-UAV selector "
            "or HM3D performance result"
        ),
    }


def _visfly_author_stress(cases: int) -> dict[str, object]:
    repos_root = LITERATURE_ROOT / "repos"
    if str(repos_root) not in sys.path:
        sys.path.insert(0, str(repos_root))
    dynamics_module = importlib.import_module(
        "visfly__SJTU_ViSYS_team.envs.base.dynamics"
    )
    dynamics_cls = dynamics_module.Dynamics
    dynamics = dynamics_cls(
        num=4,
        action_type="velocity",
        ctrl_delay=False,
        cfg="drone_state",
        device=torch.device("cpu"),
    )
    initial = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    )

    def vertical_displacement(command_z: float) -> float:
        dynamics.reset(pos=initial)
        command = torch.tensor([[0.0, 0.0, 0.0, command_z]] * 4)
        for _ in range(20):
            dynamics.step(command)
        return float((dynamics.position[:, 2] - initial[:, 2]).mean().item())

    upward_displacement = vertical_displacement(0.35)
    downward_displacement = vertical_displacement(-0.35)
    generator = torch.Generator(device="cpu").manual_seed(20260804)
    reset_max_abs_error = 0.0
    nonfinite_state_count = 0
    timings: list[int] = []
    dynamics.reset(pos=initial)
    for case in range(cases):
        if case % 25 == 0:
            dynamics.reset(pos=initial)
            reset_max_abs_error = max(
                reset_max_abs_error,
                float((dynamics.position - initial).abs().max().item()),
            )
        action = torch.rand((4, 4), generator=generator) * 0.7 - 0.35
        started = time.perf_counter_ns()
        dynamics.step(action)
        timings.append(time.perf_counter_ns() - started)
        nonfinite_state_count += int(not torch.isfinite(dynamics.state).all())
    source = VISFLY_ROOT / "envs" / "base" / "dynamics.py"
    return {
        "source_file": str(source),
        "source_sha256": _sha256_file(source),
        "step_count": cases,
        "uav_count": 4,
        "state_width": int(dynamics.state.shape[1]),
        "nonfinite_state_step_count": nonfinite_state_count,
        "reset_max_abs_position_error_m": reset_max_abs_error,
        "positive_z_command_mean_displacement_m": upward_displacement,
        "negative_z_command_mean_displacement_m": downward_displacement,
        "vertical_command_ordering_valid": upward_displacement > downward_displacement,
        "timing": _timing_summary(timings),
        "qualification_scope": (
            "author quadrotor dynamics component only; no HM3D collision, CF2X "
            "equivalence, exploration policy, or paper-table result"
        ),
    }


def _rvn_author_stress(cases: int) -> dict[str, object]:
    source = RVN_ROOT / "algo" / "ppo_vanilla.py"
    actor_cls = _load_author_classes(
        source,
        ("ActorCritic",),
        {"torch": torch, "nn": torch.nn, "F": torch.nn.functional},
    )["ActorCritic"]
    model = actor_cls(obs_dim=16, action_dim=3, hidden_dim=64)
    generator = torch.Generator(device="cpu").manual_seed(20260804)
    probability_sum_error_max = 0.0
    nonfinite_output_count = 0
    nonfinite_gradient_count = 0
    timings: list[int] = []
    for case in range(cases):
        obs = torch.randn(1 + case % 16, 16, generator=generator, requires_grad=True)
        started = time.perf_counter_ns()
        value, probabilities = model(obs)
        timings.append(time.perf_counter_ns() - started)
        probability_sum_error_max = max(
            probability_sum_error_max,
            float((probabilities.sum(dim=-1) - 1.0).abs().max().item()),
        )
        nonfinite_output_count += int(
            not torch.isfinite(value).all() or not torch.isfinite(probabilities).all()
        )
        model.zero_grad(set_to_none=True)
        (value.square().mean() - probabilities.clamp_min(1.0e-8).log().mean()).backward()
        nonfinite_gradient_count += int(
            any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )
    return {
        "source_file": str(source),
        "source_sha256": _sha256_file(source),
        "case_count": cases,
        "probability_sum_error_max": probability_sum_error_max,
        "nonfinite_output_case_count": nonfinite_output_count,
        "nonfinite_gradient_case_count": nonfinite_gradient_count,
        "timing": _timing_summary(timings),
        "author_contract": "single_jackal_three_action_sequential_point_goal",
        "qualification_scope": (
            "author PPO head stability only; observation and action contracts do not "
            "match four-UAV target-free exploration"
        ),
    }


def _falcon_author_stress(cases: int) -> dict[str, object]:
    source = FALCON_ROOT / "falcon" / "auxiliary_tasks.py"

    class DummyNet:
        output_size = 16

    gym_stub = SimpleNamespace(spaces=SimpleNamespace(Box=object))
    predictor_cls = _load_author_classes(
        source,
        ("FutureTrajectoryPrediction",),
        {"torch": torch, "nn": torch.nn, "gym": gym_stub, "Net": object},
    )["FutureTrajectoryPrediction"]
    model = predictor_cls(action_space=None, net=DummyNet())
    generator = torch.Generator(device="cpu").manual_seed(20260804)
    nonfinite_loss_count = 0
    nonfinite_gradient_count = 0
    permutation_error_max = 0.0
    permutation_dependency_count = 0
    timings: list[int] = []

    def prediction(features: torch.Tensor) -> torch.Tensor:
        lstm_output, _ = model.lstm(features)
        attention_output, _ = model.attention(
            lstm_output, lstm_output, lstm_output
        )
        return model.classifier(attention_output).view(
            features.shape[0], model.max_human_num, model.future_step, model.position_dim
        )

    for case in range(cases):
        batch_size = 2 + case % 7
        scene = torch.randn(batch_size, 16, generator=generator, requires_grad=True)
        human_num = torch.randint(0, 7, (batch_size, 1), generator=generator).float()
        trajectories = torch.randn(
            batch_size, 6, 5, 2, generator=generator
        )
        localization = torch.randn(batch_size, 3, generator=generator)
        features = torch.cat(
            (scene, human_num, trajectories[:, :, 0, :].reshape(batch_size, -1)),
            dim=-1,
        )
        permutation = torch.randperm(batch_size, generator=generator)
        started = time.perf_counter_ns()
        base_prediction = prediction(features)
        permuted_prediction = prediction(features[permutation])
        timings.append(time.perf_counter_ns() - started)
        inverse = torch.argsort(permutation)
        error = float(
            (base_prediction - permuted_prediction[inverse]).abs().max().item()
        )
        permutation_error_max = max(permutation_error_max, error)
        permutation_dependency_count += int(error > 1.0e-6)
        aux_state = {"rnn_output": scene}
        batch = {
            "observations": {
                "human_num_sensor": human_num,
                "oracle_humanoid_future_trajectory": trajectories,
                "localization_sensor": localization,
            }
        }
        loss = model(aux_state, batch)["loss"]
        nonfinite_loss_count += int(not torch.isfinite(loss))
        model.zero_grad(set_to_none=True)
        loss.backward()
        nonfinite_gradient_count += int(
            any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )
    return {
        "source_file": str(source),
        "source_sha256": _sha256_file(source),
        "case_count": cases,
        "nonfinite_loss_case_count": nonfinite_loss_count,
        "nonfinite_gradient_case_count": nonfinite_gradient_count,
        "batch_permutation_equivariance_max_abs_error": permutation_error_max,
        "batch_permutation_dependency_fraction": permutation_dependency_count / cases,
        "uses_oracle_humanoid_future_trajectory": True,
        "timing": _timing_summary(timings),
        "qualification_scope": (
            "author social-navigation auxiliary head only; oracle-human and PointNav "
            "contract excludes it from the target-free exploration main table"
        ),
    }


def _marvel_author_checkpoint_stress(cases: int) -> dict[str, object]:
    source = MARVEL_ROOT / "utils" / "model.py"
    module = _load_module_from_file("_aerocity_author_marvel_model", source)
    model = module.PolicyNet(6, 128, 36)
    checkpoint = torch.load(MARVEL_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["policy_model"], strict=True)
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(20260804)
    probability_sum_error_max = 0.0
    nonfinite_output_count = 0
    height_preprocessing_invariance_error_max = 0.0
    timings: list[int] = []
    with torch.no_grad():
        for case in range(cases):
            node_count = 12 + case % 13
            edge_count = min(10, node_count)
            xyz = torch.randn(node_count, 3, generator=generator)
            shifted_xyz = xyz.clone()
            shifted_xyz[:, 2] += 0.5 + 0.1 * (case % 8)
            extras = torch.rand(node_count, 4, generator=generator)

            def author_node_input(
                points: torch.Tensor, node_extras: torch.Tensor
            ) -> torch.Tensor:
                relative_xy = points[:, :2] - points[0, :2]
                return torch.cat((relative_xy, node_extras), dim=-1).unsqueeze(0)

            node_inputs = author_node_input(xyz, extras)
            shifted_inputs = author_node_input(shifted_xyz, extras)
            height_preprocessing_invariance_error_max = max(
                height_preprocessing_invariance_error_max,
                float((node_inputs - shifted_inputs).abs().max().item()),
            )
            node_padding_mask = torch.zeros((1, 1, node_count), dtype=torch.int16)
            edge_mask = torch.zeros((1, node_count, node_count), dtype=torch.int16)
            current_index = torch.zeros((1, 1, 1), dtype=torch.long)
            current_edge = torch.arange(edge_count).reshape(1, edge_count, 1)
            edge_padding_mask = torch.zeros((1, 1, edge_count), dtype=torch.int16)
            edge_padding_mask[0, 0, 0] = 1
            frontier_distribution = torch.rand(
                1, node_count, 36, generator=generator
            )
            headings_visited = torch.randint(
                0, 2, (1, node_count, 36), generator=generator
            ).float()
            neighbor_best_headings = torch.zeros((1, edge_count, 3, 36))
            indices = torch.randint(
                0, 36, (1, edge_count, 3, 1), generator=generator
            )
            neighbor_best_headings.scatter_(-1, indices, 1.0)
            started = time.perf_counter_ns()
            log_probabilities = model(
                node_inputs,
                node_padding_mask,
                edge_mask,
                current_index,
                current_edge,
                edge_padding_mask,
                frontier_distribution,
                headings_visited,
                neighbor_best_headings,
            )
            timings.append(time.perf_counter_ns() - started)
            probabilities = log_probabilities.exp()
            probability_sum_error_max = max(
                probability_sum_error_max,
                float((probabilities.sum(dim=-1) - 1.0).abs().max().item()),
            )
            nonfinite_output_count += int(not torch.isfinite(log_probabilities).all())
    return {
        "source_file": str(source),
        "source_sha256": _sha256_file(source),
        "checkpoint_file": str(MARVEL_CHECKPOINT),
        "checkpoint_sha256": _sha256_file(MARVEL_CHECKPOINT),
        "case_count": cases,
        "checkpoint_loaded_strictly": True,
        "probability_sum_error_max": probability_sum_error_max,
        "nonfinite_output_case_count": nonfinite_output_count,
        "height_preprocessing_invariance_error_max": height_preprocessing_invariance_error_max,
        "height_feature_slots": 0,
        "timing": _timing_summary(timings),
        "qualification_scope": (
            "author network and checkpoint inference; original XY fixed-height "
            "observation cannot represent multilevel decisions"
        ),
    }


def _ovon_contract_audit() -> dict[str, object]:
    files = tuple(
        path
        for suffix in ("*.py", "*.yaml", "*.yml", "*.sh")
        for path in OVON_ROOT.rglob(suffix)
    )
    corpus = "\n".join(
        path.read_text(encoding="utf-8", errors="replace").lower() for path in files
    )
    token_counts = {
        token: corpus.count(token)
        for token in (
            "objectgoal",
            "ddppo",
            "dagger",
            "rgb_sensor",
            "depth_sensor",
            "habitat_sim",
        )
    }
    return {
        "source_file_count": len(files),
        "token_counts": token_counts,
        "habitat_sim_import_available": importlib.util.find_spec("habitat_sim") is not None,
        "runtime_case_count": 0,
        "runtime_blocker": "Habitat-Sim is not installed in the IsaacLab environment",
        "author_contract": "single_ground_robot_open_vocabulary_objectnav",
        "qualification_scope": (
            "static author-source contract audit only; no runtime or target-free "
            "four-UAV performance claim"
        ),
    }


def _author_component_stress(cases: int) -> dict[str, object]:
    return {
        "recuriosity": _recuriosity_author_stress(cases),
        "visfly": _visfly_author_stress(cases),
        "rvn_bench": _rvn_author_stress(cases),
        "falcon": _falcon_author_stress(cases),
        "marvel": _marvel_author_checkpoint_stress(cases),
        "ovon": _ovon_contract_audit(),
    }


def _repository_source_audit() -> dict[str, object]:
    return {
        "recuriosity": _python_source_audit(RECURIOSITY_ROOT),
        "visfly": _python_source_audit(VISFLY_ROOT),
        "rvn_bench": _python_source_audit(RVN_ROOT),
        "falcon": _python_source_audit(FALCON_ROOT),
        "ovon": _python_source_audit(OVON_ROOT),
        "marvel": _python_source_audit(MARVEL_ROOT),
        "marl_ipp": _python_source_audit(DEFAULT_MARL_IPP_ROOT),
    }


def _marl_ipp_training_fidelity_audit(
    marl_ipp_root: Path = DEFAULT_MARL_IPP_ROOT,
    port_source: Path = MARL_IPP_PORT_SOURCE,
    training_source: Path = MARL_IPP_TRAINING_SOURCE,
) -> dict[str, object]:
    """Check whether the local trainer still implements the authors' PPO semantics."""

    author_driver = marl_ipp_root / "driver.py"
    author_worker = marl_ipp_root / "worker.py"
    driver_text = author_driver.read_text(encoding="utf-8", errors="replace")
    worker_text = author_worker.read_text(encoding="utf-8", errors="replace")
    author_text = f"{driver_text}\n{worker_text}"
    port_text = port_source.read_text(encoding="utf-8")
    training_text = training_source.read_text(encoding="utf-8")
    local_text = f"{port_text}\n{training_text}"

    author_contract = {
        "multi_step_episode_up_to_256_steps": "for i in range(256)" in worker_text,
        "recurrent_state_carried_between_decisions": all(
            term in worker_text
            for term in (
                "agent.LSTM_h, agent.LSTM_c = agent.network",
                "agent.experience[9] += agent.LSTM_h",
                "agent.experience[10] += agent.LSTM_c",
            )
        ),
        "old_policy_log_probability_snapshot": "old_logp" in driver_text,
        "ppo_probability_ratio": "ratios = torch.exp" in driver_text,
        "ppo_clipped_surrogate": all(
            term in driver_text
            for term in ("torch.clamp", "surr1", "surr2", "torch.min")
        ),
        "eight_ppo_epochs": "for i in range(8)" in driver_text,
        "discounted_temporal_return": all(
            term in author_text for term in ("discount(", "target_v", "value_prime")
        ),
    }
    current_port_contract = {
        "multi_step_episode_transition": all(
            term in local_text
            for term in (
                "hm3d-marl-ipp-train-transition-v2",
                '"done": done',
                '"next_public_context_hash"',
                "len(emitted_rows) != len(decisions)",
                "only the final MARL-IPP decision may be terminal",
            )
        ),
        "recurrent_state_carried_between_decisions": (
            "LSTM_h" in training_text and "LSTM_c" in training_text
        ),
        "old_policy_log_probability_snapshot": "old_logp" in local_text,
        "ppo_probability_ratio": (
            "ratios = torch.exp" in local_text or "ratio = torch.exp" in local_text
        ),
        "ppo_clipped_surrogate": all(
            term in local_text for term in ("torch.clamp", "surr1", "surr2")
        ),
        "multiple_ppo_epochs": (
            "ppo_epochs" in local_text or "num_ppo_epochs" in local_text
        ),
        "discounted_temporal_return": (
            "target_v" in training_text and "value_prime" in training_text
        ),
        "terminal_single_decision_rows": (
            "marl_ipp_training_transition" in training_text
            and "decisions[0]" in training_text
            and '"done": True' in port_text
        ),
        "recurrent_state_reset_for_every_graph": all(
            term in port_text
            for term in (
                "hidden = torch.zeros",
                "cell = torch.zeros_like(hidden)",
            )
        ),
    }
    required_local_terms = (
        "multi_step_episode_transition",
        "recurrent_state_carried_between_decisions",
        "old_policy_log_probability_snapshot",
        "ppo_probability_ratio",
        "ppo_clipped_surrogate",
        "multiple_ppo_epochs",
        "discounted_temporal_return",
    )
    blockers = [
        name for name in required_local_terms if not current_port_contract[name]
    ]
    if current_port_contract["terminal_single_decision_rows"]:
        blockers.append("terminal_single_decision_rows")
    if current_port_contract["recurrent_state_reset_for_every_graph"]:
        blockers.append("recurrent_state_reset_for_every_graph")
    author_contract_complete = all(author_contract.values())
    port_preserves_author_ppo = author_contract_complete and not blockers
    return {
        "author_driver": str(author_driver),
        "author_worker": str(author_worker),
        "author_driver_sha256": _sha256_file(author_driver),
        "author_worker_sha256": _sha256_file(author_worker),
        "port_source": str(port_source),
        "training_source": str(training_source),
        "port_source_sha256": _sha256_file(port_source),
        "training_source_sha256": _sha256_file(training_source),
        "author_ppo_contract": author_contract,
        "author_ppo_contract_complete": author_contract_complete,
        "current_port_contract": current_port_contract,
        "current_port_preserves_author_ppo_semantics": port_preserves_author_ppo,
        "current_port_main_table_qualified": False,
        "blocking_reasons": blockers,
        "classification": (
            "author_attention_network_controlled_transfer_prototype_not_yet_a_faithful_marl_ipp_ppo_transfer"
        ),
        "qualification_scope": (
            "static training-semantics audit; it does not establish HM3D performance"
        ),
    }


def _planning_author_formula_audit(cases: int) -> dict[str, object]:
    gvp_source = (
        GVP_ROOT
        / "Exploration"
        / "graph_partition"
        / "src"
        / "graph_partition.cpp"
    )
    port_source = ROOT / "src" / "aerocity_method" / "adapters" / "hm3d_external_baselines.py"
    gvp_text = gvp_source.read_text(encoding="utf-8", errors="replace")
    port_text = port_source.read_text(encoding="utf-8")
    required_author_terms = (
        "nh_private.param(ns + \"/GVD/lambda\", lambda_, 0.2)",
        "nh_private.param(ns + \"/GVD/allowance\", allowance_, 0.1)",
        "nh_private.param(ns + \"/GVD/tau\", tau_, 0.3)",
        "g = (max(g, 0.0) + (f_num * allowance_)) / (work_num + 1)",
        "g = g * exp(-d_p_it->first * lambda_)",
    )
    port_author_formula_terms = (
        "GVP_FRONTIER_ALLOWANCE",
        "delayed_competitors",
        "math.exp",
        "GVP_DISTANCE_DECAY_LAMBDA",
        "GVP_JOB_DELAY_TAU",
    )
    generator = torch.Generator(device="cpu").manual_seed(20260804)
    gvp_nonfinite_count = 0
    distance_monotonicity_failure_count = 0
    worker_monotonicity_failure_count = 0
    for _ in range(cases):
        frontier_count = float(torch.randint(1, 21, (1,), generator=generator).item())
        work_count = int(torch.randint(0, 5, (1,), generator=generator).item())
        distance = float(torch.rand(1, generator=generator).item() * 30.0)
        penalty = float(torch.rand(1, generator=generator).item() * frontier_count)

        def author_gain(
            distance_m: float,
            workers: int,
            frontier_total: float,
            current_penalty: float,
        ) -> float:
            base = max(frontier_total - current_penalty, 0.0) + frontier_total * 0.1
            return base / (workers + 1) * math.exp(-distance_m * 0.2)

        value = author_gain(distance, work_count, frontier_count, penalty)
        gvp_nonfinite_count += int(not math.isfinite(value))
        distance_monotonicity_failure_count += int(
            author_gain(distance + 1.0, work_count, frontier_count, penalty)
            > value + 1.0e-12
        )
        worker_monotonicity_failure_count += int(
            author_gain(distance, work_count + 1, frontier_count, penalty)
            > value + 1.0e-12
        )

    c2_hgrid = (
        C2_ROOT
        / "src"
        / "swarm_exploration"
        / "active_perception"
        / "src"
        / "hgrid.cpp"
    )
    c2_grid = (
        C2_ROOT
        / "src"
        / "swarm_exploration"
        / "active_perception"
        / "src"
        / "uniform_grid.cpp"
    )
    c2_manager = (
        C2_ROOT
        / "src"
        / "swarm_exploration"
        / "exploration_manager"
        / "src"
        / "c2_exploration_manager.cpp"
    )
    hgrid_text = c2_hgrid.read_text(encoding="utf-8", errors="replace")
    grid_text = c2_grid.read_text(encoding="utf-8", errors="replace")
    manager_text = c2_manager.read_text(encoding="utf-8", errors="replace")
    c2_symmetry_failure_count = 0
    c2_monotonicity_failure_count = 0
    for _ in range(cases):
        dz = float(torch.rand(1, generator=generator).item() * 8.0)
        vz = 0.1 + float(torch.rand(1, generator=generator).item() * 4.0)
        positive = abs(dz) / vz
        negative = abs(-dz) / vz
        c2_symmetry_failure_count += int(abs(positive - negative) > 1.0e-12)
        c2_monotonicity_failure_count += int(abs(dz + 0.5) / vz < positive)
    return {
        "gvp_mrep": {
            "author_source_file": str(gvp_source),
            "author_source_sha256": _sha256_file(gvp_source),
            "expected_author_source_sha256": GVP_MREP_GRAPH_PARTITION_SHA256,
            "author_source_commit": GVP_MREP_AUTHOR_COMMIT,
            "port_source_file": str(port_source),
            "port_source_sha256": _sha256_file(port_source),
            "author_formula_source_terms_present": all(
                term in gvp_text for term in required_author_terms
            ),
            "port_author_formula_terms_present": {
                term: term in port_text for term in port_author_formula_terms
            },
            "port_source_identity_matches": (
                _sha256_file(gvp_source) == GVP_MREP_GRAPH_PARTITION_SHA256
            ),
            "formula_probe_case_count": cases,
            "formula_nonfinite_count": gvp_nonfinite_count,
            "distance_monotonicity_failure_count": distance_monotonicity_failure_count,
            "worker_monotonicity_failure_count": worker_monotonicity_failure_count,
            "task_fit": "high_native_target_free_multi_uav_unknown_3d_exploration",
            "current_port_mechanism_qualified": (
                all(term in gvp_text for term in required_author_terms)
                and all(term in port_text for term in port_author_formula_terms)
                and _sha256_file(gvp_source) == GVP_MREP_GRAPH_PARTITION_SHA256
                and gvp_nonfinite_count == 0
                and distance_monotonicity_failure_count == 0
                and worker_monotonicity_failure_count == 0
            ),
            "current_port_main_table_qualified": False,
            "blocking_reason": (
                "the author-sourced controlled transfer still needs paired multi-scene "
                "HM3D/CF2X performance, failure and tuning evidence"
            ),
        },
        "c2_explorer": {
            "hgrid_source_file": str(c2_hgrid),
            "hgrid_source_sha256": _sha256_file(c2_hgrid),
            "uniform_grid_source_sha256": _sha256_file(c2_grid),
            "manager_source_sha256": _sha256_file(c2_manager),
            "three_dimensional_grid_index_present": (
                "Eigen::Vector3i" in grid_text and "posToIndex" in grid_text
            ),
            "vertical_time_cost_present": all(
                term in hgrid_text
                for term in (
                    "w_global_z_change",
                    "global_vz_",
                    "std::abs(path_merged[k + 1].z() - path_merged[k].z()) / vz",
                )
            ),
            "acvrp_assignment_present": "ACVRP" in manager_text,
            "vertical_formula_probe_case_count": cases,
            "vertical_sign_symmetry_failure_count": c2_symmetry_failure_count,
            "vertical_magnitude_monotonicity_failure_count": c2_monotonicity_failure_count,
            "task_fit": "high_native_multi_uav_3d_exploration_but_full_coordination_port_is_large",
            "current_port_main_table_qualified": False,
            "blocking_reason": (
                "no shared-candidate adapter exists for the author ACVRP, hierarchical "
                "grid, and connectivity coordination stack"
            ),
        },
        "qualification_scope": (
            "author-source equation and task-contract probes; not an original ROS "
            "runtime reproduction or HM3D performance result"
        ),
    }


def _repository_qualification() -> dict[str, object]:
    candidates = {
        "recuriosity": {
            "root": RECURIOSITY_ROOT,
            "license": "LICENSE",
            "task_fit": "target_free_exploration_but_single_ground_habitat_agent",
            "runtime_test": (
                "author_linear_memory_stress_tested_but_full_habitat_runtime_unavailable"
            ),
        },
        "falcon": {
            "root": FALCON_ROOT,
            "license": "LICENSE",
            "task_fit": "single_agent_social_pointnav",
            "runtime_test": "excluded_from_target_free_multi_uav_main_table",
        },
        "ovon": {
            "root": OVON_ROOT,
            "license": None,
            "task_fit": "single_ground_robot_objectnav",
            "runtime_test": "excluded_and_no_declared_repository_license",
        },
        "rvn_bench": {
            "root": RVN_ROOT,
            "license": None,
            "task_fit": "single_jackal_sequential_point_goal",
            "runtime_test": "excluded_and_no_declared_repository_license",
        },
        "visfly": {
            "root": VISFLY_ROOT,
            "license": "LICENSE",
            "task_fit": "hm3d_capable_quadrotor_rl_platform_not_same_exploration_method",
            "runtime_test": "platform_precedent_not_algorithm_baseline",
        },
        "ir2": {
            "root": Path(r"E:\github_repos\IR2-Multi-Robot-RL-Exploration-master"),
            "license": "LICENSE",
            "task_fit": "multi_robot_exploration_with_intermittent_communication_but_2d",
            "runtime_test": "communication_ablation_candidate_not_multilevel_main_baseline",
        },
        "marvel": {
            "root": MARVEL_ROOT,
            "license": "LICENSE",
            "task_fit": "multi_robot_exploration_but_author_environment_is_xy_fixed_altitude",
            "runtime_test": (
                "author_network_and_official_checkpoint_stress_tested_but_height_is_absent"
            ),
        },
        "marl_ipp": {
            "root": DEFAULT_MARL_IPP_ROOT,
            "license": "LICENSE",
            "task_fit": (
                "native_3d_multi_robot_dynamic_graph_ipp_but_original_reward_is_target_mapping"
            ),
            "runtime_test": "author_network_and_checkpoint_stress_tested_on_public_candidate_graph",
        },
        "gvp_mrep": {
            "root": Path(
                r"E:\Outcome_Grounded_Repertoire_Literature_2026\official_repositories\GVP-MREP-main"
            ),
            "license": "LICENSE",
            "task_fit": "native_target_free_multi_uav_unknown_3d_exploration",
            "runtime_test": (
                "controlled_transfer_stress_tested_original_ros_runtime_not_reproduced_here"
            ),
        },
    }
    result: dict[str, object] = {}
    for name, raw in candidates.items():
        root = Path(raw["root"])
        license_name = raw["license"]
        license_candidates = () if license_name is None else tuple(root.glob("LICENSE*"))
        result[name] = {
            "checkout_present": root.is_dir(),
            "license_declared": any(path.is_file() for path in license_candidates),
            "task_fit": raw["task_fit"],
            "runtime_test": raw["runtime_test"],
            "source_root": str(root),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--marl-ipp-root", type=Path, default=DEFAULT_MARL_IPP_ROOT)
    parser.add_argument(
        "--marl-ipp-checkpoint", type=Path, default=DEFAULT_MARL_IPP_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _deterministic_evidence(value: object) -> object:
    """Remove wall-clock measurements before hashing scientific audit evidence."""
    if isinstance(value, dict):
        return {
            key: _deterministic_evidence(item)
            for key, item in value.items()
            if key not in {"timing", "elapsed_wall_s"}
            and not key.endswith("_inference_ms")
        }
    if isinstance(value, list):
        return [_deterministic_evidence(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    if args.cases < 100:
        raise ValueError("candidate stress audit requires at least 100 cases")
    if not args.marl_ipp_root.is_dir() or not args.marl_ipp_checkpoint.is_file():
        raise FileNotFoundError("MARL-IPP author source or official checkpoint is missing")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    result: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "synthetic": True,
        "paper_result_eligible": False,
        "claim_limit": (
            "This audit tests four-UAV 3D interface stability and selector separation. "
            "It is not an HM3D/CF2X performance result and cannot enter the paper table."
        ),
        "case_count": args.cases,
        "seed": args.seed,
        "source": {
            "marl_ipp_attention_net_sha256": _sha256_file(
                args.marl_ipp_root / "attention_net.py"
            ),
            "marl_ipp_official_checkpoint_sha256": _sha256_file(
                args.marl_ipp_checkpoint
            ),
        },
        "repository_qualification": _repository_qualification(),
        "repository_source_audit": _repository_source_audit(),
        "marl_ipp_training_fidelity_audit": _marl_ipp_training_fidelity_audit(
            args.marl_ipp_root
        ),
        "author_component_stress": _author_component_stress(args.cases),
        "planning_author_formula_audit": _planning_author_formula_audit(args.cases),
        "learning_stress": _learning_stress(
            args.cases,
            marl_ipp_root=args.marl_ipp_root,
            marl_ipp_checkpoint=args.marl_ipp_checkpoint,
        ),
        "planning_stress": _planning_stress(args.cases),
        "elapsed_wall_s": time.time() - started,
    }
    result["deterministic_evidence_sha256"] = canonical_sha256(
        _deterministic_evidence(result)
    )
    result["audit_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
