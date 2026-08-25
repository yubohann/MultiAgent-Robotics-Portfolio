"""Shared SplitGNN / hybrid utilities."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import yaml

from .metric_utils import (
    DEFAULT_FIXED_PRECISION_TARGET,
    compute_binary_metrics,
    find_best_threshold_metrics,
    gmean_from_confusion,
    threshold_candidates,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def resolve_from_script_dir(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def resolve_graph_path(data_dir: str, dataset: str) -> str:
    candidates = [os.path.join(data_dir, f"{dataset}.dgl")]
    if dataset == "comp":
        candidates.append(os.path.join(data_dir, "FDCompCN", "comp.dgl"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'Cannot find graph file for dataset "{dataset}" in {data_dir}')


def setup_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yelp")
    parser.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--tb_dir", type=str, default="../tb_runs", help="TensorBoard log dir (relative to src/)")
    parser.add_argument("--run_name", type=str, default="", help="Optional run name for TensorBoard")
    args_input = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, "config", args_input.dataset + ".yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    args = argparse.Namespace(**config)

    args.data_path = resolve_from_script_dir(args.data_path)
    args.result_path = resolve_from_script_dir(args.result_path)
    args.tb = bool(args_input.tb)
    args.tb_dir = resolve_from_script_dir(args_input.tb_dir)
    args.run_name = str(args_input.run_name or "")

    print("----------------------------------")
    print("              args")
    print("----------------------------------")
    print(f"dataset:\t{args.dataset}")
    print(f"seed:\t{args.seed}")
    print(f"epoch:\t{args.epoch}")
    print(f"early_stop:\t{args.early_stop}")
    print(f"lr:\t{args.lr}")
    print(f"weigth_decay:{args.weight_decay}")
    print(f"gamma:\t{args.gamma}")
    print(f"C:\t{args.C}")
    print(f"K:\t{args.K}")
    print(f"intra_dim:\t{args.intra_dim}")
    print(f"dropout:\t{args.dropout}")
    print(f"cuda:\t{args.cuda}")
    print(f"tb:\t{args.tb}")
    if args.tb:
        print(f"tb_dir:\t{args.tb_dir}")
        if args.run_name:
            print(f"run_name:\t{args.run_name}")
    print("----------------------------------")
    return args


class EarlyStop:
    """Simple early stopper."""

    def __init__(self, early_stop: int, if_more: bool = True) -> None:
        self.best_eval = 0
        self.best_epoch = 0
        self.if_more = if_more
        self.early_stop = early_stop
        self.stop_steps = 0

    def step(self, current_eval: float, current_epoch: int) -> tuple[bool, bool]:
        do_stop = False
        do_store = False
        if self.if_more:
            if current_eval > self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        else:
            if current_eval < self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        return do_store, do_stop


def conf_gmean(conf: np.ndarray) -> float:
    return gmean_from_confusion(conf)


def prob2pred(prob: np.ndarray, threshhold: float = 0.5, threshold: float | None = None) -> np.ndarray:
    if threshold is None:
        threshold = threshhold
    pred = np.zeros_like(prob, dtype=np.int32)
    pred[prob >= threshold] = 1
    pred[prob < threshold] = 0
    return pred


def _threshold_candidates(probs: np.ndarray) -> np.ndarray:
    return threshold_candidates(probs)


def _compute_binary_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    precision_target: float = DEFAULT_FIXED_PRECISION_TARGET,
) -> dict[str, object]:
    metrics = compute_binary_metrics(
        labels,
        probs,
        threshold,
        precision_target=precision_target,
    )
    return {
        **metrics,
        "preds": prob2pred(np.asarray(probs, dtype=np.float64), threshold=threshold),
        "probs": np.asarray(probs, dtype=np.float64),
    }


def find_best_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    precision_target: float = DEFAULT_FIXED_PRECISION_TARGET,
) -> dict[str, object]:
    best_metrics = find_best_threshold_metrics(
        labels,
        probs,
        precision_target=precision_target,
    )
    return {
        **best_metrics,
        "preds": prob2pred(np.asarray(probs, dtype=np.float64), threshold=best_metrics["threshold"]),
        "probs": np.asarray(probs, dtype=np.float64),
    }


def evaluate(
    labels: np.ndarray,
    logits: torch.Tensor,
    result_path: str = "",
    threshold: float | None = None,
    return_details: bool = False,
    precision_target: float = DEFAULT_FIXED_PRECISION_TARGET,
):
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    if threshold is None:
        metrics = find_best_threshold(labels, probs, precision_target=precision_target)
    else:
        metrics = _compute_binary_metrics(
            labels,
            probs,
            float(threshold),
            precision_target=precision_target,
        )
    if result_path:
        np.save(result_path + "_result_preds", metrics["preds"])
        np.save(result_path + "_result_probs", probs)
    if return_details:
        return {key: value for key, value in metrics.items() if key not in {"preds", "probs"}}
    return metrics["f1_macro"], metrics["auc"], metrics["gmean"], metrics["recall"]


def hinge_loss(labels: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    margin = 1
    ls = labels * scores
    loss = F.relu(margin - ls)
    return loss.mean()


def normalize(mx: np.ndarray) -> np.ndarray:
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)
