from __future__ import annotations

from importlib import import_module

from .config import (
    BattleConfig,
    DATA_DIR,
    DEFAULT_THETA,
    FIG_DIR,
    MEDIA_DIR,
    ROOT,
    config_from_args,
    policy_params,
    sigmoid
)
from .sim import (
    LargeScaleBattle50v50
)

_LAZY_EXPORTS = {
    "evaluate": ("train", "evaluate"),
    "load_checkpoint": ("train", "load_checkpoint"),
    "side_fitness": ("train", "side_fitness"),
    "summarize_episodes": ("train", "summarize_episodes"),
    "train": ("train", "train"),
    "_font": ("render", "_font"),
    "render_frame": ("render", "render_frame"),
    "render_video": ("render", "render_video"),
    "world_to_px": ("render", "world_to_px"),
    "make_figures": ("report", "make_figures"),
    "write_report": ("report", "write_report"),
    "build_parser": ("cli", "build_parser"),
    "main": ("cli", "main"),
    "run_all": ("cli", "run_all"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    return getattr(import_module(f".{module_name}", __name__), attribute)

__all__ = [
    'BattleConfig',
    'DATA_DIR',
    'DEFAULT_THETA',
    'FIG_DIR',
    'LargeScaleBattle50v50',
    'MEDIA_DIR',
    'ROOT',
    '_font',
    'build_parser',
    'config_from_args',
    'evaluate',
    'load_checkpoint',
    'main',
    'make_figures',
    'policy_params',
    'render_frame',
    'render_video',
    'run_all',
    'side_fitness',
    'sigmoid',
    'summarize_episodes',
    'train',
    'world_to_px',
    'write_report'
]
