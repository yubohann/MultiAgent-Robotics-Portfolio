from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch


def clone_state_dict_to_cpu(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    cloned: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[str(key)] = value.detach().cpu().clone()
        else:
            cloned[str(key)] = copy.deepcopy(value)
    return cloned


def snapshot_model_state_to_cpu(model: torch.nn.Module) -> OrderedDict[str, Any]:
    return clone_state_dict_to_cpu(model.state_dict())
