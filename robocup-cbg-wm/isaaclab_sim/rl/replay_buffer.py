from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    belief_state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    next_belief_state: torch.Tensor
    dones: torch.Tensor
    rule_risks: torch.Tensor
