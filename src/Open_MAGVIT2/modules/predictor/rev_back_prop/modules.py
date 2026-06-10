"""Base classes for memory-efficient reversible backward."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def seed_generator() -> int:
    if (
        hasattr(torch.cuda, "default_generators")
        and len(torch.cuda.default_generators) > 0
    ):
        device_idx = torch.cuda.current_device()
        return torch.cuda.default_generators[device_idx].seed()
    return int(torch.seed() % sys.maxsize)


def employ_seed(custom_backward: bool, seeds: dict, key: str, safe: bool) -> None:
    if not custom_backward:
        return
    if key in seeds:
        seed = seeds[key] if safe else seeds.pop(key)
        torch.manual_seed(seed)
    else:
        seeds[key] = seed_generator()
        torch.manual_seed(seeds[key])


class ReversibleModule(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        self.seeds: dict = {}
        self.custom_backward = True
        super().__init__(*args, **kwargs)

    @abstractmethod
    def backward_pass(self, y: torch.Tensor, dy: torch.Tensor):
        pass

    def seed_cuda(self, key: str, safe: bool = False) -> None:
        employ_seed(self.custom_backward, self.seeds, key, safe)


class NotReversibleModule(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        self.seeds: dict = {}
        self.custom_backward = True
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward_for_backward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def seed_cuda(self, key: str, safe: bool = False) -> None:
        employ_seed(self.custom_backward, self.seeds, key, safe)
