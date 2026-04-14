from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd

class Archetype(str, Enum):
    CONTINUOUS = "continuous"
    BURSTY = "bursty"
    BINARY = "binary"

_REGISTRY: dict[str, "Profile"] = {}

class Profile(ABC):
    archetype: Archetype
    name: str

    @abstractmethod
    def sample(self, start: pd.Timestamp, n_ticks: int, rng: np.random.Generator) -> np.ndarray: ...

def register(profile: "Profile") -> "Profile":
    _REGISTRY[profile.name] = profile
    return profile

def get_profile(name: str) -> "Profile":
    if name not in _REGISTRY:
        raise KeyError(f"unknown profile: {name}")
    return _REGISTRY[name]

def list_profiles() -> list[str]:
    return sorted(_REGISTRY)
