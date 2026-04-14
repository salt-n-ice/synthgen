# src/sensorgen/rng.py
from __future__ import annotations
import hashlib
import numpy as np

class RootRng:
    def __init__(self, seed: int) -> None:
        self._seed = int(seed)

    def child(self, label: str) -> np.random.Generator:
        h = hashlib.blake2b(label.encode(), digest_size=8).digest()
        offset = int.from_bytes(h, "big")
        return np.random.default_rng(self._seed ^ offset)
