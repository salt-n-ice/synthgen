# src/sensorgen/anomalies/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from ..labels import LabelRecord
from ..profiles.base import Archetype

@dataclass
class AnomalyContext:
    sensor_id: str
    capability: str
    unit: str
    archetype: Archetype
    start: pd.Timestamp
    signal: np.ndarray

_REG: dict[str, type["Anomaly"]] = {}

class Anomaly(ABC):
    name: str
    supports: set[Archetype]
    detector_hint: str = ""

    @abstractmethod
    def apply(self, ctx: AnomalyContext, *, at: pd.Timestamp, duration_sec: int, params: dict[str, Any]) -> LabelRecord: ...

def register(cls: type[Anomaly]) -> type[Anomaly]:
    _REG[cls.name] = cls
    return cls

def get_anomaly(name: str) -> type[Anomaly]:
    if name not in _REG:
        raise KeyError(f"unknown anomaly: {name}")
    return _REG[name]

def list_anomalies() -> list[str]:
    return sorted(_REG)

def _idx(ctx: AnomalyContext, at: pd.Timestamp) -> int:
    return int((at - ctx.start).total_seconds())

def _slice(ctx: AnomalyContext, at: pd.Timestamp, duration_sec: int) -> slice:
    i0 = max(0, _idx(ctx, at))
    i1 = min(len(ctx.signal), i0 + duration_sec)
    return slice(i0, i1)

def _record(ctx: AnomalyContext, at: pd.Timestamp, duration_sec: int, atype: str, hint: str, params: dict) -> LabelRecord:
    end = at + pd.Timedelta(seconds=duration_sec)
    return LabelRecord(ctx.sensor_id, ctx.capability, at, end, atype, hint, dict(params))
