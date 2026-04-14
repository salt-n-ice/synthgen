from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from .profiles.base import Archetype

@dataclass
class EmitConfig:
    sensor_id: str
    capability: str
    unit: str
    archetype: Archetype
    delta_threshold: float = 1.0
    heartbeat_sec: int = 300

@dataclass
class Event:
    timestamp: pd.Timestamp
    sensor_id: str
    capability: str
    value: float
    unit: str

def emit_events(signal: np.ndarray, start: pd.Timestamp, cfg: EmitConfig) -> list[Event]:
    out: list[Event] = []
    if len(signal) == 0:
        return out
    last_emit_idx = 0
    last_val = float(signal[0])
    out.append(Event(start, cfg.sensor_id, cfg.capability, last_val, cfg.unit))
    for i in range(1, len(signal)):
        v = float(signal[i])
        crossed = False
        if cfg.archetype == Archetype.BINARY:
            if v != last_val:
                crossed = True
        else:
            if abs(v - last_val) >= cfg.delta_threshold:
                crossed = True
        if crossed or (i - last_emit_idx) >= cfg.heartbeat_sec:
            out.append(Event(start + pd.Timedelta(seconds=i), cfg.sensor_id, cfg.capability, v, cfg.unit))
            last_emit_idx = i
            last_val = v
    return out

def events_to_dataframe(events: Sequence[Event]) -> pd.DataFrame:
    return pd.DataFrame([{
        "timestamp": e.timestamp.isoformat(),
        "sensor_id": e.sensor_id,
        "capability": e.capability,
        "value": e.value,
        "unit": e.unit,
    } for e in events])
