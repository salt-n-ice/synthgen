# src/sensorgen/scenario.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re
import yaml
import pandas as pd
from . import profiles as _profiles_pkg  # ensure registration  # noqa: F401
from . import anomalies as _anomalies_pkg  # noqa: F401
from .profiles.base import get_profile, list_profiles
from .anomalies.base import list_anomalies, list_transport_anomalies

_DURATION_RE = re.compile(r"^\s*(\d+)\s*(s|sec|min|h|d)?\s*$")

def _parse_duration(v: Any) -> int:
    if isinstance(v, int): return v
    m = _DURATION_RE.match(str(v))
    if not m:
        raise ValueError(f"invalid duration: {v!r}")
    n = int(m.group(1)); u = (m.group(2) or "s").lower()
    return n * {"s":1,"sec":1,"min":60,"h":3600,"d":86400}[u]

@dataclass
class EmitSpec:
    delta_threshold: float = 1.0
    heartbeat_sec: int = 300

@dataclass
class SensorSpec:
    id: str
    profile: str
    capability: str
    unit: str = ""
    emit: EmitSpec = field(default_factory=EmitSpec)

@dataclass
class AnomalySpec:
    type: str
    sensor: str
    at: pd.Timestamp
    duration_sec: int
    params: dict

@dataclass
class Scenario:
    seed: int
    start: pd.Timestamp
    duration_sec: int
    sensors: list[SensorSpec]
    anomalies: list[AnomalySpec]

def load_scenario(path: Path) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text())
    if "seed" not in raw:
        raise ValueError("scenario missing required field: seed")
    start = pd.Timestamp(raw["start"])
    duration_sec = _parse_duration(raw["duration"])
    valid_profiles = set(list_profiles())
    signal_anomalies = set(list_anomalies())
    transport_anomalies = set(list_transport_anomalies())
    all_anomalies = signal_anomalies | transport_anomalies
    sensors = []
    for s in raw.get("sensors", []):
        if s["profile"] not in valid_profiles:
            raise ValueError(f"unknown profile: {s['profile']!r} (known: {sorted(valid_profiles)})")
        em = s.get("emit", {}) or {}
        sensors.append(SensorSpec(
            id=s["id"], profile=s["profile"], capability=s["capability"], unit=s.get("unit",""),
            emit=EmitSpec(
                delta_threshold=float(em.get("delta_threshold",1.0)),
                heartbeat_sec=_parse_duration(em.get("heartbeat","5min")),
            ),
        ))
    sensor_ids = {s.id for s in sensors}
    anomalies = []
    for a in raw.get("anomalies", []):
        if a["type"] not in all_anomalies:
            raise ValueError(f"unknown anomaly type: {a['type']!r}")
        if a["sensor"] not in sensor_ids:
            raise ValueError(f"unknown sensor id: {a['sensor']!r}")
        anomalies.append(AnomalySpec(
            type=a["type"], sensor=a["sensor"],
            at=pd.Timestamp(a["at"]),
            duration_sec=_parse_duration(a["duration"]),
            params=a.get("params", {}) or {},
        ))
    return Scenario(int(raw["seed"]), start, duration_sec, sensors, anomalies)
