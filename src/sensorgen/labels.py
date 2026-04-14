# src/sensorgen/labels.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import pandas as pd

@dataclass
class LabelRecord:
    sensor_id: str
    capability: str
    start: pd.Timestamp
    end: pd.Timestamp
    anomaly_type: str
    detector_hint: str = ""
    params: dict[str, Any] = field(default_factory=dict)

def write_labels(records: list[LabelRecord], path: Path) -> None:
    rows = [{
        "sensor_id": r.sensor_id,
        "capability": r.capability,
        "start": r.start.isoformat(),
        "end": r.end.isoformat(),
        "anomaly_type": r.anomaly_type,
        "detector_hint": r.detector_hint,
        "params_json": json.dumps(r.params, default=str),
    } for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=[
        "sensor_id","capability","start","end","anomaly_type","detector_hint","params_json",
    ]).to_csv(path, index=False)
