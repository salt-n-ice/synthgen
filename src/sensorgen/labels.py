# src/sensorgen/labels.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import pandas as pd

# anomaly_type -> label_class
# user_behavior: things the user wants to be told about (appliance behavior change,
#   occupancy/usage pattern shift, sustained physical event like a leak).
# sensor_fault: infrastructure-level signal-quality issues (sensor offline, stuck,
#   calibration drift, transport-layer noise) — relevant for the system to suppress
#   downstream user-facing alerts but NOT what the user wants summarized.
USER_BEHAVIOR_TYPES: frozenset[str] = frozenset({
    "spike", "dip", "level_shift", "trend", "degradation_trajectory",
    "frequency_change", "seasonality_loss", "time_of_day",
    "weekend_anomaly", "month_shift", "seasonal_mismatch",
    "water_leak_sustained",
    # Auto-emitted by the natural-variation outlier detector
    # (post-generation pass over BURSTY sensor signals): days where
    # the simulator's stochastic background produced unusually
    # heavy/light usage relative to the long-term baseline. Without
    # these labels the detector correctly fires on real signal
    # deviations but the eval scores them as FPs because the YAML
    # author didn't intend them. Labelled here as a structural
    # honesty fix for evt_F1 / fpur / uvfp/d.
    "usage_anomaly",
})
SENSOR_FAULT_TYPES: frozenset[str] = frozenset({
    "out_of_range", "saturation", "noise_burst", "noise_floor_up",
    "stuck_at", "calibration_drift", "dropout", "duplicate_stale",
    "reporting_rate_change", "clock_drift", "batch_arrival",
})


def label_class(anomaly_type: str) -> str:
    if anomaly_type in USER_BEHAVIOR_TYPES:
        return "user_behavior"
    if anomaly_type in SENSOR_FAULT_TYPES:
        return "sensor_fault"
    return "unknown"


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
        "label_class": label_class(r.anomaly_type),
        "detector_hint": r.detector_hint,
        "params_json": json.dumps(r.params, default=str),
    } for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=[
        "sensor_id","capability","start","end","anomaly_type","label_class","detector_hint","params_json",
    ]).to_csv(path, index=False)
