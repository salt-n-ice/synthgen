# src/sensorgen/match.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Interval:
    sensor_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    anomaly_type: str = ""

def _overlaps(a: Interval, b: Interval) -> bool:
    return a.sensor_id == b.sensor_id and a.start < b.end and b.start < a.end

def match_intervals(ground_truth: list[Interval], detections: list[Interval]) -> tuple[list[Interval], list[Interval], list[Interval]]:
    tp, fp, fn = [], [], []
    matched_det = set()
    for g in ground_truth:
        hit = False
        for i, d in enumerate(detections):
            if i in matched_det: continue
            if _overlaps(g, d):
                tp.append(g); matched_det.add(i); hit = True; break
        if not hit:
            fn.append(g)
    for i, d in enumerate(detections):
        if i not in matched_det:
            fp.append(d)
    return tp, fp, fn

def load_intervals_from_csv(path) -> list[Interval]:
    df = pd.read_csv(path)
    return [Interval(r.sensor_id, pd.Timestamp(r.start), pd.Timestamp(r.end), r.anomaly_type) for r in df.itertuples()]
