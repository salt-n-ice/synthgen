# tests/test_end_to_end.py
import pandas as pd
from pathlib import Path
from sensorgen.generator import run_scenario
from sensorgen.scenario import load_scenario

YAML = """
seed: 7
start: 2026-03-01T00:00:00Z
duration: 1h
sensors:
  - id: outlet
    profile: fridge
    capability: power
    unit: W
    emit: { delta_threshold: 5.0, heartbeat: 5min }
anomalies:
  - type: spike
    sensor: outlet
    at: 2026-03-01T00:30:00Z
    duration: 30s
    params: { magnitude: 500 }
  - type: dropout
    sensor: outlet
    at: 2026-03-01T00:40:00Z
    duration: 5min
    params: {}
"""

def test_run_produces_expected_artifacts(tmp_path: Path):
    p = tmp_path/"s.yaml"; p.write_text(YAML)
    out = tmp_path/"out"
    run_scenario(load_scenario(p), out)
    ev = pd.read_csv(out/"events.csv")
    lb = pd.read_csv(out/"labels.csv")
    assert len(ev) > 0
    assert set(lb["anomaly_type"]) == {"spike","dropout"}
    # dropout window has no events
    lo = pd.Timestamp("2026-03-01T00:40:00Z"); hi = lo + pd.Timedelta(minutes=5)
    ts = pd.to_datetime(ev["timestamp"], utc=True)
    assert not ((ts >= lo) & (ts < hi)).any()

def test_same_seed_byte_identical(tmp_path: Path):
    p = tmp_path/"s.yaml"; p.write_text(YAML)
    a = tmp_path/"a"; b = tmp_path/"b"
    run_scenario(load_scenario(p), a)
    run_scenario(load_scenario(p), b)
    assert (a/"events.csv").read_bytes() == (b/"events.csv").read_bytes()
    assert (a/"labels.csv").read_bytes() == (b/"labels.csv").read_bytes()
