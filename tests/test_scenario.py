# tests/test_scenario.py
import pytest
from pathlib import Path
from sensorgen.scenario import load_scenario

YAML_OK = """
seed: 42
start: 2026-02-01T00:00:00Z
duration: 1d
sensors:
  - id: outlet
    profile: fridge
    capability: power
    unit: W
    emit: { delta_threshold: 2.0, heartbeat: 5min }
anomalies:
  - type: spike
    sensor: outlet
    at: 2026-02-01T12:00:00Z
    duration: 2min
    params: { magnitude: 300 }
"""

def test_load_valid(tmp_path: Path):
    p = tmp_path/"s.yaml"; p.write_text(YAML_OK)
    sc = load_scenario(p)
    assert sc.seed == 42
    assert sc.duration_sec == 86400
    assert sc.sensors[0].profile == "fridge"
    assert sc.sensors[0].emit.heartbeat_sec == 300
    assert sc.anomalies[0].type == "spike"
    assert sc.anomalies[0].duration_sec == 120

def test_unknown_profile_rejected(tmp_path: Path):
    p = tmp_path/"s.yaml"
    p.write_text(YAML_OK.replace("fridge","not_a_profile"))
    with pytest.raises(ValueError, match="profile"):
        load_scenario(p)

def test_unknown_anomaly_rejected(tmp_path: Path):
    p = tmp_path/"s.yaml"
    p.write_text(YAML_OK.replace("spike","not_a_type"))
    with pytest.raises(ValueError, match="anomaly"):
        load_scenario(p)

def test_missing_seed_rejected(tmp_path: Path):
    p = tmp_path/"s.yaml"
    p.write_text(YAML_OK.replace("seed: 42\n",""))
    with pytest.raises(ValueError, match="seed"):
        load_scenario(p)
