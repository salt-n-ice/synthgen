# tests/test_anomalies_point.py
import numpy as np
import pandas as pd
from sensorgen.anomalies.base import get_anomaly, AnomalyContext
from sensorgen.profiles.base import Archetype

def _ctx(signal):
    return AnomalyContext(
        sensor_id="s1", capability="x", unit="",
        archetype=Archetype.CONTINUOUS,
        start=pd.Timestamp("2026-03-01T00:00:00Z"),
        signal=signal,
    )

def test_spike_applies_magnitude_and_labels_interval():
    ctx = _ctx(np.full(600, 10.0))
    cls = get_anomaly("spike")
    at = pd.Timestamp("2026-03-01T00:05:00Z")  # t=300s
    rec = cls().apply(ctx, at=at, duration_sec=10, params={"magnitude": 50})
    assert ctx.signal[300:310].mean() > 55
    assert ctx.signal[0:300].mean() == 10.0
    assert ctx.signal[310:].mean() == 10.0
    assert rec.anomaly_type == "spike"
    assert rec.start == at

def test_dip_subtracts_magnitude():
    ctx = _ctx(np.full(600, 10.0))
    cls = get_anomaly("dip")
    at = pd.Timestamp("2026-03-01T00:05:00Z")
    cls().apply(ctx, at=at, duration_sec=10, params={"magnitude": 5})
    assert ctx.signal[300] == 5.0

def test_out_of_range_overwrites_with_value():
    ctx = _ctx(np.full(600, 10.0))
    cls = get_anomaly("out_of_range")
    at = pd.Timestamp("2026-03-01T00:05:00Z")
    cls().apply(ctx, at=at, duration_sec=5, params={"value": 9999.0})
    assert (ctx.signal[300:305] == 9999.0).all()

def test_saturation_clamps_to_max():
    ctx = _ctx(np.linspace(0, 100, 600))
    cls = get_anomaly("saturation")
    at = pd.Timestamp("2026-03-01T00:05:00Z")
    cls().apply(ctx, at=at, duration_sec=50, params={"max": 25.0})
    assert ctx.signal[300:350].max() <= 25.0
