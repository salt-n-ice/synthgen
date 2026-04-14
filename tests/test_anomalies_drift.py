# tests/test_anomalies_drift.py
import numpy as np
import pandas as pd
from sensorgen.anomalies.base import get_anomaly, AnomalyContext
from sensorgen.profiles.base import Archetype

def _ctx(n):
    return AnomalyContext("s1","x","",Archetype.CONTINUOUS,
                          pd.Timestamp("2026-03-01T00:00:00Z"),
                          np.zeros(n, dtype=float))

def test_calibration_drift_adds_persistent_bias():
    ctx = _ctx(1200)
    get_anomaly("calibration_drift")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:05:00Z"), duration_sec=300,
        params={"bias": 2.0})
    assert ctx.signal[0] == 0.0
    assert abs(ctx.signal[400] - 2.0) < 1e-9  # bias persists after window
    assert abs(ctx.signal[-1] - 2.0) < 1e-9

def test_trend_is_linear_ramp_within_window():
    ctx = _ctx(1000)
    get_anomaly("trend")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:00:00Z"), duration_sec=1000,
        params={"slope_per_sec": 0.01})
    assert abs(ctx.signal[100] - 1.0) < 1e-6
    assert abs(ctx.signal[500] - 5.0) < 1e-6

def test_level_shift_step_persistent():
    ctx = _ctx(600)
    get_anomaly("level_shift")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:05:00Z"), duration_sec=60,
        params={"offset": 3.0})
    assert ctx.signal[200] == 0.0
    assert ctx.signal[300] == 3.0
    assert ctx.signal[-1] == 3.0

def test_degradation_is_very_slow_ramp():
    ctx = _ctx(100000)
    get_anomaly("degradation_trajectory")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:00:00Z"), duration_sec=100000,
        params={"slope_per_sec": 1e-5})
    assert 0.0 < ctx.signal[-1] < 2.0
