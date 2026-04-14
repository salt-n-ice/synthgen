# tests/test_anomalies_shape.py
import numpy as np
import pandas as pd
from sensorgen.anomalies.base import get_anomaly, AnomalyContext
from sensorgen.profiles.base import Archetype

def _ctx(signal):
    return AnomalyContext("s1","x","",Archetype.CONTINUOUS,
                          pd.Timestamp("2026-03-01T00:00:00Z"), signal)

def test_noise_burst_increases_local_std():
    base = np.full(1200, 10.0)
    ctx = _ctx(base.copy())
    get_anomaly("noise_burst")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:05:00Z"), duration_sec=300,
        params={"sigma": 3.0, "seed": 7})
    assert ctx.signal[300:600].std() > 1.0
    assert ctx.signal[0:300].std() < 0.01

def test_noise_floor_up_multiplies_residual():
    rng = np.random.default_rng(0)
    base = 10.0 + rng.normal(0, 0.1, 1200)
    ctx = _ctx(base.copy())
    get_anomaly("noise_floor_up")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:05:00Z"), duration_sec=300,
        params={"factor": 20.0})
    assert ctx.signal[300:600].std() > 1.0
