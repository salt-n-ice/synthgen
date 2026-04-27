# tests/test_anomalies_state.py
import numpy as np
import pandas as pd
from sensorgen.anomalies.base import get_anomaly, AnomalyContext
from sensorgen.profiles.base import Archetype

def test_stuck_at_freezes_last_value():
    sig = np.arange(1000, dtype=float)
    ctx = AnomalyContext("s1","x","",Archetype.CONTINUOUS,
                         pd.Timestamp("2026-03-01T00:00:00Z"), sig.copy())
    get_anomaly("stuck_at")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:05:00Z"), duration_sec=100, params={})
    assert (ctx.signal[300:400] == ctx.signal[299]).all()

def test_water_leak_sustained_sets_wet_for_duration():
    sig = np.zeros(3600, dtype=np.uint8)
    ctx = AnomalyContext("s1","water","",Archetype.BINARY,
                         pd.Timestamp("2026-03-01T00:00:00Z"), sig.copy())
    rec = get_anomaly("water_leak_sustained")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:10:00Z"), duration_sec=600, params={})
    assert (ctx.signal[600:1200] == 1).all()
    assert ctx.signal[0] == 0 and ctx.signal[-1] == 0
    assert rec.anomaly_type == "water_leak_sustained"

