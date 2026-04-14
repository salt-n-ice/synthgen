# tests/test_anomalies_pattern.py
import numpy as np
import pandas as pd
from sensorgen.anomalies.base import get_anomaly, AnomalyContext
from sensorgen.profiles.base import Archetype

def _ctx(sig):
    return AnomalyContext("s1","x","",Archetype.CONTINUOUS,
                          pd.Timestamp("2026-03-01T00:00:00Z"), sig)

def test_frequency_change_adds_oscillation():
    sig = np.full(10_000, 0.0)
    ctx = _ctx(sig.copy())
    get_anomaly("frequency_change")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:00:00Z"), duration_sec=10_000,
        params={"freq_hz": 0.01, "amp": 1.0})
    # FFT magnitude at 0.01 Hz should dominate
    spec = np.abs(np.fft.rfft(ctx.signal))
    freqs = np.fft.rfftfreq(len(ctx.signal), d=1.0)
    peak = freqs[np.argmax(spec)]
    assert 0.005 < peak < 0.02

def test_seasonality_loss_flattens_segment():
    t = np.arange(3600)
    sig = 10.0 + 3.0*np.sin(2*np.pi*t/600)
    ctx = _ctx(sig.copy())
    get_anomaly("seasonality_loss")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:00:00Z"), duration_sec=3600,
        params={})
    assert ctx.signal.std() < 0.01

def test_time_of_day_adds_during_hour_window():
    n = 86400 * 3
    ctx = _ctx(np.zeros(n))
    get_anomaly("time_of_day")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:00:00Z"), duration_sec=n,
        params={"hour_start": 3, "hour_end": 4, "magnitude": 5.0})
    # only seconds falling in hour [3,4) get +5
    hits = (ctx.signal == 5.0).sum()
    assert hits == 3 * 3600

def test_weekend_anomaly_only_affects_weekend():
    # start Sunday 2026-03-01 00:00; 7 days
    n = 86400*7
    ctx = AnomalyContext("s1","x","",Archetype.CONTINUOUS,
                         pd.Timestamp("2026-03-01T00:00:00Z"), np.zeros(n))
    get_anomaly("weekend_anomaly")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:00:00Z"), duration_sec=n,
        params={"magnitude": 2.0, "target": "weekend"})
    # March 1 2026 is Sunday (weekend), March 2–6 weekdays, March 7 Saturday
    assert ctx.signal[0] == 2.0         # Sun
    assert ctx.signal[86400] == 0.0     # Mon
    assert ctx.signal[86400*6] == 2.0   # Sat

def test_month_shift_applies_within_month():
    # start Feb 15 2026 for 45 days; month_shift on Feb should hit days 0..13
    n = 86400*45
    start = pd.Timestamp("2026-02-15T00:00:00Z")
    ctx = AnomalyContext("s1","x","",Archetype.CONTINUOUS, start, np.zeros(n))
    get_anomaly("month_shift")().apply(ctx,
        at=start, duration_sec=n,
        params={"month": 2, "offset": 1.5})
    assert ctx.signal[0] == 1.5
    # March 1 onwards = offset 0
    mar1 = int((pd.Timestamp("2026-03-01T00:00:00Z") - start).total_seconds())
    assert ctx.signal[mar1] == 0.0

def test_seasonal_mismatch_inverts_detrended_component():
    t = np.arange(3600)
    sig = 10.0 + np.sin(2*np.pi*t/600)
    ctx = _ctx(sig.copy())
    get_anomaly("seasonal_mismatch")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:00:00Z"), duration_sec=3600,
        params={})
    # detrended component flipped → correlation between original and modified is negative
    corr = np.corrcoef(sig - sig.mean(), ctx.signal - ctx.signal.mean())[0,1]
    assert corr < -0.9
