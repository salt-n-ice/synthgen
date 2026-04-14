# tests/test_anomalies_transport.py
import pandas as pd
from sensorgen.anomalies.base import get_transport_anomaly, TransportContext
from sensorgen.emitter import Event

def _events(n, step=60, v=10.0):
    base = pd.Timestamp("2026-03-01T00:00:00Z")
    return [Event(base + pd.Timedelta(seconds=i*step), "s1","x", v, "") for i in range(n)]

def _ctx(evts):
    return TransportContext(
        sensor_id="s1", capability="x",
        start=pd.Timestamp("2026-03-01T00:00:00Z"),
        end=pd.Timestamp("2026-03-02T00:00:00Z"),
        events=evts,
    )

def test_dropout_suppresses_events_in_window():
    ctx = _ctx(_events(60))
    rec = get_transport_anomaly("dropout")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:10:00Z"), duration_sec=600, params={})
    ts = [e.timestamp for e in ctx.events]
    window_start = pd.Timestamp("2026-03-01T00:10:00Z")
    window_end = window_start + pd.Timedelta(seconds=600)
    assert not any(window_start <= t < window_end for t in ts)
    assert rec.anomaly_type == "dropout"

def test_duplicate_stale_reissues_last_event():
    ctx = _ctx(_events(10))
    get_transport_anomaly("duplicate_stale")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:05:00Z"), duration_sec=60, params={})
    # should contain at least one duplicate timestamp or value
    vals = [(e.timestamp, e.value) for e in ctx.events]
    assert len(vals) > len(set(vals))

def test_reporting_rate_change_subsamples():
    ctx = _ctx(_events(60))
    get_transport_anomaly("reporting_rate_change")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:00:00Z"), duration_sec=3600,
        params={"keep_ratio": 0.25})
    assert 10 < len(ctx.events) < 40

def test_clock_drift_shifts_timestamps_linearly():
    ctx = _ctx(_events(100))
    get_transport_anomaly("clock_drift")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:00:00Z"), duration_sec=6000,
        params={"drift_sec_per_sec": 0.01})
    # last event's shift should be ~60s (0.01 * 6000)
    expected_last = pd.Timestamp("2026-03-01T00:00:00Z") + pd.Timedelta(seconds=99*60 + 99*60*0.01)
    assert abs((ctx.events[-1].timestamp - expected_last).total_seconds()) < 1.0

def test_batch_arrival_collapses_events_into_one_burst():
    ctx = _ctx(_events(60))
    get_transport_anomaly("batch_arrival")().apply(ctx,
        at=pd.Timestamp("2026-03-01T00:00:00Z"), duration_sec=3600,
        params={"release_at": "2026-03-01T00:59:00Z"})
    release = pd.Timestamp("2026-03-01T00:59:00Z")
    # lots of events stacked near release time
    near = sum(1 for e in ctx.events if abs((e.timestamp - release).total_seconds()) < 5)
    assert near >= 30
