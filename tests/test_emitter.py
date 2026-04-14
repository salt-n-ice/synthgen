import numpy as np
import pandas as pd
from sensorgen.emitter import EmitConfig, emit_events
from sensorgen.profiles.base import Archetype

def _cfg(arch, **kw):
    defaults = dict(
        sensor_id="s1", capability="x", unit="", archetype=arch,
        delta_threshold=1.0, heartbeat_sec=300,
    )
    defaults.update(kw)
    return EmitConfig(**defaults)

def test_numeric_emits_on_delta_and_heartbeat():
    start = pd.Timestamp("2026-03-01T00:00:00Z")
    # constant signal for 600s; heartbeat every 300s → 3 events (t=0, 300, 600)
    sig = np.full(601, 10.0)
    evts = emit_events(sig, start, _cfg(Archetype.CONTINUOUS, delta_threshold=0.5, heartbeat_sec=300))
    assert len(evts) == 3
    assert evts[0].value == 10.0 and evts[-1].value == 10.0

def test_numeric_suppresses_below_delta():
    start = pd.Timestamp("2026-03-01T00:00:00Z")
    sig = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 11.6])  # only last crosses
    evts = emit_events(sig, start, _cfg(Archetype.CONTINUOUS, delta_threshold=1.0, heartbeat_sec=3600))
    # first event emitted (initial), then the 11.6 crossing
    assert len(evts) == 2
    assert evts[-1].value == 11.6

def test_binary_emits_transitions_only_plus_heartbeat():
    start = pd.Timestamp("2026-03-01T00:00:00Z")
    sig = np.array([0,0,0,1,1,1,0,0], dtype=np.uint8)
    evts = emit_events(sig, start, _cfg(Archetype.BINARY, heartbeat_sec=10_000))
    vals = [e.value for e in evts]
    # initial (0), transition to 1 (index 3), transition to 0 (index 6)
    assert vals == [0, 1, 0]
