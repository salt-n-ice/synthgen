import numpy as np
import pandas as pd
from sensorgen.profiles.base import Archetype, get_profile

TICKS_PER_DAY = 86400

def test_water_leak_is_mostly_dry():
    prof = get_profile("water_leak")
    assert prof.archetype == Archetype.BINARY
    sig = prof.sample(pd.Timestamp("2026-03-01T00:00:00Z"), TICKS_PER_DAY*7, np.random.default_rng(0))
    assert set(np.unique(sig)).issubset({0, 1})
    assert sig.mean() < 0.001  # essentially always dry

def test_light_switch_has_multiple_transitions_per_day():
    prof = get_profile("light_switch")
    sig = prof.sample(pd.Timestamp("2026-03-01T00:00:00Z"), TICKS_PER_DAY, np.random.default_rng(0))
    transitions = np.sum(np.abs(np.diff(sig.astype(int))))
    assert transitions >= 4

