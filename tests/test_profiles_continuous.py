import numpy as np
import pandas as pd
from sensorgen.profiles.base import Archetype, get_profile

TICKS_PER_DAY = 86400

def test_room_temperature_has_diurnal_amplitude():
    prof = get_profile("room_temperature")
    assert prof.archetype == Archetype.CONTINUOUS
    sig = prof.sample(pd.Timestamp("2026-03-01T00:00:00Z"), TICKS_PER_DAY * 3, np.random.default_rng(0))
    assert sig.shape == (TICKS_PER_DAY * 3,)
    daily_peaks = [sig[i*TICKS_PER_DAY:(i+1)*TICKS_PER_DAY].max() for i in range(3)]
    daily_troughs = [sig[i*TICKS_PER_DAY:(i+1)*TICKS_PER_DAY].min() for i in range(3)]
    amplitude = np.mean(np.array(daily_peaks) - np.array(daily_troughs))
    assert amplitude > 1.0  # at least 1 °C diurnal swing

def test_battery_drain_is_monotonic_decreasing_on_average():
    prof = get_profile("battery_drain")
    sig = prof.sample(pd.Timestamp("2026-03-01T00:00:00Z"), TICKS_PER_DAY * 30, np.random.default_rng(0))
    assert sig[0] > sig[-1]
    assert 0 <= sig.min() <= 100
    assert sig.max() <= 100

def test_voltage_mains_is_near_constant():
    prof = get_profile("voltage_mains")
    sig = prof.sample(pd.Timestamp("2026-03-01T00:00:00Z"), TICKS_PER_DAY, np.random.default_rng(0))
    assert 115 < sig.mean() < 125
    assert sig.std() < 2.0

def test_same_seed_byte_identical():
    prof = get_profile("room_temperature")
    a = prof.sample(pd.Timestamp("2026-03-01T00:00:00Z"), 3600, np.random.default_rng(123))
    b = prof.sample(pd.Timestamp("2026-03-01T00:00:00Z"), 3600, np.random.default_rng(123))
    assert np.array_equal(a, b)
