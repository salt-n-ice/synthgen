import numpy as np
import pandas as pd
from sensorgen.profiles.base import Archetype, get_profile

def _k2_means(x, iters=20):
    lo, hi = np.quantile(x, 0.1), np.quantile(x, 0.9)
    c = np.array([lo, hi], dtype=float)
    for _ in range(iters):
        d = np.abs(x[:, None] - c[None, :])
        a = d.argmin(axis=1)
        for k in (0, 1):
            if (a == k).any():
                c[k] = x[a == k].mean()
    return c, a

TICKS_PER_DAY = 86400

def test_fridge_has_two_modes_and_idle_mode_near_zero():
    prof = get_profile("fridge")
    assert prof.archetype == Archetype.BURSTY
    sig = prof.sample(pd.Timestamp("2026-03-01T00:00:00Z"), TICKS_PER_DAY*2, np.random.default_rng(0))
    centers, _ = _k2_means(sig)
    assert centers.min() < 5.0
    assert centers.max() > 60.0
    assert centers.max() - centers.min() > 50.0

def test_kettle_is_mostly_idle():
    prof = get_profile("kettle")
    sig = prof.sample(pd.Timestamp("2026-03-01T00:00:00Z"), TICKS_PER_DAY, np.random.default_rng(0))
    assert (sig < 5).mean() > 0.95

def test_tv_evening_biased():
    prof = get_profile("tv")
    sig = prof.sample(pd.Timestamp("2026-03-01T00:00:00Z"), TICKS_PER_DAY, np.random.default_rng(0))
    evening = sig[18*3600:24*3600]
    morning = sig[6*3600:12*3600]
    assert (evening > 20).sum() > (morning > 20).sum()
