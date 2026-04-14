from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Archetype, Profile, register
from .continuous import SEC_PER_DAY, _time_of_day_fraction

def _bursts(n, rng, p_start, mean_len, idle_val, active_val, jitter):
    out = np.full(n, idle_val, dtype=float)
    i = 0
    while i < n:
        # dwell in idle
        gap = int(rng.exponential(1.0 / max(p_start, 1e-9)))
        i += gap
        if i >= n:
            break
        length = max(1, int(rng.exponential(mean_len)))
        j = min(i + length, n)
        out[i:j] = active_val + rng.normal(0, jitter, j-i)
        i = j
    return np.clip(out, 0, None) + rng.normal(0, jitter*0.1, n)

class Fridge(Profile):
    archetype = Archetype.BURSTY
    name = "fridge"
    def sample(self, start, n_ticks, rng):
        # compressor: ~20 min ON every ~60 min
        p_start = 1.0 / (40*60)      # avg 40 min idle
        mean_len = 20*60             # 20 min active
        return _bursts(n_ticks, rng, p_start, mean_len, idle_val=1.5, active_val=90.0, jitter=3.0)

class TV(Profile):
    archetype = Archetype.BURSTY
    name = "tv"
    def sample(self, start, n_ticks, rng):
        tod = _time_of_day_fraction(start, n_ticks)
        # evening-biased start probability
        hour = tod * 24
        weight = np.exp(-((hour-20)**2)/6.0)
        # decide activity second-by-second with prob ~ weight * tiny factor, sessions last long
        base = _bursts(n_ticks, rng, p_start=1.0/(6*3600), mean_len=90*60,
                       idle_val=0.3, active_val=140.0, jitter=5.0)
        # zero out activity during daytime hours to enforce evening bias
        day_mask = (hour > 6) & (hour < 16)
        base[day_mask & (base > 20)] = 0.3
        return base

class Kettle(Profile):
    archetype = Archetype.BURSTY
    name = "kettle"
    def sample(self, start, n_ticks, rng):
        # short bursts (~3 min), a few times a day
        return _bursts(n_ticks, rng, p_start=1.0/(6*3600), mean_len=180,
                       idle_val=0.0, active_val=1500.0, jitter=40.0)

class GenericBursty(Profile):
    archetype = Archetype.BURSTY
    name = "generic_bursty"
    def sample(self, start, n_ticks, rng):
        return _bursts(n_ticks, rng, p_start=1.0/(30*60), mean_len=10*60,
                       idle_val=2.0, active_val=80.0, jitter=4.0)

register(Fridge()); register(TV()); register(Kettle()); register(GenericBursty())
