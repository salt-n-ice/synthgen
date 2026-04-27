from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Archetype, Profile, register
from .continuous import _time_of_day_fraction

def _alternating(n, rng, p_on_per_sec, mean_on_dur):
    out = np.zeros(n, dtype=np.uint8)
    i = 0
    while i < n:
        gap = int(rng.exponential(1.0 / max(p_on_per_sec, 1e-12)))
        i += gap
        if i >= n:
            break
        dur = max(1, int(rng.exponential(mean_on_dur)))
        j = min(i + dur, n)
        out[i:j] = 1
        i = j
    return out

class WaterLeak(Profile):
    archetype = Archetype.BINARY
    name = "water_leak"
    def sample(self, start, n_ticks, rng):
        return np.zeros(n_ticks, dtype=np.uint8)

class LightSwitch(Profile):
    archetype = Archetype.BINARY
    name = "light_switch"
    def sample(self, start, n_ticks, rng):
        tod = _time_of_day_fraction(start, n_ticks) * 24
        base = _alternating(n_ticks, rng, p_on_per_sec=1.0/(3*3600), mean_on_dur=45*60)
        # force off during middle of night
        base[(tod > 1) & (tod < 6)] = 0
        return base

register(WaterLeak()); register(LightSwitch())
