# src/sensorgen/anomalies/shape.py
from __future__ import annotations
import numpy as np
from ..profiles.base import Archetype
from .base import Anomaly, register, _slice, _record

ANY_NUMERIC = {Archetype.CONTINUOUS, Archetype.BURSTY}

@register
class NoiseBurst(Anomaly):
    name = "noise_burst"
    supports = ANY_NUMERIC
    detector_hint = "batch"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        rng = np.random.default_rng(int(params.get("seed", 0)))
        ctx.signal[s] = ctx.signal[s] + rng.normal(0, float(params["sigma"]), s.stop - s.start)
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class NoiseFloorUp(Anomaly):
    name = "noise_floor_up"
    supports = ANY_NUMERIC
    detector_hint = "batch"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        seg = ctx.signal[s]
        mean = seg.mean()
        ctx.signal[s] = mean + (seg - mean) * float(params["factor"])
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)
