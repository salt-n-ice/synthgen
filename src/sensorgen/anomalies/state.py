# src/sensorgen/anomalies/state.py
from __future__ import annotations
import numpy as np
from ..profiles.base import Archetype
from .base import Anomaly, register, _idx, _slice, _record

ALL = {Archetype.CONTINUOUS, Archetype.BURSTY, Archetype.BINARY}

@register
class StuckAt(Anomaly):
    name = "stuck_at"
    supports = ALL
    detector_hint = "stuck_at_future"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        if s.start == 0:
            freeze_val = ctx.signal[0]
        else:
            freeze_val = ctx.signal[s.start - 1]
        ctx.signal[s] = freeze_val
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class WaterLeakSustained(Anomaly):
    name = "water_leak_sustained"
    supports = {Archetype.BINARY}
    detector_hint = "state_transition"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        ctx.signal[s] = 1
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

