# src/sensorgen/anomalies/point.py
from __future__ import annotations
import numpy as np
from ..profiles.base import Archetype
from .base import Anomaly, AnomalyContext, register, _slice, _record

ANY_NUMERIC = {Archetype.CONTINUOUS, Archetype.BURSTY}

@register
class Spike(Anomaly):
    name = "spike"
    supports = ANY_NUMERIC
    detector_hint = "pca"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        ctx.signal[s] = ctx.signal[s] + float(params["magnitude"])
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class Dip(Anomaly):
    name = "dip"
    supports = ANY_NUMERIC
    detector_hint = "pca"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        ctx.signal[s] = ctx.signal[s] - float(params["magnitude"])
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class OutOfRange(Anomaly):
    name = "out_of_range"
    supports = ANY_NUMERIC
    detector_hint = "data_quality_gate"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        ctx.signal[s] = float(params["value"])
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class Saturation(Anomaly):
    name = "saturation"
    supports = ANY_NUMERIC
    detector_hint = "data_quality_gate"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        ctx.signal[s] = np.minimum(ctx.signal[s], float(params["max"]))
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)
