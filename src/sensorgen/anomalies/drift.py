# src/sensorgen/anomalies/drift.py
from __future__ import annotations
import numpy as np
from ..profiles.base import Archetype
from .base import Anomaly, register, _idx, _slice, _record

ALL = {Archetype.CONTINUOUS, Archetype.BURSTY, Archetype.BINARY}
NUMERIC = {Archetype.CONTINUOUS, Archetype.BURSTY}

@register
class CalibrationDrift(Anomaly):
    name = "calibration_drift"
    supports = NUMERIC
    detector_hint = "cusum"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        bias = float(params["bias"])
        ctx.signal[s] = ctx.signal[s] + bias
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class Trend(Anomaly):
    name = "trend"
    supports = NUMERIC
    detector_hint = "cusum"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        slope = float(params["slope_per_sec"])
        t = np.arange(s.stop - s.start, dtype=float)
        ctx.signal[s] = ctx.signal[s] + slope * t
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class LevelShift(Anomaly):
    name = "level_shift"
    supports = NUMERIC
    detector_hint = "cusum_pca"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        off = float(params["offset"])
        ctx.signal[s] = ctx.signal[s] + off
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class DegradationTrajectory(Anomaly):
    name = "degradation_trajectory"
    supports = NUMERIC
    detector_hint = "cusum"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        slope = float(params["slope_per_sec"])
        t = np.arange(s.stop - s.start, dtype=float)
        ctx.signal[s] = ctx.signal[s] + slope * t
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)
