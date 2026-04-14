# src/sensorgen/anomalies/pattern.py
from __future__ import annotations
import numpy as np
import pandas as pd
from ..profiles.base import Archetype
from .base import Anomaly, register, _slice, _record

ALL = {Archetype.CONTINUOUS, Archetype.BURSTY, Archetype.BINARY}
NUMERIC = {Archetype.CONTINUOUS, Archetype.BURSTY}

@register
class FrequencyChange(Anomaly):
    name = "frequency_change"
    supports = NUMERIC
    detector_hint = "batch"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        n = s.stop - s.start
        t = np.arange(n, dtype=float)
        ctx.signal[s] = ctx.signal[s] + float(params["amp"]) * np.sin(2*np.pi*float(params["freq_hz"])*t)
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class SeasonalityLoss(Anomaly):
    name = "seasonality_loss"
    supports = NUMERIC
    detector_hint = "batch"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        ctx.signal[s] = ctx.signal[s].mean()
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

def _timestamps(ctx, s):
    i = np.arange(s.start, s.stop)
    return ctx.start + pd.to_timedelta(i, unit="s")

@register
class TimeOfDay(Anomaly):
    name = "time_of_day"
    supports = ALL
    detector_hint = "temporal_profiles"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        ts = _timestamps(ctx, s)
        mask = (ts.hour >= int(params["hour_start"])) & (ts.hour < int(params["hour_end"]))
        ctx.signal[s] = ctx.signal[s] + np.where(mask, float(params["magnitude"]), 0.0)
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class WeekendAnomaly(Anomaly):
    name = "weekend_anomaly"
    supports = ALL
    detector_hint = "temporal_profiles"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        ts = _timestamps(ctx, s)
        is_weekend = ts.dayofweek >= 5
        target_weekend = params.get("target","weekend") == "weekend"
        mask = is_weekend if target_weekend else ~is_weekend
        ctx.signal[s] = ctx.signal[s] + np.where(mask, float(params["magnitude"]), 0.0)
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class MonthShift(Anomaly):
    name = "month_shift"
    supports = ALL
    detector_hint = "temporal_profiles"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        ts = _timestamps(ctx, s)
        mask = ts.month == int(params["month"])
        ctx.signal[s] = ctx.signal[s] + np.where(mask, float(params["offset"]), 0.0)
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register
class SeasonalMismatch(Anomaly):
    name = "seasonal_mismatch"
    supports = NUMERIC
    detector_hint = "temporal_profiles"
    def apply(self, ctx, *, at, duration_sec, params):
        s = _slice(ctx, at, duration_sec)
        seg = ctx.signal[s]
        ctx.signal[s] = 2*seg.mean() - seg
        return _record(ctx, at, duration_sec, self.name, self.detector_hint, params)
