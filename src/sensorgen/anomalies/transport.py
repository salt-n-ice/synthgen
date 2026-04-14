# src/sensorgen/anomalies/transport.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import TransportAnomaly, register_transport, _tr_record

@register_transport
class Dropout(TransportAnomaly):
    name = "dropout"
    detector_hint = "data_quality_gate"
    def apply(self, ctx, *, at, duration_sec, params):
        end = at + pd.Timedelta(seconds=duration_sec)
        ctx.events[:] = [e for e in ctx.events if not (at <= e.timestamp < end)]
        return _tr_record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register_transport
class DuplicateStale(TransportAnomaly):
    name = "duplicate_stale"
    detector_hint = "data_quality_gate"
    def apply(self, ctx, *, at, duration_sec, params):
        end = at + pd.Timedelta(seconds=duration_sec)
        last = None
        extras = []
        for e in ctx.events:
            if e.timestamp < at: last = e
            elif at <= e.timestamp < end and last is not None:
                extras.append(type(e)(e.timestamp, e.sensor_id, e.capability, last.value, e.unit))
        ctx.events.extend(extras)
        ctx.events.sort(key=lambda e: e.timestamp)
        return _tr_record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register_transport
class ReportingRateChange(TransportAnomaly):
    name = "reporting_rate_change"
    detector_hint = "data_quality_gate"
    def apply(self, ctx, *, at, duration_sec, params):
        end = at + pd.Timedelta(seconds=duration_sec)
        keep = float(params["keep_ratio"])
        rng = np.random.default_rng(int(params.get("seed", 0)))
        kept = []
        for e in ctx.events:
            if at <= e.timestamp < end:
                if rng.random() < keep:
                    kept.append(e)
            else:
                kept.append(e)
        ctx.events[:] = kept
        return _tr_record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register_transport
class ClockDrift(TransportAnomaly):
    name = "clock_drift"
    detector_hint = "data_quality_gate"
    def apply(self, ctx, *, at, duration_sec, params):
        end = at + pd.Timedelta(seconds=duration_sec)
        rate = float(params["drift_sec_per_sec"])
        new_events = []
        for e in ctx.events:
            if at <= e.timestamp < end:
                elapsed = (e.timestamp - at).total_seconds()
                shift = pd.Timedelta(seconds=elapsed * rate)
                new_events.append(type(e)(e.timestamp + shift, e.sensor_id, e.capability, e.value, e.unit))
            else:
                new_events.append(e)
        new_events.sort(key=lambda x: x.timestamp)
        ctx.events[:] = new_events
        return _tr_record(ctx, at, duration_sec, self.name, self.detector_hint, params)

@register_transport
class BatchArrival(TransportAnomaly):
    name = "batch_arrival"
    detector_hint = "data_quality_gate"
    def apply(self, ctx, *, at, duration_sec, params):
        end = at + pd.Timedelta(seconds=duration_sec)
        release = pd.Timestamp(params["release_at"])
        new_events = []
        for e in ctx.events:
            if at <= e.timestamp < end:
                new_events.append(type(e)(release, e.sensor_id, e.capability, e.value, e.unit))
            else:
                new_events.append(e)
        new_events.sort(key=lambda x: x.timestamp)
        ctx.events[:] = new_events
        return _tr_record(ctx, at, duration_sec, self.name, self.detector_hint, params)
