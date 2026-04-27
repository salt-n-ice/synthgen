# src/sensorgen/generator.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .scenario import Scenario
from .rng import RootRng
from .profiles.base import get_profile
from .emitter import EmitConfig, emit_events, events_to_dataframe
from .anomalies.base import (
    get_anomaly, get_transport_anomaly, AnomalyContext, TransportContext,
    list_anomalies, list_transport_anomalies,
)
from .labels import write_labels
from .auto_labels import compute_usage_anomaly_labels

def run_scenario(sc: Scenario, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = RootRng(sc.seed)
    signal_types = set(list_anomalies())
    transport_types = set(list_transport_anomalies())

    all_events = []
    all_labels = []
    sensor_archetypes: dict[tuple[str, str], object] = {}

    for sensor in sc.sensors:
        prof = get_profile(sensor.profile)
        sensor_archetypes[(sensor.id, sensor.capability)] = prof.archetype
        rng_sig = root.child(f"profile:{sensor.id}")
        signal = prof.sample(sc.start, sc.duration_sec, rng_sig)
        ctx_sig = AnomalyContext(sensor.id, sensor.capability, sensor.unit,
                                 prof.archetype, sc.start, signal)
        # signal-level anomalies (in declared order)
        for a in sc.anomalies:
            if a.sensor != sensor.id or a.type not in signal_types:
                continue
            cls = get_anomaly(a.type)
            if prof.archetype not in cls.supports:
                raise ValueError(f"anomaly {a.type} not supported for archetype {prof.archetype}")
            all_labels.append(cls().apply(ctx_sig, at=a.at, duration_sec=a.duration_sec, params=a.params))
        # emit
        cfg = EmitConfig(sensor.id, sensor.capability, sensor.unit, prof.archetype,
                         sensor.emit.delta_threshold, sensor.emit.heartbeat_sec)
        events = emit_events(signal, sc.start, cfg)
        # transport anomalies
        ctx_tr = TransportContext(sensor.id, sensor.capability, sc.start,
                                  sc.start + pd.Timedelta(seconds=sc.duration_sec), events)
        for a in sc.anomalies:
            if a.sensor != sensor.id or a.type not in transport_types:
                continue
            cls = get_transport_anomaly(a.type)
            all_labels.append(cls().apply(ctx_tr, at=a.at, duration_sec=a.duration_sec, params=a.params))
        all_events.extend(ctx_tr.events)

    all_events.sort(key=lambda e: (e.timestamp, e.sensor_id, e.capability))
    # Auto-label natural-variation outlier days (BURSTY sensors only).
    # Runs after all signal/transport anomalies are applied so the
    # baseline is computed on the same event stream the detector sees,
    # and so user-labelled windows are correctly excluded from the
    # outlier baseline.
    all_labels.extend(compute_usage_anomaly_labels(
        all_events, all_labels, sensor_archetypes))
    events_to_dataframe(all_events).to_csv(out_dir/"events.csv", index=False)
    write_labels(all_labels, out_dir/"labels.csv")
