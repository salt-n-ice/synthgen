"""Auto-emit `usage_anomaly` labels for natural-variation outlier days.

Synth-gen's BURSTY profiles use Poisson-stochastic bursts whose long-tail
variance occasionally produces days with usage well outside the typical
range — heavy-TV days, quiet-kettle weeks, etc. The downstream anomaly
detector correctly fires on these (real signal deviations from the
bootstrap-window baseline), but the YAML author did not flag them as
labelled anomalies, so evt_F1 counts them as FPs.

This module runs after signal generation and event emission. For each
BURSTY sensor it bins events by calendar day, builds a robust baseline
(median + MAD) over days NOT covered by any user-specified anomaly,
then flags any unlabelled day whose event count is ≥3 MAD-scaled SD
from the baseline. Each flagged day becomes a `usage_anomaly` label
spanning 00:00 → 24:00 UTC on that day.

Tunables:
- USAGE_ANOMALY_Z_THRESHOLD = 3.0 — robust z-score floor.
- MIN_UNLABELED_DAYS = 10 — need this many clean days for a stable
  median; if too few, skip the sensor (avoids flagging in scenarios
  where labels dominate).
"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd
from .labels import LabelRecord
from .profiles.base import Archetype


USAGE_ANOMALY_Z_THRESHOLD = 3.0
MIN_UNLABELED_DAYS = 10
# Magnitude gate: a day must also deviate from baseline by ≥30%
# (relative) to be flagged. Without this, low-variance sensors like
# fridge (MAD ~3 events on baseline 322) trigger z≥3 on tiny absolute
# deviations (~14 extra events/day, 4% relative) that the duty-cycle
# detector can't see. The 30% floor matches the detector's effective
# sensitivity on duty-shift anomalies.
USAGE_ANOMALY_MIN_RELATIVE_DELTA = 0.30
# Skip the detector-side bootstrap window: the anomaly-detection
# pipeline isn't live during the first 14 days (it's fitting the
# baseline). Auto-labels in this window can't be matched to detector
# chains, so they only inflate label count and tank incR.
DETECTOR_BOOTSTRAP_DAYS = 14


def compute_usage_anomaly_labels(
    events,
    existing_labels: list[LabelRecord],
    sensor_archetypes: dict[tuple[str, str], Archetype],
) -> list[LabelRecord]:
    if not events:
        return []
    scenario_start = min(ev.timestamp for ev in events)
    bootstrap_end = (
        scenario_start.normalize()
        + pd.Timedelta(days=DETECTOR_BOOTSTRAP_DAYS)
    )
    user_windows: dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    for lbl in existing_labels:
        user_windows[(lbl.sensor_id, lbl.capability)].append((lbl.start, lbl.end))

    by_sensor: dict[tuple[str, str], list] = defaultdict(list)
    for ev in events:
        by_sensor[(ev.sensor_id, ev.capability)].append(ev.timestamp)

    out: list[LabelRecord] = []
    for (sid, cap), timestamps in by_sensor.items():
        if sensor_archetypes.get((sid, cap)) != Archetype.BURSTY:
            continue
        ts = pd.to_datetime(pd.Series(timestamps), utc=True)
        df = pd.DataFrame({"date": ts.dt.normalize()})
        daily = df.groupby("date").size().reset_index(name="count")
        windows = user_windows.get((sid, cap), [])

        def _is_labeled(d_start: pd.Timestamp) -> bool:
            d_end = d_start + pd.Timedelta(days=1)
            return any(ws < d_end and we > d_start for ws, we in windows)

        daily["labeled"] = daily["date"].apply(_is_labeled)
        clean = daily[~daily["labeled"]]
        if len(clean) < MIN_UNLABELED_DAYS:
            continue
        med = clean["count"].median()
        mad = (clean["count"] - med).abs().median()
        if mad == 0:
            continue
        scale = 1.4826 * mad
        for _, row in daily.iterrows():
            if row["labeled"]:
                continue
            d_start = row["date"]
            if d_start < bootstrap_end:
                continue
            z = (row["count"] - med) / scale
            if abs(z) < USAGE_ANOMALY_Z_THRESHOLD:
                continue
            if med <= 0:
                continue
            rel = abs(row["count"] - med) / med
            if rel < USAGE_ANOMALY_MIN_RELATIVE_DELTA:
                continue
            d_end = d_start + pd.Timedelta(days=1)
            direction = "high" if z > 0 else "low"
            out.append(LabelRecord(
                sid, cap, d_start, d_end, "usage_anomaly",
                "duty_cycle_shift_6h",
                {"z_score": float(round(z, 2)),
                 "direction": direction,
                 "event_count": int(row["count"]),
                 "baseline_median": int(med),
                 "baseline_mad": float(round(mad, 1))},
            ))
    return out
