# src/sensorgen/viz.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .match import Interval, match_intervals

_COLORS = {
    "spike":"rgba(255,80,80,0.25)", "dip":"rgba(80,80,255,0.25)",
    "out_of_range":"rgba(255,140,0,0.25)", "noise_burst":"rgba(180,0,180,0.25)",
    "noise_floor_up":"rgba(200,0,120,0.25)", "calibration_drift":"rgba(0,180,140,0.25)",
    "trend":"rgba(0,150,0,0.25)", "level_shift":"rgba(0,100,220,0.25)",
    "degradation_trajectory":"rgba(120,80,0,0.25)", "frequency_change":"rgba(220,60,160,0.25)",
    "seasonality_loss":"rgba(120,120,120,0.25)", "time_of_day":"rgba(255,200,0,0.25)",
    "weekend_anomaly":"rgba(0,200,255,0.25)", "month_shift":"rgba(180,160,80,0.25)",
    "seasonal_mismatch":"rgba(200,100,200,0.25)", "stuck_at":"rgba(80,80,80,0.25)",
    "water_leak_sustained":"rgba(0,120,255,0.35)", "unusual_occupancy":"rgba(220,120,60,0.25)",
    "dropout":"rgba(160,40,40,0.25)", "duplicate_stale":"rgba(80,160,160,0.25)",
    "reporting_rate_change":"rgba(150,150,30,0.25)", "clock_drift":"rgba(100,50,160,0.25)",
    "batch_arrival":"rgba(200,80,40,0.25)", "saturation":"rgba(240,100,100,0.25)",
}
def _color(t: str) -> str: return _COLORS.get(t, "rgba(120,120,120,0.2)")

def build_figure(events: pd.DataFrame, labels: pd.DataFrame,
                 detections: pd.DataFrame | None = None) -> go.Figure:
    ev = events.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True, format="ISO8601")
    keys = list(ev.groupby(["sensor_id","capability"]).groups.keys())
    fig = make_subplots(rows=max(1,len(keys)), cols=1, shared_xaxes=True,
                        subplot_titles=[f"{s} · {c}" for s,c in keys])
    for i, (sid, cap) in enumerate(keys, start=1):
        sub = ev[(ev.sensor_id==sid) & (ev.capability==cap)].sort_values("timestamp")
        is_binary = set(sub["value"].unique()).issubset({0,1,0.0,1.0})
        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["value"],
            mode="lines+markers",
            line=dict(shape="hv") if is_binary else dict(),
            name=f"{sid}:{cap}", legendgroup=f"{sid}:{cap}",
        ), row=i, col=1)

    def _add_rect(start, end, color, name, hover):
        fig.add_vrect(x0=start, x1=end, fillcolor=color, line_width=0,
                      annotation_text=name, annotation_position="top left",
                      annotation=dict(font_size=10, font_color="#333"),
                      layer="below")

    lb = labels.copy()
    lb["start"] = pd.to_datetime(lb["start"], utc=True)
    lb["end"] = pd.to_datetime(lb["end"], utc=True)

    if detections is None:
        for _, r in lb.iterrows():
            _add_rect(r["start"], r["end"], _color(r["anomaly_type"]), r["anomaly_type"], r["anomaly_type"])
    else:
        det = detections.copy()
        det["start"] = pd.to_datetime(det["start"], utc=True)
        det["end"] = pd.to_datetime(det["end"], utc=True)
        gt_iv = [Interval(r.sensor_id, r.start, r.end, r.anomaly_type) for r in lb.itertuples()]
        det_iv = [Interval(r.sensor_id, r.start, r.end, r.anomaly_type) for r in det.itertuples()]
        tp, fp, fn = match_intervals(gt_iv, det_iv)
        matched_gt = set(id(x) for x in tp)
        for x in tp: _add_rect(x.start, x.end, "rgba(0,180,0,0.25)", "TP", x.anomaly_type)
        for x in fn: _add_rect(x.start, x.end, "rgba(220,0,0,0.25)", "FN", x.anomaly_type)
        for x in fp: _add_rect(x.start, x.end, "rgba(255,140,0,0.25)", "FP", x.anomaly_type)
        # also shade detection intervals that matched (TP detections)
        tp_det, _, _ = match_intervals(det_iv, gt_iv)
        for x in tp_det: _add_rect(x.start, x.end, "rgba(0,180,0,0.15)", "det", x.anomaly_type)

    fig.update_xaxes(rangeslider=dict(visible=True), row=max(1,len(keys)), col=1)
    fig.update_layout(height=300*max(1,len(keys)), hovermode="x unified", showlegend=True)
    return fig

def write_html(fig: go.Figure, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
