# tests/test_viz.py
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sensorgen.viz import build_figure, write_html

def _events():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-03-01T00:00:00Z","2026-03-01T00:01:00Z","2026-03-01T00:02:00Z"], utc=True),
        "sensor_id": ["s1"]*3,
        "capability": ["power"]*3,
        "value": [10.0, 15.0, 12.0],
        "unit": ["W"]*3,
    })

def _labels():
    return pd.DataFrame([{
        "sensor_id":"s1","capability":"power",
        "start":"2026-03-01T00:00:30Z","end":"2026-03-01T00:01:30Z",
        "anomaly_type":"spike","detector_hint":"pca","params_json":"{}",
    }])

def test_build_figure_has_trace_and_label_shape():
    fig = build_figure(_events(), _labels())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert len(fig.layout.shapes) >= 1

def test_build_figure_with_detections_adds_more_shapes():
    det = pd.DataFrame([{
        "sensor_id":"s1","capability":"power",
        "start":"2026-03-01T00:00:45Z","end":"2026-03-01T00:01:15Z",
        "anomaly_type":"detected","detector_hint":"","params_json":"{}",
    }])
    fig = build_figure(_events(), _labels(), detections=det)
    # at minimum: label shape + overlap shape
    assert len(fig.layout.shapes) >= 2

def test_write_html_creates_file(tmp_path: Path):
    out = tmp_path / "viz.html"
    write_html(build_figure(_events(), _labels()), out)
    text = out.read_text(encoding="utf-8")
    assert "<html" in text.lower() or "plotly" in text.lower()
