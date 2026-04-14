# tests/test_labels.py
import pandas as pd
from pathlib import Path
from sensorgen.labels import LabelRecord, write_labels

def test_write_labels_roundtrip(tmp_path: Path):
    recs = [
        LabelRecord(
            sensor_id="s1", capability="power",
            start=pd.Timestamp("2026-01-01T00:00:00Z"),
            end=pd.Timestamp("2026-01-01T00:05:00Z"),
            anomaly_type="spike", detector_hint="pca",
            params={"magnitude": 500},
        ),
    ]
    p = tmp_path / "labels.csv"
    write_labels(recs, p)
    df = pd.read_csv(p)
    assert list(df.columns) == ["sensor_id","capability","start","end","anomaly_type","detector_hint","params_json"]
    assert df.iloc[0]["anomaly_type"] == "spike"
    assert "magnitude" in df.iloc[0]["params_json"]
