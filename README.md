# sensorgen

Synthetic sensor data + anomaly generator for the smart-home anomaly-detection pipeline.

## Install

```bash
pip install -e .[dev]
```

## Run

```bash
sensorgen run scenarios/outlet_demo.yaml --out out/outlet
sensorgen viz out/outlet/events.csv --labels out/outlet/labels.csv --output out/outlet/viz.html
```
