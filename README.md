# sensorgen

Synthetic sensor data + anomaly generator for the smart-home anomaly-detection pipeline.

## Install

```bash
pip install -e .[dev]
```

## Run a scenario

```bash
sensorgen run scenarios/outlet_demo.yaml --out out/outlet
sensorgen run scenarios/waterleak_demo.yaml --out out/leak
```

Each run produces:
- `events.csv` — irregular event stream
- `labels.csv` — ground-truth anomaly intervals

## Visualize

```bash
sensorgen viz out/outlet/events.csv --labels out/outlet/labels.csv --output out/outlet/viz.html
```

With pipeline detections:

```bash
sensorgen viz out/outlet/events.csv --labels out/outlet/labels.csv \
    --detections out/outlet/detections.csv --output out/outlet/matched.html
```

Open the HTML file in a browser. Use the range slider and zoom for specific windows. Anomalies are shaded per type; in match mode, green=TP, red=FN, orange=FP.

## Add a new sensor profile

Add a subclass of `Profile` in `src/sensorgen/profiles/{continuous,bursty,binary}.py` and register it. One file, no other changes.

## Add a new anomaly

Add a subclass of `Anomaly` (or `TransportAnomaly`) in the appropriate `src/sensorgen/anomalies/*.py` module and register it. Add one test.

## Tests

```bash
pytest -q
```
