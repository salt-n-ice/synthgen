# sensorgen

Synthetic sensor data + anomaly generator for the smart-home anomaly-detection pipeline.

## Install

```bash
pip install -e .[dev]
```

## Run a scenario

```bash
sensorgen run scenarios/household_120d.yaml --out out/household_120d
sensorgen run scenarios/leak_30d.yaml --out out/leak_30d
```

Available scenarios in `scenarios/`: `household_60d`, `household_120d`,
`household_dense_90d`, `household_sparse_60d`, `holdout_household_45d`,
`leak_30d`, `single_outlet_fridge_30d`, `training_patterns_60d`,
`training_rich_60d`.

Each run produces:
- `events.csv` — irregular event stream
- `labels.csv` — ground-truth anomaly intervals

## Visualize

```bash
sensorgen viz out/household_120d/events.csv \
    --labels out/household_120d/labels.csv \
    --output out/household_120d/viz.html
```

With pipeline detections:

```bash
sensorgen viz out/household_120d/events.csv \
    --labels out/household_120d/labels.csv \
    --detections out/household_120d/detections.csv \
    --output out/household_120d/matched.html
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
