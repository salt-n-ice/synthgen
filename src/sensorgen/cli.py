# src/sensorgen/cli.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .scenario import load_scenario
from .generator import run_scenario
from .viz import build_figure, write_html

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sensorgen")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run a scenario")
    r.add_argument("scenario")
    r.add_argument("--out", required=True)

    v = sub.add_parser("viz", help="Build HTML visualizer")
    v.add_argument("events")
    v.add_argument("--labels", required=True)
    v.add_argument("--detections", default=None)
    v.add_argument("--output", required=True)

    args = p.parse_args(argv)

    if args.cmd == "run":
        sc = load_scenario(Path(args.scenario))
        run_scenario(sc, Path(args.out))
        return 0

    if args.cmd == "viz":
        events = pd.read_csv(args.events)
        labels = pd.read_csv(args.labels)
        det = pd.read_csv(args.detections) if args.detections else None
        fig = build_figure(events, labels, detections=det)
        write_html(fig, Path(args.output))
        return 0

    return 2
