# tests/test_match.py
import pandas as pd
from sensorgen.match import Interval, match_intervals

def iv(s, e, sensor="s1", t="spike"): return Interval(sensor, pd.Timestamp(s), pd.Timestamp(e), t)

def test_full_overlap_is_tp():
    gt = [iv("2026-03-01T00:00:00Z","2026-03-01T00:05:00Z")]
    det = [iv("2026-03-01T00:01:00Z","2026-03-01T00:04:00Z")]
    tp, fp, fn = match_intervals(gt, det)
    assert len(tp) == 1 and len(fp) == 0 and len(fn) == 0

def test_disjoint_is_fp_and_fn():
    gt = [iv("2026-03-01T00:00:00Z","2026-03-01T00:01:00Z")]
    det = [iv("2026-03-01T01:00:00Z","2026-03-01T01:01:00Z")]
    tp, fp, fn = match_intervals(gt, det)
    assert len(tp) == 0 and len(fp) == 1 and len(fn) == 1

def test_different_sensors_do_not_match():
    gt = [iv("2026-03-01T00:00:00Z","2026-03-01T00:05:00Z", sensor="a")]
    det = [iv("2026-03-01T00:01:00Z","2026-03-01T00:04:00Z", sensor="b")]
    tp, fp, fn = match_intervals(gt, det)
    assert len(tp) == 0 and len(fp) == 1 and len(fn) == 1
