# tests/test_rng.py
import numpy as np
from sensorgen.rng import RootRng

def test_child_rngs_are_deterministic_and_independent():
    r1 = RootRng(42)
    r2 = RootRng(42)
    a1 = r1.child("profile:outlet").random()
    b1 = r1.child("anomaly:spike").random()
    a2 = r2.child("profile:outlet").random()
    b2 = r2.child("anomaly:spike").random()
    assert a1 == a2
    assert b1 == b2
    assert a1 != b1

def test_same_label_returns_same_rng():
    r = RootRng(7)
    v1 = r.child("x").integers(0, 10_000)
    v2 = r.child("x").integers(0, 10_000)
    assert v1 == v2

def test_returns_numpy_generator():
    assert isinstance(RootRng(0).child("x"), np.random.Generator)
