from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Archetype, Profile, register

SEC_PER_DAY = 86400

def _time_of_day_fraction(start: pd.Timestamp, n: int) -> np.ndarray:
    sec0 = start.hour*3600 + start.minute*60 + start.second
    t = (np.arange(n) + sec0) % SEC_PER_DAY
    return t / SEC_PER_DAY

def _day_index(start: pd.Timestamp, n: int) -> np.ndarray:
    sec0 = start.hour*3600 + start.minute*60 + start.second
    return (np.arange(n) + sec0) // SEC_PER_DAY + start.dayofweek

class RoomTemperature(Profile):
    archetype = Archetype.CONTINUOUS
    name = "room_temperature"
    def sample(self, start, n_ticks, rng):
        tod = _time_of_day_fraction(start, n_ticks)
        diurnal = 3.0 * np.sin(2*np.pi*(tod - 0.25))  # warmest mid-afternoon
        weekly = 0.5 * np.sin(2*np.pi*_day_index(start, n_ticks)/7.0)
        noise = rng.normal(0, 0.15, n_ticks)
        return 21.0 + diurnal + weekly + noise

class BatteryDrain(Profile):
    archetype = Archetype.CONTINUOUS
    name = "battery_drain"
    def sample(self, start, n_ticks, rng):
        # ~1% per day baseline drain with tiny noise, reported as integer percent
        days = np.arange(n_ticks) / SEC_PER_DAY
        raw = 100.0 - days * 1.0 + rng.normal(0, 0.05, n_ticks)
        return np.clip(raw, 0.0, 100.0)

class VoltageMains(Profile):
    archetype = Archetype.CONTINUOUS
    name = "voltage_mains"
    def sample(self, start, n_ticks, rng):
        return 120.0 + rng.normal(0, 0.4, n_ticks)

register(RoomTemperature())
register(BatteryDrain())
register(VoltageMains())
