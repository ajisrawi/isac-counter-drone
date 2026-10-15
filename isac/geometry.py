"""Bistatic geometry helpers (local ENU coordinates, metres)."""
from __future__ import annotations

import numpy as np

EARTH_R = 6_378_137.0


def bistatic_excess(p, tx, rx):
    p, tx, rx = (np.asarray(a, float) for a in (p, tx, rx))
    return (np.linalg.norm(p - tx, axis=-1) + np.linalg.norm(p - rx, axis=-1)
            - np.linalg.norm(tx - rx, axis=-1))


def bistatic_rate(p, v, tx, rx):
    p, v, tx, rx = (np.asarray(a, float) for a in (p, v, tx, rx))
    ut = (p - tx) / np.linalg.norm(p - tx, axis=-1, keepdims=True)
    ur = (p - rx) / np.linalg.norm(p - rx, axis=-1, keepdims=True)
    return np.sum((ut + ur) * v, axis=-1)


def enu_to_geodetic(enu, lat0, lon0, h0=0.0):
    """Small-area (flat-earth) conversion, adequate for a few tens of km."""
    e, n, u = np.asarray(enu, float)
    lat = lat0 + np.degrees(n / EARTH_R)
    lon = lon0 + np.degrees(e / (EARTH_R * np.cos(np.radians(lat0))))
    return float(lat), float(lon), float(h0 + u)


def geodetic_to_enu(lat, lon, h, lat0, lon0, h0=0.0):
    n = np.radians(lat - lat0) * EARTH_R
    e = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    return np.array([e, n, h - h0])
