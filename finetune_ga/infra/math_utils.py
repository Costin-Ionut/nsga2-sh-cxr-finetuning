"""math_utils.py — Stateless numeric helpers with no framework dependencies."""
from __future__ import annotations

import math
import random


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    """Sample uniformly in log-space between *low* and *high*."""
    return float(10 ** rng.uniform(math.log10(low), math.log10(high)))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def steps_safe(num_images: int, batch: int, factor: float) -> int:
    """Steps per epoch scaled by *factor*, minimum 1."""
    base = max(1, int(math.ceil(num_images / float(batch))))
    return max(1, int(math.ceil(base * factor)))


def full_steps(num_images: int, batch: int) -> int:
    """Full steps for one pass through *num_images* at *batch* size."""
    return max(1, int(math.ceil(num_images / float(batch))))



def pareto_dominates_values(a_values, b_values) -> bool:
    """Return True when objective vector *a_values* Pareto-dominates *b_values*."""
    return all(ax <= bx for ax, bx in zip(a_values, b_values)) and any(ax < bx for ax, bx in zip(a_values, b_values))
