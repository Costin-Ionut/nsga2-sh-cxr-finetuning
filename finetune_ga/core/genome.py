"""genome.py — Genome dataclass and all genome-level operations.

A Genome encodes one candidate hyperparameter configuration.  This module is
deliberately free of TensorFlow imports so it can be used in analysis scripts
that run without a GPU.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

from finetune_ga.infra.math_utils import log_uniform, clamp


OBJECTIVE_NAMES = ('mean_val_auc_loss_obj', 'selection_time_s_obj', 'trainable_params_m_obj')


@dataclass
class Genome:
    base_lr1: float
    base_lr2: float
    n_last_layers: int
    dense_units: int
    dropout: float
    l2_weight: float
    batch_size: int
    aug_rotation: float
    aug_zoom: float
    aug_contrast: float
    lr1_mul: Dict[str, float]
    lr2_mul: Dict[str, float]
    genome_id: str = ''
    last_budget: str = 'B0'
    objectives: Tuple[float, float, float] = (1e9, 1e9, 1e9)
    rank: int = 0
    crowd: float = 0.0


def genome_to_dict(g: Genome) -> Dict[str, Any]:
    d = asdict(g)
    d['objective_names'] = list(OBJECTIVE_NAMES)
    d['objectives'] = list(g.objectives)
    d['mean_val_auc_loss_obj'] = float(g.objectives[0])
    d['mean_val_auc'] = float(1.0 - g.objectives[0])
    d['selection_time_s_obj'] = float(g.objectives[1])
    d['trainable_params_m_obj'] = float(g.objectives[2])
    return d


def _normalize_objectives(value: Any) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)):
        return (1e9, 1e9, 1e9)
    normalized: List[float] = []
    for item in list(value)[:3]:
        try:
            v = float(item)
        except (TypeError, ValueError):
            v = 1e9
        if not math.isfinite(v):
            v = 1e9
        normalized.append(v)
    while len(normalized) < 3:
        normalized.append(1e9)
    return (normalized[0], normalized[1], normalized[2])


def genome_from_dict(d: Dict[str, Any]) -> Genome:
    return Genome(
        base_lr1=float(d['base_lr1']),
        base_lr2=float(d['base_lr2']),
        n_last_layers=int(d['n_last_layers']),
        dense_units=int(d['dense_units']),
        dropout=float(d['dropout']),
        l2_weight=float(d['l2_weight']),
        batch_size=int(d['batch_size']),
        aug_rotation=float(d['aug_rotation']),
        aug_zoom=float(d['aug_zoom']),
        aug_contrast=float(d['aug_contrast']),
        lr1_mul={str(k).lower(): float(v) for k, v in d['lr1_mul'].items()},
        lr2_mul={str(k).lower(): float(v) for k, v in d['lr2_mul'].items()},
        genome_id=str(d.get('genome_id', '')),
        last_budget=str(d.get('last_budget', 'B0')),
        objectives=_normalize_objectives(d.get('objectives', [1e9, 1e9, 1e9])),
        rank=int(d.get('rank', 0)),
        crowd=float(d.get('crowd', 0.0)),
    )


def genome_fingerprint(g: Genome, model_names: List[str]) -> str:
    lr1m = ','.join([f"{k}:{g.lr1_mul[k.lower()]:.6f}" for k in model_names])
    lr2m = ','.join([f"{k}:{g.lr2_mul[k.lower()]:.6f}" for k in model_names])
    return (
        f"lr1={g.base_lr1:.12e}|lr2={g.base_lr2:.12e}|unf={g.n_last_layers}|"
        f"dense={g.dense_units}|drop={g.dropout:.4f}|l2={g.l2_weight:.12e}|bs={g.batch_size}|"
        f"rot={g.aug_rotation:.4f}|zoom={g.aug_zoom:.4f}|con={g.aug_contrast:.4f}|"
        f"lr1m={lr1m}|lr2m={lr2m}"
    )


def assign_genome_id(g: Genome, model_names: List[str]) -> str:
    """Assign a genome ID derived from a SHA-256 hash of the genome fingerprint.

    Using 12 hex characters (48 bits) instead of CRC32 (32 bits) makes
    birthday-problem collisions negligible even for large population sizes.
    """
    fp = genome_fingerprint(g, model_names)
    gid = hashlib.sha256(fp.encode('utf-8')).hexdigest()[:12]
    g.genome_id = gid
    return gid


def random_multipliers(
    rng: random.Random, model_names: List[str], lo: float, hi: float
) -> Dict[str, float]:
    return {b.lower(): float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) for b in model_names}


def sample_genome(cfg: Dict[str, Any], rng: random.Random) -> Genome:
    ss = cfg['search_space']
    model_names = cfg['model_names']
    lr_mul_lo, lr_mul_hi = ss['lr_mul_bounds']
    g = Genome(
        base_lr1=float(rng.choice(ss['base_lr1'])),
        base_lr2=float(rng.choice(ss['base_lr2'])),
        n_last_layers=int(rng.choice(ss['n_last_layers'])),
        dense_units=int(rng.choice(ss['dense_units'])),
        dropout=float(rng.uniform(ss['dropout'][0], ss['dropout'][1])),
        l2_weight=float(rng.choice(ss['l2_weight'])),
        batch_size=int(rng.choice(ss['batch_size'])),
        aug_rotation=float(rng.uniform(ss['aug_rotation'][0], ss['aug_rotation'][1])),
        aug_zoom=float(rng.uniform(ss['aug_zoom'][0], ss['aug_zoom'][1])),
        aug_contrast=float(rng.uniform(ss['aug_contrast'][0], ss['aug_contrast'][1])),
        lr1_mul=random_multipliers(rng, model_names, lr_mul_lo, lr_mul_hi),
        lr2_mul=random_multipliers(rng, model_names, lr_mul_lo, lr_mul_hi),
    )
    assign_genome_id(g, model_names)
    return g


def hyper_for_backbone(g: Genome, backbone: str, unfreeze_cap: Dict[str, int]) -> Dict[str, Any]:
    b = backbone.lower()
    lr1 = clamp(g.base_lr1 * g.lr1_mul.get(b, 1.0), 1e-6, 2e-3)
    lr2 = clamp(g.base_lr2 * g.lr2_mul.get(b, 1.0), 1e-8, 1e-4)
    cap = int(unfreeze_cap.get(b, 80))
    n_last = int(clamp(g.n_last_layers, 0, cap))
    return {'lr1': lr1, 'lr2': lr2, 'n_last_layers': n_last}
