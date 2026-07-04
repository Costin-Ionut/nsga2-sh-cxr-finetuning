from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


def resolve_test_candidate_tags(cfg: Dict[str, Any]) -> List[str]:
    raw_tags = cfg.get("test_from_tags") or ["nsga2", "random_sh", "nsga2_no_conditioning", "random_full_budget"]
    tags: List[str] = []
    for tag in raw_tags:
        name = str(tag).strip()
        if not name or name.startswith("ablation_"):
            continue
        if name not in tags:
            tags.append(name)
    return tags


def build_test_protocol_id(cfg: Dict[str, Any]) -> str:
    payload = {
        "candidate_tags": resolve_test_candidate_tags(cfg),
        "topk": int(cfg.get("test_topk_per_tag", 1)),
        "test_e1": int(cfg.get("test_e1", 11)),
        "test_e2": int(cfg.get("test_e2", 5)),
        "final_retrain_mode": cfg.get("final_retrain_mode", "train_plus_val_fixed_epochs"),
        "final_retrain_uses_validation": bool(cfg.get("final_retrain_uses_validation", False)),
        "policy": cfg.get("test_selection_policy", {}),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]
