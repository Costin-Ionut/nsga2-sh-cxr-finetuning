from __future__ import annotations

import os
from typing import Any, Dict

_WORKER_GPU_ID: str | None = None


def init_search_worker(gpu_id: str) -> None:
    global _WORKER_GPU_ID
    _WORKER_GPU_ID = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = _WORKER_GPU_ID
    os.environ["MULTI_GPU_MODE"] = "single"


def run_search_task(task: Dict[str, Any]) -> Dict[str, Any]:
    if _WORKER_GPU_ID is None:
        raise RuntimeError("Search worker GPU was not initialized.")

    from finetune_ga.core.genome import genome_from_dict
    from finetune_ga.core.training import clear_tf_memory, train_one_backbone

    try:
        mode = str(task.get("mode", "search")).lower()
        if mode not in {"search", "final"}:
            raise ValueError(f"Unsupported task mode {mode!r}. Expected 'search' or 'final'.")

        genome = genome_from_dict(task["genome"])
        row = train_one_backbone(
            task["cfg"],
            genome,
            task["backbone"],
            task["budget"],
            int(task["rep"]),
            root_dir=task["root_dir"],
            tag=task["tag"],
            gen_idx=int(task["gen_idx"]),
            active_seed=int(task["active_seed"]),
            mode=mode,
        )
        row.update({"gen": int(task["gen_idx"]), "tag": task["tag"], "worker_gpu_id": _WORKER_GPU_ID})
        return row
    finally:
        clear_tf_memory()
