"""Shared GPU task executor for search/baseline training tasks.

The executor assigns one long-lived worker process to each visible GPU and runs
independent training tasks in parallel. Workers return result rows; callers are
responsible for writing JSONL files from the main process only.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import math
import os
from typing import Any, Callable, Dict, List

from finetune_ga.search.worker import init_search_worker, run_search_task


def _resolve_search_worker_count(cfg: Dict[str, Any]) -> int:
    requested = int(os.environ.get("SEARCH_PARALLEL_WORKERS", cfg.get("search_parallel_workers", 0) or 0))
    if requested > 0:
        return max(1, requested)

    env_num_gpus = os.environ.get("NUM_GPUS", "").strip()
    if env_num_gpus.isdigit():
        return max(1, int(env_num_gpus))

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        gpu_ids = [chunk.strip() for chunk in visible.split(",") if chunk.strip()]
        return max(1, len(gpu_ids))

    default_workers = int(cfg.get("search_parallel_workers_default", 1) or 1)
    return max(1, default_workers)


def _resolve_visible_gpu_ids(max_workers: int) -> List[str]:
    """Return visible GPU identifiers as CUDA_VISIBLE_DEVICES-compatible strings.

    CUDA_VISIBLE_DEVICES may contain integer ordinals, UUIDs, or MIG identifiers.
    Preserve the exact tokens instead of forcing int conversion.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpu_ids: List[str] = []
    if visible:
        for chunk in visible.split(","):
            value = chunk.strip()
            if value:
                gpu_ids.append(value)

    if not gpu_ids:
        gpu_ids = [str(i) for i in range(int(max_workers))]
    return gpu_ids[:max_workers]


def is_parallel_search_enabled(cfg: Dict[str, Any]) -> bool:
    worker_count = _resolve_search_worker_count(cfg)
    gpu_ids = _resolve_visible_gpu_ids(worker_count)
    return worker_count > 1 and len(gpu_ids) > 1


def _pin_local_single_gpu(gpu_ids: List[str]) -> None:
    """Force serial fallback to use exactly one visible GPU before TF import."""
    os.environ["MULTI_GPU_MODE"] = "single"
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if "," in visible:
        first = next((chunk.strip() for chunk in visible.split(",") if chunk.strip()), "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = first



def force_pin_current_process_to_first_gpu() -> None:
    """Pin this process to the first visible GPU before any TensorFlow access.

    Use for intentionally serial code paths even when SEARCH_PARALLEL_WORKERS=2
    is set globally for other baselines.
    """
    gpu_ids = _resolve_visible_gpu_ids(1)
    _pin_local_single_gpu(gpu_ids)


def pin_single_gpu_if_single_worker(cfg: Dict[str, Any]) -> None:
    """Pin the current process before any TensorFlow access when running serially.

    This prevents the serial fallback path from accidentally creating a
    MirroredStrategy over multiple visible GPUs if CUDA_VISIBLE_DEVICES=0,1
    but SEARCH_PARALLEL_WORKERS/config requests only one worker.
    """
    worker_count = _resolve_search_worker_count(cfg)
    gpu_ids = _resolve_visible_gpu_ids(worker_count)
    if worker_count <= 1 or len(gpu_ids) <= 1:
        _pin_local_single_gpu(gpu_ids)


def _run_search_task_local(task: Dict[str, Any]) -> Dict[str, Any]:
    """Serial fallback used when only one worker/GPU is requested."""
    os.environ["MULTI_GPU_MODE"] = "single"

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
        row.update({"gen": int(task["gen_idx"]), "tag": task["tag"], "worker_gpu_id": -1})
        return row
    finally:
        clear_tf_memory()


class SearchTaskExecutor:
    def __init__(self, cfg: Dict[str, Any]):
        self.worker_count = _resolve_search_worker_count(cfg)
        self.gpu_ids = _resolve_visible_gpu_ids(self.worker_count)
        self.local_only = self.worker_count <= 1 or len(self.gpu_ids) <= 1
        self._ctx = None
        # One single-process executor per GPU gives an explicit, stable
        # worker->GPU mapping and avoids queue-consumption corner cases if a
        # worker fails during initialization.
        self._executors: List[ProcessPoolExecutor] = []

    def __enter__(self) -> "SearchTaskExecutor":
        if self.local_only:
            _pin_local_single_gpu(self.gpu_ids)
            print(f"[SEARCH] serial GPU mode | gpu={os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU/auto')}")
            return self

        self._ctx = mp.get_context("spawn")
        self._executors = []
        for gpu_id in self.gpu_ids:
            self._executors.append(
                ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=self._ctx,
                    initializer=init_search_worker,
                    initargs=(str(gpu_id),),
                )
            )
        print(f"[SEARCH] persistent worker pool ready | workers={len(self._executors)} | gpus={self.gpu_ids}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=False)
        self._executors = []
        self._ctx = None

    def run_tasks(self, tasks: List[Dict[str, Any]], on_result: Callable[[Dict[str, Any]], None] | None = None) -> List[Dict[str, Any]]:
        if not tasks:
            return []
        if self.local_only or not self._executors:
            _pin_local_single_gpu(self.gpu_ids)
            results: List[Dict[str, Any]] = []
            total = len(tasks)
            for idx, task in enumerate(tasks, start=1):
                row = _run_search_task_local(task)
                results.append(row)
                if on_result is not None:
                    on_result(row)
                self._log_completion(row, idx, total)
            return results

        total = len(tasks)
        next_task_idx = 0
        future_to_slot = {}

        def submit_next(slot: int) -> None:
            nonlocal next_task_idx
            if next_task_idx >= total:
                return
            future = self._executors[slot].submit(run_search_task, tasks[next_task_idx])
            future_to_slot[future] = slot
            next_task_idx += 1

        for slot in range(min(len(self._executors), total)):
            submit_next(slot)

        results: List[Dict[str, Any]] = []
        done_count = 0
        while future_to_slot:
            for future in as_completed(list(future_to_slot)):
                slot = future_to_slot.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    for pending in future_to_slot:
                        pending.cancel()
                    for executor in self._executors:
                        executor.shutdown(wait=False, cancel_futures=True)
                    self._executors = []
                    raise RuntimeError(
                        f"Parallel search task failed after {len(results)}/{total} completed tasks. "
                        "Completed task results are persisted incrementally when an on_result callback is configured; "
                        "rerun with SEARCH_PARALLEL_WORKERS=1 to reproduce the failing task serially."
                    ) from exc
                results.append(row)
                if on_result is not None:
                    on_result(row)
                done_count += 1
                self._log_completion(row, done_count, total)
                submit_next(slot)
                break
        return results

    @staticmethod
    def _log_completion(row: Dict[str, Any], done_count: int, total: int) -> None:
        def _finite(value: Any, default: float = 0.0) -> float:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return float(default)
            return value_f if math.isfinite(value_f) else float(default)

        print(
            f"[SEARCH] completed {done_count}/{total} | "
            f"genome={row.get('genome_id')} | bb={row.get('backbone')} | "
            f"budget={row.get('budget')} | rep={row.get('rep')} | "
            f"val_auc={_finite(row.get('best_val_auc'), 0.0):.4f} | "
            f"time={_finite(row.get('time_s'), 0.0) / 60.0:.2f}m | "
            f"gpu={row.get('worker_gpu_id', 'local')}"
        )
