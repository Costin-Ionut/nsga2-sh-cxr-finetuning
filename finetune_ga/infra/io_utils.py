"""io_utils.py — File I/O helpers: atomic JSON writes, JSONL cache, experiment state."""
from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover
    orjson = None

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover
    fcntl = None

try:
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover
    msvcrt = None

# In-memory read cache for JSONL files.  Avoids re-reading large files inside
# tight loops (e.g. genome × budget × backbone).  Invalidated on every append.
_RUNS_CACHE: Dict[str, Tuple[Tuple[int, int], List[Dict[str, Any]]]] = {}


def clear_runs_cache() -> None:
    """Evict all entries from the in-memory JSONL cache.

    Call this between independent seed runs to prevent memory accumulation.
    With 3+ seeds × 10 generations × large populations the cache can grow
    to hundreds of MB if never cleared.
    """
    _RUNS_CACHE.clear()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_loads(line: str) -> Dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def _json_dumps_line(row: Dict[str, Any]) -> str:
    if orjson is not None:
        return orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE).decode('utf-8')
    return json.dumps(row, allow_nan=False) + "\n"


def _cache_token(path: str) -> Tuple[int, int]:
    stat = os.stat(path)
    return int(stat.st_mtime_ns), int(stat.st_size)


@contextmanager
def _locked_file(path: str, mode: str) -> Iterator[Any]:
    with open(path, mode, encoding='utf-8') as f:
        locked_size = 0
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover
            try:
                f.seek(0, os.SEEK_END)
                locked_size = f.tell()
                if locked_size <= 0:
                    locked_size = 1
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, locked_size)
            except OSError:
                locked_size = 0
                pass
        try:
            yield f
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover
                try:
                    if locked_size > 0:
                        f.seek(0)
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, locked_size)
                except OSError:
                    pass


def _as_plain_scalar(value: Any) -> Any:
    """Return Python scalars for numpy/tensorflow scalar-like values when possible."""
    if isinstance(value, (str, bytes, bytearray, dict, list, tuple, set)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value


def sanitize_json_value(value: Any) -> Any:
    """Make nested values safe for strict JSON output.

    Search/evaluation code can occasionally produce NaN/inf metrics when a
    training run is unstable or an aggregate is computed from an empty slice.
    JSONL must stay valid even in that case, so non-finite floats are stored
    as null instead of crashing after a long generation.
    """
    value = _as_plain_scalar(value)

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json_value(v) for v in value]
    return value


def _validate_json_value(value: Any, *, path: str, json_path: str = "$") -> None:
    value = _as_plain_scalar(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f'Non-finite float encountered while writing JSON to {path} at {json_path}'
        )
    if isinstance(value, dict):
        for key, nested in value.items():
            _validate_json_value(nested, path=path, json_path=f"{json_path}.{key}")
        return
    if isinstance(value, (list, tuple, set)):
        for idx, nested in enumerate(value):
            _validate_json_value(nested, path=path, json_path=f"{json_path}[{idx}]")


def atomic_write_json(path: str, data: Any, *, validate_finite: bool = False) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    else:
        directory = '.'
    # Always sanitize before writing. This keeps checkpoint/state/report JSON
    # valid even when an intermediate metric is NaN/inf.
    data = sanitize_json_value(data)
    if validate_finite:
        _validate_json_value(data, path=path)
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(path) + '.',
        suffix='.tmp',
        dir=directory,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Could not read JSON from {path}: {type(exc).__name__}: {exc}")
        return {} if default is None else default


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    row = sanitize_json_value(row)
    _validate_json_value(row, path=path)
    payload = _json_dumps_line(row)
    with _locked_file(path, 'a') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    # Invalidate cache so the next read_jsonl() reflects the new row.
    _RUNS_CACHE.pop(path, None)



def _write_jsonl_rows_atomic(path: str, rows: List[Dict[str, Any]]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    rows = [sanitize_json_value(r) for r in rows]
    for row in rows:
        _validate_json_value(row, path=path)
    directory = d or '.'
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(path) + '.',
        suffix='.tmp',
        dir=directory,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(_json_dumps_line(row))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    _RUNS_CACHE.pop(path, None)


def upsert_jsonl(path: str, row: Dict[str, Any], key_fields: Tuple[str, ...]) -> None:
    """Insert or replace one JSONL row by key, also removing older duplicates."""
    if not key_fields:
        append_jsonl(path, row)
        return
    row = sanitize_json_value(row)
    key = tuple(row.get(field) for field in key_fields)
    existing = read_jsonl(path)
    kept: List[Dict[str, Any]] = []
    for old in existing:
        old_key = tuple(old.get(field) for field in key_fields)
        if old_key == key:
            continue
        kept.append(old)
    kept.append(row)
    _write_jsonl_rows_atomic(path, kept)


def dedupe_jsonl(path: str, key_fields: Tuple[str, ...]) -> int:
    """Deduplicate an existing JSONL file in-place; returns removed row count."""
    if not os.path.exists(path) or not key_fields:
        return 0
    rows = read_jsonl(path)
    seen: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    order: List[Tuple[Any, ...]] = []
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key not in seen:
            order.append(key)
        seen[key] = row
    deduped = [seen[key] for key in order]
    removed = len(rows) - len(deduped)
    if removed > 0:
        _write_jsonl_rows_atomic(path, deduped)
    return removed

def read_jsonl(path: str, *, strict: bool = False) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        _RUNS_CACHE[path] = ((0, 0), [])
        return []
    token = _cache_token(path)
    cached = _RUNS_CACHE.get(path)
    if cached is not None and cached[0] == token:
        return list(cached[1])
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_json_loads(line))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                message = f"Malformed JSONL row in {path} at line {line_no}: {type(exc).__name__}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                warnings.warn(message)
    _RUNS_CACHE[path] = (token, rows)
    return list(rows)




def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value_f if math.isfinite(value_f) else float(default)


def safe_float_or_none(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None

# ---------------------------------------------------------------------------
# Output-path management
# ---------------------------------------------------------------------------

def prepare_output_paths(
    root_dir: str, tag: str, backbone: Optional[str] = None
) -> Dict[str, str]:
    exp_dir = os.path.join(root_dir, tag)
    per_model_dir = os.path.join(exp_dir, 'per_model', backbone) if backbone else None
    shared_dir = os.path.join(root_dir, 'shared')
    best_dir = os.path.join(root_dir, 'best')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(shared_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    if per_model_dir:
        os.makedirs(per_model_dir, exist_ok=True)
    return {
        'root_dir': root_dir,
        'exp_dir': exp_dir,
        'shared_dir': shared_dir,
        'best_dir': best_dir,
        'per_model_dir': per_model_dir,
    }



def save_run_config_snapshot(exp_dir: str, cfg: Dict[str, Any], *, tag: str | None = None, active_seed: int | None = None) -> None:
    """Persist the exact run configuration for reproducibility/debugging."""
    payload = {
        "tag": tag,
        "active_seed": active_seed,
        "config": cfg,
    }
    atomic_write_json(os.path.join(exp_dir, 'run_config_snapshot.json'), payload, validate_finite=True)


def save_experiment_state(root_dir: str, tag: str, state: Dict[str, Any]) -> None:
    paths = prepare_output_paths(root_dir, tag)
    atomic_write_json(os.path.join(paths['exp_dir'], 'checkpoint_state.json'), state)


def load_experiment_state(root_dir: str, tag: str) -> Dict[str, Any]:
    paths = prepare_output_paths(root_dir, tag)
    return load_json(os.path.join(paths['exp_dir'], 'checkpoint_state.json'), default={})
