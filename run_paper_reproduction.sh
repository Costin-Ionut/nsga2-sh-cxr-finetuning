#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Must be set before Python starts for deterministic hashing.
export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "[RUN] Python interpreter not found." >&2
    exit 1
  fi
fi


if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
else
    GPU_COUNT=0
fi


if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]]; then
    GPU_COUNT=0
fi


if [[ -z "${NUM_GPUS:-}" ]]; then
    export NUM_GPUS="$GPU_COUNT"
fi


if [ "$NUM_GPUS" -gt 0 ]; then
    echo "GPU detected: $NUM_GPUS GPU(s)"
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))"
    fi
else
    echo "No GPU detected. Running on CPU."
fi

VISIBLE_GPU_COUNT=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    VISIBLE_GPU_COUNT="$(python - <<'PY'
import os
visible=os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
ids=[x.strip() for x in visible.split(',') if x.strip()]
print(len(ids))
PY
)"
fi

export ENABLE_XLA=${ENABLE_XLA:-0}
export ENABLE_MIXED_PRECISION=${ENABLE_MIXED_PRECISION:-1}
export ENABLE_DATASET_CACHE=${ENABLE_DATASET_CACHE:-1}
export CACHE_ON_DISK=${CACHE_ON_DISK:-0}
export TF_DATA_CACHE_DIR=${TF_DATA_CACHE_DIR:-/kaggle/working/finetuned_ga_tf_cache}
export RESUME=${RESUME:-1}
export SEARCH_PARALLEL_WORKERS=${SEARCH_PARALLEL_WORKERS:-$(( VISIBLE_GPU_COUNT > 0 ? VISIBLE_GPU_COUNT : 1 ))}
export TRAIN_LOG_MODE=${TRAIN_LOG_MODE:-minimal}
export TRAIN_FIT_VERBOSE=${TRAIN_FIT_VERBOSE:-0}
export AUTO_SPLIT_DATASET=${AUTO_SPLIT_DATASET:-0}
export FORCE_REBUILD_SPLIT=${FORCE_REBUILD_SPLIT:-0}
PRIMARY_DATASET_ROOT=${PRIMARY_DATASET_ROOT:-/kaggle/input/datasets/romanatanase/pneumonia-balanced-dataset}
SECONDARY_DATASET_ROOT=${SECONDARY_DATASET_ROOT:-/kaggle/input/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset}
if [[ -z "${DATASET_ROOT:-}" ]]; then
  for candidate in "$PRIMARY_DATASET_ROOT" "$SECONDARY_DATASET_ROOT"; do
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      export DATASET_ROOT="$candidate"
      break
    fi
  done
fi
export DATASET_ROOT=${DATASET_ROOT:-$PRIMARY_DATASET_ROOT}
export DATASET_SPLIT_ROOT=${DATASET_SPLIT_ROOT:-${DATASET_ROOT}_auto_split_80_10_10}
export DATASET_SPLIT_SEED=${DATASET_SPLIT_SEED:-42}
export DATASET_SPLIT_MODE=${DATASET_SPLIT_MODE:-copy}
export REQUIRE_DETERMINISTIC_DATASET=${REQUIRE_DETERMINISTIC_DATASET:-1}
export ALLOW_KAGGLEHUB_DOWNLOAD=${ALLOW_KAGGLEHUB_DOWNLOAD:-0}
export ALLOW_UNPINNED_KAGGLE_FALLBACK=${ALLOW_UNPINNED_KAGGLE_FALLBACK:-0}

echo "[RUN] SCRIPT_DIR=$SCRIPT_DIR"
echo "[RUN] PYTHON_BIN=$PYTHON_BIN"
echo "[RUN] NUM_GPUS=$NUM_GPUS"
echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[RUN] TRAIN_FIT_VERBOSE=$TRAIN_FIT_VERBOSE"
echo "[RUN] TRAIN_LOG_MODE=$TRAIN_LOG_MODE"
echo "[RUN] SEARCH_PARALLEL_WORKERS=$SEARCH_PARALLEL_WORKERS"
echo "[RUN] PRIMARY_DATASET_ROOT=$PRIMARY_DATASET_ROOT"
echo "[RUN] SECONDARY_DATASET_ROOT=$SECONDARY_DATASET_ROOT"
echo "[RUN] DATASET_ROOT=$DATASET_ROOT"
echo "[RUN] AUTO_SPLIT_DATASET=$AUTO_SPLIT_DATASET"
echo "[RUN] FORCE_REBUILD_SPLIT=$FORCE_REBUILD_SPLIT"
echo "[RUN] DATASET_SPLIT_ROOT=$DATASET_SPLIT_ROOT"
echo "[RUN] REQUIRE_DETERMINISTIC_DATASET=$REQUIRE_DETERMINISTIC_DATASET"
echo "[RUN] ALLOW_KAGGLEHUB_DOWNLOAD=$ALLOW_KAGGLEHUB_DOWNLOAD"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[RUN] Official dataset root not found: $DATASET_ROOT" >&2
  echo "[RUN] Expected Kaggle path: /kaggle/input/datasets/romanatanase/pneumonia-balanced-dataset" >&2
  echo "[RUN] You can still override DATASET_ROOT manually if needed." >&2
  exit 1
fi

"$PYTHON_BIN" -m experiments.run_paper_suite
