"""Lazy TensorFlow runtime bootstrap per process.

This module intentionally avoids importing TensorFlow or building any
strategy at import time. Workers can set CUDA_VISIBLE_DEVICES before any
training code touches TensorFlow, then the first runtime access will import
and configure TensorFlow inside that process.
"""
from __future__ import annotations

import os
import warnings
from types import ModuleType
from typing import Any

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')

from finetune_ga.infra.config import ENABLE_XLA, ENABLE_MIXED_PRECISION

TF_RUNTIME_CONFIG_EXCEPTIONS = (RuntimeError, ValueError, TypeError, AttributeError)

_TF_MODULE: ModuleType | None = None
_RUNTIME_CONFIGURED = False
_STRATEGY: Any | None = None
_SOFTMAX_BINARY_AUC_CLASS: type | None = None
_IMPORT_FAILED = False


class _NoopStrategy:
    num_replicas_in_sync = 1

    class _Scope:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def scope(self):
        return self._Scope()


def _import_tensorflow_or_none() -> ModuleType | None:
    global _TF_MODULE, _IMPORT_FAILED
    if _TF_MODULE is not None:
        return _TF_MODULE
    if _IMPORT_FAILED:
        return None
    try:
        import tensorflow as tensorflow_module
    except ModuleNotFoundError:
        _IMPORT_FAILED = True
        return None
    tensorflow_module.get_logger().setLevel('ERROR')
    _TF_MODULE = tensorflow_module
    return _TF_MODULE


def get_tf_module_or_none() -> ModuleType | None:
    return _import_tensorflow_or_none()


def _require_tensorflow() -> None:
    if get_tf_module_or_none() is None:
        raise ModuleNotFoundError(
            'TensorFlow is required for training-time functionality but is not installed.'
        )


def _get_mixed_precision_module():
    tensorflow_module = get_tf_module_or_none()
    if tensorflow_module is None:
        return None
    try:
        return tensorflow_module.keras.mixed_precision
    except AttributeError:
        return None


def apply_precision_policy() -> None:
    """Apply the configured Keras dtype policy, even after clear_session()."""
    mixed_precision = _get_mixed_precision_module()
    if mixed_precision is None:
        return
    try:
        mixed_precision.set_global_policy('mixed_float16' if ENABLE_MIXED_PRECISION else 'float32')
    except TF_RUNTIME_CONFIG_EXCEPTIONS as exc:
        warnings.warn(f"Could not configure precision policy: {type(exc).__name__}: {exc}")


def configure_runtime() -> None:
    global _RUNTIME_CONFIGURED
    tensorflow_module = get_tf_module_or_none()
    if tensorflow_module is None:
        return
    if _RUNTIME_CONFIGURED:
        # Keras clear_session() may reset only the dtype policy; keep it fresh.
        apply_precision_policy()
        return

    if ENABLE_XLA:
        try:
            tensorflow_module.config.optimizer.set_jit(True)
        except TF_RUNTIME_CONFIG_EXCEPTIONS as exc:
            warnings.warn(f"Could not enable XLA JIT: {type(exc).__name__}: {exc}")

    try:
        gpus = tensorflow_module.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tensorflow_module.config.experimental.set_memory_growth(gpu, True)
            except TF_RUNTIME_CONFIG_EXCEPTIONS as exc:
                warnings.warn(
                    f"Could not enable memory growth for GPU {gpu}: {type(exc).__name__}: {exc}"
                )
    except TF_RUNTIME_CONFIG_EXCEPTIONS as exc:
        warnings.warn(f"Could not enumerate GPUs: {type(exc).__name__}: {exc}")

    apply_precision_policy()

    _RUNTIME_CONFIGURED = True


def _visible_gpu_count() -> int:
    tensorflow_module = get_tf_module_or_none()
    if tensorflow_module is None:
        return 0
    configure_runtime()
    try:
        return len(tensorflow_module.config.list_logical_devices('GPU'))
    except TF_RUNTIME_CONFIG_EXCEPTIONS:
        return 0


def _build_strategy():
    tensorflow_module = get_tf_module_or_none()
    if tensorflow_module is None:
        return _NoopStrategy()

    configure_runtime()
    multi_gpu_mode = os.environ.get('MULTI_GPU_MODE', 'auto').strip().lower()
    gpu_count = _visible_gpu_count()

    if multi_gpu_mode == 'single' or gpu_count <= 1:
        return tensorflow_module.distribute.get_strategy()

    try:
        return tensorflow_module.distribute.MirroredStrategy()
    except TF_RUNTIME_CONFIG_EXCEPTIONS as exc:
        warnings.warn(f"Falling back to default TF strategy: {type(exc).__name__}: {exc}")
        return tensorflow_module.distribute.get_strategy()


def get_strategy():
    global _STRATEGY
    if _STRATEGY is None:
        _STRATEGY = _build_strategy()
    return _STRATEGY


class _StrategyProxy:
    def scope(self):
        return get_strategy().scope()

    @property
    def num_replicas_in_sync(self):
        return getattr(get_strategy(), 'num_replicas_in_sync', 1)


STRATEGY = _StrategyProxy()


def get_softmax_binary_auc_class():
    global _SOFTMAX_BINARY_AUC_CLASS
    if _SOFTMAX_BINARY_AUC_CLASS is not None:
        return _SOFTMAX_BINARY_AUC_CLASS

    tensorflow_module = get_tf_module_or_none()
    if tensorflow_module is None:
        class _UnavailableSoftmaxBinaryAUC:
            def __init__(self, *args, **kwargs):
                _require_tensorflow()
        _SOFTMAX_BINARY_AUC_CLASS = _UnavailableSoftmaxBinaryAUC
        return _SOFTMAX_BINARY_AUC_CLASS

    configure_runtime()

    class _SoftmaxBinaryAUC(tensorflow_module.keras.metrics.AUC):
        def __init__(self, name='auc', curve='ROC', num_thresholds=1000, **kwargs):
            super().__init__(name=name, curve=curve, num_thresholds=int(num_thresholds), **kwargs)

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tensorflow_module.cast(tensorflow_module.reshape(y_true, [-1]), tensorflow_module.float32)
            y_pred = tensorflow_module.cast(tensorflow_module.convert_to_tensor(y_pred), tensorflow_module.float32)
            if y_pred.shape.rank is not None and y_pred.shape.rank >= 2:
                y_pred_pos = y_pred[:, 1]
            else:
                y_pred_pos = y_pred
            y_pred_pos = tensorflow_module.reshape(y_pred_pos, [-1])
            return super().update_state(y_true, y_pred_pos, sample_weight=sample_weight)

    _SOFTMAX_BINARY_AUC_CLASS = _SoftmaxBinaryAUC
    return _SOFTMAX_BINARY_AUC_CLASS


class _SoftmaxBinaryAUCProxy:
    def __call__(self, *args, **kwargs):
        return get_softmax_binary_auc_class()(*args, **kwargs)


SoftmaxBinaryAUC = _SoftmaxBinaryAUCProxy()


class _TensorFlowProxy:
    def __bool__(self):
        return get_tf_module_or_none() is not None

    def __getattr__(self, item):
        tensorflow_module = get_tf_module_or_none()
        if tensorflow_module is None:
            raise AttributeError(item)
        configure_runtime()
        return getattr(tensorflow_module, item)


# Backward-compatible lazy handle for existing imports.
tf = _TensorFlowProxy()
