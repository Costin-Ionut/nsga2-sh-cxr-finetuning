import os
from pathlib import Path
from typing import Iterable, Optional

from finetune_ga.infra.utils import env_flag

DEFAULT_WINDOWS_ROOT = ''
DEFAULT_KAGGLE_ROOT = '/kaggle/input/datasets/romanatanase/pneumonia-balanced-dataset'
DEFAULT_KAGGLE_ROOT_ALT = '/kaggle/input/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset'
DEFAULT_KAGGLE_DATASET_HANDLE = 'romanatanase/pneumonia-balanced-dataset'
DEFAULT_KAGGLE_DATASET_VERSION: Optional[str] = None
CONFIG_JSON_PATH = Path(__file__).with_name('config.json')
if not CONFIG_JSON_PATH.exists():
    CONFIG_JSON_PATH = Path(__file__).resolve().parents[2] / 'config.json'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')


def _load_config_json_defaults() -> dict:
    if not CONFIG_JSON_PATH.exists():
        return {}
    try:
        import json
        with CONFIG_JSON_PATH.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


_CONFIG_DEFAULTS = _load_config_json_defaults()
_CONFIG_VERBOSE = env_flag('CONFIG_VERBOSE', '0')


def _local_kagglehub_root() -> str:
    """
    Returns the local KaggleHub cache root directory.
    """
    import os
    from pathlib import Path
    return os.environ.get(
        'KAGGLEHUB_CACHE_ROOT',
        str(Path.home() / '.cache' / 'kagglehub')
    ).strip().rstrip('/\\')


def _dataset_handle() -> str:
    return str(_CONFIG_DEFAULTS.get('kaggle_dataset_handle', DEFAULT_KAGGLE_DATASET_HANDLE)).strip()


def _dataset_version() -> Optional[str]:
    raw = _CONFIG_DEFAULTS.get('kaggle_dataset_version', DEFAULT_KAGGLE_DATASET_VERSION)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _runtime_environment() -> str:
    """Return the effective runtime environment: kaggle or local."""
    forced = os.environ.get('RUN_ENV', os.environ.get('DATASET_RUNTIME_ENV', 'auto')).strip().lower()
    if forced in {'kaggle', 'local'}:
        return forced
    if os.path.exists('/kaggle/input') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        return 'kaggle'
    return 'local'


def _allow_unpinned_kaggle_fallback() -> bool:
    """Opt-in unpinned KaggleHub download fallback.

    Disabled by default for publication-safe, deterministic dataset resolution.
    Set ALLOW_UNPINNED_KAGGLE_FALLBACK=1 to permit the fallback behavior.
    """
    return env_flag('ALLOW_UNPINNED_KAGGLE_FALLBACK', '0')


def _deterministic_dataset_required() -> bool:
    """Whether non-deterministic dataset downloads should be rejected by default."""
    return env_flag('REQUIRE_DETERMINISTIC_DATASET', '1')


def _default_dataset_root_source() -> str:
    if os.environ.get('DATASET_ROOT', '').strip():
        return 'env'
    if _runtime_environment() == 'kaggle':
        return 'kaggle_mount_or_config'
    return 'config_or_default'


def _maybe_download_dataset_via_kagglehub() -> Optional[str]:
    """Download a Kaggle dataset only when explicitly allowed and needed.

    Publication-safe default behavior is deterministic:
    - pinned version requested when supported
    - unpinned fallback disabled unless explicitly opted in
    """
    allow_download = env_flag('ALLOW_KAGGLEHUB_DOWNLOAD', '0')
    if not allow_download:
        return None

    dataset_handle = os.environ.get('KAGGLE_DATASET_HANDLE', _dataset_handle()).strip()
    dataset_version_env = os.environ.get('KAGGLE_DATASET_VERSION')
    dataset_version = (dataset_version_env.strip() if dataset_version_env is not None else _dataset_version())
    if not dataset_handle:
        return None

    try:
        import kagglehub
    except (ImportError, ModuleNotFoundError) as exc:
        if _CONFIG_VERBOSE:
            print(f'[CONFIG][WARN] kagglehub unavailable: {type(exc).__name__}: {exc}')
        return None

    kwargs = {'handle': dataset_handle}
    if dataset_version:
        try:
            kwargs['version'] = int(dataset_version)
        except ValueError:
            kwargs['version'] = dataset_version
    try:
        path = kagglehub.dataset_download(**kwargs)
    except TypeError as exc:
        if not _allow_unpinned_kaggle_fallback():
            if _CONFIG_VERBOSE or _deterministic_dataset_required():
                print(
                    '[CONFIG][WARN] Pinned kagglehub download is unsupported by this kagglehub version; '
                    'set ALLOW_UNPINNED_KAGGLE_FALLBACK=1 to permit unpinned non-deterministic download fallback.'
                )
            return None
        print(
            '[WARNING] Falling back to unpinned Kaggle dataset download. '
            'Reproducibility is NOT guaranteed.'
        )
        if _CONFIG_VERBOSE:
            print(
                '[CONFIG][WARN] kagglehub.dataset_download() does not support '
                f'pinning version={dataset_version!r}; falling back to unpinned handle. '
                f'{type(exc).__name__}: {exc}'
            )
        try:
            path = kagglehub.dataset_download(dataset_handle)
        except (OSError, RuntimeError, ValueError) as inner_exc:
            if _CONFIG_VERBOSE:
                print(f'[CONFIG][WARN] kagglehub fallback download failed: {type(inner_exc).__name__}: {inner_exc}')
            return None
    except (OSError, RuntimeError, ValueError) as exc:
        if _CONFIG_VERBOSE:
            print(f'[CONFIG][WARN] kagglehub download failed: {type(exc).__name__}: {exc}')
        return None

    return str(path).rstrip('/\\') if path else None


def _pick_dataset_root() -> str:
    # Dataset resolution prefers explicit env/config paths and existing Kaggle mounts.
    env_root = os.environ.get('DATASET_ROOT', '').strip()
    if env_root:
        return env_root.rstrip('/\\')

    config_root_kaggle = str(_CONFIG_DEFAULTS.get('dataset_root_kaggle', DEFAULT_KAGGLE_ROOT)).strip().rstrip('/\\')
    config_root_windows = str(_CONFIG_DEFAULTS.get('dataset_root_windows', DEFAULT_WINDOWS_ROOT)).strip().rstrip('/\\')
    config_root_kaggle_alt = str(_CONFIG_DEFAULTS.get('dataset_root_kaggle_alt', DEFAULT_KAGGLE_ROOT_ALT)).strip().rstrip('/\\')

    runtime_env = _runtime_environment()
    ordered_candidates = []
    if runtime_env == 'kaggle':
        ordered_candidates.extend([config_root_kaggle, DEFAULT_KAGGLE_ROOT, config_root_kaggle_alt, DEFAULT_KAGGLE_ROOT_ALT])
    else:
        ordered_candidates.extend([config_root_windows, config_root_kaggle, DEFAULT_KAGGLE_ROOT, config_root_kaggle_alt, DEFAULT_KAGGLE_ROOT_ALT])

    unique_candidates = []
    seen = set()
    for candidate in ordered_candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
        if os.path.exists(candidate):
            return candidate

    downloaded_root = _maybe_download_dataset_via_kagglehub()
    if downloaded_root:
        return downloaded_root

    return unique_candidates[0] if unique_candidates else ''


def _find_split_folder(root: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    return None


def _has_train_test_layout(root: Path) -> bool:
    train_dir = _find_split_folder(root, ('train',))
    test_dir = _find_split_folder(root, ('test',))
    return bool(train_dir and test_dir)


def _looks_like_split_root(root: Path) -> bool:
    train_dir = _find_split_folder(root, ('train',))
    val_dir = _find_split_folder(root, ('val', 'valid', 'validation'))
    test_dir = _find_split_folder(root, ('test',))
    return bool(train_dir and val_dir and test_dir)


def _resolve_dataset_root(root: str) -> str:
    """Return the directory that actually contains the split folders.

    Some Kaggle datasets add one extra nesting level after extraction/publication.
    Accept both train/val/test and train/test layouts while searching.
    """
    p = Path(root)
    if _looks_like_split_root(p) or _has_train_test_layout(p):
        return str(p)

    candidates = []
    if p.is_dir():
        try:
            candidates.extend([x for x in p.iterdir() if x.is_dir()])
            for child in list(candidates):
                try:
                    candidates.extend([x for x in child.iterdir() if x.is_dir()])
                except OSError:
                    continue
        except OSError:
            return str(p)
    for cand in candidates:
        if _looks_like_split_root(cand) or _has_train_test_layout(cand):
            return str(cand)
    return str(p)


def _pick_split_folder(root: str, preferred: str, *aliases: str) -> str:
    for name in (preferred,) + aliases:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return os.path.join(root, preferred)


def _count_images_safe(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    total = 0
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            total += len([f for f in os.listdir(class_path) if f.lower().endswith(IMAGE_EXTENSIONS)])
    return total


def _default_split_root(base_root: str) -> str:
    env_out = os.environ.get('DATASET_SPLIT_ROOT', '').strip()
    if env_out:
        return env_out.rstrip('/\\')
    if os.path.exists('/kaggle/working'):
        return '/kaggle/working/chest_xray_auto_split_safe'
    return base_root.rstrip('/\\') + '_auto_split_safe'


def _ensure_auto_split_if_needed(base_root: str) -> str:
    """Resolve the dataset root in a publication-safe, reproducible way."""
    auto_split = env_flag('AUTO_SPLIT_DATASET', '0')
    resolved_root = _resolve_dataset_root(base_root)
    if not auto_split:
        return resolved_root

    resolved_path = Path(resolved_root)

    if not (_looks_like_split_root(resolved_path) or _has_train_test_layout(resolved_path)):
        return resolved_root

    output_root = _default_split_root(base_root)
    output_path = Path(output_root)
    report_path = output_path / 'SPLIT_REPORT.md'
    force_rebuild = env_flag('FORCE_REBUILD_SPLIT', '0')

    if _looks_like_split_root(output_path) and report_path.exists() and not force_rebuild:
        if _CONFIG_VERBOSE:
            print(f'[AUTO_SPLIT] Reusing frozen derived split: {output_path}')
        return str(output_path)

    try:
        from finetune_ga.data.split import build_split_map, copy_split, write_report
    except (ImportError, ModuleNotFoundError) as exc:
        if _CONFIG_VERBOSE:
            print(f'[AUTO_SPLIT][WARN] Could not import split utilities: {type(exc).__name__}: {exc}')
        return resolved_root

    split_seed = int(os.environ.get('DATASET_SPLIT_SEED', '42'))
    split_mode = os.environ.get('DATASET_SPLIT_MODE', 'copy').strip().lower()
    if split_mode not in {'copy', 'hardlink'}:
        split_mode = 'copy'

    if output_path.exists() and (force_rebuild or not report_path.exists()):
        import shutil
        shutil.rmtree(output_path, ignore_errors=True)

    if not output_path.exists():
        if _CONFIG_VERBOSE:
            print(f'[AUTO_SPLIT] Building frozen leakage-safe split from source dataset: {resolved_root}')
            print(f'[AUTO_SPLIT] Output split root: {output_path}')
        split_map, source_layout = build_split_map(resolved_path, seed=split_seed)
        copy_split(split_map, output_path, mode=split_mode)
        write_report(split_map, output_path, resolved_path, split_seed, split_mode, source_layout)

    return str(output_path)


def _initial_dataset_root_without_heavy_io() -> str:
    """Return a cheap dataset-root candidate without scanning the filesystem.

    Full dataset resolution can inspect candidate folders, download via KaggleHub
    when explicitly enabled, count images, and optionally build an auto-split.
    Those operations are deferred until the dataset is actually needed through
    refresh_dataset_config(), get_dataset_paths(), get_dataset_counts(), or
    validate_dataset_layout().
    """
    env_root = os.environ.get('DATASET_ROOT', '').strip()
    if env_root:
        return env_root.rstrip('/\\')

    forced = os.environ.get('RUN_ENV', os.environ.get('DATASET_RUNTIME_ENV', 'auto')).strip().lower()
    looks_like_kaggle = forced == 'kaggle' or bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))

    if looks_like_kaggle:
        root = str(_CONFIG_DEFAULTS.get('dataset_root_kaggle', DEFAULT_KAGGLE_ROOT)).strip().rstrip('/\\')
    else:
        root = str(_CONFIG_DEFAULTS.get('dataset_root_windows', DEFAULT_WINDOWS_ROOT)).strip().rstrip('/\\')

    return root or DEFAULT_KAGGLE_ROOT


_INITIAL_LAZY_DATASET_ROOT = _initial_dataset_root_without_heavy_io()

_RUNTIME_DATASET_STATE = {
    'dataset_root': _INITIAL_LAZY_DATASET_ROOT,
    'train_folder': '',
    'val_folder': '',
    'test_folder': '',
    'num_train_images': 0,
    'num_val_images': 0,
    'num_test_images': 0,
    'is_resolved': False,
}


def _resolve_runtime_dataset_state(dataset_root: str) -> dict:
    resolved_root = _ensure_auto_split_if_needed(dataset_root)
    train_folder = _pick_split_folder(resolved_root, 'train')
    val_folder = _pick_split_folder(resolved_root, 'val', 'valid', 'validation')
    test_folder = _pick_split_folder(resolved_root, 'test')
    return {
        'dataset_root': resolved_root,
        'train_folder': train_folder,
        'val_folder': val_folder,
        'test_folder': test_folder,
        'num_train_images': _count_images_safe(train_folder),
        'num_val_images': _count_images_safe(val_folder),
        'num_test_images': _count_images_safe(test_folder),
        'is_resolved': True,
    }


def _ensure_dataset_state_initialized() -> dict:
    if not _RUNTIME_DATASET_STATE['is_resolved']:
        return refresh_dataset_config()
    return dict(_RUNTIME_DATASET_STATE)


def refresh_dataset_config() -> dict:
    env_root = os.environ.get('DATASET_ROOT', '').strip()
    current_root = str(_RUNTIME_DATASET_STATE.get('dataset_root', '')).strip()

    if env_root:
        dataset_root = env_root.rstrip('/\\')
    elif not current_root or current_root == _INITIAL_LAZY_DATASET_ROOT:
        # Do the full candidate scan only on demand. This preserves previous
        # behavior for normal runs while keeping module import cheap.
        dataset_root = _pick_dataset_root()
    else:
        # Tests and advanced callers may inject a runtime dataset root directly.
        dataset_root = current_root.rstrip('/\\')

    state = _resolve_runtime_dataset_state(dataset_root)
    _RUNTIME_DATASET_STATE.update(state)
    try:
        from finetune_ga.data.loader import clear_dataset_manifest_cache
        clear_dataset_manifest_cache()
    except (ImportError, ModuleNotFoundError):
        pass

    globals()['DATASET_ROOT'] = state['dataset_root']
    globals()['TRAIN_FOLDER'] = state['train_folder']
    globals()['VAL_FOLDER'] = state['val_folder']
    globals()['TEST_FOLDER'] = state['test_folder']
    globals()['NUM_TRAIN_IMAGES'] = int(state['num_train_images'])
    globals()['NUM_VAL_IMAGES'] = int(state['num_val_images'])
    globals()['NUM_TEST_IMAGES'] = int(state['num_test_images'])
    return dict(state)


def get_dataset_paths(refresh: bool = False) -> dict:
    if refresh:
        state = refresh_dataset_config()
    else:
        state = _ensure_dataset_state_initialized()
    return {
        'dataset_root': state['dataset_root'],
        'train_folder': state['train_folder'] or os.path.join(state['dataset_root'], 'train'),
        'val_folder': state['val_folder'] or os.path.join(state['dataset_root'], 'val'),
        'test_folder': state['test_folder'] or os.path.join(state['dataset_root'], 'test'),
    }


def get_dataset_counts(refresh: bool = False) -> dict:
    if refresh:
        state = refresh_dataset_config()
    else:
        state = _ensure_dataset_state_initialized()
    return {
        'train': int(state['num_train_images']),
        'val': int(state['num_val_images']),
        'test': int(state['num_test_images']),
    }


def dataset_config_snapshot(refresh: bool = False) -> dict:
    paths = get_dataset_paths(refresh=refresh)
    counts = get_dataset_counts(refresh=False)
    return {
        'dataset_root': paths['dataset_root'],
        'train_folder': paths['train_folder'],
        'val_folder': paths['val_folder'],
        'test_folder': paths['test_folder'],
        'num_train_images': int(counts['train']),
        'num_val_images': int(counts['val']),
        'num_test_images': int(counts['test']),
        'auto_split_dataset': bool(env_flag('AUTO_SPLIT_DATASET', '0')),
        'dataset_handle': os.environ.get('KAGGLE_DATASET_HANDLE', _dataset_handle()).strip(),
        'dataset_version': (lambda v: v.strip() if v is not None else None)(os.environ.get('KAGGLE_DATASET_VERSION') or _dataset_version()),
        'dataset_root_source': _default_dataset_root_source(),
        'runtime_environment': _runtime_environment(),
        'deterministic_dataset_required': bool(_deterministic_dataset_required()),
        'allow_unpinned_kaggle_fallback': bool(_allow_unpinned_kaggle_fallback()),
        'allow_kagglehub_download': bool(env_flag('ALLOW_KAGGLEHUB_DOWNLOAD', '0')),
        'kagglehub_cache_root': _local_kagglehub_root(),
        'kaggle_dataset_handle': _dataset_handle(),
        'kaggle_dataset_version': _dataset_version(),
    }


def _log_dataset_config() -> None:
    if not _CONFIG_VERBOSE:
        return
    snap = dataset_config_snapshot(refresh=True)
    print(f"[CONFIG] DATASET_ROOT={snap['dataset_root']}")
    print(f"[CONFIG] TRAIN_FOLDER={snap['train_folder']}")
    print(f"[CONFIG] VAL_FOLDER={snap['val_folder']}")
    print(f"[CONFIG] TEST_FOLDER={snap['test_folder']}")
    print(
        f"[CONFIG] NUM_TRAIN_IMAGES={snap['num_train_images']} | "
        f"NUM_VAL_IMAGES={snap['num_val_images']} | NUM_TEST_IMAGES={snap['num_test_images']}"
    )
    print(
        f"[CONFIG] DATASET_HANDLE={snap['dataset_handle']} | "
        f"DATASET_VERSION={snap['dataset_version']} | SOURCE={snap['dataset_root_source']}"
    )


_INITIAL_DATASET_SNAPSHOT = dict(_RUNTIME_DATASET_STATE)
DATASET_ROOT = _RUNTIME_DATASET_STATE['dataset_root']
TRAIN_FOLDER = os.path.join(DATASET_ROOT, 'train')
VAL_FOLDER = os.path.join(DATASET_ROOT, 'val')
TEST_FOLDER = os.path.join(DATASET_ROOT, 'test')
SEARCH_IMG_SIZE = int(os.environ.get('SEARCH_IMG_SIZE', _CONFIG_DEFAULTS.get('search_img_size', 160)))
FINAL_IMG_SIZE = int(os.environ.get('FINAL_IMG_SIZE', _CONFIG_DEFAULTS.get('final_img_size', 224)))
IMG_SIZE = SEARCH_IMG_SIZE
BATCH_SIZE = 32
NUM_TRAIN_IMAGES = 0
NUM_VAL_IMAGES = 0
NUM_TEST_IMAGES = 0
PREFERRED_NUM_GPUS = int(os.environ.get('NUM_GPUS', '2'))
ENABLE_XLA = env_flag('ENABLE_XLA', '1')
ENABLE_MIXED_PRECISION = env_flag('ENABLE_MIXED_PRECISION', '1')
ENABLE_DATASET_CACHE = env_flag('ENABLE_DATASET_CACHE', '1')
CACHE_ON_DISK = env_flag('CACHE_ON_DISK', '1')
TF_DATA_CACHE_DIR = os.environ.get('TF_DATA_CACHE_DIR', '/tmp/finetuned_ga_tf_cache')

if _CONFIG_VERBOSE:
    _log_dataset_config()


def validate_dataset_layout(strict: bool = True, refresh: bool = False) -> bool:
    """Validate that train/val/test folders exist and contain images."""
    snap = dataset_config_snapshot(refresh=refresh)
    problems = []
    for split_name, folder, count in (
        ('train', snap['train_folder'], snap['num_train_images']),
        ('val', snap['val_folder'], snap['num_val_images']),
        ('test', snap['test_folder'], snap['num_test_images']),
    ):
        if not os.path.isdir(folder):
            problems.append(f'{split_name} folder missing: {folder}')
        elif int(count) <= 0:
            problems.append(f'{split_name} folder has no images: {folder}')

    if problems and strict:
        available_kaggle_inputs = []
        if os.path.isdir("/kaggle/input"):
            try:
                available_kaggle_inputs = os.listdir("/kaggle/input")
            except OSError:
                available_kaggle_inputs = []

        raise FileNotFoundError(
            "Invalid dataset layout:\n"
            "- " + "\n- ".join(problems) +
            "\n\nExpected structure:\n"
            f"{snap['dataset_root']}/train/\n"
            f"{snap['dataset_root']}/val/ or {snap['dataset_root']}/valid/ or {snap['dataset_root']}/validation/\n"
            f"{snap['dataset_root']}/test/\n"
            "\nHow to fix:\n"
            "1. Set DATASET_ROOT explicitly:\n"
            "   export DATASET_ROOT=/path/to/your/dataset\n\n"
            "2. On Kaggle, check the mounted dataset folder:\n"
            "   python - <<'PY'\n"
            "   import os\n"
            "   print(os.listdir('/kaggle/input'))\n"
            "   PY\n\n"
            "3. If using the expected Kaggle datasets, add one of:\n"
            "   - costinraduionut/pneumonia-balanced-dataset\n"
            "   - yusufmurtaza01/chest-xray-pneumonia-balanced-dataset\n\n"
            "4. Optional auto-download:\n"
            "   export ALLOW_KAGGLEHUB_DOWNLOAD=1\n\n"
            f"Available /kaggle/input folders: {available_kaggle_inputs}\n"
            f"[dataset_root={snap['dataset_root']}]\n"
            f"[dataset_root_source={snap['dataset_root_source']}]\n"
            f"[runtime_environment={snap['runtime_environment']}]\n"
            f"[dataset_handle={snap['dataset_handle']}]\n"
            f"[dataset_version={snap['dataset_version']}]"
        )
    return not problems
