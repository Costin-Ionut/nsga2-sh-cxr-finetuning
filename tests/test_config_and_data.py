import json
from pathlib import Path

import pytest

from finetune_ga.data.split import (
    build_split_map,
    collect_images,
    copy_split,
    detect_layout,
    write_report,
)
from finetune_ga.infra import config as cfgmod
from finetune_ga.infra.experiment_config import get_config_path, validate_config
from finetune_ga.infra.io_utils import append_jsonl, read_jsonl


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"img")


@pytest.fixture
def toy_dataset(tmp_path):
    root = tmp_path / "dataset"
    for split in ["train", "val", "test"]:
        for cls in ["normal", "pneumonia"]:
            for i in range(4):
                _touch(root / split / cls / f"{split}_{cls}_{i}.jpg")
    return root


def test_detect_layout_supports_train_val_test(toy_dataset):
    assert detect_layout(toy_dataset) == "train_val_test"


def test_collect_images_reads_single_split_only(toy_dataset):
    out = collect_images(toy_dataset, split_name="train")
    assert sorted(out) == ["normal", "pneumonia"]
    assert len(out["normal"]) == 4
    assert all("/train/" in str(p).replace("\\", "/") for p in out["normal"])


def test_build_split_map_preserves_existing_train_val_test_without_reshuffle(toy_dataset):
    split_map_1, layout_1 = build_split_map(toy_dataset, seed=42)
    split_map_2, layout_2 = build_split_map(toy_dataset, seed=7)
    assert layout_1 == layout_2 == "train_val_test_preserved_official_split"
    assert split_map_1 == split_map_2
    assert len(split_map_1["train"]["normal"]) == 4
    assert len(split_map_1["val"]["normal"]) == 4
    assert len(split_map_1["test"]["normal"]) == 4


def test_copy_split_and_write_report_materialize_expected_files(toy_dataset, tmp_path):
    split_map, layout = build_split_map(toy_dataset, seed=7)
    out_root = tmp_path / "out"
    copy_split(split_map, out_root, mode="copy")
    write_report(split_map, out_root, toy_dataset, 7, "copy", layout)
    assert (out_root / "train" / "normal").is_dir()
    assert (out_root / "val.txt").exists()
    report = (out_root / "SPLIT_REPORT.md").read_text(encoding="utf-8")
    assert "preserve existing official train/val/test split" in report
    assert "| train | normal |" in report


def test_resolve_dataset_root_discovers_nested_split_folder(tmp_path):
    nested = tmp_path / "outer" / "inner_dataset"
    for split in ["train", "test"]:
        for cls in ["a", "b"]:
            _touch(nested / split / cls / "x.jpg")
    resolved = cfgmod._resolve_dataset_root(str(tmp_path / "outer"))
    assert resolved == str(nested)


def test_validate_dataset_layout_can_be_strict_or_non_strict(tmp_path, monkeypatch):
    root = tmp_path / "bad_dataset"
    (root / "train" / "a").mkdir(parents=True)
    monkeypatch.setitem(cfgmod._RUNTIME_DATASET_STATE, "dataset_root", str(root))
    monkeypatch.setitem(cfgmod._RUNTIME_DATASET_STATE, "is_resolved", False)
    assert cfgmod.validate_dataset_layout(strict=False, refresh=True) is False
    with pytest.raises(FileNotFoundError):
        cfgmod.validate_dataset_layout(strict=True, refresh=True)


def test_get_config_path_prefers_explicit_relative_file(tmp_path, monkeypatch):
    config_file = tmp_path / "my_config.json"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert get_config_path("my_config.json") == str(config_file.resolve())


def test_validate_config_rejects_duplicate_model_names():
    cfg = {
        "num_classes": 2,
        "model_names": ["m", "m"],
        "budgets": [{"name": "B0", "e1": 1, "e2": 0, "steps_factor": 1.0, "reps": 1}],
        "out_dir": "out",
        "search_space": {},
    }
    with pytest.raises(ValueError, match="model_names must be unique"):
        validate_config(cfg)


def test_validate_config_requires_new_seed_fields_only():
    cfg = {
        "num_classes": 2,
        "model_names": ["m", "r"],
        "budgets": [{"name": "B0", "e1": 1, "e2": 0, "steps_factor": 1.0, "reps": 1}],
        "out_dir": "out",
        "search_space": {},
        "search_seeds": [9],
        "final_seeds": [9],
        "selection_source_seed": 9,
    }
    out = validate_config(cfg)
    assert out["search_seeds"] == [9]
    assert out["final_seeds"] == [9]
    assert out["selection_source_seed"] == 9


def test_append_jsonl_invalidates_cache(tmp_path):
    path = tmp_path / "cache.jsonl"
    append_jsonl(str(path), {"x": 1})
    assert read_jsonl(str(path)) == [{"x": 1}]
    append_jsonl(str(path), {"x": 2})
    assert read_jsonl(str(path)) == [{"x": 1}, {"x": 2}]


def test_build_split_map_preserves_test_and_derives_val_from_train(tmp_path):
    root = tmp_path / "dataset_tt"
    for split, count in [("train", 10), ("test", 3)]:
        for cls in ["normal", "pneumonia"]:
            for i in range(count):
                _touch(root / split / cls / f"{split}_{cls}_{i}.jpg")
    split_map, layout = build_split_map(root, seed=42)
    assert layout == "train_test_preserved_test_split_train_into_train_val"
    assert len(split_map["test"]["normal"]) == 3
    assert len(split_map["train"]["normal"]) == 9
    assert len(split_map["val"]["normal"]) == 1


def test_cache_path_changes_when_dataset_root_changes(tmp_path):
    root_a = tmp_path / "dataset_a"
    root_b = tmp_path / "dataset_b"
    for root in [root_a, root_b]:
        for split in ["train", "val", "test"]:
            for cls in ["normal", "pneumonia"]:
                _touch(root / split / cls / f"{split}_{cls}.jpg")

    state = cfgmod._resolve_runtime_dataset_state(str(root_a))
    cfgmod._RUNTIME_DATASET_STATE.update(state)
    cfgmod._RUNTIME_DATASET_STATE["is_resolved"] = True

    from finetune_ga.data.loader import _cache_path

    cache_a = _cache_path("train", target_size=224, seed=42)

    state = cfgmod._resolve_runtime_dataset_state(str(root_b))
    cfgmod._RUNTIME_DATASET_STATE.update(state)
    cfgmod._RUNTIME_DATASET_STATE["is_resolved"] = True

    cache_b = _cache_path("train", target_size=224, seed=42)
    assert cache_a != cache_b


def test_refresh_dataset_config_mutates_public_module_constants(tmp_path, monkeypatch):
    original_dataset_root = cfgmod.DATASET_ROOT
    root = tmp_path / "dataset"
    for split in ["train", "val", "test"]:
        for cls in ["normal", "pneumonia"]:
            _touch(root / split / cls / f"{split}_{cls}.jpg")

    monkeypatch.setitem(cfgmod._RUNTIME_DATASET_STATE, "dataset_root", str(root))
    monkeypatch.setitem(cfgmod._RUNTIME_DATASET_STATE, "is_resolved", False)
    out = cfgmod.refresh_dataset_config()

    assert out["dataset_root"] == str(root)
    assert cfgmod.DATASET_ROOT != original_dataset_root
    assert cfgmod.DATASET_ROOT == str(root)
    assert cfgmod.TRAIN_FOLDER == str(root / 'train')
    assert cfgmod.VAL_FOLDER == str(root / 'val')
    assert cfgmod.TEST_FOLDER == str(root / 'test')



def test_pick_dataset_root_accepts_alt_kaggle_root(monkeypatch):
    def fake_exists(path):
        return path == cfgmod.DEFAULT_KAGGLE_ROOT_ALT

    monkeypatch.setattr(cfgmod.os.path, 'exists', fake_exists)
    monkeypatch.delenv('DATASET_ROOT', raising=False)
    assert cfgmod._pick_dataset_root() == cfgmod.DEFAULT_KAGGLE_ROOT_ALT


def test_dataset_manifest_token_changes_when_file_content_changes_in_place(tmp_path):
    root = tmp_path / "dataset"
    for split in ["train", "val", "test"]:
        for cls in ["normal", "pneumonia"]:
            _touch(root / split / cls / f"{split}_{cls}.jpg")

    state = cfgmod._resolve_runtime_dataset_state(str(root))
    cfgmod._RUNTIME_DATASET_STATE.update(state)
    cfgmod._RUNTIME_DATASET_STATE["is_resolved"] = True

    from finetune_ga.data.loader import _dataset_manifest_token

    token1 = _dataset_manifest_token("train")
    target = root / "train" / "normal" / "train_normal.jpg"
    target.write_bytes(b"changed-content")
    token2 = _dataset_manifest_token("train")
    assert token1 != token2


def test_validate_config_supports_distinct_search_and_final_seeds():
    cfg = {
        "num_classes": 2,
        "model_names": ["m", "r"],
        "budgets": [{"name": "B0", "e1": 1, "e2": 0, "steps_factor": 1.0, "reps": 1}],
        "out_dir": "out",
        "search_space": {},
        "search_seeds": [42],
        "final_seeds": [42, 999, 1337],
        "selection_source_seed": 42,
    }
    out = validate_config(cfg)
    assert out["search_seeds"] == [42]
    assert out["final_seeds"] == [42, 999, 1337]
    assert out["search_seeds"] == [42]
    assert out["selection_source_seed"] == 42


def test_validate_config_rejects_selection_seed_outside_search_seeds():
    cfg = {
        "num_classes": 2,
        "model_names": ["m", "r"],
        "budgets": [{"name": "B0", "e1": 1, "e2": 0, "steps_factor": 1.0, "reps": 1}],
        "out_dir": "out",
        "search_space": {},
        "search_seeds": [42],
        "final_seeds": [42, 999, 1337],
        "selection_source_seed": 999,
    }
    with pytest.raises(ValueError, match="selection_source_seed"):
        validate_config(cfg)


def test_validate_config_rejects_legacy_seed_keys():
    cfg = {
        "num_classes": 2,
        "model_names": ["m", "r"],
        "budgets": [{"name": "B0", "e1": 1, "e2": 0, "steps_factor": 1.0, "reps": 1}],
        "out_dir": "out",
        "search_space": {},
        "seed": 9,
        "search_seeds": [9],
        "final_seeds": [9],
        "selection_source_seed": 9,
    }
    with pytest.raises(ValueError, match="Legacy seed keys are not allowed"):
        validate_config(cfg)


def test_run_utils_missing_keys_have_clear_errors():
    from finetune_ga.infra.run_utils import get_search_seeds, get_final_seeds, get_selection_source_seed

    for fn, key in [
        (get_search_seeds, 'search_seeds'),
        (get_final_seeds, 'final_seeds'),
        (get_selection_source_seed, 'selection_source_seed'),
    ]:
        try:
            fn({})
            assert False, f"Expected KeyError for {key}"
        except KeyError as exc:
            assert key in str(exc)


def test_read_jsonl_cache_refreshes_when_file_changes_in_place(tmp_path):
    path = tmp_path / "cache_refresh.jsonl"
    path.write_text('{"x": 1}\n', encoding='utf-8')
    assert read_jsonl(str(path)) == [{"x": 1}]
    path.write_text('{"x": 1}\n{"x": 2}\n', encoding='utf-8')
    assert read_jsonl(str(path)) == [{"x": 1}, {"x": 2}]


def test_read_jsonl_strict_raises_on_malformed_row(tmp_path):
    path = tmp_path / "bad_rows.jsonl"
    path.write_text('{"a": 1}\nnot-json\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Malformed JSONL row'):
        read_jsonl(str(path), strict=True)
