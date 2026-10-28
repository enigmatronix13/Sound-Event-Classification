"""Synthetic-audio regression tests. These do not measure DataSEC accuracy."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from src.data_prep import (audit_audio, build_table, create_split, infer_parent,
                           read_manifest, validate_manifest)
from src.dataset import AudioConfig, LogMelDataset
from src.evaluation import classification_metrics
from src.model import EffNetB0Audio
from src.train_efficientnet import compute_norm_stats, debug_subset, make_loader


@pytest.fixture
def audio_table(tmp_path):
    rng = np.random.default_rng(15)
    rows = []
    for label_id, label in enumerate(["Bells", "Birds"]):
        for i, split in enumerate(["train", "train", "val", "test"]):
            relative = f"{label}/{i}.wav"
            path = tmp_path / relative
            path.parent.mkdir(exist_ok=True)
            sf.write(path, rng.normal(0, 0.1, 4000), 16000, subtype="FLOAT")
            rows.append(dict(path=relative, label=label, label_id=label_id, split=split))
    return tmp_path, pd.DataFrame(rows), {"Bells": 0, "Birds": 1}


def test_nested_subclasses_map_to_parent():
    assert infer_parent("DATASEC/Propeller aircrafts/Helicopters/a.wav") == "Propeller aircrafts"
    assert infer_parent("DATASEC/Grinder/a.wav") == "Workshop"
    with pytest.raises(ValueError, match="Unknown/conflicting"):
        infer_parent("Bells/Birds/a.wav")


def test_scan_and_missing_file_fail_closed(audio_table):
    root, df, _ = audio_table
    assert set(build_table(root).label) == {"Bells", "Birds"}
    (root / df.iloc[0].path).unlink()
    with pytest.raises(ValueError, match="Missing audio"):
        validate_manifest(df, {"Bells": 0, "Birds": 1}, root, 2)


def test_shared_sincnet_manifest_keeps_assignments(audio_table):
    root, df, labels = audio_table
    shared = df.rename(columns={"path": "relative_path", "label_id": "target"})
    shared.to_csv(root / "shared.csv", index=False)
    (root / "labels.json").write_text(json.dumps(labels))
    actual, mapping = read_manifest(root / "shared.csv", root / "labels.json", root, 2)
    assert actual.split.tolist() == df.split.tolist()
    assert actual.label_id.tolist() == df.label_id.tolist()
    assert mapping == labels


@pytest.mark.parametrize("problem", ["duplicate_path", "label_id", "missing_class", "group", "escape"])
def test_bad_manifest_rejected(audio_table, problem):
    root, df, labels = audio_table
    if problem == "duplicate_path":
        df.loc[1, "path"] = df.loc[0, "path"]
    elif problem == "label_id":
        df.loc[0, "label_id"] = 1
    elif problem == "missing_class":
        df = df.drop(df[(df.label == "Bells") & (df.split == "val")].index)
    elif problem == "group":
        df["group_id"] = "same-session"
    elif problem == "escape":
        df.loc[0, "path"] = "../outside.wav"
    with pytest.raises(ValueError):
        validate_manifest(df, labels, root, 2)


def test_exact_duplicates_across_splits_rejected(audio_table):
    root, df, _ = audio_table
    (root / "Bells/3.wav").write_bytes((root / "Bells/0.wav").read_bytes())
    with pytest.raises(ValueError, match="Duplicate audio leakage"):
        audit_audio(df, root)


def test_nonfinite_audio_rejected(audio_table):
    root, df, _ = audio_table
    sf.write(root / "Bells/0.wav", np.array([float("nan"), 1.0]), 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="non-finite"):
        audit_audio(df, root)


def test_conflicting_duplicate_labels_rejected(audio_table):
    root, df, _ = audio_table
    (root / "Birds/0.wav").write_bytes((root / "Bells/0.wav").read_bytes())
    with pytest.raises(ValueError, match="conflicting labels"):
        audit_audio(df, root)


def test_split_reproducible_and_disjoint():
    df = pd.DataFrame({"path": [f"{i}.wav" for i in range(80)], "label": ["a"] * 40 + ["b"] * 40})
    a, b = create_split(df), create_split(df)
    pd.testing.assert_frame_equal(a, b)
    assert a.path.nunique() == len(df)
    assert a.split.value_counts().to_dict() == {"train": 56, "val": 12, "test": 12}


def test_eval_features_and_short_batch(audio_table):
    root, df, _ = audio_table
    cfg = AudioConfig()
    ds = LogMelDataset(df, root, cfg)
    x, _ = ds[0]
    assert x.shape == (1, 128, 501)
    assert torch.isfinite(x).all()
    assert torch.equal(x, ds[0][0])
    loader = make_loader(df.iloc[:1], root, cfg, 0, 1, 32, 0, torch.device("cpu"), train=True)
    assert len(loader) == 1
    assert next(iter(loader))[0].shape[0] == 1


def test_normalization_only_accepts_training(audio_table):
    root, df, _ = audio_table
    with pytest.raises(ValueError, match="training-only"):
        compute_norm_stats(df, root, AudioConfig(), 0)
    mean, std = compute_norm_stats(df[df.split == "train"], root, AudioConfig(), 0)
    assert np.isfinite(mean) and std > 0


def test_macro_metrics_include_absent_classes():
    metrics = classification_metrics(np.array([0, 0]), np.array([0, 0]), 2)
    assert metrics["accuracy"] == 1
    assert metrics["macro_f1"] == 0.5


def test_debug_subset_keeps_every_class(audio_table):
    _, df, _ = audio_table
    assert set(debug_subset(df, 2, 42).label) == {"Bells", "Birds"}
    with pytest.raises(ValueError):
        debug_subset(df, 1, 42)


def test_real_efficientnet_shape_and_backward():
    torch.set_num_threads(2)
    model = EffNetB0Audio(22, pretrained=False)
    x = torch.randn(1, 1, 128, 501)
    out = model(x)
    assert out.shape == (1, 22)
    torch.nn.functional.cross_entropy(out, torch.tensor([3])).backward()
    assert torch.isfinite(model.net.classifier[1].weight.grad).all()
    groups = model.param_groups(1e-3, 1e-4, 0.01)
    assert not ({id(p) for p in groups[0]["params"]} & {id(p) for p in groups[1]["params"]})


def test_training_checkpoint_evaluation_and_prediction(audio_table):
    """One real CPU epoch, serialization, fixed test evaluation, and standalone prediction."""
    root, df, labels = audio_table
    df.to_csv(root / "splits.csv", index=False)
    (root / "label_map.json").write_text(json.dumps(labels))
    run = root / "run"
    base = [sys.executable, "-m"]
    subprocess.run(base + ["src.train_efficientnet", "--data_dir", str(root),
        "--splits", str(root / "splits.csv"), "--label_map", str(root / "label_map.json"),
        "--out_dir", str(run), "--epochs", "1", "--batch_size", "8", "--expected_classes", "2",
        "--no_pretrained", "--class_weights", "--device", "cpu"], check=True)
    summary = json.loads((run / "metrics.json").read_text())
    assert summary["test"] is None
    assert summary["best_epoch"] == 1
    subprocess.run(base + ["src.evaluate_efficientnet", "--checkpoint", str(run / "best.pt"),
                          "--data_dir", str(root), "--device", "cpu"], check=True)
    assert (run / "comparison_row.csv").is_file()
    summary = json.loads((run / "metrics.json").read_text())
    assert summary["test"]["recordings"] == 2
    predictions = pd.read_csv(run / "test/predictions.csv")
    assert predictions.path.tolist() == df.loc[df.split == "test", "path"].tolist()
    result = subprocess.run(base + ["src.predict", "--checkpoint", str(run / "best.pt"),
        "--audio", str(root / "Bells/0.wav"), "--device", "cpu"], check=True, text=True, capture_output=True)
    result = json.loads(result.stdout)
    assert len(result["predictions"]) == 2
    assert abs(sum(p["probability"] for p in result["predictions"]) - 1) < 1e-5
    # A second final test call must not replace saved results.
    repeated = subprocess.run(base + ["src.evaluate_efficientnet", "--checkpoint", str(run / "best.pt"),
        "--data_dir", str(root)], text=True, capture_output=True)
    assert repeated.returncode != 0 and "already exist" in repeated.stderr


def test_data_prep_cli_generates_audited_split(tmp_path):
    rng = np.random.default_rng(2)
    for label in ("Bells", "Birds"):
        folder = tmp_path / "audio" / label
        folder.mkdir(parents=True)
        for i in range(20):
            sf.write(folder / f"{i}.wav", rng.normal(0, 0.05, 200), 16000, subtype="FLOAT")
    prepared = tmp_path / "prepared"
    subprocess.run([sys.executable, "-m", "src.data_prep", "--data_dir", str(tmp_path / "audio"),
                    "--out_dir", str(prepared), "--expected_classes", "2"], check=True)
    df, labels = read_manifest(prepared / "splits.csv", prepared / "label_map.json", tmp_path / "audio", 2)
    assert len(df) == 40 and len(labels) == 2
    assert {"audio_hash", "duration", "sample_rate"} <= set(df.columns)
    assert json.loads((prepared / "audit.json").read_text())["recordings"] == 40


def test_evaluator_rejects_debug_checkpoint_before_audio_access(tmp_path, monkeypatch):
    from src import evaluate_efficientnet as module
    monkeypatch.setattr(module, "load_checkpoint", lambda *args: (None, {"run_config": {"debug_run": True}}))
    with pytest.raises(ValueError, match="Debug checkpoints"):
        module.evaluate_checkpoint(tmp_path / "best.pt", tmp_path, "cpu")


def test_evaluator_detects_changed_recording(audio_table, monkeypatch):
    from src import evaluate_efficientnet as module
    from src.data_prep import manifest_fingerprint
    root, df, labels = audio_table
    checked = audit_audio(df, root)
    ckpt = {"run_config": {"debug_run": False}, "label_map": labels,
            "norm_stats": {"split_sha256": manifest_fingerprint(checked)}}
    df.to_csv(root / "splits.csv", index=False)
    sf.write(root / "Bells/0.wav", np.arange(500, dtype=float) / 1000, 16000, subtype="FLOAT")
    monkeypatch.setattr(module, "load_checkpoint", lambda *args: (None, ckpt))
    with pytest.raises(ValueError, match="changed since training"):
        module.evaluate_checkpoint(root / "best.pt", root, "cpu")
