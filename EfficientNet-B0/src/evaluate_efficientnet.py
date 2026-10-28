"""Evaluate the fixed best checkpoint on its held-out test recordings, once per run folder."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .data_prep import audit_audio, manifest_fingerprint, validate_manifest
from .dataset import AudioConfig
from .evaluation import evaluate, save_evaluation
from .train_efficientnet import choose_device, load_checkpoint, make_loader


def evaluate_checkpoint(checkpoint_path, data_dir, device_name="auto", batch_size=32, workers=0):
    checkpoint_path = Path(checkpoint_path)
    out_dir = checkpoint_path.parent
    if batch_size < 1 or workers < 0:
        raise ValueError("Need positive batch size and nonnegative workers.")
    if (out_dir / "test").exists():
        raise ValueError("Test outputs already exist. Review them without rerunning or overwriting.")
    device = choose_device(device_name)
    model, ckpt = load_checkpoint(checkpoint_path, device)
    if ckpt["run_config"]["debug_run"]:
        raise ValueError("Debug checkpoints are plumbing checks, not project test results. Run full training first.")
    df = pd.read_csv(out_dir / "splits.csv", dtype={"group_id": str})
    df = validate_manifest(df, ckpt["label_map"], data_dir, len(ckpt["label_map"]))
    # Verify dataset identity, not just filenames, against the saved training run.
    df = audit_audio(df, data_dir)
    if manifest_fingerprint(df) != ckpt["norm_stats"]["split_sha256"]:
        raise ValueError("Manifest or decoded audio changed since training.")
    test = df[df.split == "test"].reset_index(drop=True)
    stats = ckpt["norm_stats"]
    loader = make_loader(test, data_dir, AudioConfig(**ckpt["audio_config"]), stats["mean"], stats["std"],
                         batch_size, workers, device)
    metrics, y, prob = evaluate(model, loader, device, len(ckpt["label_map"]))
    save_evaluation(out_dir / "test", test, ckpt["label_map"], metrics, y, prob, "EfficientNet-B0: held-out test")
    summary = json.loads((out_dir / "metrics.json").read_text())
    summary.update(test=metrics, status="test_complete")
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    row = {"model": "EfficientNet-B0", "parameters": summary["parameters"],
           "best_epoch": summary["best_epoch"], "split_sha256": ckpt["norm_stats"]["split_sha256"], **metrics}
    pd.DataFrame([row]).to_csv(out_dir / "comparison_row.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    evaluate_checkpoint(a.checkpoint, a.data_dir, a.device, a.batch_size, a.workers)


if __name__ == "__main__":
    main()
