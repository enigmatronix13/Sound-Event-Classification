"""Train Model 2 and select by validation macro-F1; test evaluation is opt-in."""
import argparse
from dataclasses import asdict
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_prep import audit_audio, manifest_fingerprint, read_manifest
from .dataset import AudioConfig, LogMelDataset
from .evaluation import evaluate, plot_history, save_evaluation, synchronize
from .model import EffNetB0Audio


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Select a GPU runtime or use --device cpu.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable.")
    return torch.device(name)


def compute_norm_stats(df, data_dir, cfg, workers):
    """Unaugmented centre crops from TRAIN only, accumulated in float64."""
    if df.empty or set(df.split) != {"train"}:
        raise ValueError("Normalization requires nonempty training-only rows.")
    loader = DataLoader(LogMelDataset(df, data_dir, cfg), batch_size=32, num_workers=workers)
    count, total, total_sq = 0, 0.0, 0.0
    for x, _ in tqdm(loader, desc="Training normalization"):
        x = x.double()
        count += x.numel()
        total += x.sum().item()
        total_sq += x.square().sum().item()
    mean = total / count
    std = float(np.sqrt(max(0.0, total_sq / count - mean ** 2)))
    if not np.isfinite([mean, std]).all() or std <= 1e-8:
        raise ValueError("Invalid/constant training features; inspect the audio.")
    return mean, std


def make_loader(df, data_dir, cfg, mean, std, batch_size, workers, device, train=False, seed=42):
    ds = LogMelDataset(df, data_dir, cfg, train=train, mean=mean, std=std)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                      pin_memory=device.type == "cuda", drop_last=False, worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=workers > 0)


def debug_subset(df, limit, seed):
    """Keep at least one example per class even in a short plumbing check."""
    if limit <= 0 or limit >= len(df):
        return df.reset_index(drop=True)
    if limit < df.label_id.nunique():
        raise ValueError("--limit must be at least the number of classes.")
    first = df.groupby("label_id", group_keys=False).sample(n=1, random_state=seed)
    rest = df.drop(first.index).sample(n=limit - len(first), random_state=seed)
    return pd.concat([first, rest]).sort_values("path").reset_index(drop=True)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Legacy weights-only checkpoint: retrain to save labels and preprocessing together.")
    model = EffNetB0Audio(len(checkpoint["label_map"]), pretrained=False,
                         dropout=checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--splits", default=Path("outputs/prepared/splits.csv"), type=Path)
    ap.add_argument("--label_map", default=Path("outputs/prepared/label_map.json"), type=Path)
    ap.add_argument("--out_dir", default=Path("outputs/efficientnet_b0"), type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--class_weights", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Debug only: maximum clips per split; at least one per class")
    ap.add_argument("--expected_classes", type=int, default=22)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--evaluate_test", action="store_true", help="Use only for the final selected experiment")
    a = ap.parse_args()
    if min(a.epochs, a.batch_size, a.patience) < 1 or min(a.workers, a.limit, a.mixup_alpha, a.weight_decay) < 0:
        ap.error("Epochs, batch size and patience must be positive; counts/regularization must be nonnegative.")
    if not 0 <= a.dropout < 1 or min(a.lr_head, a.lr_backbone) <= 0:
        ap.error("Need dropout in [0,1) and positive learning rates.")
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        ap.error("out_dir is nonempty. Choose a new run folder; existing results will not be overwritten.")
    set_seed(a.seed)
    device = choose_device(a.device)
    cfg = AudioConfig()
    splits, label_map = read_manifest(a.splits, a.label_map, a.data_dir, a.expected_classes)
    print("Auditing audio and checking recording leakage...")
    splits = audit_audio(splits, a.data_dir)
    parts = {s: debug_subset(splits[splits.split == s], a.limit, a.seed) for s in ("train", "val", "test")}
    used_splits = pd.concat(parts.values(), ignore_index=True)
    split_hash = manifest_fingerprint(used_splits)
    n_classes = len(label_map)
    # Always recompute: no normalization cache can leak a previous/debug split into this run.
    mean, std = compute_norm_stats(parts["train"], a.data_dir, cfg, a.workers)
    model = EffNetB0Audio(n_classes, pretrained=not a.no_pretrained, dropout=a.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    used_splits.to_csv(a.out_dir / "splits.csv", index=False)
    (a.out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))
    stats = {"mean": mean, "std": std, "training_recordings": len(parts["train"]), "split_sha256": split_hash}
    (a.out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(audio=asdict(cfg), device=str(device), parameters=n_params, split_sha256=split_hash,
                  torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                  torchaudio_version=str(torchaudio.__version__), debug_run=bool(a.limit),
                  evaluation_policy="one deterministic centre crop per recording",
                  normalization="training-only unaugmented centre crops; no RGB ImageNet normalization")
    (a.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    loaders = {s: make_loader(df, a.data_dir, cfg, mean, std, a.batch_size, a.workers,
                             device, train=s == "train", seed=a.seed) for s, df in parts.items()}
    weights = None
    if a.class_weights:
        counts = np.bincount(parts["train"].label_id, minlength=n_classes)
        weights = torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32, device=device)
    # Per-example reduction makes weighted mixup a well-defined convex sum.
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
    optimizer = torch.optim.AdamW(model.param_groups(a.lr_head, a.lr_backbone, a.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, best_f1, stale, best_epoch = [], -1.0, 0, 0
    start_training = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        synchronize(device)
        tick = time.perf_counter()
        model.train()
        loss_sum, count = 0.0, 0
        for x, y in tqdm(loaders["train"], desc=f"Epoch {epoch}/{a.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if a.mixup_alpha:
                    lam = float(np.random.beta(a.mixup_alpha, a.mixup_alpha))
                    permutation = torch.randperm(len(y), device=device)
                    logits = model(lam * x + (1 - lam) * x[permutation])
                    loss = (lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[permutation])).mean()
                else:
                    loss = criterion(model(x), y).mean()
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)
            count += len(y)
        scheduler.step()
        val, _, _ = evaluate(model, loaders["val"], device, n_classes)
        synchronize(device)
        row = {"epoch": epoch, "train_loss": loss_sum / count,
               **{f"val_{k}": val[k] for k in ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")},
               "epoch_time_s": time.perf_counter() - tick}
        history.append(row)
        pd.DataFrame(history).to_csv(a.out_dir / "history.csv", index=False)
        print(f"Epoch {epoch}: loss={row['train_loss']:.4f}, val macro-F1={val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1, stale, best_epoch = val["macro_f1"], 0, epoch
            torch.save({"model_state": model.state_dict(), "label_map": label_map,
                        "audio_config": asdict(cfg), "norm_stats": stats, "dropout": a.dropout,
                        "epoch": epoch, "val_macro_f1": best_f1, "run_config": config}, a.out_dir / "best.pt")
        else:
            stale += 1
        if stale >= a.patience:
            break
    training_seconds = time.perf_counter() - start_training
    model, _ = load_checkpoint(a.out_dir / "best.pt", device)
    val, y, prob = evaluate(model, loaders["val"], device, n_classes)
    save_evaluation(a.out_dir / "validation", parts["val"], label_map, val, y, prob,
                    "EfficientNet-B0: validation")
    plot_history(history, a.out_dir)
    summary = {"model": "EfficientNet-B0", "status": "debug_only" if a.limit else "validation_complete",
               "best_epoch": best_epoch, "parameters": n_params, "training_seconds": training_seconds,
               "avg_epoch_time_s": float(np.mean([r["epoch_time_s"] for r in history])),
               "val": val, "test": None, "config": config}
    (a.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Best epoch: {best_epoch}. Saved to {a.out_dir}. Test metrics remain pending.")
    if a.evaluate_test:
        from .evaluate_efficientnet import evaluate_checkpoint
        evaluate_checkpoint(a.out_dir / "best.pt", a.data_dir, str(device), a.batch_size, a.workers)


if __name__ == "__main__":
    main()
