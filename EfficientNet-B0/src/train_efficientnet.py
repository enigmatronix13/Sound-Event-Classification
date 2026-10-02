"""
Train / validate / test EfficientNet-B0 on DataSEC log-Mel spectrograms.

Example:
  python -m src.data_prep --data_dir data/DataSEC
  python -m src.train_efficientnet --data_dir data/DataSEC

Artifacts are written to outputs/efficientnet_b0/ :
  best.pt, history.csv, metrics.json, classification_report.txt,
  confusion_matrix.csv/.png, per_class_f1.csv, misclassified.csv, norm_stats.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import AudioConfig, LogMelDataset
from .model import EffNetB0Audio


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def compute_norm_stats(df, data_dir, cfg, workers):
    """Mean/std of dB-Mel values over the TRAIN split only (no augmentation)."""
    ds = LogMelDataset(df, data_dir, cfg, train=False)
    dl = DataLoader(ds, batch_size=32, num_workers=workers)
    n, s, ss = 0, 0.0, 0.0
    for x, _ in tqdm(dl, desc="norm stats"):
        n += x.numel(); s += x.sum().item(); ss += (x.double() ** 2).sum().item()
    mean = s / n
    return mean, float(np.sqrt(ss / n - mean ** 2))


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    ys, ps, tot_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        if criterion is not None:
            tot_loss += criterion(out, y).item() * len(y)
        n += len(y)
        ys.append(y.cpu().numpy()); ps.append(out.argmax(1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"loss": tot_loss / max(n, 1), "acc": accuracy_score(y, p),
            "macro_precision": pr, "macro_recall": rc, "macro_f1": f1}, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--splits", default=Path("outputs/splits.csv"), type=Path)
    ap.add_argument("--label_map", default=Path("outputs/label_map.json"), type=Path)
    ap.add_argument("--out_dir", default=Path("outputs/efficientnet_b0"), type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--mixup_alpha", type=float, default=0.2, help="0 disables mixup")
    ap.add_argument("--class_weights", action="store_true", help="class-weighted loss for imbalance")
    ap.add_argument("--no_pretrained", action="store_true", help="train from scratch (ablation)")
    ap.add_argument("--patience", type=int, default=8, help="early stopping on val macro-F1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N clips per split")
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = AudioConfig()
    label_map = json.loads(a.label_map.read_text())
    id2label = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)

    splits = pd.read_csv(a.splits)
    parts = {s: splits[splits["split"] == s] for s in ("train", "val", "test")}
    if a.limit:
        parts = {s: d.sample(min(len(d), a.limit), random_state=a.seed) for s, d in parts.items()}

    stats_path = a.out_dir / "norm_stats.json"
    if stats_path.exists() and not a.limit:
        mean, std = json.loads(stats_path.read_text()).values()
    else:
        mean, std = compute_norm_stats(parts["train"], a.data_dir, cfg, a.workers)
        stats_path.write_text(json.dumps({"mean": mean, "std": std}))
    print(f"norm stats: mean={mean:.3f} std={std:.3f}")

    def loader(split, train):
        ds = LogMelDataset(parts[split], a.data_dir, cfg, train=train, mean=mean, std=std)
        return DataLoader(ds, batch_size=a.batch_size, shuffle=train, num_workers=a.workers,
                          pin_memory=device.type == "cuda", drop_last=train, persistent_workers=a.workers > 0)
    dl_tr, dl_va, dl_te = loader("train", True), loader("val", False), loader("test", False)

    model = EffNetB0Audio(n_classes, pretrained=not a.no_pretrained, dropout=a.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"EfficientNet-B0: {n_params/1e6:.2f}M params, device={device}")

    weight = None
    if a.class_weights:
        counts = np.bincount(parts["train"]["label_id"], minlength=n_classes)
        weight = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.param_groups(a.lr_head, a.lr_backbone, a.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_f1, bad, epoch_times = [], -1.0, 0, []
    for epoch in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run_loss = 0.0
        for x, y in tqdm(dl_tr, desc=f"epoch {epoch}/{a.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if a.mixup_alpha > 0:
                    lam = np.random.beta(a.mixup_alpha, a.mixup_alpha)
                    idx = torch.randperm(x.size(0), device=device)
                    out = model(lam * x + (1 - lam) * x[idx])
                    loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y[idx])
                else:
                    loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item()
        sched.step()
        epoch_times.append(time.time() - t0)

        val, _, _ = evaluate(model, dl_va, device, criterion)
        row = {"epoch": epoch, "train_loss": run_loss / len(dl_tr), **{f"val_{k}": v for k, v in val.items()},
               "epoch_time_s": epoch_times[-1]}
        history.append(row)
        print(f"ep {epoch:02d} train_loss {row['train_loss']:.3f} | val acc {val['acc']:.4f} "
              f"macro-F1 {val['macro_f1']:.4f} | {epoch_times[-1]:.0f}s")
        pd.DataFrame(history).to_csv(a.out_dir / "history.csv", index=False)

        if val["macro_f1"] > best_f1:
            best_f1, bad = val["macro_f1"], 0
            torch.save(model.state_dict(), a.out_dir / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break

    # ---- final evaluation with the best checkpoint ----
    model.load_state_dict(torch.load(a.out_dir / "best.pt", map_location=device))
    val, _, _ = evaluate(model, dl_va, device)
    t0 = time.time()
    test, y, p = evaluate(model, dl_te, device)
    infer_ms = (time.time() - t0) / len(y) * 1000

    labels = list(range(n_classes)); names = [id2label[i] for i in labels]
    (a.out_dir / "classification_report.txt").write_text(
        classification_report(y, p, labels=labels, target_names=names, zero_division=0))
    cm = confusion_matrix(y, p, labels=labels)
    pd.DataFrame(cm, index=names, columns=names).to_csv(a.out_dir / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im)
    ax.set_xticks(labels); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(labels); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("EfficientNet-B0 - test confusion matrix")
    fig.tight_layout(); fig.savefig(a.out_dir / "confusion_matrix.png", dpi=200); plt.close(fig)

    _, _, f1c, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    pd.DataFrame({"class": names, "f1": f1c, "support": sup}).sort_values("f1").to_csv(
        a.out_dir / "per_class_f1.csv", index=False)
    wrong = parts["test"].reset_index(drop=True)
    # NOTE: loader order == df order because test loader has shuffle=False
    wrong = wrong.assign(pred=[id2label[i] for i in p])
    wrong[wrong["label_id"].values != p][["path", "label", "pred"]].to_csv(a.out_dir / "misclassified.csv", index=False)

    metrics = {"model": "EfficientNet-B0", "pretrained": not a.no_pretrained, "params_million": n_params / 1e6,
               "avg_epoch_time_s": float(np.mean(epoch_times)), "inference_ms_per_clip": infer_ms,
               "best_val_macro_f1": best_f1, "val": val, "test": test, "config": {k: str(v) for k, v in vars(a).items()}}
    (a.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nTEST:", json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
