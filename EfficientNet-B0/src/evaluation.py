"""Shared Model 2 metrics and figures; all class IDs are included in macro scores."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def classification_metrics(y, predictions, n_classes):
    p, r, f, _ = precision_recall_fscore_support(
        y, predictions, labels=list(range(n_classes)), average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y, predictions)), "macro_precision": float(p),
            "macro_recall": float(r), "macro_f1": float(f)}


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    targets, probabilities = [], []
    total_loss, forward_seconds, count = 0.0, 0.0, 0
    synchronize(device)
    started = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_device = y.to(device, non_blocking=True)
        synchronize(device)
        tick = time.perf_counter()
        logits = model(x)
        synchronize(device)
        forward_seconds += time.perf_counter() - tick
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite model outputs.")
        total_loss += torch.nn.functional.cross_entropy(logits, y_device, reduction="sum").item()
        count += len(y)
        targets.append(y.numpy())
        probabilities.append(logits.softmax(1).cpu().numpy())
    if count == 0:
        raise ValueError("Cannot evaluate an empty split.")
    end_to_end_seconds = time.perf_counter() - started
    y, prob = np.concatenate(targets), np.concatenate(probabilities)
    metrics = classification_metrics(y, prob.argmax(1), n_classes)
    metrics.update(loss=total_loss / count, recordings=count,
                   forward_ms_per_clip=forward_seconds * 1000 / count,
                   end_to_end_ms_per_clip=end_to_end_seconds * 1000 / count)
    return metrics, y, prob


def save_evaluation(out_dir, df, label_map, metrics, y, prob, title="EfficientNet-B0"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    labels = list(range(len(names)))
    pred = prob.argmax(1)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (out_dir / "classification_report.txt").write_text(
        classification_report(y, pred, labels=labels, target_names=names, zero_division=0, digits=4))
    cm = confusion_matrix(y, pred, labels=labels)
    pd.DataFrame(cm, index=names, columns=names).to_csv(out_dir / "confusion_matrix.csv")
    normalized = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float),
                           where=cm.sum(1, keepdims=True) != 0)
    for suffix, values in (("", cm), ("_normalized", normalized)):
        fig, ax = plt.subplots(figsize=(12, 11))
        im = ax.imshow(values, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(xticks=labels, yticks=labels, xlabel="Predicted class", ylabel="True class", title=title)
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_matrix{suffix}.png", dpi=160)
        plt.close(fig)
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    pd.DataFrame({"class": names, "precision": p, "recall": r, "f1": f, "support": support}).to_csv(
        out_dir / "per_class_f1.csv", index=False)
    predictions = df.reset_index(drop=True)[["path", "label", "label_id"]].copy()
    if len(predictions) != len(y) or not np.array_equal(predictions.label_id, y):
        raise ValueError("Prediction order differs from the manifest.")
    predictions["pred"] = [names[i] for i in pred]
    predictions["confidence"] = prob.max(1)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    predictions.loc[y != pred].sort_values("confidence", ascending=False).to_csv(
        out_dir / "misclassified.csv", index=False)


def plot_history(history, out_dir):
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h.epoch, h.train_loss, label="Train (augmented, mixup if enabled)")
    axes[0].plot(h.epoch, h.val_loss, label="Validation (unweighted)")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend(fontsize=8)
    axes[1].plot(h.epoch, h.val_accuracy, label="Validation accuracy")
    axes[1].plot(h.epoch, h.val_macro_f1, label="Validation macro-F1")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "learning_curves.png", dpi=160)
    plt.close(fig)
