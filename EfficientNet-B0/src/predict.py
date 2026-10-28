"""Predict one WAV using the checkpoint's exact label map and preprocessing."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .dataset import AudioConfig, LogMelDataset
from .train_efficientnet import choose_device, load_checkpoint


@torch.no_grad()
def predict(checkpoint, audio, top_k=5, device_name="auto"):
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    device = choose_device(device_name)
    model, ckpt = load_checkpoint(checkpoint, device)
    audio = Path(audio).resolve()
    stats = ckpt["norm_stats"]
    ds = LogMelDataset(pd.DataFrame([{"path": audio.name, "label_id": 0}]), audio.parent,
                       AudioConfig(**ckpt["audio_config"]), mean=stats["mean"], std=stats["std"])
    x, _ = ds[0]
    probs = model(x.unsqueeze(0).to(device)).softmax(1)[0].cpu()
    values, indices = probs.topk(min(top_k, len(probs)))
    id2label = {v: k for k, v in ckpt["label_map"].items()}
    return {"audio": str(audio), "crop_policy": "one centre crop, padded if shorter than 5 seconds",
            "debug_checkpoint": ckpt["run_config"]["debug_run"],
            "predictions": [{"class": id2label[int(i)], "probability": float(p)} for p, i in zip(values, indices)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    a = ap.parse_args()
    print(json.dumps(predict(a.checkpoint, a.audio, a.top_k, a.device), indent=2))


if __name__ == "__main__":
    main()
