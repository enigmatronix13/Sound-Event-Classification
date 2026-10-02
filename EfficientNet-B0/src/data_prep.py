"""
Step 1 (COMMON to all four models): build the label map and ONE shared
stratified train/val/test split for DataSEC.

Two ways to describe the dataset:
  (a) folder-per-class:   data_dir/<class_name>/*.wav      (default)
  (b) a metadata CSV:     --csv meta.csv --path_col <col> --label_col <col>
      (paths in the CSV are relative to --data_dir)

Outputs (in --out_dir):
  label_map.json  {class_name: id}
  splits.csv      path,label,label_id,split   (split in train/val/test)
All models must read this same splits.csv so results are comparable.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_table(data_dir: Path, csv=None, path_col="filename", label_col="class"):
    if csv:
        meta = pd.read_csv(csv)
        df = pd.DataFrame({"path": meta[path_col].astype(str), "label": meta[label_col].astype(str)})
    else:
        files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() == ".wav")
        df = pd.DataFrame({
            "path": [str(p.relative_to(data_dir)) for p in files],
            "label": [p.parent.name for p in files],
        })
    # drop files that do not exist (typos / missing downloads)
    exists = df["path"].map(lambda p: (data_dir / p).exists())
    if (~exists).any():
        print(f"[warn] dropping {(~exists).sum()} rows whose file is missing")
    return df[exists].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("outputs"), type=Path)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--path_col", default="filename")
    ap.add_argument("--label_col", default="class")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = build_table(a.data_dir, a.csv, a.path_col, a.label_col)
    classes = sorted(df["label"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["label"].map(label_map)

    print(f"{len(df)} clips, {len(classes)} classes")
    print(df["label"].value_counts().to_string())
    small = df["label"].value_counts()
    if (small < 3).any():
        print("[warn] classes with <3 clips cannot be split stratified:", list(small[small < 3].index))

    # stratified 70/15/15 (two-step split, same seed for everyone)
    train_val, test = train_test_split(
        df, test_size=a.test_frac, stratify=df["label_id"], random_state=a.seed)
    rel_val = a.val_frac / (1 - a.test_frac)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val["label_id"], random_state=a.seed)

    train, val, test = train.copy(), val.copy(), test.copy()
    train["split"], val["split"], test["split"] = "train", "val", "test"
    out = pd.concat([train, val, test]).reset_index(drop=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / "splits.csv", index=False)
    (a.out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))
    print(out["split"].value_counts().to_string())
    print(f"saved -> {a.out_dir/'splits.csv'}, {a.out_dir/'label_map.json'}")


if __name__ == "__main__":
    main()
