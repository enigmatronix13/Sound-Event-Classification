"""Model 2 data preparation. Reuse a shared split without changing its assignments."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

PARENTS = [
    "Bells", "Birds", "Cat fights and moans", "Chicken coop", "Cicadas and crickets",
    "Crows seagulls and magpies", "Dog barkings and howlings", "Glass breaking", "Horn",
    "Jet aircrafts", "Lawn mower brush cutter and olive shaker", "Music", "Propeller aircrafts",
    "Sirens and alarms", "Thunder fireworks and gunshot", "Train",
    "Vacuum cleaner fan and hairdryer", "Vehicle idling", "Vehicle pass-by", "Voices",
    "Wind turbine", "Workshop",
]
SUBCLASSES = {
    "Cicadas and crickets": ["Cicadas", "Crickets"],
    "Crows seagulls and magpies": ["Crows", "Seagulls", "Magpies"],
    "Lawn mower brush cutter and olive shaker": ["Lawn mower", "Brush cutter", "Olive shaker"],
    "Propeller aircrafts": ["Airplanes", "Helicopters", "Propeller airplanes"],
    "Sirens and alarms": ["Sirens", "Alarms"],
    "Thunder fireworks and gunshot": ["Thunder", "Fireworks", "Gunshot", "Gunshots"],
    "Vacuum cleaner fan and hairdryer": ["Vacuum cleaner", "Fan", "Hairdryer"],
    "Vehicle idling": ["Car-truck idling", "Car idling", "Truck idling", "Motorbike idling"],
    "Vehicle pass-by": ["Car pass-by", "Motorbike pass-by", "Truck pass-by"],
    "Workshop": ["Air compressor", "Drill", "Grinder", "Jackhammer", "Saw"],
}


def normalize_label(value):
    words = re.findall(r"[a-z]+", value.lower())
    return "".join(w[:-1] if w.endswith("s") else w for w in words if w != "and")


ALIASES = {normalize_label(p): p for p in PARENTS}
for parent, children in SUBCLASSES.items():
    ALIASES.update({normalize_label(c): parent for c in children})


def infer_parent(relative_path):
    candidates = {ALIASES[normalize_label(p)] for p in Path(relative_path).parts[:-1]
                  if normalize_label(p) in ALIASES}
    if len(candidates) != 1:
        raise ValueError(f"Unknown/conflicting class folders for {relative_path}. Supply a CSV with parent labels.")
    return candidates.pop()


def safe_path(data_dir, value):
    root = Path(data_dir).resolve()
    path = (root / str(value)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Audio path is outside data_dir: {value}")
    if not path.is_file():
        raise ValueError(f"Missing audio file: {value}")
    return path


def clean_table(df, data_dir):
    df = df.copy()
    if not {"path", "label"} <= set(df.columns) or df.empty:
        raise ValueError("Need a nonempty table with path and label columns.")
    if df[["path", "label"]].isna().any().any():
        raise ValueError("Missing path or label; no recordings are silently excluded.")
    df["path"] = [safe_path(data_dir, p).relative_to(Path(data_dir).resolve()).as_posix() for p in df.path]
    df["label"] = df.label.astype(str).str.strip()
    if (df.label == "").any() or df.path.duplicated().any():
        raise ValueError("Empty label or repeated recording path in the manifest.")
    return df.reset_index(drop=True)


def build_table(data_dir, csv=None, path_col="filename", label_col="class"):
    if csv:
        meta = pd.read_csv(csv, dtype={"group_id": str})
        if not {path_col, label_col} <= set(meta.columns):
            raise ValueError(f"CSV needs {path_col!r} and {label_col!r} columns.")
        df = meta.rename(columns={path_col: "path", label_col: "label"})
        # Explicit metadata must already identify the 22 parent classes.
    else:
        files = sorted(p for p in Path(data_dir).rglob("*")
                       if p.suffix.lower() == ".wav" and not p.name.startswith("._"))
        paths = [p.relative_to(data_dir).as_posix() for p in files]
        df = pd.DataFrame({"path": paths, "label": [infer_parent(p) for p in paths]})
    return clean_table(df, data_dir)


def validate_manifest(df, label_map, data_dir, expected_classes=22):
    df = clean_table(df, data_dir)
    if not {"label_id", "split"} <= set(df.columns):
        raise ValueError("Manifest needs label_id and split (train/val/test).")
    if (len(label_map) != expected_classes or
            any(type(v) is not int for v in label_map.values()) or
            sorted(label_map.values()) != list(range(len(label_map)))):
        raise ValueError(f"Label map must contain {expected_classes} classes with unique contiguous integer IDs.")
    if expected_classes == 22 and set(label_map) != set(PARENTS):
        raise ValueError("The DataSEC label map must use the 22 parent labels, not subclasses.")
    expected = df.label.map(label_map)
    if expected.isna().any() or not np.array_equal(expected.to_numpy(), df.label_id.to_numpy()):
        raise ValueError("Manifest label IDs disagree with label_map.json.")
    if set(df.split) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain exactly train, val and test split names.")
    for split in ("train", "val", "test"):
        if set(df.loc[df.split == split, "label"]) != set(label_map):
            raise ValueError(f"Missing classes in {split}; agree a valid split before training.")
    if "group_id" in df:
        if df.group_id.isna().any() or df.group_id.astype(str).str.strip().eq("").any():
            raise ValueError("Missing source group IDs.")
        if df.groupby("group_id").split.nunique().max() > 1:
            raise ValueError("Source-group leakage across splits.")
    return df


def read_manifest(path, label_map_path, data_dir, expected_classes=22):
    df = pd.read_csv(path, dtype={"group_id": str})
    # Accept the existing SincNet export without changing that notebook.
    if "relative_path" in df:
        df["path"] = df.relative_path
    if "target" in df and "label_id" not in df:
        df["label_id"] = df.target
    label_map = json.loads(Path(label_map_path).read_text())
    return validate_manifest(df, label_map, data_dir, expected_classes), label_map


def audit_audio(df, data_dir):
    """Decode every recording; reject corruption and detect exact audio duplicates."""
    records = []
    for row in df.itertuples():
        path = safe_path(data_dir, row.path)
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Unreadable audio: {row.path}: {exc}") from exc
        if not len(wav) or not np.isfinite(wav).all():
            raise ValueError(f"Empty/non-finite audio: {row.path}")
        digest = hashlib.sha256(f"{sr}:{wav.shape}".encode())
        digest.update(np.ascontiguousarray(wav, dtype="<f4").tobytes())
        records.append({"path": row.path, "audio_hash": digest.hexdigest(),
                        "duration": len(wav) / sr, "sample_rate": sr, "channels": wav.shape[1]})
    audit = pd.DataFrame(records)
    checked = df.drop(columns=[c for c in audit if c != "path" and c in df], errors="ignore").merge(
        audit, on="path", validate="one_to_one", sort=False)
    if checked.groupby("audio_hash").label.nunique().max() > 1:
        raise ValueError("Identical decoded audio has conflicting labels; resolve the dataset inventory.")
    if "split" in checked and checked.groupby("audio_hash").split.nunique().max() > 1:
        raise ValueError("Duplicate audio leakage across train/val/test.")
    return checked


def create_split(df, seed=42, val_frac=0.15, test_frac=0.15):
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one.")
    if "group_id" in df:
        raise ValueError("Source groups supplied: use an agreed group-aware --shared_split; do not split groups randomly.")
    try:
        train_val, test = train_test_split(df, test_size=test_frac, stratify=df.label, random_state=seed)
        train, val = train_test_split(train_val, test_size=val_frac / (1 - test_frac),
                                      stratify=train_val.label, random_state=seed)
    except ValueError as exc:
        raise ValueError(f"Too few recordings for the requested stratified split: {exc}") from exc
    return pd.concat([train.assign(split="train"), val.assign(split="val"),
                      test.assign(split="test")], ignore_index=True)


def manifest_fingerprint(df):
    cols = [c for c in ("path", "label", "label_id", "split", "audio_hash", "group_id") if c in df]
    return hashlib.sha256(df[cols].sort_values("path").to_csv(index=False).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("outputs/prepared"), type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--path_col", default="filename")
    ap.add_argument("--label_col", default="class")
    ap.add_argument("--shared_split", type=Path, help="Reuse existing team assignments (including SincNet format)")
    ap.add_argument("--label_map", type=Path, help="Existing name-to-ID JSON for --shared_split")
    ap.add_argument("--drop_duplicates", action="store_true", help="Explicitly exclude exact duplicates when creating a NEW split")
    ap.add_argument("--expected_classes", type=int, default=22)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        ap.error("out_dir is nonempty; use a new folder to preserve the previous manifest.")
    if a.shared_split and (not a.label_map or a.csv or a.drop_duplicates):
        ap.error("--shared_split needs --label_map and cannot be combined with --csv/--drop_duplicates.")
    if a.label_map and not a.shared_split:
        ap.error("--label_map is only used with --shared_split.")
    if a.shared_split:
        df, label_map = read_manifest(a.shared_split, a.label_map, a.data_dir, a.expected_classes)
    else:
        df = build_table(a.data_dir, a.csv, a.path_col, a.label_col)
        if "split" in df:
            ap.error("CSV already has splits: use --shared_split and --label_map to preserve them.")
        label_map = {c: i for i, c in enumerate(sorted(df.label.unique()))}
        df["label_id"] = df.label.map(label_map)
    df = audit_audio(df, a.data_dir)
    duplicates = df[df.audio_hash.duplicated(keep="first")]
    if not a.shared_split:
        if len(duplicates) and not a.drop_duplicates:
            ap.error(f"Found {len(duplicates)} exact duplicates. Agree exclusions, then use --drop_duplicates.")
        df = df.drop_duplicates("audio_hash", keep="first")
        df = create_split(df, a.seed, a.val_frac, a.test_frac)
    df = validate_manifest(df, label_map, a.data_dir, a.expected_classes)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_dir / "splits.csv", index=False)
    (a.out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))
    duplicates.to_csv(a.out_dir / "duplicate_recordings.csv", index=False)
    counts = pd.crosstab(df.label, df.split).reindex(columns=["train", "val", "test"])
    counts.to_csv(a.out_dir / "class_distribution.csv")
    summary = {"recordings": len(df), "classes": len(label_map), "hours": float(df.duration.sum() / 3600),
               "shared_split": str(a.shared_split) if a.shared_split else None,
               "seed": a.seed, "exact_duplicates_found": len(duplicates),
               "source_groups_available": "group_id" in df, "sha256": manifest_fingerprint(df),
               "note": "Exact hashes do not detect near-duplicates or source sessions without group IDs."}
    (a.out_dir / "audit.json").write_text(json.dumps(summary, indent=2))
    print(counts.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
