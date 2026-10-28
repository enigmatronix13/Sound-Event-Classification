# Model 2: EfficientNet-B0 for DataSEC

**Owner: Venisa Ivan Tellis (230911064), ICT 4442.** Everything in this folder is for Model 2. The other model folders and the repository-level files are unchanged.

This extends the existing Model 2 starter with a notebook, audited data preparation, reproducible training, held-out evaluation, prediction and regression tests. **Full DataSEC training and verified project scores are pending.** No old report score is treated as a measured result of this implementation.

## Start here

Open [`EfficientNet_B0_DataSEC.ipynb`](EfficientNet_B0_DataSEC.ipynb) in Jupyter or Google Colab. In Colab select a GPU runtime, run setup, and set the dataset and shared-manifest paths. It calls the same Python modules described below, so there is only one training implementation to maintain.

For a local terminal, start inside `EfficientNet-B0/`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use Python 3.10 or newer. Install compatible PyTorch, torchvision and torchaudio builds together; use the [PyTorch installation selector](https://pytorch.org/get-started/locally/) if your CUDA environment requires a specific wheel. CPU works for checks but a GPU is recommended for full training. `--device mps` is available for Apple Silicon; it has not been validated for a full training run.

## 1. Prepare the agreed dataset and split

Obtain DataSEC from the [dataset record](https://doi.org/10.5281/zenodo.15340689). The interim report describes a different observed inventory for SincNet than the synopsis, so use the **same release, cleaned recordings, parent labels and split assignments as the team**. The code reports actual counts; it does not assume that 4,292 files must exist. Dataset audio and checkpoints are excluded from Git.

Preferred: reuse the team's existing split and mapping. The SincNet export (`relative_path,label,target,split`, `class_mapping.json`) is supported directly. The original Model 2 format (`path,label,label_id,split`) is also supported.

```bash
python -m src.data_prep \
  --data_dir data/DataSEC \
  --shared_split data/shared/split_manifest.csv \
  --label_map data/shared/class_mapping.json \
  --out_dir outputs/prepared
```

Paths are relative to `--data_dir`. Use the same dataset root level that the manifest used. The importer preserves every supplied label ID and split assignment. A CSV must identify the 22 parent classes, not the 28 subclasses. Source `group_id` values, when present, must remain within one split.

If the team has **no agreed split yet**, create a candidate split and share it for agreement:

```bash
python -m src.data_prep --data_dir data/DataSEC --out_dir outputs/prepared
# Explicit metadata with parent labels is also supported:
python -m src.data_prep --data_dir data/DataSEC --csv data/meta.csv \
  --path_col filename --label_col class --out_dir outputs/prepared_from_csv
```

Folder scanning maps known parent/subclass names to the 22 parent classes and stops on ambiguity. New splits are stratified approximately 70/15/15 with seed 42. Different splitting algorithms can produce different assignments even with the same seed: share the actual CSV. If exact duplicate recordings are found, preparation stops; after the team agrees exclusions, `--drop_duplicates` explicitly retains the first path in each duplicate group and saves excluded rows. This flag is forbidden when importing a shared split. Group-aware split generation is left to the team; metadata with source groups requires an existing grouped split.

Missing, empty, unreadable or non-finite audio; inconsistent labels; duplicate audio across splits; and source-group overlap cause an error. Same-split duplicates in an existing shared manifest are recorded and retained to preserve its inventory. Exact audio hashes do not detect near-duplicates or unknown recording sessions.

Preparation writes `splits.csv`, `label_map.json`, `audit.json`, `duplicate_recordings.csv` and `class_distribution.csv` under its output directory. Generated outputs are ignored by Git. Once the team agrees, share the exact manifest and map; if they belong in the common `data/split.csv` location, the team should add them there. This Model 2 change does not write shared repository files.

## 2. Train and select on validation results

```bash
python -m src.train_efficientnet --data_dir data/DataSEC
```

Defaults use `outputs/prepared/splits.csv` and `outputs/prepared/label_map.json`. Override `--splits` and `--label_map` to use another location. Every run requires an empty/new output folder and recomputes normalization from its training recordings; old statistics and results cannot be reused accidentally.

For a short check that downloads no pretrained weights:

```bash
python -m src.train_efficientnet --data_dir data/DataSEC \
  --no_pretrained --limit 64 --epochs 2 --batch_size 8 --workers 0 \
  --out_dir outputs/debug
```

`--limit` keeps at least one example per class and marks the run as debug-only. A debug checkpoint cannot produce final test metrics. Real DataSEC defaults require all 22 classes in each split. `--expected_classes` exists for synthetic tests only.

Compare the requested ablations on **validation**, using separate run folders:

```bash
python -m src.train_efficientnet --data_dir data/DataSEC \
  --no_pretrained --out_dir outputs/effnet_scratch
python -m src.train_efficientnet --data_dir data/DataSEC \
  --class_weights --out_dir outputs/effnet_class_weights
```

## 3. Evaluate the final selected experiment

After choosing settings using validation results, evaluate the best checkpoint on the held-out test split:

```bash
python -m src.evaluate_efficientnet --data_dir data/DataSEC \
  --checkpoint outputs/efficientnet_b0/best.pt
```

This checks the saved recording inventory and decoded-audio fingerprints, then evaluates one centre crop per test recording. It refuses to overwrite an existing test evaluation. `--evaluate_test` on the training command is a convenience for an already fixed final experiment. Do not use test scores to choose ablations.

## 4. Predict a new WAV file

```bash
python -m src.predict --checkpoint outputs/efficientnet_b0/best.pt \
  --audio data/example.wav --top_k 5
```

The checkpoint contains the model, class mapping, audio settings, normalization statistics, selected epoch and run configuration. Prediction uses those saved settings. A five-second centre crop may miss events elsewhere in a long recording; this is recording classification, not temporal event detection. Softmax probabilities are not calibrated certainty estimates.

## Model and experiment settings

| Component | Setting |
|---|---|
| Input | Mono, 32 kHz; random 5-second training crop, centre evaluation crop; right zero-padding |
| Spectrogram | Power log-Mel, FFT 1,024, hop 320, 128 bins, 50–16,000 Hz, 80 dB range; shape `1 × 128 × 501` |
| Normalization | Mean/std of all unaugmented training centre-crop Mel values only |
| CNN input | Bilinear resize to `224 × 224`, replicate into three channels |
| Backbone | torchvision EfficientNet-B0 with ImageNet-1K V1 weights, or scratch ablation |
| Head | Dropout 0.2, Linear 1,280 → 22; logits for cross-entropy, softmax for prediction |
| Augmentation | Training only: circular shift ±10%, gain ±6 dB, frequency mask <24 bins, time mask <48 frames, mixup α=0.2 |
| Optimizer | AdamW: head LR 0.001, backbone LR 0.0001, weight decay 0.01 |
| Schedule | Cosine, batch 32, at most 30 epochs, patience 8 on validation macro-F1, seed 42 |
| Stability | Gradient clipping at 5; mixed precision on CUDA only |
| Metrics | Accuracy, macro-precision/recall/F1 across all class IDs, per-class scores and confusion matrices |

The model uses audio training-set normalization as specified in the interim report; it does not additionally use the ImageNet RGB normalization or image centre-crop transform. Fine-tuning adapts the pretrained features to this input. The original 1,000-class ImageNet model's parameter count is not the count of the adapted 22-class model: the adapted 22-class count is 4,035,730 and is also recorded in each run.

## Saved outputs

In `outputs/efficientnet_b0/` (or your selected run folder):

- `best.pt`, `config.json`, `norm_stats.json`, `splits.csv`, `label_map.json` preserve the experiment.
- `history.csv` and `learning_curves.png` show training and validation progress.
- `validation/` and, after explicit evaluation, `test/` contain metrics, class reports, predictions, per-class F1, misclassified recordings, and count/normalized confusion matrices.
- `metrics.json` records the selected epoch, actual parameters, elapsed training time, average epoch time and final results. `test` is null until evaluated.
- `comparison_row.csv` is created only after final evaluation, with Model 2's row for the team comparison.

Scores are fractions from 0 to 1, not percentages. Epoch time includes training plus validation. `forward_ms_per_clip` measures synchronized batched model forwards, excluding preprocessing and device transfer; `end_to_end_ms_per_clip` includes loading, preprocessing, transfer and metrics. Neither is a standalone single-clip latency benchmark. Compare hardware, batch size and crop policies along with these numbers. Reproducibility can vary across devices and library versions.

## Your remaining work

1. Confirm the shared DataSEC release, exclusions, 22-class map and split with teammates.
2. Run the short check, then train the baseline and desired validation ablations on a GPU.
3. Evaluate the selected model once; inspect weak classes and high-confidence errors.
4. Use measured outputs for your methodology, results, error analysis and computational comparison.
5. Explain MBConv, squeeze-and-excitation, transfer learning, log-Mel input, augmentation and macro-F1 in your presentation. Review and understand the code before claiming the work as your contribution.

No full-run score is supplied here. The interim report's unverified numerical entries must not be copied as results of this version.

## Verification and attribution

```bash
python -m pip install pytest
python -m pytest -q
```

Tests use synthetic WAVs and real EfficientNet forward/backward passes, including a one-epoch CPU training/evaluation/prediction cycle. They check operation, not environmental-sound classification quality. See `VALIDATION.md` for the checks actually run.

The original starter is retained and extended. ChatGPT Codex assisted with the updates to data validation, training/evaluation, prediction, tests, notebook and this documentation. The owner must review the implementation, run the real experiments and accurately disclose this assistance in the report. No reviewed/signed contribution statement or training result is claimed on the owner's behalf.

References:

- [Tan & Le (2019), EfficientNet](https://proceedings.mlr.press/v97/tan19a.html).
- [torchvision EfficientNet-B0 implementation and weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).
- [DataSEC dataset and taxonomy](https://zenodo.org/records/17033970) (use the team's agreed version).
- [Park et al. (2019), SpecAugment](https://arxiv.org/abs/1904.08779).
- [Zhang et al. (2018), mixup](https://arxiv.org/abs/1710.09412).
