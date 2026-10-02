# DataSEC Environmental Sound Classification — EfficientNet-B0 (Model 2)

ICT 4442 Deep Learning Mini Project. Owner: Venisa Ivan Tellis (230911064).

## Setup
```bash
pip install -r requirements.txt
```
Download DataSEC from Zenodo (DOI 10.5281/zenodo.15340689) into `data/DataSEC/`.

## 1. Common split (all models must use this)
```bash
# folder-per-class layout
python -m src.data_prep --data_dir data/DataSEC
# or, if labels are in a CSV
python -m src.data_prep --data_dir data/DataSEC --csv meta.csv --path_col filename --label_col class
```
Creates `outputs/splits.csv` (stratified 70/15/15, seed 42) and `outputs/label_map.json`.
Commit these two files so teammates use the identical split.

## 2. Train + evaluate EfficientNet-B0
```bash
python -m src.train_efficientnet --data_dir data/DataSEC
# quick debug run
python -m src.train_efficientnet --data_dir data/DataSEC --limit 64 --epochs 2 --workers 0
# ablations
python -m src.train_efficientnet --data_dir data/DataSEC --no_pretrained --out_dir outputs/effnet_scratch
python -m src.train_efficientnet --data_dir data/DataSEC --class_weights --out_dir outputs/effnet_cw
```
Results go to `outputs/efficientnet_b0/` (`metrics.json`, `confusion_matrix.png`,
`classification_report.txt`, `per_class_f1.csv`, `misclassified.csv`, `history.csv`).

## Defaults (match the interim report)
32 kHz, 5 s clips, n_fft 1024, hop 320, 128 Mel bins, 50 Hz–16 kHz, dB scale, train-set
mean/std normalisation, 1→3 channels, 224×224; SpecAugment masking, random shift/gain, mixup α=0.2;
ImageNet-pretrained EfficientNet-B0, dropout 0.2, AdamW (head 1e-3, backbone 1e-4, wd 1e-2),
batch 32, 30 epochs, cosine schedule, early stopping on val macro-F1, seed 42.

## Academic integrity note
Parts of this code were written with LLM assistance (Claude). State this, and where it was used,
in the report as required by the course guidelines.
