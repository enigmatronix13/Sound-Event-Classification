# Model 2 verification

Verified on 28 October 2026, using Python 3.11.9 on macOS Apple Silicon, **CPU** execution. Libraries: torch 2.14.1, torchvision 0.29.1, torchaudio 2.11.0, NumPy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1 and matplotlib 3.11.2. These record the tested environment; they are not a claim of CUDA or MPS validation.

## Checks completed

- 21 regression tests passed (18 initial tests, then 3 additional data-integrity tests).
- Real EfficientNet-B0 forward/backward pass with a 22-class head, including a one-example training batch.
- One-epoch CPU training on eight synthetic WAV files across two classes; four train, two validation and two test recordings. The run exercises class-weighted mixup, checkpoint selection, safe checkpoint reload, explicit test evaluation, artifact export and single-file prediction.
- ImageNet-1K pretrained weights downloaded and loaded successfully; one `1 × 1 × 128 × 501` input produced finite `1 × 22` logits.
- Adapted model parameter count measured: **4,035,730**.
- Parent/subclass mapping, exact decoded-audio duplicates, conflicting labels, missing/non-finite audio, source-group leakage, class-map mismatch and path validation checked.
- Existing SincNet-format manifest imported with unchanged labels and split assignments.
- New split preparation exercised through its command-line entry point on 40 synthetic recordings.
- Training-only normalization, deterministic evaluation features, all-class macro metrics, small batches and stratified debug sampling checked.
- Debug checkpoints refused final evaluation; changing audio after training was detected; repeated test exports were refused.
- Notebook schema validated with nbformat; all 12 code cells parsed successfully and contain no claimed execution outputs.
- Generated confusion-matrix output inspected. Repository diff checked for whitespace issues and confined to `EfficientNet-B0/`.

Run the suite from this folder with `python -m pytest -q` after installing requirements and pytest. On a CPU, setting `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` can reduce contention in the synthetic integration test.

## Still pending

Full DataSEC acquisition/audit on the team's agreed release, agreed shared manifest, full baseline and ablation training, CUDA mixed-precision execution, final held-out metrics, and the complete notebook workflow on real data. Synthetic test outputs are temporary test fixtures and are **not** committed as project results. No real accuracy/F1 claim or trained DataSEC checkpoint is supplied.
