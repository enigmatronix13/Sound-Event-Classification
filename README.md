# Sound Event Classification

Comparative study of four deep learning architectures — **SincNet**, **EfficientNet-B0**,
**CRNN-BiGRU**, and **Audio Spectrogram Transformer (AST)** — for environmental sound event
classification, built for the ICT 4442 Deep Learning Mini Project.

## Project overview

Environmental sound classification aims to automatically identify sound events — vehicles,
aircraft, animals, human activity, mechanical equipment — in real-world audio. Unlike speech or
music, these sounds vary widely in duration, frequency content, and background conditions, making
them a challenging classification problem.

This project trains and evaluates four architecturally distinct models under a single, shared
experimental protocol (same dataset split, same metrics) to compare how different approaches to
audio representation and feature learning affect classification performance, generalization, and
computational cost.

## Dataset

**DataSEC** — Dataset for Sound Event Classification of Environmental Noise
- Source: [Zenodo, DOI 10.5281/zenodo.15340689](https://doi.org/10.5281/zenodo.15340689)
- ~4,292–5,024 real-world recordings, ~18–23 hours total
- 22 environmental sound classes, 28 subclasses
- Mono-channel `.wav`, 44.1 kHz, one sound event per recording

## Models

| Model | Input representation | Owner |
|---|---|---|
| SincNet | Raw waveform, learnable sinc band-pass filters | Kaaviya Kalyanakumar |
| EfficientNet-B0 | Log-Mel spectrogram | Venisa Ivan Tellis |
| CRNN-BiGRU | CNN + bidirectional GRU over spectral features | Harshini Rebala |
| Audio Spectrogram Transformer (AST) | Self-attention over spectrogram patches | Ridhima Verma |


## Shared evaluation protocol

To keep the comparison meaningful, all four models are trained and tested on the **same**
stratified 70/15/15 train/val/test split (`data/split.csv`, seed=42) and reported on the same
metrics: accuracy, macro-precision, macro-recall, macro-F1, and confusion matrices.


Any use of external code, tutorials, or LLMs is cited in the report, with notes on exactly where
it was used, per course guidelines.
