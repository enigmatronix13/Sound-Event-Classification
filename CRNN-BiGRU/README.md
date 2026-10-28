# Model 3: CRNN-BiGRU (Harshini Rebala)

CNN (3 conv blocks) on log-Mel spectrograms followed by a 2-layer bidirectional GRU.

- `fetch_subset.py` / `remote_zip.py`: download the first 5 s of up to 25 random clips per class from the DataSEC zip (subset of 307 clips, 16 kHz).
- `crnn_bigru.py`: log-Mel features, 70/15/15 stratified split (seed 42), training (Adam, 30 epochs, CPU) and evaluation.
- `results.txt`, `curves.png`, `confusion_matrix.png`: test results.

Test accuracy 0.702, macro-F1 0.644, 322,806 parameters, 0.3 min training on CPU.
Note: small subset, so scores are rough.
