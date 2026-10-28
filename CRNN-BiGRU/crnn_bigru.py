# CRNN-BiGRU for environmental sound classification (DataSEC subset)
import time, numpy as np, librosa, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

torch.manual_seed(42); np.random.seed(42)
SR, EPOCHS = 16000, 30

# 1. Load audio and turn each clip into a log-Mel spectrogram
d = np.load("subset.npz")
waves, y, classes = d["X"], d["y"], list(d["classes"])

def logmel(w):
    m = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=1024, hop_length=512, n_mels=64)
    return librosa.power_to_db(m, ref=np.max)

X = np.array([logmel(w) for w in waves])          # (N, 64 mels, 157 frames)
X = (X - X.mean()) / X.std()
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)   # (N, 1, 64, 157)
y = torch.tensor(y)

# 2. Split 70 / 15 / 15, stratified by class
idx = np.arange(len(y))
tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=42)

# 3. Model: CNN blocks -> bidirectional GRU -> linear classifier
class CRNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d((2, 2)))
        self.cnn = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))   # 64 mels -> 8
        self.gru = nn.GRU(64 * 8, 64, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x):
        x = self.cnn(x)                                # (B, C, mels, time)
        x = x.permute(0, 3, 1, 2).flatten(2)           # (B, time, C*mels)
        x, _ = self.gru(x)
        return self.fc(self.drop(x.mean(dim=1)))       # average over time

model = CRNN(len(classes))
params = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def run(ids, train):
    model.train(train)
    order = np.random.permutation(ids) if train else ids
    tot, correct, preds = 0.0, 0, []
    for i in range(0, len(order), 32):
        b = order[i:i + 32]
        with torch.set_grad_enabled(train):
            out = model(X[b]); loss = loss_fn(out, y[b])
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
        p = out.argmax(1); preds += p.tolist(); correct += (p == y[b]).sum().item()
    return tot / len(ids), correct / len(ids), preds

# 4. Train, keep the best model on validation accuracy
hist = {"tl": [], "vl": [], "ta": [], "va": []}
best, t0 = 0, time.time()
for ep in range(EPOCHS):
    tl, ta, _ = run(tr, True); vl, vacc, _ = run(va, False)
    for k, v in zip(hist, (tl, vl, ta, vacc)): hist[k].append(v)
    if vacc > best:
        best = vacc; torch.save(model.state_dict(), "best_crnn.pt")
    print(f"epoch {ep+1:2d}  train loss {tl:.3f} acc {ta:.3f}  val loss {vl:.3f} acc {vacc:.3f}")
train_time = time.time() - t0

# 5. Test the best model
model.load_state_dict(torch.load("best_crnn.pt"))
_, _, pred = run(te, False)
true = y[te].numpy()
res = {"Accuracy": accuracy_score(true, pred),
       "Macro-precision": precision_score(true, pred, average="macro", zero_division=0),
       "Macro-recall": recall_score(true, pred, average="macro", zero_division=0),
       "Macro-F1": f1_score(true, pred, average="macro", zero_division=0)}
lines = [f"{k}: {v:.4f}" for k, v in res.items()]
lines += [f"Parameters: {params:,}", f"Training time: {train_time/60:.1f} min (CPU)",
          f"Train/val/test clips: {len(tr)}/{len(va)}/{len(te)}"]
open("results.txt", "w").write("\n".join(lines))
print("\n".join(lines))

# 6. Plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(hist["tl"], label="train"); ax[0].plot(hist["vl"], label="val"); ax[0].set_title("Loss")
ax[1].plot(hist["ta"], label="train"); ax[1].plot(hist["va"], label="val"); ax[1].set_title("Accuracy")
for a in ax: a.set_xlabel("epoch"); a.legend()
plt.tight_layout(); plt.savefig("curves.png", dpi=120)

fig, a = plt.subplots(figsize=(11, 11))
ConfusionMatrixDisplay(confusion_matrix(true, pred, labels=range(len(classes))),
                       display_labels=[c[:14] for c in classes]).plot(ax=a, xticks_rotation=90, colorbar=False)
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120)
