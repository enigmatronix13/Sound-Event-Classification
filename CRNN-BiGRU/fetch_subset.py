# Downloads only the first 5 s of N random clips per class from the remote zip.
import struct, zlib, random, numpy as np, librosa, requests, time
from concurrent.futures import ThreadPoolExecutor
from remote_zip import *

N_PER_CLASS, SR, SECS = 25, 16000, 5
random.seed(42)
zf = zipfile.ZipFile(HTTPFile(URL))
files = [i for i in zf.infolist() if i.filename.endswith(".wav")]
classes = sorted({i.filename.split("/")[1] for i in files})
chosen = []
for c in classes:
    fs = [i for i in files if i.filename.split("/")[1] == c]
    chosen += random.sample(fs, min(N_PER_CLASS, len(fs)))

def get(a, b):
    for _ in range(5):
        r = requests.get(URL, headers={"Range": f"bytes={a}-{b}"}, timeout=60)
        if r.status_code == 206 and len(r.content) == b - a + 1:
            return r.content
        time.sleep(2)
    raise RuntimeError(f"bad range response {r.status_code} {len(r.content)} for {a}-{b}")

def load(info):
    try:
        return load_one(info)
    except Exception as e:
        print("skip", info.filename, e); return None

def load_one(info):
    h = get(info.header_offset, info.header_offset + 29)
    n, m = struct.unpack("<HH", h[26:30])
    pos = info.header_offset + 30 + n + m
    end = info.header_offset + 30 + n + m + info.compress_size - 1
    d, buf = zlib.decompressobj(-15), b""
    while pos <= end:
        stop = min(pos + 131071, end)
        buf += d.decompress(get(pos, stop)); pos = stop + 1
        i = buf.find(b"data")
        if i > 0 and len(buf) - (i + 8) >= 44100 * 2 * SECS: break   # enough for 5 s at 44.1 kHz
    i = buf.find(b"data")
    raw = buf[i + 8:]
    pcm = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
    sr = struct.unpack("<I", buf[24:28])[0]
    y = librosa.resample(pcm[: sr * SECS], orig_sr=sr, target_sr=SR)
    y = np.pad(y, (0, max(0, SR * SECS - len(y))))[: SR * SECS]
    return y

t = time.time()
with ThreadPoolExecutor(16) as ex:
    X = list(ex.map(load, chosen))
ok = [k for k, x in enumerate(X) if x is not None]
X = [X[k] for k in ok]
y = [classes.index(chosen[k].filename.split("/")[1]) for k in ok]
np.savez("subset.npz", X=np.array(X, dtype=np.float32), y=np.array(y), classes=np.array(classes))
print("done", len(X), "clips in", round(time.time() - t), "s")
