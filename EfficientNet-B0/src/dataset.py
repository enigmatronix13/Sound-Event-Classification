"""Log-Mel dataset for DataSEC (used by the EfficientNet-B0 model)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass
class AudioConfig:
    sample_rate: int = 32000
    duration: float = 5.0        # seconds, fixed clip length
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 128
    f_min: float = 50.0
    f_max: float = 16000.0
    top_db: float = 80.0
    # augmentation (train only)
    max_shift: float = 0.1       # fraction of clip for random circular shift
    gain_db: float = 6.0         # random gain in [-gain_db, +gain_db]
    freq_mask: int = 24
    time_mask: int = 48

    @property
    def n_samples(self):
        return int(self.sample_rate * self.duration)


class LogMelDataset(Dataset):
    """Returns (log-Mel tensor [1, n_mels, T], label_id)."""

    def __init__(self, df, data_dir, cfg: AudioConfig, train=False, mean=0.0, std=1.0):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.train = train
        self.mean, self.std = float(mean), float(std)
        self._resamplers = {}
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2.0)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=cfg.top_db)
        self.fmask = torchaudio.transforms.FrequencyMasking(cfg.freq_mask)
        self.tmask = torchaudio.transforms.TimeMasking(cfg.time_mask)

    def __len__(self):
        return len(self.df)

    def _load(self, rel_path):
        wav, sr = sf.read(str(self.data_dir / rel_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))                 # mono
        if sr != self.cfg.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.cfg.sample_rate)
            wav = self._resamplers[sr](wav)
        return wav

    def _fix_length(self, wav):
        n = self.cfg.n_samples
        if wav.numel() >= n:
            start = np.random.randint(0, wav.numel() - n + 1) if self.train else (wav.numel() - n) // 2
            return wav[start:start + n]
        return torch.nn.functional.pad(wav, (0, n - wav.numel()))   # zero-pad short clips

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = self._fix_length(self._load(row["path"]))
        if self.train:
            shift = int(np.random.uniform(-self.cfg.max_shift, self.cfg.max_shift) * wav.numel())
            wav = torch.roll(wav, shift)
            wav = wav * 10 ** (np.random.uniform(-self.cfg.gain_db, self.cfg.gain_db) / 20)
        spec = self.to_db(self.mel(wav)).unsqueeze(0)             # [1, n_mels, T]
        spec = (spec - self.mean) / (self.std + 1e-8)
        if self.train:                                            # SpecAugment masking
            spec = self.tmask(self.fmask(spec))
        return spec, int(row["label_id"])
