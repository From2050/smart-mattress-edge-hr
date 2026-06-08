#!/usr/bin/env python3
"""
thesis_unified/common.py
統一復驗共用模組 — 所有模型、資料載入、訓練、評估函數

33 subjects (EXCLUDE=[100, 200]), CNN+MLP 為主模型
所有實驗共用相同的超參數與後處理管線
"""

import os, sys, glob, time, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal as sp_signal, ndimage, stats as sp_stats
from scipy.fft import fft, fftfreq
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════
# Paths & Constants
# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR    = os.path.join(ROOT_DIR, 'data', 'ESP32_recored')
S1_OUT      = os.path.join(ROOT_DIR, 'outputs', 'stage1')
OUTPUT_ROOT = os.path.join(ROOT_DIR, 'outputs', 'thesis_unified')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

SAMPLING_RATE   = 13.5
WINDOW_SAMPLES  = 128
OVERLAP_SAMPLES = 64
NUM_CHANNELS    = 32
SENSOR_ROWS     = 8
SENSOR_COLS     = 4
EXCLUDE_SUBJECTS = [100, 200]      # 33 subjects

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 0.001
SEED       = 42

# Phase alignment delay model path
_DELAY_MODEL_PATH = os.path.join(OUTPUT_ROOT, 'exp11_phase_aligned', 'delay_model.json')
_DELAY_TABLE = None   # lazy-loaded cache: np.array (32,) fractional samples


def _load_delay_table():
    """Load per-channel fractional-sample delays (lazy, cached)."""
    global _DELAY_TABLE
    if _DELAY_TABLE is not None:
        return _DELAY_TABLE
    import json
    if not os.path.exists(_DELAY_MODEL_PATH):
        raise FileNotFoundError(
            f"Delay model not found: {_DELAY_MODEL_PATH}\n"
            "Run calibrate_phase_delay.py first."
        )
    with open(_DELAY_MODEL_PATH, 'r') as f:
        model = json.load(f)
    _DELAY_TABLE = np.array(model['delays_frac_samples'], dtype=np.float64)
    return _DELAY_TABLE


def align_phase(raw_adc, delay_table=None):
    """Compensate inter-channel phase offset via FFT phase rotation.

    Each channel c was sampled delay_table[c] samples AFTER ch0.
    To align all channels to ch0's time, we advance channel c by
    delay_table[c] fractional samples using the Fourier shift theorem:

        X_aligned[k] = X[k] · exp(+j·2π·k·Δn/N)

    Parameters
    ----------
    raw_adc : np.ndarray, shape (N, 32)
        Raw ADC window (time × channels)
    delay_table : np.ndarray, shape (32,), optional
        Fractional sample delays. If None, loads from delay_model.json.

    Returns
    -------
    np.ndarray, shape (N, 32) — phase-aligned signals (real-valued)
    """
    if delay_table is None:
        delay_table = _load_delay_table()

    N, C = raw_adc.shape
    aligned = np.empty_like(raw_adc, dtype=np.float64)
    freqs = np.fft.fftfreq(N)  # k/N for k = 0..N-1

    for c in range(C):
        dn = delay_table[c]
        if abs(dn) < 1e-8:
            aligned[:, c] = raw_adc[:, c]
            continue
        X = np.fft.fft(raw_adc[:, c].astype(np.float64))
        # Advance by dn samples: multiply by exp(+j·2π·freq·dn)
        X *= np.exp(1j * 2 * np.pi * freqs * dn)
        aligned[:, c] = np.fft.ifft(X).real

    return aligned


# ═══════════════════════════════════════════════════════════════════════
# Signal Processing
# ═══════════════════════════════════════════════════════════════════════
def calculate_spectrum(sig, fs=13.5, n_fft=2048):
    sig = sig - np.mean(sig)
    sig = sig * np.hanning(len(sig))
    yf = fft(sig, n=n_fft)
    xf = fftfreq(n_fft, 1 / fs)
    m = xf > 0
    return xf[m], np.abs(yf[m]) ** 2


# ═══════════════════════════════════════════════════════════════════════
# SQI Filter (replicates stage2/ppg_sqi.py logic)
# ═══════════════════════════════════════════════════════════════════════
def compute_channel_skewness(raw_ch, fs=13.5):
    """Bandpass filter + skewness (same as ppg_sqi.calculate_sqi_metrics)."""
    nyq = fs / 2
    b, a = sp_signal.butter(4, [0.5 / nyq, 4.0 / nyq], btype='band')
    try:
        filt = sp_signal.filtfilt(b, a, raw_ch)
    except Exception:
        return 0.0
    return float(sp_stats.skew(filt))


def apply_sqi_filter(weights, raw_adc, skew_thresh=0.05):
    """Zero out channels with skewness <= threshold."""
    mask = np.zeros(NUM_CHANNELS)
    for ch in range(NUM_CHANNELS):
        skew = compute_channel_skewness(raw_adc[:, ch])
        if skew > skew_thresh:
            mask[ch] = weights[ch]
    return mask


# ═══════════════════════════════════════════════════════════════════════
# SpectrumFuser (unified: sigma=0.0, alpha=0.8, Top-3 peak voting)
# ═══════════════════════════════════════════════════════════════════════
class SpectrumFuser:
    """Gaussian Peak Voting + Viterbi — unified version."""

    def __init__(self, rows=8, cols=4, sigma=0.0, alpha=0.8):
        self.rows  = rows
        self.cols  = cols
        self.sigma = sigma
        self.alpha = alpha
        self.prev_weights = None

    def reset_history(self):
        self.prev_weights = None

    def calculate_spatial_weights(self, valid_mask):
        grid = valid_mask.reshape(self.rows, self.cols).astype(float)
        if self.sigma > 0:
            grid = ndimage.gaussian_filter(grid, self.sigma)
        current = grid.flatten()
        if self.prev_weights is not None:
            final = self.alpha * current + (1 - self.alpha) * self.prev_weights
        else:
            final = current
        self.prev_weights = final
        return final

    def build_consensus_spectrum(self, spectra_list, weights,
                                 hr_range=(0.75, 3.0)):
        grid_res = 0.01
        grid_freqs = np.arange(hr_range[0], hr_range[1], grid_res)
        consensus  = np.zeros_like(grid_freqs)
        sigma = 0.05
        has_votes = False

        for i, (f, p) in enumerate(spectra_list):
            w = weights[i]
            if w <= 0.01:
                continue
            mask = (f >= hr_range[0]) & (f <= hr_range[1])
            vf, vp = f[mask], p[mask]
            if len(vp) == 0 or np.max(vp) == 0:
                continue
            peaks, _ = sp_signal.find_peaks(vp, height=np.max(vp) * 0.3)
            if len(peaks) > 0:
                top3 = peaks[np.argsort(vp[peaks])[::-1][:3]]
                for idx in top3:
                    consensus += w * np.exp(
                        -0.5 * ((grid_freqs - vf[idx]) / sigma) ** 2)
                    has_votes = True

        if not has_votes:
            return grid_freqs, np.zeros_like(grid_freqs)
        return grid_freqs, consensus

    def build_peaked_spectrum(self, hr_bpm, hr_range=(0.75, 3.0)):
        """Convert a single HR estimate to a peaked Gaussian spectrum."""
        grid_res = 0.01
        grid_freqs = np.arange(hr_range[0], hr_range[1], grid_res)
        freq = np.clip(hr_bpm / 60.0, hr_range[0], hr_range[1])
        sigma = 0.05
        spectrum = np.exp(-0.5 * ((grid_freqs - freq) / sigma) ** 2)
        return grid_freqs, spectrum

    def get_raw_hr(self, grid_freqs, consensus):
        if consensus is None or np.max(consensus) == 0:
            return 0.0
        peaks, _ = sp_signal.find_peaks(
            consensus, height=np.max(consensus) * 0.3)
        if len(peaks) == 0:
            return grid_freqs[np.argmax(consensus)] * 60.0
        return grid_freqs[peaks[np.argmax(consensus[peaks])]] * 60.0

    def solve_viterbi(self, observations, grid_freqs):
        T = len(observations)
        if T == 0:
            return []
        N = len(grid_freqs)
        dp  = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)

        trans_sigma = 0.05
        fd2 = (grid_freqs[:, None] - grid_freqs[None, :]) ** 2
        lt  = np.log(np.exp(-0.5 * fd2 / trans_sigma ** 2) + 1e-10)

        dp[0] = np.log(observations[0] + 1e-10)

        for t in range(1, T):
            ol  = np.log(observations[t] + 1e-10)
            mat = dp[t - 1][:, None] + lt
            bp  = np.argmax(mat, axis=0)
            dp[t]  = mat[bp, np.arange(N)] + ol
            psi[t] = bp

        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(dp[-1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return grid_freqs[path] * 60.0


# ═══════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════

# ── 1. SpatialAttentionCNN (Full: CNN + Transformer, ~355K) ──
class SpatialAttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.flat_size = 64 * 16
        self.fc_feat   = nn.Linear(self.flat_size, 64)
        self.dropout   = nn.Dropout(0.5)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=2048,
            dropout=0.1, batch_first=True,
        )
        self.fc_out  = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, L = x.size()
        x = x.view(-1, 1, L)
        h = self.pool(torch.relu(self.conv1(x)))
        h = self.pool(torch.relu(self.conv2(h)))
        h = self.pool(torch.relu(self.conv3(h)))
        h = h.view(h.size(0), -1)
        h = torch.relu(self.fc_feat(h))
        h = self.dropout(h)
        h = h.view(B, C, -1)
        h = self.transformer(h)
        return self.sigmoid(self.fc_out(h)).squeeze(-1)


# ── 2. NoAttentionCNN (CNN only, ~74K) ──
class NoAttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.flat_size = 64 * 16
        self.fc_feat   = nn.Linear(self.flat_size, 64)
        self.dropout   = nn.Dropout(0.5)
        self.fc_out  = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, L = x.size()
        x = x.view(-1, 1, L)
        h = self.pool(torch.relu(self.conv1(x)))
        h = self.pool(torch.relu(self.conv2(h)))
        h = self.pool(torch.relu(self.conv3(h)))
        h = h.view(h.size(0), -1)
        h = torch.relu(self.fc_feat(h))
        h = self.dropout(h)
        out = self.sigmoid(self.fc_out(h))
        return out.view(B, C)


# ── 3. NoCNN_Transformer (Linear + Transformer) ──
class NoCNN_Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_proj = nn.Linear(128, 64)
        self.dropout     = nn.Dropout(0.5)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=2048,
            dropout=0.1, batch_first=True,
        )
        self.fc_out  = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = torch.relu(self.linear_proj(x))
        h = self.dropout(h)
        h = self.transformer(h)
        return self.sigmoid(self.fc_out(h)).squeeze(-1)


# ── 3b. NoCNN_MLP_Large (MLP-only, parameter-matched to ~353K) ──
class NoCNN_MLP_Large(nn.Module):
    def __init__(self):
        super().__init__()
        # Per-channel MLP on raw 128-point signal without any CNN/Transformer.
        # Widths chosen to keep capacity close to NoAttentionCNN_Large.
        self.fc1 = nn.Linear(128, 480)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(480, 480)
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(480, 128)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        B, C, L = x.size()
        h = x.view(-1, L)
        h = torch.relu(self.fc1(h))
        h = self.drop1(h)
        h = torch.relu(self.fc2(h))
        h = self.drop2(h)
        h = torch.relu(self.fc3(h))
        return torch.sigmoid(self.fc_out(h)).view(B, C)


# ── 4. NoAttentionCNN_Large (Capacity-matched CNN+MLP, ~353K) ── ★ Main Model
class NoAttentionCNN_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.flat_size = 64 * 16
        self.fc1     = nn.Linear(self.flat_size, 256)
        self.drop1   = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 256)
        self.drop2   = nn.Dropout(0.1)
        self.fc3     = nn.Linear(256, 64)
        self.fc_out  = nn.Linear(64, 1)

    def forward(self, x):
        B, C, L = x.size()
        x = x.view(-1, 1, L)
        h = self.pool(torch.relu(self.conv1(x)))
        h = self.pool(torch.relu(self.conv2(h)))
        h = self.pool(torch.relu(self.conv3(h)))
        h = h.view(h.size(0), -1)
        h = torch.relu(self.fc1(h))
        h = self.drop1(h)
        h = torch.relu(self.fc2(h))
        h = self.drop2(h)
        h = torch.relu(self.fc3(h))
        return torch.sigmoid(self.fc_out(h)).view(B, C)


# ── 5. DeepCNN_5L (5-layer deep CNN + MLP) ──
class DeepCNN_5L(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv1d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv1d(64, 64, 3, padding=1)
        self.flat_size = 64 * 4
        self.fc1    = nn.Linear(self.flat_size, 256)
        self.drop1  = nn.Dropout(0.5)
        self.fc2    = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        B, C, L = x.size()
        x = x.view(-1, 1, L)
        h = self.pool(torch.relu(self.conv1(x)))
        h = self.pool(torch.relu(self.conv2(h)))
        h = self.pool(torch.relu(self.conv3(h)))
        h = self.pool(torch.relu(self.conv4(h)))
        h = self.pool(torch.relu(self.conv5(h)))
        h = h.view(h.size(0), -1)
        h = torch.relu(self.fc1(h))
        h = self.drop1(h)
        h = torch.relu(self.fc2(h))
        return torch.sigmoid(self.fc_out(h)).view(B, C)


# ── 6. WiderCNN (wider 3-layer CNN + MLP) ──
class WiderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, padding=3)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.flat_size = 128 * 16
        self.fc1    = nn.Linear(self.flat_size, 128)
        self.drop1  = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        B, C, L = x.size()
        x = x.view(-1, 1, L)
        h = self.pool(torch.relu(self.conv1(x)))
        h = self.pool(torch.relu(self.conv2(h)))
        h = self.pool(torch.relu(self.conv3(h)))
        h = h.view(h.size(0), -1)
        h = torch.relu(self.fc1(h))
        h = self.drop1(h)
        return torch.sigmoid(self.fc_out(h)).view(B, C)


# ── 7. CNN_CrossMLP (CNN + cross-channel MLP) ──
class CNN_CrossMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.flat_size = 64 * 16
        self.fc_feat = nn.Linear(self.flat_size, 64)
        self.drop    = nn.Dropout(0.5)
        self.cross_fc1 = nn.Linear(64 * 32, 512)
        self.cross_fc2 = nn.Linear(512, 256)
        self.cross_fc3 = nn.Linear(256, 32)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        B, C, L = x.size()
        x = x.view(-1, 1, L)
        h = self.pool(torch.relu(self.conv1(x)))
        h = self.pool(torch.relu(self.conv2(h)))
        h = self.pool(torch.relu(self.conv3(h)))
        h = h.view(h.size(0), -1)
        h = torch.relu(self.fc_feat(h))
        h = self.drop(h)
        h = h.view(B, C * 64)
        h = torch.relu(self.cross_fc1(h))
        h = torch.relu(self.cross_fc2(h))
        return self.sigmoid(self.cross_fc3(h))


# ── 8. PaperCNN_Regression (11-layer AllConv, MSE, ~16K) ──
class PaperCNN_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 8, 7, stride=2, padding=3),
            nn.BatchNorm1d(8), nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 8, 5, stride=1, padding=2),
            nn.BatchNorm1d(8), nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 16, 5, stride=2, padding=2),
            nn.BatchNorm1d(16), nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 16, 3, stride=2, padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 2, stride=2, padding=0),
            nn.BatchNorm1d(32), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 1, 1, stride=1),
        )

    def forward(self, x):
        B, C, L = x.size()
        x = x.view(-1, 1, L)
        h = self.layers(x)
        return h.view(B, C)


# ── 9. PaperCNN_Reliability (11-layer AllConv + sigmoid, ~16K) ──
class PaperCNN_Reliability(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 8, 7, stride=2, padding=3),
            nn.BatchNorm1d(8), nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 8, 5, stride=1, padding=2),
            nn.BatchNorm1d(8), nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 16, 5, stride=2, padding=2),
            nn.BatchNorm1d(16), nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 16, 3, stride=2, padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 2, stride=2, padding=0),
            nn.BatchNorm1d(32), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 1, 1, stride=1),
        )

    def forward(self, x):
        B, C, L = x.size()
        x = x.view(-1, 1, L)
        h = self.layers(x)
        return torch.sigmoid(h.view(B, C))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════
class ReliabilityDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return torch.FloatTensor(it['signal']), torch.FloatTensor(it['labels'])


class RegressionDataset(Dataset):
    def __init__(self, items, hr_mean, hr_std):
        self.items = items
        self.hr_mean = hr_mean
        self.hr_std = hr_std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        z_hr = (it['true_hr'] - self.hr_mean) / (self.hr_std + 1e-6)
        target = np.full(NUM_CHANNELS, z_hr, dtype=np.float32)
        return torch.FloatTensor(it['signal']), torch.FloatTensor(target)


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════
def load_all_data(csv_path=None, data_root=None, exclude_subjects=None,
                  phase_align=False):
    if csv_path is None:
        csv_path = os.path.join(S1_OUT, 'channel_level_annotations.csv')
    if data_root is None:
        data_root = DATA_DIR
    if exclude_subjects is None:
        exclude_subjects = EXCLUDE_SUBJECTS

    df = pd.read_csv(csv_path)
    df = df[~df['subject'].isin(exclude_subjects)]
    df = df[(df['true_hr'] > 40) & (df['true_hr'] < 160)]

    label_map, hr_map = {}, {}
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Build label map"):
        key = (r['subject'], r['posture'], r['window_idx'])
        if key not in label_map:
            label_map[key] = np.zeros(NUM_CHANNELS)
            hr_map[key] = r['true_hr']
        ch = int(r['channel'])
        if 0 <= ch < NUM_CHANNELS:
            label_map[key][ch] = 1.0 if r['is_reliable'] else 0.0

    items = []
    grouped = df.groupby(['subject', 'posture'])
    stride = WINDOW_SAMPLES - OVERLAP_SAMPLES

    for (subj, post), grp in tqdm(grouped, desc="Load raw signals"):
        spath = os.path.join(data_root, str(subj), post)
        files = glob.glob(os.path.join(spath, '*.txt'))
        if not files:
            continue
        try:
            with open(files[0], 'r') as f:
                lines = f.readlines()
            adc = []
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 33:
                    adc.append([float(x) for x in parts[1:33]])
            adc = np.array(adc)
        except Exception:
            continue

        for wi in grp['window_idx'].unique():
            key = (subj, post, wi)
            if key not in label_map:
                continue
            s, e = wi * stride, wi * stride + WINDOW_SAMPLES
            if e > len(adc):
                continue
            raw = adc[s:e, :]
            if phase_align:
                raw = align_phase(raw)
            proc = np.zeros((NUM_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
            for c in range(NUM_CHANNELS):
                sig = raw[:, c]
                med = np.median(sig)
                mad = np.median(np.abs(sig - med))
                proc[c] = (sig - med) / (mad + 1e-6)
            items.append({
                'subject': subj, 'posture': post, 'window_idx': wi,
                'signal': proc, 'labels': label_map[key],
                'true_hr': hr_map[key], 'raw_adc': raw,
            })
    return items


# ═══════════════════════════════════════════════════════════════════════
# Seed & Training
# ═══════════════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_reliability(model, train_items, epochs=EPOCHS, bs=BATCH_SIZE, lr=LR):
    """Train: BCE loss on reliability labels."""
    model.train()
    loader = DataLoader(ReliabilityDataset(train_items), batch_size=bs,
                        shuffle=True, num_workers=0,
                        pin_memory=(DEVICE.type == 'cuda'))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def train_regression(model, train_items, epochs=EPOCHS, bs=BATCH_SIZE, lr=LR):
    """Train paper model: MSE loss on Z-scored true HR.
    Returns (model, hr_mean, hr_std).
    """
    hrs = np.array([it['true_hr'] for it in train_items])
    hr_mean, hr_std = float(hrs.mean()), float(hrs.std())

    model.train()
    loader = DataLoader(RegressionDataset(train_items, hr_mean, hr_std),
                        batch_size=bs, shuffle=True, num_workers=0,
                        pin_memory=(DEVICE.type == 'cuda'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model, hr_mean, hr_std


# ═══════════════════════════════════════════════════════════════════════
# Evaluation — Reliability-based (BCE models)
# ═══════════════════════════════════════════════════════════════════════
def evaluate_reliability(model, test_items, fuser, use_sqi=False):
    """Reliability weights → spectrum fusion → Viterbi."""
    model.eval()
    by_posture = {}
    for it in test_items:
        by_posture.setdefault(it['posture'], []).append(it)

    results = []
    for pos, p_items in by_posture.items():
        p_items.sort(key=lambda x: x['window_idx'])

        loader = DataLoader(ReliabilityDataset(p_items), batch_size=32,
                            shuffle=False)
        all_weights = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(DEVICE)
                all_weights.append(model(X_batch).cpu().numpy())
        all_weights = np.concatenate(all_weights, axis=0)

        fuser.reset_history()
        consensus_buf, raw_hrs, grid_ref = [], [], None

        for k, it in enumerate(p_items):
            weights = all_weights[k]
            if use_sqi:
                weights = apply_sqi_filter(weights, it['raw_adc'])
            smoothed = fuser.calculate_spatial_weights(weights)
            spectra = [calculate_spectrum(it['raw_adc'][:, ch])
                       for ch in range(NUM_CHANNELS)]
            gf, cs = fuser.build_consensus_spectrum(spectra, smoothed)
            grid_ref = gf
            consensus_buf.append(cs)
            raw_hrs.append(fuser.get_raw_hr(gf, cs))

        if consensus_buf and grid_ref is not None:
            vit_hrs = fuser.solve_viterbi(consensus_buf, grid_ref)
            for j, it in enumerate(p_items):
                if j < len(vit_hrs):
                    results.append({
                        'subject': it['subject'], 'posture': pos,
                        'window_idx': it['window_idx'],
                        'true_hr': it['true_hr'],
                        'raw_hr': raw_hrs[j], 'viterbi_hr': vit_hrs[j],
                        'abs_err_raw': abs(raw_hrs[j] - it['true_hr']),
                        'abs_err_viterbi': abs(vit_hrs[j] - it['true_hr']),
                    })
    return results


# ═══════════════════════════════════════════════════════════════════════
# Evaluation — Regression-based (Paper method: MSE)
# ═══════════════════════════════════════════════════════════════════════
def evaluate_regression(model, test_items, fuser, hr_mean, hr_std):
    """Paper: per-channel HR regression → median → Viterbi."""
    model.eval()
    by_posture = {}
    for it in test_items:
        by_posture.setdefault(it['posture'], []).append(it)

    results = []
    for pos, p_items in by_posture.items():
        p_items.sort(key=lambda x: x['window_idx'])

        loader = DataLoader(ReliabilityDataset(p_items), batch_size=32,
                            shuffle=False)
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(DEVICE)
                all_preds.append(model(X_batch).cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)

        all_hr_preds = all_preds * hr_std + hr_mean
        median_hrs = np.clip(np.median(all_hr_preds, axis=1), 45, 180)

        fuser.reset_history()
        consensus_buf, raw_hrs, grid_ref = [], [], None
        for k, it in enumerate(p_items):
            gf, spec = fuser.build_peaked_spectrum(median_hrs[k])
            grid_ref = gf
            consensus_buf.append(spec)
            raw_hrs.append(float(median_hrs[k]))

        if consensus_buf and grid_ref is not None:
            vit_hrs = fuser.solve_viterbi(consensus_buf, grid_ref)
            for j, it in enumerate(p_items):
                if j < len(vit_hrs):
                    results.append({
                        'subject': it['subject'], 'posture': pos,
                        'window_idx': it['window_idx'],
                        'true_hr': it['true_hr'],
                        'raw_hr': raw_hrs[j], 'viterbi_hr': vit_hrs[j],
                        'abs_err_raw': abs(raw_hrs[j] - it['true_hr']),
                        'abs_err_viterbi': abs(vit_hrs[j] - it['true_hr']),
                    })
    return results


# ═══════════════════════════════════════════════════════════════════════
# Evaluation — Conventional baselines
# ═══════════════════════════════════════════════════════════════════════
def get_conventional_weights(method, raw_adc, use_sqi=False):
    w = np.zeros(NUM_CHANNELS)
    sqi_pass = np.ones(NUM_CHANNELS, dtype=bool)
    if use_sqi:
        for ch in range(NUM_CHANNELS):
            skew = compute_channel_skewness(raw_adc[:, ch])
            if skew <= 0.05:
                sqi_pass[ch] = False

    if method == 'avg':
        if use_sqi and sqi_pass.any():
            w[sqi_pass] = 1.0
        else:
            w[:] = 1.0
    elif method == 'max':
        variances = np.var(raw_adc, axis=0)
        if use_sqi and sqi_pass.any():
            masked_var = np.where(sqi_pass, variances, -1.0)
            w[np.argmax(masked_var)] = 1.0
        else:
            w[np.argmax(variances)] = 1.0
    return w


def evaluate_conventional(method, test_items, fuser, use_sqi=False):
    by_posture = {}
    for it in test_items:
        by_posture.setdefault(it['posture'], []).append(it)

    results = []
    for pos, p_items in by_posture.items():
        p_items.sort(key=lambda x: x['window_idx'])
        fuser.reset_history()
        consensus_buf, raw_hrs, grid_ref, meta = [], [], None, []

        for it in p_items:
            w = get_conventional_weights(method, it['raw_adc'], use_sqi)
            sw = fuser.calculate_spatial_weights(w)
            spectra = [calculate_spectrum(it['raw_adc'][:, ch])
                       for ch in range(NUM_CHANNELS)]
            gf, cs = fuser.build_consensus_spectrum(spectra, sw)
            grid_ref = gf
            consensus_buf.append(cs)
            raw_hrs.append(fuser.get_raw_hr(gf, cs))
            meta.append(it)

        if consensus_buf and grid_ref is not None:
            vit_hrs = fuser.solve_viterbi(consensus_buf, grid_ref)
            for j, it in enumerate(meta):
                if j < len(vit_hrs):
                    results.append({
                        'subject': it['subject'], 'posture': pos,
                        'window_idx': it['window_idx'],
                        'true_hr': it['true_hr'],
                        'raw_hr': raw_hrs[j], 'viterbi_hr': vit_hrs[j],
                        'abs_err_raw': abs(raw_hrs[j] - it['true_hr']),
                        'abs_err_viterbi': abs(vit_hrs[j] - it['true_hr']),
                    })
    return results


# ═══════════════════════════════════════════════════════════════════════
# Summary Helpers
# ═══════════════════════════════════════════════════════════════════════
def make_fuser():
    """Create standard SpectrumFuser with unified settings."""
    return SpectrumFuser(rows=SENSOR_ROWS, cols=SENSOR_COLS,
                         sigma=0.0, alpha=0.8)


def summarize_results(df, group_col='experiment'):
    """Compute per-condition summary: MAE, Std, Acc5 (raw and Viterbi)."""
    return (df.groupby(group_col)
            .agg(
                MAE_raw=('abs_err_raw', 'mean'),
                Std_raw=('abs_err_raw', 'std'),
                MAE_viterbi=('abs_err_viterbi', 'mean'),
                Std_viterbi=('abs_err_viterbi', 'std'),
                Acc5_raw=('abs_err_raw', lambda x: (x < 5).mean() * 100),
                Acc5_viterbi=('abs_err_viterbi',
                              lambda x: (x < 5).mean() * 100),
                N=('abs_err_raw', 'count'),
            )
            .reset_index())
