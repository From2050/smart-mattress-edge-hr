#!/usr/bin/env python3
"""
Shared library for the Smart Mattress BCG platform.
Model definition, signal processing, cached loaders, and Plotly figure builders.
Imported by every page under app/views/.
"""

import os
import glob

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from scipy.fft import fft, fftfreq
from scipy import signal as sp_signal

# ── Paths ──────────────────────────────────────────────────────────────
APP_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.abspath(os.path.join(APP_DIR, '..'))
DATA_DIR    = os.path.join(ROOT_DIR, 'data', 'ESP32_recored')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'deploy', 'weights')
WEIGHT_PATH = os.path.join(WEIGHTS_DIR, 'papercnn_16k.pth')

# ── Signal constants ────────────────────────────────────────────────────
FS           = 13.5         # Hz (32-channel firmware default)
WIN          = 128          # samples per window
STRIDE       = 64           # 50% overlap
N_CH         = 32
ROWS, COLS   = 8, 4         # internal grid used by spatial smoother
HR_LO, HR_HI = 0.75, 3.0    # Hz  (45-180 BPM)

EXCLUDE_SUBJECTS = {100, 200}
VOTE_FLOOR       = 0.01     # channels with weight <= this are excluded from voting

# Pre-computed BCG bandpass filter coefficients (heart-rate band)
_BCG_B, _BCG_A = sp_signal.butter(3, [HR_LO / (FS / 2), HR_HI / (FS / 2)], btype='bandpass')

# Edge AI deployment stats (from EDGE_DEPLOY_REPORT.md)
EDGE_SIZE_KB    = 36.7
EDGE_PARAMS     = 16_057
EDGE_LATENCY_MS = 57.98
EDGE_RAM_KB     = 16.2
WINDOW_PERIOD_S = WIN / FS  # 9.48 s

# Theme palette
CLR_PRIMARY = '#1976D2'
CLR_ACCENT  = '#E53935'
CLR_GOOD    = '#2E7D32'


# ══════════════════════════════════════════════════════════════════════
# Model (copied verbatim from pipeline/common.py — PaperCNN_Reliability)
# ══════════════════════════════════════════════════════════════════════
class PaperCNN_Reliability(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 8, 7, stride=2, padding=3),
            nn.BatchNorm1d(8),  nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 8, 5, stride=1, padding=2),
            nn.BatchNorm1d(8),  nn.LeakyReLU(inplace=True),
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
        h = self.layers(x.view(-1, 1, L))
        return torch.sigmoid(h.view(B, C))


# ══════════════════════════════════════════════════════════════════════
# Signal Processing
# ══════════════════════════════════════════════════════════════════════
def calculate_spectrum(sig, fs=FS, n_fft=2048):
    sig = (sig - np.mean(sig)) * np.hanning(len(sig))
    yf  = fft(sig, n=n_fft)
    xf  = fftfreq(n_fft, 1 / fs)
    m   = xf > 0
    return xf[m], np.abs(yf[m]) ** 2


def ppg_to_hr(ppg_window, fs=FS):
    freqs, power = calculate_spectrum(ppg_window, fs=fs, n_fft=1024)
    mask = (freqs >= HR_LO) & (freqs <= HR_HI)
    if not mask.any():
        return 0.0
    return freqs[mask][np.argmax(power[mask])] * 60.0


def preprocess_channel(sig):
    med = np.median(sig)
    mad = np.median(np.abs(sig - med))
    return (sig - med) / (mad + 1e-6)


def bcg_filter(sig):
    """Bandpass a 1-D signal to the heart-rate band (0.75-3 Hz)."""
    try:
        return sp_signal.filtfilt(_BCG_B, _BCG_A, sig)
    except Exception:
        return np.zeros_like(sig)


def bcg_rms_per_channel(adc_window):
    """Per-channel RMS amplitude of the bandpassed BCG signal."""
    rms = np.zeros(N_CH)
    for ch in range(N_CH):
        filt = bcg_filter(adc_window[:, ch])
        rms[ch] = np.sqrt(np.mean(filt ** 2))
    return rms


def build_consensus(spectra, weights):
    """Top-3 peak voting -> consensus spectrum on HR grid."""
    grid  = np.arange(HR_LO, HR_HI, 0.01)
    cons  = np.zeros_like(grid)
    sigma = 0.05
    for ch in range(N_CH):
        if weights[ch] <= VOTE_FLOOR:
            continue
        f, p = spectra[ch]
        mask = (f >= HR_LO) & (f <= HR_HI)
        vf, vp = f[mask], p[mask]
        if len(vp) == 0 or vp.max() == 0:
            continue
        pks, _ = sp_signal.find_peaks(vp, height=vp.max() * 0.3)
        if len(pks):
            for idx in pks[np.argsort(vp[pks])[::-1][:3]]:
                cons += weights[ch] * np.exp(-0.5 * ((grid - vf[idx]) / sigma) ** 2)
    return grid, cons


def viterbi_smooth(consensus_list, grid):
    """Viterbi temporal smoothing over a sequence of consensus spectra."""
    T, N = len(consensus_list), len(grid)
    dp   = np.full((T, N), -np.inf)
    psi  = np.zeros((T, N), dtype=int)
    sigma_t = 0.05
    fd2  = (grid[:, None] - grid[None, :]) ** 2
    lt   = np.log(np.exp(-0.5 * fd2 / sigma_t ** 2) + 1e-10)
    dp[0] = np.log(consensus_list[0] + 1e-10)
    for t in range(1, T):
        ol = np.log(consensus_list[t] + 1e-10)
        mat = dp[t - 1][:, None] + lt
        bp  = np.argmax(mat, axis=0)
        dp[t]  = mat[bp, np.arange(N)] + ol
        psi[t] = bp
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return grid[path] * 60.0


# ══════════════════════════════════════════════════════════════════════
# Cached Loaders
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    model = PaperCNN_Reliability()
    sd = torch.load(WEIGHT_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_raw_session(subject: int, posture: str):
    path  = os.path.join(DATA_DIR, str(subject), posture)
    files = glob.glob(os.path.join(path, '*.txt'))
    if not files:
        return None, None
    adc_rows, ppg_vals = [], []
    with open(files[0], 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 34:
                continue
            try:
                adc_rows.append([float(x) for x in parts[1:33]])
                ppg_vals.append(float(parts[33]))
            except ValueError:
                continue
    if not adc_rows:
        return None, None
    return np.array(adc_rows, np.float32), np.array(ppg_vals, np.float32)


@st.cache_data(show_spinner=False)
def run_inference(subject: int, posture: str):
    """Sliding-window CNN -> Viterbi -> per-window results dict."""
    adc, ppg = load_raw_session(subject, posture)
    if adc is None:
        return None

    N = len(adc)
    starts = list(range(0, N - WIN + 1, STRIDE))

    # ── Preprocess & batch inference ──
    X = np.zeros((len(starts), N_CH, WIN), np.float32)
    for i, s in enumerate(starts):
        for ch in range(N_CH):
            X[i, ch] = preprocess_channel(adc[s:s + WIN, ch])

    with torch.no_grad():
        rel_all = load_model()(torch.FloatTensor(X)).numpy()   # (T, 32)

    spectra_all, ppg_hrs, cons_all, weights_all = [], [], [], []
    prev_w, alpha = None, 0.8

    for i, s in enumerate(starts):
        raw_w = adc[s:s + WIN]
        spectra = [calculate_spectrum(raw_w[:, ch]) for ch in range(N_CH)]
        spectra_all.append(spectra)
        ppg_hrs.append(ppg_to_hr(ppg[s:s + WIN]))

        cur = rel_all[i].copy()
        w   = alpha * cur + (1 - alpha) * prev_w if prev_w is not None else cur
        prev_w = w
        weights_all.append(w)

        grid, cons = build_consensus(spectra, w)
        cons_all.append(cons)

    grid = np.arange(HR_LO, HR_HI, 0.01)
    vit_hrs = list(viterbi_smooth(cons_all, grid)) if cons_all else []

    ppg_clean = ppg_hrs.copy()
    for i in range(1, len(ppg_clean)):
        if abs(ppg_clean[i] - ppg_clean[i - 1]) > 20:
            ppg_clean[i] = ppg_clean[i - 1]

    t_axis = [(s + WIN / 2) / FS for s in starts]

    return {
        'adc': adc, 'starts': starts, 't_axis': t_axis,
        'rel_all': rel_all, 'weights_all': weights_all,
        'ppg_hrs': ppg_hrs, 'ppg_clean': ppg_clean, 'vit_hrs': vit_hrs,
        'spectra_all': spectra_all, 'cons_all': cons_all, 'grid': grid,
    }


def list_subjects():
    """32-channel @ 13.5 Hz subjects only.

    The 16-channel variant (e.g. '224_16ch', 27 Hz) is excluded because the
    shipped model and FS constants assume the 32-channel layout.
    """
    if not os.path.isdir(DATA_DIR):
        return []
    out = []
    for d in os.listdir(DATA_DIR):
        if not d.isdigit():            # skips '224_16ch' and stray files
            continue
        sid = int(d)
        if sid not in EXCLUDE_SUBJECTS:
            out.append(sid)
    return [str(s) for s in sorted(out)]


def list_postures(subject_dir: str):
    base = os.path.join(DATA_DIR, str(subject_dir))
    if not os.path.isdir(base):
        return []
    return sorted([p for p in os.listdir(base)
                   if os.path.isdir(os.path.join(base, p))])


# ══════════════════════════════════════════════════════════════════════
# Plotly helpers
# ══════════════════════════════════════════════════════════════════════
_LAYOUT_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(248,249,250,0.85)',
    margin=dict(l=8, r=8, t=36, b=36),
    font=dict(family='Inter, sans-serif', size=12),
)


def _to_physical_grid(values_32):
    """Map 32 channels to the physical 2x16 mattress layout.

    Row 0 (head/top):  ch0  ch1  ... ch15   (left -> right)
    Row 1 (foot/bot):  ch31 ch30 ... ch16   (right -> left)
    """
    g = np.zeros((2, 16))
    g[0, :] = values_32[0:16]
    g[1, :] = values_32[16:32][::-1]
    return g


_CH_ID = np.zeros((2, 16), dtype=int)
_CH_ID[0, :] = np.arange(16)
_CH_ID[1, :] = np.arange(31, 15, -1)


def fig_heatmap(adc_window, weights):
    """Stacked 2x16 heatmaps: pressure DC (top) and BCG AC amplitude (bottom)."""
    avg_adc  = adc_window.mean(axis=0)
    pressure = np.clip(22000 - avg_adc, 0, None)
    bcg_rms  = bcg_rms_per_channel(adc_window)

    press_grid = _to_physical_grid(pressure)
    bcg_grid   = _to_physical_grid(bcg_rms)

    hover_p = [[f'ch{_CH_ID[r, c]}<br>pressure: {press_grid[r, c]:.0f}'
                for c in range(16)] for r in range(2)]
    hover_b = [[f'ch{_CH_ID[r, c]}<br>BCG RMS: {bcg_grid[r, c]:.1f} ADC'
                for c in range(16)] for r in range(2)]

    y_labels = ['ch0-15', 'ch31-16']
    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.22,
        subplot_titles=['Pressure (body weight, DC)',
                        'BCG amplitude (heartbeat, 0.75-3 Hz AC)'],
    )

    fig.add_trace(go.Heatmap(
        z=press_grid, text=hover_p, hovertemplate='%{text}<extra></extra>',
        colorscale='Blues',
        colorbar=dict(len=0.44, y=0.79, thickness=13,
                      title=dict(text='ADC', font=dict(size=10))),
        zmin=0, xgap=2, ygap=10,
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=bcg_grid, text=hover_b, hovertemplate='%{text}<extra></extra>',
        colorscale=[[0, '#FFFFFF'], [0.25, '#FFF9C4'],
                    [0.65, '#FF9800'], [1, '#B71C1C']],
        colorbar=dict(len=0.44, y=0.21, thickness=13,
                      title=dict(text='RMS', font=dict(size=10))),
        xgap=2, ygap=10,
    ), row=2, col=1)

    for r in [1, 2]:
        fig.update_yaxes(ticktext=y_labels, tickvals=[0, 1],
                         autorange='reversed', tickfont=dict(size=10),
                         row=r, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(tickmode='array', tickvals=list(range(0, 16, 2)),
                     ticktext=[str(i) for i in range(0, 16, 2)],
                     title_text='Column (0-15)', tickfont=dict(size=10),
                     row=2, col=1)
    fig.update_layout(**_LAYOUT_BASE, height=340)
    fig.update_layout(margin=dict(l=8, r=58, t=36, b=28))
    return fig


def fig_hr_timeseries(t_axis, vit_hrs, ppg_hrs, current_win):
    fig = go.Figure()
    valid = [(t, hr) for t, hr in zip(t_axis, ppg_hrs) if 40 < hr < 160]
    if valid:
        tv, hv = zip(*valid)
        fig.add_trace(go.Scatter(
            x=list(tv), y=list(hv), mode='lines',
            name='PPG reference (ground truth)',
            line=dict(color='#9E9E9E', dash='dot', width=1.5)))
    fig.add_trace(go.Scatter(
        x=t_axis, y=vit_hrs, mode='lines', name='Estimated HR (Edge AI)',
        line=dict(color=CLR_PRIMARY, width=2.5)))
    if 0 <= current_win < len(t_axis):
        fig.add_vline(x=t_axis[current_win], line_dash='dash',
                      line_color=CLR_ACCENT, line_width=2,
                      annotation_text='  now', annotation_font_color=CLR_ACCENT)
    fig.update_layout(**_LAYOUT_BASE, height=340, xaxis_title='Time (s)',
                      yaxis_title='Heart Rate (BPM)', yaxis=dict(range=[40, 130]),
                      legend=dict(orientation='h', y=1.12))
    return fig


def fig_bcg_waveform(adc_window, weights):
    """Filtered BCG waveform for the top-3 channels by reliability weight."""
    t = np.arange(WIN) / FS
    top3 = np.argsort(weights)[::-1][:3]
    palette = [CLR_PRIMARY, CLR_GOOD, CLR_ACCENT]
    fig = go.Figure()
    for rank, ch in enumerate(top3):
        filt = bcg_filter(adc_window[:, ch])
        mx = np.abs(filt).max() + 1e-6
        fig.add_trace(go.Scatter(
            x=t, y=filt / mx + rank * 2.5, mode='lines',
            name=f'ch{ch}  (w={weights[ch]:.2f})',
            line=dict(color=palette[rank], width=1.5)))
    fig.update_layout(**_LAYOUT_BASE, height=240,
                      xaxis_title='Time within window (s)',
                      yaxis=dict(showticklabels=False, title='BCG (normalized)'),
                      legend=dict(orientation='h', y=1.18, font=dict(size=10)))
    return fig


def fig_reliability_bar(weights):
    """Per-channel CNN reliability bar with the algorithm-native cutoff (w > 0.01)."""
    colors = [CLR_GOOD if w > VOTE_FLOOR else '#E0E0E0' for w in weights]
    fig = go.Figure(go.Bar(
        x=[f'ch{i}' for i in range(N_CH)], y=weights, marker_color=colors,
        hovertemplate='Channel %{x}<br>Score: %{y:.3f}<extra></extra>'))
    fig.add_hline(y=VOTE_FLOOR, line_dash='dot', line_color='#555',
                  annotation_text=f'  voting floor ({VOTE_FLOOR})',
                  annotation_font_color='#555')
    fig.update_layout(**_LAYOUT_BASE, height=230, xaxis_title='Sensor channel',
                      yaxis=dict(range=[0, 1], title='CNN reliability score'),
                      xaxis=dict(tickfont=dict(size=9)))
    return fig


def fig_spectrum(spectra, weights, grid, consensus):
    fig = go.Figure()
    top3 = np.argsort(weights)[::-1][:3]
    palette = ['#BBDEFB', '#90CAF9', '#64B5F6']
    for rank, ch in enumerate(top3):
        f, p = spectra[ch]
        mask = (f >= HR_LO) & (f <= HR_HI)
        if mask.any():
            pn = p[mask] / (p[mask].max() + 1e-10)
            fig.add_trace(go.Scatter(x=f[mask] * 60, y=pn, mode='lines',
                                     name=f'ch{ch} (rank {rank+1})',
                                     line=dict(color=palette[rank], width=1),
                                     opacity=0.7))
    if consensus.max() > 0:
        cn = consensus / consensus.max()
        fig.add_trace(go.Scatter(x=grid * 60, y=cn, mode='lines',
                                 name='Consensus (voted)', fill='tozeroy',
                                 fillcolor='rgba(25,118,210,0.15)',
                                 line=dict(color=CLR_PRIMARY, width=2.5)))
        peak_bpm = grid[np.argmax(consensus)] * 60
        fig.add_vline(x=peak_bpm, line_dash='dash', line_color=CLR_ACCENT,
                      annotation_text=f'  {peak_bpm:.0f} BPM',
                      annotation_font_color=CLR_ACCENT)
    fig.update_layout(**_LAYOUT_BASE, height=230, xaxis_title='Heart Rate (BPM)',
                      yaxis_title='Spectrum (normalized)',
                      legend=dict(orientation='h', y=1.15, font=dict(size=10)))
    return fig
