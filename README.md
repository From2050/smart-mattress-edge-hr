# Smart Mattress Edge Heart-Rate Platform

**Posture-Robust Lightweight Heart-Rate Monitoring on a Low-Cost 32-Channel Pressure Sensor Array**

> Master's Thesis — National Yang Ming Chiao Tung University, 2026  
> [繁體中文版](README_zh.md)

---

## Overview

A complete, end-to-end platform for **wearable-free heart-rate monitoring** from a pressure mattress — covering the data-collection firmware, a Streamlit research dashboard, an ML training pipeline, and the edge deployment artifacts.

The system uses **32 FSR (force-sensitive resistor) channels** sampled at 13.5 Hz.  
A lightweight 16 K-parameter CNN scores per-channel signal reliability; a Top-3 Peak Voting + Viterbi HMM post-processor fuses the trusted channels into a final heart-rate estimate — no wrist-worn or chest-worn sensor required at inference time.

```
ESP32-S3 (FreeRTOS)                         ← firmware/
  32ch FSR + MAX30105 PPG (ground truth only)
         │ Serial 921600 baud
         ▼
  PC Receiver (receiver/PPG_read.py)         ← receiver/
         │  .txt files — 35 col / row
         ▼
  Streamlit Platform  (app/)                 ← app/   ★ NEW
    Collect · Observe · Train · Deploy
         │
         ▼
  ML Pipeline (pipeline/)                    ← pipeline/
    Stage 1: PPG FFT → per-window HR label
    Stage 2: CNN → per-channel reliability score
    Stage 3: Top-3 Voting + Viterbi → final HR
         │
         ▼
  Edge Deploy (deploy/)                      ← deploy/
    Int8 ONNX  36.7 KB
    Peak RAM   ~16.2 KB
    Latency    57.98 ms / window
```

---

## Results

| Model | Params | LOSO MAE | Acc@5 | Notes |
|-------|--------|----------|-------|-------|
| NoAttentionCNN_Large (offline) | 353,601 | **7.58 BPM** | 66.7% | 33-fold LOSO, 5 postures |
| PaperCNN_Reliability (deployed) | 16,057 | 7.81 BPM | 63.3% | Wilcoxon p = 0.72 — no significant difference |

Both models share the identical post-processing pipeline.  
The deployed model is **22× smaller** with no statistically significant accuracy loss.

---

## Repository Structure

```
smart-mattress-edge-hr/
├── app/                   Streamlit research platform (5 pages)
│   ├── dashboard.py       Entry point — st.navigation wiring
│   ├── lib.py             Shared model, signal processing, figure builders
│   └── views/
│       ├── overview.py    Pipeline overview & design rationale
│       ├── collect.py     Firmware wiring + interactive config generator
│       ├── observe.py     Interactive BCG dashboard (play/scrub)
│       ├── train.py       Architecture, labelling rule, LOSO, live score check
│       └── deploy.py      Export path, artifact sizes, MCU resource budget
├── firmware/              ESP32-S3 ESP-IDF data collector
├── receiver/              PC-side Python serial receiver
├── pipeline/              ML training + LOSO evaluation scripts
├── deploy/                Pre-trained weights (PyTorch / ONNX / Int8)
└── docs/                  Extended hardware & method documentation
```

---

## Streamlit Platform

The fastest way to understand and operate the whole system.

### Setup

```bash
# From project root
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt

# Download the dataset (needed for Observation and Training pages)
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='m46012002/smart-mattress-bcg',
    repo_type='dataset',
    local_dir='data',
)
EOF

streamlit run app/dashboard.py
```

Open **http://localhost:8501**.

### Pages

| Page | What you can do |
|------|-----------------|
| 🏠 **Overview** | Understand the four-stage pipeline and the per-channel-vote design rationale in two minutes. |
| 📡 **Data Collection** | See the hardware wiring diagram. Use the **interactive config generator** to pick channel count and per-channel acquisition time, see the resulting sample rate and window length, and download a ready-to-paste `sensor_config.h`. View the PC receiver workflow. |
| 🔬 **Live Observation** | Pick any subject and posture from the dataset. Use **▶ Play** or drag the slider to scrub through the session window by window. Each frame shows: (1) pressure heatmap (body weight DC), (2) BCG amplitude heatmap (heart AC, 0.75–3 Hz), (3) BCG waveform of the top reliable channels with visible heartbeat oscillations, (4) voted frequency spectrum with the BPM peak, (5) per-channel CNN reliability bar chart, (6) full-session HR vs PPG ground truth with a live cursor. |
| 🧠 **Training** | Inspect the 11-layer all-convolutional architecture. Try the **interactive label simulator** — enter a channel FFT peak, PPG reference, and SNR to see whether the channel gets labelled reliable. Run the training commands. Check the live **CNN score distribution histogram** on any session. |
| 🚀 **Edge Deployment** | Follow the PyTorch → ONNX → Int8 → TFLite export path. Inspect **real on-disk artifact sizes**. See the timing budget chart (worst-case 32-channel compute vs. window period). |

---

## Quick Start — Individual Components

### 1. Flash Firmware

```bash
cd firmware
idf.py set-target esp32s3
idf.py build flash monitor
```

Requires ESP-IDF v5.x. See [firmware/README.md](firmware/README.md) for wiring and build details.

### 2. Collect Data

```bash
pip install pyserial matplotlib numpy
python receiver/PPG_read.py
# Press S to start recording a posture, E to end and save.
```

Each `.txt` file has 35 comma-separated columns per row: `timestamp, ch0–ch31, ppg, esp32_us`.

### 3. Train

```bash
cd pipeline
pip install -r requirements.txt
python run_exp1_main.py          # main model, 33-fold LOSO  (~2 h GPU / ~8 h CPU)
python run_exp5_lightweight.py   # deployment model comparison
python run_edge_deploy.py        # ONNX + Int8 export + latency profile
```

### 4. Run Inference (Python)

```python
from pipeline.common import PaperCNN_Reliability
import torch

model = PaperCNN_Reliability()
model.load_state_dict(torch.load('deploy/weights/papercnn_16k.pth'))
model.eval()

# x: (1, 32, 128) — batch=1, 32 channels, 128 samples per window
x = torch.randn(1, 32, 128)
reliability = model(x)   # (1, 32) per-channel reliability scores in [0, 1]
```

---

## Hardware

| Component | Specification |
|-----------|---------------|
| MCU | ESP32-S3 (Xtensa LX7, 240 MHz, 512 KB SRAM) |
| Pressure | FSR × 32 — 16-column × 2-row grid, ADS1115 × 2 + 4-to-1 MUX |
| PPG (reference) | MAX30105 fingertip clip, Green LED, 800 SPS |
| Sample rate | 13.5 Hz (32ch) / 27.03 Hz (16ch variant) |
| ADC | ADS1115 16-bit, 860 SPS single-ended |

> **ADC polarity**: higher value = less pressure; lower = more pressure.  
> BCG signal = tiny dynamic fluctuation (~100–1000 ADC counts) riding on a large static DC bias.

See [docs/data_description.md](docs/data_description.md) for full hardware details.

---

## Dataset

33 healthy subjects (age 18–35) · 5 postures each · ~270 MB preprocessed

| Posture | Description |
|---------|-------------|
| Front | Supine (face up) |
| Back | Prone (face down) |
| LeftSide / RightSide | Lateral |
| Leave | Supine → leave mattress → return |

**Download**: [huggingface.co/datasets/m46012002/smart-mattress-bcg](https://huggingface.co/datasets/m46012002/smart-mattress-bcg)

---

## Pre-trained Weights

| File | Size | Description |
|------|------|-------------|
| `deploy/weights/papercnn_16k.pth` | 86.9 KB | Deployment model FP32 (PyTorch) |
| `deploy/weights/papercnn_16k.onnx` | 65.7 KB | ONNX FP32 |
| `deploy/weights/papercnn_16k_int8.onnx` | 36.7 KB | **ONNX Int8 — edge deployment target** |
| `deploy/weights/main_353k.pth` | 1.4 MB | Offline model FP32 (higher accuracy, not for MCU) |

---

## Method Summary

**Why per-channel independent scoring?**  
Experiments show that across all postures, BCG channels are mutually uninformative — even after phase correction, one channel cannot improve another's estimate. The system therefore scores each channel independently with a shared-weight CNN, then fuses the trusted channels via weighted spectral voting.

**Training labels**  
A channel-window is labelled *reliable* if its BCG FFT peak matches the PPG ground truth within ±5 BPM **and** the spectral SNR > 3. Only 13.1% of all 296,448 channel-windows are reliable — a heavily imbalanced binary classification task.

**Deployment**  
The all-convolutional architecture has no FC layers and no cross-channel operations. On an MCU it streams one channel at a time: 512 bytes peak activation, ~16 KB weights always resident. The 32-channel pass takes ~1.85 s total — well within the 9.48 s window budget.

See [docs/method.md](docs/method.md) for the full pipeline description.

---

## Citation

```bibtex
@mastersthesis{wu2026smartmattress,
  author  = {Wu, Jie-Neng},
  title   = {Posture-Robust Lightweight Smart Mattress Edge Heart Rate Monitoring
             on Low-Cost Pressure Sensor Arrays},
  school  = {National Yang Ming Chiao Tung University},
  year    = {2026}
}
```

---

## License

[MIT License](LICENSE) — © 2026 Wu Jie-Neng, NYCU
