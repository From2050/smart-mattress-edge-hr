# Smart Mattress Edge Heart Rate Monitoring

**Posture-Robust Lightweight Heart Rate Monitoring on a Low-Cost 32-Channel Pressure Sensor Array**

> Master's Thesis — National Yang Ming Chiao Tung University, 2026  
> [繁體中文版](README_zh.md)

---

## Overview

This repository contains the full hardware design, data collection firmware, machine learning pipeline, and edge deployment artifacts for a BCG-based (ballistocardiography) heart rate monitoring system embedded in a pressure-sensor mattress.

The system uses **32 FSR (force-sensitive resistor) channels** sampled at 13.5 Hz. A lightweight CNN model scores per-channel signal reliability; a Top-3 Peak Voting + Viterbi HMM post-processor fuses the reliable channels into a final heart rate estimate — all without any wrist-worn or chest-worn sensors.

```
ESP32-S3 (FreeRTOS)
  32ch FSR + MAX30105 PPG (ground truth only)
         │ Serial 921600 baud
         ▼
  PC Receiver (PPG_read.py)  →  .txt files
         │
         ▼
  ML Pipeline (pipeline/)
    Stage 1: PPG FFT → per-window HR label
    Stage 2: CNN → per-channel reliability score
    Stage 3: Top-3 Voting + Viterbi → final HR
         │
         ▼
  Edge Deploy (deploy/)
    Int8 ONNX  36.7 KB
    Peak RAM   ~16.2 KB
    Latency    57.98 ms / window
```

---

## Results

| Model | Params | LOSO MAE | Notes |
|-------|--------|----------|-------|
| NoAttentionCNN_Large (offline) | 353,601 | **7.58 BPM** | 33-fold LOSO, 5 postures |
| PaperCNN_Reliability (deployed) | 16,057 | 7.81 BPM | Wilcoxon p = 0.72, no significant difference |

Both models share the identical post-processing pipeline. The deployed model is **22× smaller** with no statistically significant accuracy loss.

---

## Repository Structure

```
smart-mattress-edge-hr/
├── firmware/          ESP32-S3 ESP-IDF data collector
├── receiver/          PC-side Python serial receiver
├── pipeline/          ML training + LOSO evaluation
├── deploy/            Pre-trained weights (ONNX / Int8)
└── docs/              Extended hardware & method documentation
```

---

## Quick Start

### 1. Flash Firmware

```bash
cd firmware
idf.py build flash monitor
```

Requires ESP-IDF v5.x. See [firmware/README.md](firmware/README.md) for wiring and build details.

### 2. Collect Data

```bash
pip install pyserial matplotlib numpy
python receiver/PPG_read.py
```

One `.txt` file per posture is saved; each row is 35 columns: timestamp, ch0–ch31 ADC, PPG, esp32_us.

### 3. Train the Model

```bash
cd pipeline
pip install -r requirements.txt
python run_exp1_main.py          # main model 33-fold LOSO
python run_exp5_lightweight.py   # deployment model comparison
```

### 4. Export to Edge

```bash
python pipeline/run_edge_deploy.py
# outputs: papercnn_16k.onnx, papercnn_16k_int8.onnx, EDGE_DEPLOY_REPORT.md
```

### 5. Run Inference (Python)

```python
from pipeline.common import PaperCNN_Reliability
import torch

model = PaperCNN_Reliability()
model.load_state_dict(torch.load('deploy/weights/papercnn_16k.pth'))
model.eval()

# x: (batch, 32, 128) — 32 channels, 128 samples per window
x = torch.randn(1, 32, 128)
reliability = model(x)   # (batch, 32) per-channel reliability scores
```

---

## Hardware

| Component | Specification |
|-----------|---------------|
| MCU | ESP32-S3 (Xtensa LX7, 240 MHz, 512 KB SRAM) |
| Pressure | FSR × 32 — 16-column × 2-row grid, ADS1115 × 2 + MUX × 4 |
| PPG (reference) | MAX30105 fingertip clip, Green LED, 800 SPS |
| Sample rate | 13.5 Hz (32ch) / 27.03 Hz (16ch variant) |
| ADC | ADS1115 16-bit, 860 SPS single-ended |

> **ADC polarity note**: higher ADC value = less pressure; lower = more pressure.  
> BCG signal = small dynamic fluctuation riding on the static DC bias.

See [docs/data_description.md](docs/data_description.md) for full hardware details.

---

## Dataset

The dataset contains 33 healthy subjects (age 18–35) across 5 postures:
`Front`, `Back`, `LeftSide`, `RightSide`, `Leave`.

Raw data is not included in this repository due to size (~270 MB preprocessed).  
**Download**: https://huggingface.co/datasets/m46012002/smart-mattress-bcg

See [docs/data_description.md](docs/data_description.md) for the full collection protocol and file format.

---

## Pre-trained Weights

Pre-trained weights are in `deploy/weights/`:

| File | Size | Description |
|------|------|-------------|
| `papercnn_16k.pth` | 86.9 KB | Deployment model FP32 (PyTorch) |
| `papercnn_16k.onnx` | 65.7 KB | ONNX FP32 |
| `papercnn_16k_int8.onnx` | 36.7 KB | ONNX Int8 quantized (edge target) |
| `main_353k.pth` | 1.4 MB | Full offline model FP32 |

---

## Citation

If you use this work, please cite:

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
