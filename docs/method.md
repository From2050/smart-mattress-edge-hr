# Method and Edge Deployment

> [繁體中文版](method_zh.md)

---

## 1. System Overview

```
32ch pressure mattress + PPG fingertip clip
         ↓ raw ADC (35-column .txt)
    ┌──────────────────────────────────────┐
    │  Stage 1: Ground Truth + Labelling   │
    │  PPG FFT → per-window HR label       │
    │  Per-channel FFT → reliable/not      │
    └──────────────────────────────────────┘
         ↓ channel_level_annotations.csv
    ┌──────────────────────────────────────┐
    │  Stage 2: Channel Reliability Model  │
    │  CNN backbone → sigmoid(reliable)    │
    └──────────────────────────────────────┘
         ↓ per-channel reliability scores
    ┌──────────────────────────────────────┐
    │  Stage 3: Multi-channel Spectrum     │
    │  Fusion + Viterbi                    │
    │  Top-3 Peak Voting → Viterbi → HR   │
    └──────────────────────────────────────┘
```

**Core idea**: instead of end-to-end heart rate regression, we first decide *which channels are trustworthy*, then vote from the spectra of those trusted channels.

---

## 2. Model Architectures

### 2.1 Offline Model — `NoAttentionCNN_Large`

| Item | Specification |
|------|---------------|
| Parameters | **353,601** |
| Architecture | 3-layer CNN (16→32→64) + MaxPool + 4-layer MLP (1024→256→256→64→1) |
| Input | Single channel `(1, 128)` time-domain signal; 32 channels share weights |
| Output | `sigmoid → [0, 1]` reliability score |
| Training | BCEWithLogitsLoss + Adam, 30 epochs |
| Use | Offline analysis, primary LOSO validation result |

```
(1,128) → Conv1d(1→16, k=5) → MaxPool(2) → (16,64)
       → Conv1d(16→32, k=5) → MaxPool(2) → (32,32)
       → Conv1d(32→64, k=3) → MaxPool(2) → (64,16)
       → Flatten(1024) → FC(256) → Dropout(0.5)
       → FC(256) → Dropout(0.1) → FC(64) → FC(1) → sigmoid
```

### 2.2 Deployment Model — `PaperCNN_Reliability`

| Item | Specification |
|------|---------------|
| Parameters | **16,057** (4.5% of the offline model) |
| Architecture | 11-layer All-Convolutional (no FC, no Transformer) |
| Input / Output | Same as above |
| Use | **MCU edge deployment target** |

```
(1,128) → Conv(1→8, k=7, s=2)  → BN → LReLU → (8,64)
        → Conv(8→8, k=5, s=1)  → BN → LReLU → (8,64)
        → Conv(8→16, k=5, s=2) → BN → LReLU → (16,32)
        → Conv(16→16,k=3, s=1) → BN → LReLU → (16,32)
        → Conv(16→16,k=3, s=2) → BN → LReLU → (16,16)
        → Conv(16→32,k=3, s=1) → BN → LReLU → (32,16)
        → Conv(32→32,k=3, s=2) → BN → LReLU → (32,8)
        → Conv(32→32,k=3, s=2) → BN → LReLU → (32,4)
        → Conv(32→32,k=3, s=2) → BN → LReLU → (32,2)
        → Conv(32→32,k=2, s=2) → BN → LReLU → (32,1)
        → Conv(32→1, k=1, s=1) → sigmoid
```

> All spatial downsampling uses stride — no pooling layers, no FC layers.  
> Fully supported by TFLite Micro with no operator decomposition needed.

### 2.3 Accuracy Comparison

| Evaluation | Offline (353K) | Deployment (16K) | Difference |
|-----------|:--------------:|:----------------:|:----------:|
| LOSO 33-fold MAE | **7.58 BPM** | 7.81 BPM | +0.23 |
| LOSO Acc@5 | **66.7%** | 63.3% | −3.4% |
| Wilcoxon p-value | — | 0.72 | no significant difference |
| Holdout 5-subject MAE | 8.02 BPM | **7.48 BPM** | −0.54 |

> Both models share the identical post-processing pipeline; only the backbone differs.

---

## 3. Post-Processing Pipeline (shared by both models)

```
Model output: reliability[32]  (per-channel score 0–1)
    ↓
(1) Spatial smoothing: reshape to 8×4 grid → EMA weighting
    ↓
(2) Per-channel FFT: 128-pt Hanning → 0.75–3.0 Hz band
    ↓
(3) Top-3 Peak Voting:
    Each channel: extract top-3 spectral peaks
    → weight each with Gaussian kernel (σ = 0.05 Hz)
    → multiply by reliability score
    → accumulate into consensus spectrum
    ↓
(4) Viterbi temporal smoothing:
    State space = discrete frequencies in HR band
    Emission probability = consensus spectrum
    Transition probability = Gaussian(Δf, σ = 0.05)
    → global optimal path → final HR (BPM)
```

**Code entry point**: `pipeline/common.py → evaluate_reliability() → SpectrumFuser`

---

## 4. Ground-Truth Labelling

| Step | Details |
|------|---------|
| PPG processing | Remove DC + Hanning window → 1024-pt zero-padded FFT |
| Reference HR | Peak in 0.75–3.0 Hz × 60 = BPM |
| Outlier correction | Jumps > 20 BPM between adjacent windows → forward fill |
| Channel label | Per-channel FFT dominant freq vs PPG HR: **bpm_error < 5 AND SNR > 3 → reliable** |

Result: 296,448 channel-level annotations; only 13.1% are reliable (heavily imbalanced).

---

## 5. Edge Deployment Specifications

### 5.1 Target Platform

| Item | Specification |
|------|---------------|
| MCU | ESP32-S3 (Xtensa LX7, 240 MHz) |
| SRAM | 512 KB |
| Flash | 4–16 MB |
| Framework | TFLite Micro / ESP-NN |

### 5.2 Deployed Model Resources

| Metric | Value |
|--------|-------|
| Parameters | 16,057 |
| FP32 weights (.pth) | 86.9 KB |
| FP32 ONNX | 65.7 KB |
| **Int8 ONNX** | **36.7 KB** |
| Peak inference RAM (streaming Int8) | **~16.2 KB** (512 bytes activation + 16 KB weights) |
| CPU streaming latency | 57.98 ms (<< 9.48 s window period) |
| Int8 quantization error | mean diff 0.0026 (negligible) |

### 5.3 Streaming Inference Strategy

```
PaperCNN is purely convolutional; 32 channels are fully independent:

for ch in 0..31:
    reliability[ch] = AllConv_11layer(adc_window[ch])  # (128,) → scalar
    // Peak intermediate tensor: 512 bytes (Int8)
    // No feature buffer needed

→ feed into SpectrumFuser + Viterbi (~4 KB extra RAM, C implementation)
```

> No Transformer, no FC layers, no cross-channel interaction. Channel-by-channel streaming inference is natural.

### 5.4 Deployment Path

```
papercnn_16k.pth (86.9 KB)
    ↓  torch.onnx.export (opset 17)
papercnn_16k.onnx (65.7 KB)
    ↓  onnxruntime quantize_dynamic
papercnn_16k_int8.onnx (36.7 KB)
    ↓  onnx2tf → tflite_converter
model.tflite (~37 KB)
    ↓  xxd -i
model_data.h → embedded in ESP32-S3 Flash
```

> **Note**: the TFLite Micro integration firmware is not yet complete (see future work in README).

---

## 6. Reproducing Key Results

### Run main model LOSO
```bash
python pipeline/run_exp1_main.py
# → outputs/exp1_main/
```

### Run deployment model comparison
```bash
python pipeline/run_exp5_lightweight.py   # requires exp1 to finish first
# → outputs/exp5_lightweight/
```

### Run edge deployment validation
```bash
python pipeline/run_edge_deploy.py
# → outputs/edge_deploy/
#    papercnn_16k.onnx, papercnn_16k_int8.onnx, EDGE_DEPLOY_REPORT.md
```
