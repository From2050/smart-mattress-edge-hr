# Deploy — Pre-trained Weights & Edge Deployment

> [繁體中文](#中文說明) | [English](#english-guide)

---

## English Guide

### Pre-trained Weights (`weights/`)

| File | Size | Description |
|------|------|-------------|
| `papercnn_16k.pth` | 86.9 KB | Deployment model FP32 (PyTorch state dict) |
| `papercnn_16k.onnx` | 65.7 KB | ONNX FP32 |
| `papercnn_16k_int8.onnx` | 36.7 KB | ONNX Int8 dynamic quantization (edge target) |
| `main_353k.pth` | 1.4 MB | Offline model FP32 (higher accuracy, not for MCU) |

All weights were trained with 33-fold LOSO cross-validation on the full 33-subject dataset.

### Running ONNX Inference

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('weights/papercnn_16k_int8.onnx')
# input: (1, 128) float32 — single channel, 128 samples
x = np.random.randn(1, 128).astype(np.float32)
reliability = sess.run(None, {'input': x})[0]   # (1,) float32
```

Repeat for each of the 32 channels, then pass the reliability array to the SpectrumFuser.

### Deployment Path to ESP32-S3

```
papercnn_16k_int8.onnx
    ↓  onnx2tf
model.tflite  (~37 KB)
    ↓  xxd -i model.tflite > model_data.h
model_data.h  → embed in ESP32-S3 Flash, run with TFLite Micro
```

> **Status**: TFLite Micro firmware integration is not yet complete.  
> The ONNX Int8 model runs correctly on PC with ONNX Runtime.

### Edge Resource Budget

| Metric | Value |
|--------|-------|
| Int8 model size | 36.7 KB |
| Peak activation RAM (per channel, streaming) | 512 bytes |
| Weights RAM (Int8) | ~16 KB |
| Total inference RAM | **~16.2 KB** |
| Inference latency (CPU, single channel) | 57.98 ms |
| Window period (128 samples @ 13.5 Hz) | 9.48 s |

---

## 中文說明

### 預訓練權重

| 檔案 | 大小 | 說明 |
|------|------|------|
| `papercnn_16k.pth` | 86.9 KB | 部署模型 FP32（PyTorch state dict）|
| `papercnn_16k.onnx` | 65.7 KB | ONNX FP32 |
| `papercnn_16k_int8.onnx` | 36.7 KB | ONNX Int8 動態量化（邊緣部署目標）|
| `main_353k.pth` | 1.4 MB | 離線主模型 FP32（精度較高，非 MCU 用）|

所有權重以 33 名受試者全資料集的 33-fold LOSO 交叉驗證訓練。

### 邊緣資源預算

| 指標 | 數值 |
|------|------|
| Int8 模型大小 | 36.7 KB |
| 峰值啟動 RAM（單通道串流）| 512 bytes |
| 權重 RAM（Int8）| ~16 KB |
| 推論總 RAM | **~16.2 KB** |
| 推論延遲（CPU，單通道）| 57.98 ms |
| 窗口週期（128 samples @ 13.5 Hz）| 9.48 s |
