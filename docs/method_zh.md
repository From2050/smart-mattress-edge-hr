# 方法與邊緣部署說明（工程交接用）

---

## 1. 系統總覽

```
32ch 壓力床墊 + PPG 指夾
         ↓ 原始 ADC (35 col txt)
    ┌────────────────────────────────────┐
    │  Stage 1: Ground Truth + 標註       │
    │  PPG FFT → HR label                │
    │  Per-channel FFT → reliable/unreliable │
    └────────────────────────────────────┘
         ↓ channel_level_annotations.csv
    ┌────────────────────────────────────┐
    │  Stage 2: 通道可靠度模型訓練         │
    │  CNN backbone → sigmoid(reliable)  │
    └────────────────────────────────────┘
         ↓ per-channel reliability scores
    ┌────────────────────────────────────┐
    │  Stage 3: 多通道頻譜融合 + Viterbi   │
    │  Top-3 Peak Voting → Viterbi → HR  │
    └────────────────────────────────────┘
```

**核心觀點**：不做端到端心率回歸，而是先判「哪些通道可信」，再用可信通道的頻譜投票出心率。

---

## 2. 兩種模型骨幹

### 2.1 主模型 — `NoAttentionCNN_Large`（論文表 4-1 ProposedCNN Reliability）

| 項目 | 規格 |
|------|------|
| 參數量 | **353,601** |
| 架構 | 3-layer CNN (16→32→64) + MaxPool + 4-layer MLP (1024→256→256→64→1) |
| 輸入 | 單通道 `(1, 128)` 時域訊號，32ch 共享權重 |
| 輸出 | `sigmoid → [0,1]` 可靠度分數 |
| 訓練 | BCEWithLogitsLoss + Adam, 30 epochs |
| 用途 | 離線分析、LOSO 驗證主結果 |

```
(1,128) → Conv1d(1→16, k=5) → MaxPool(2) → (16,64)
       → Conv1d(16→32, k=5) → MaxPool(2) → (32,32)
       → Conv1d(32→64, k=3) → MaxPool(2) → (64,16)
       → Flatten(1024) → FC(256) → Drop(0.5)
       → FC(256) → Drop(0.1) → FC(64) → FC(1) → sigmoid
```

### 2.2 部署模型 — `PaperCNN_Reliability`（論文表 4-6 Tramontano's CNN）

| 項目 | 規格 |
|------|------|
| 參數量 | **16,057**（主模型的 4.5%）|
| 架構 | 11-layer All-Convolutional（無 FC、無 Transformer）|
| 輸入/輸出 | 同上 |
| 用途 | **MCU 邊緣部署目標** |

```
(1,128) → Conv(1→8,k=7,s=2)→BN→LReLU → (8,64)
        → Conv(8→8,k=5,s=1)→BN→LReLU → (8,64)
        → Conv(8→16,k=5,s=2)→BN→LReLU → (16,32)
        → Conv(16→16,k=3,s=1)→BN→LReLU → (16,32)
        → Conv(16→16,k=3,s=2)→BN→LReLU → (16,16)
        → Conv(16→32,k=3,s=1)→BN→LReLU → (32,16)
        → Conv(32→32,k=3,s=2)→BN→LReLU → (32,8)
        → Conv(32→32,k=3,s=2)→BN→LReLU → (32,4)
        → Conv(32→32,k=3,s=2)→BN→LReLU → (32,2)
        → Conv(32→32,k=2,s=2)→BN→LReLU → (32,1)
        → Conv(32→1,k=1,s=1) → sigmoid
```

> 全部使用 stride 下採樣，無 pooling 層、無 FC 層。  
> TFLite Micro 完全支援，無需拆解特殊算子。

### 2.3 精度比較

| 評估方式 | 主模型 (353K) | 部署模型 (16K) | 差距 |
|---------|:---:|:---:|:---:|
| LOSO 33-fold MAE | **7.58 BPM** | 7.81 BPM | +0.23 |
| LOSO Acc@5 | **66.7%** | 63.3% | -3.4% |
| Wilcoxon p-value | — | 0.72 | 無顯著差異 |
| Holdout 5人 MAE | 8.02 BPM | **7.48 BPM** | -0.54 |

> 兩模型共享完全相同的後處理管線，僅骨幹不同。

---

## 3. 後處理管線（兩模型共用）

```
模型輸出: reliability[32]  (每通道 0~1 分數)
    ↓
(1) 空間平滑: 8×4 grid reshape → EMA 加權
    ↓
(2) Per-channel FFT: 128pt Hanning → 0.75~3.0 Hz 頻段
    ↓
(3) Top-3 Peak Voting:
    每通道取頻譜前 3 強峰 → 以 Gaussian kernel (σ=0.05 Hz) 投票
    → 乘以 reliability score → 累加為 consensus spectrum
    ↓
(4) Viterbi 時序平滑:
    狀態空間 = HR 頻段離散頻率
    發射機率 = consensus spectrum
    轉移機率 = Gaussian(Δf, σ=0.05)
    → 全局最佳路徑 → 最終 HR (BPM)
```

**程式碼入口**: `common.py → evaluate_reliability() → SpectrumFuser`

---

## 4. 標註方法（Ground Truth 建立）

| 步驟 | 說明 |
|------|------|
| PPG 處理 | 去 DC + Hanning window → 1024pt zero-padding FFT |
| 參考 HR | 0.75–3.0 Hz 最大峰 × 60 = BPM |
| 異常修正 | 相鄰窗口跳變 > 20 BPM → forward fill |
| 通道標註 | 每通道 FFT 主頻 vs PPG HR：**bpm_error < 5 且 SNR > 3 → reliable** |

結果：296,448 筆通道標註，其中 reliable 僅 13.1%（高度不均衡）。

---

## 5. 邊緣部署規格

### 5.1 目標平台

| 項目 | 規格 |
|------|------|
| MCU | ESP32-S3 (Xtensa LX7, 240 MHz) |
| SRAM | 512 KB |
| Flash | 4–16 MB |
| Framework | TFLite Micro / ESP-NN |

### 5.2 部署模型資源

| 指標 | 數值 |
|------|------|
| 參數量 | 16,057 |
| FP32 weight (.pth) | 86.9 KB |
| FP32 ONNX | 65.7 KB |
| **Int8 ONNX** | **36.7 KB** |
| 推論峰值 RAM（串流 Int8） | **~16.2 KB**（512 bytes activation + 16K weights）|
| CPU 串流延遲 | 57.98 ms（<< 9.48s 窗口週期）|
| Int8 量化損失 | mean diff 0.0026（可忽略）|

### 5.3 串流推論策略

```
PaperCNN 為純卷積，32 通道完全獨立：

for ch in 0..31:
    reliability[ch] = AllConv_11layer(adc_window[ch])  # (128,) → scalar
    // 峰值中間張量: 512 bytes (Int8)
    // 不需 feature buffer

→ 進入 SpectrumFuser + Viterbi（C 實作 ~4 KB 額外 RAM）
```

> 無 Transformer、無 FC 層、無跨通道交互。逐通道串流推論自然成立。

### 5.4 部署路徑

```
papercnn_16k.pth (86.9 KB)
    ↓  torch.onnx.export (opset 17)
papercnn_16k.onnx (65.7 KB)
    ↓  onnxruntime quantize_dynamic
papercnn_16k_int8.onnx (36.7 KB)
    ↓  onnx2tf → tflite_converter
model.tflite (~37 KB)
    ↓  xxd -i
model_data.h → 嵌入 ESP32-S3 Flash
```

---

## 6. 產出檔案索引

### 訓練與驗證腳本

| 腳本 | 功能 |
|------|------|
| `thesis_unified/run_exp1_main.py` | 主模型 33-fold LOSO |
| `thesis_unified/run_exp5_lightweight.py` | 部署模型 vs 主模型 LOSO 比較 |
| `thesis_unified/run_edge_deploy.py` | 部署可行性驗證（ONNX/Int8/延遲/RAM）|
| `thesis_unified/common.py` | 所有模型定義、訓練、評估、融合管線 |

### 模型權重與 ONNX

| 檔案 | 路徑 |
|------|------|
| 部署模型 FP32 | `outputs/thesis_unified/edge_deploy/papercnn_16k.pth` |
| 部署模型 ONNX FP32 | `outputs/thesis_unified/edge_deploy/papercnn_16k.onnx` |
| 部署模型 ONNX Int8 | `outputs/thesis_unified/edge_deploy/papercnn_16k_int8.onnx` |
| 主模型 FP32 | `outputs/thesis_unified/edge_deploy/main_353k.pth` |

### 實驗結果

| 檔案 | 內容 |
|------|------|
| `outputs/thesis_unified/exp1_main/full_results.csv` | 主模型 LOSO 全結果 |
| `outputs/thesis_unified/exp5_lightweight/summary.csv` | 部署模型 vs 主模型摘要 |
| `outputs/thesis_unified/edge_deploy/EDGE_DEPLOY_REPORT.md` | 完整部署可行性報告 |
| `outputs/thesis_unified/edge_deploy/holdout_papercnn.csv` | 部署模型 holdout 結果 |

---

## 7. 快速上手

### 跑主模型 LOSO
```bash
python thesis_unified/run_exp1_main.py
# → outputs/thesis_unified/exp1_main/
```

### 跑部署模型比較
```bash
python thesis_unified/run_exp5_lightweight.py   # 需先跑 exp1
# → outputs/thesis_unified/exp5_lightweight/
```

### 跑邊緣部署驗證
```bash
python thesis_unified/run_edge_deploy.py
# → outputs/thesis_unified/edge_deploy/
#    papercnn_16k.onnx, papercnn_16k_int8.onnx, EDGE_DEPLOY_REPORT.md
```

### 載入模型做推論
```python
from thesis_unified.common import PaperCNN_Reliability, make_fuser
import torch

model = PaperCNN_Reliability()
model.load_state_dict(torch.load('outputs/thesis_unified/edge_deploy/papercnn_16k.pth'))
model.eval()

# x: (1, 32, 128) — 1 window, 32 channels, 128 samples
x = torch.randn(1, 32, 128)
reliability = model(x)  # (1, 32) — per-channel reliability scores
```
