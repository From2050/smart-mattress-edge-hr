# 智慧床墊邊緣心率監測

**抗姿勢干擾之輕量化智慧床墊邊緣心率監測 — 低成本 32 通道壓力感測陣列**

> 國立陽明交通大學 碩士論文，2026  
> [English Version](README.md)

---

## 專案概述

本倉庫包含完整的硬體設計、資料收集韌體、機器學習訓練管線，以及邊緣部署成品，用於實作以 BCG（彈震描記法）為基礎的床墊心率監測系統。

系統使用 **32 個 FSR（力敏電阻）通道**，以 13.5 Hz 取樣率記錄體重微小震動。輕量 CNN 模型對每個通道的訊號品質評分，再透過「Top-3 峰值投票 + Viterbi HMM」後處理管線融合可靠通道，輸出最終心率——**全程不需穿戴式感測器**。

```
ESP32-S3 (FreeRTOS)
  32ch FSR + MAX30105 PPG（僅作 Ground Truth）
         │ Serial 921600 baud
         ▼
  PC 接收軟體 (PPG_read.py)  →  .txt 資料檔
         │
         ▼
  ML 管線 (pipeline/)
    Stage 1：PPG FFT → 每窗口 HR 標籤
    Stage 2：CNN → 每通道可靠度分數
    Stage 3：Top-3 投票 + Viterbi → 最終 HR
         │
         ▼
  邊緣部署 (deploy/)
    Int8 ONNX  36.7 KB
    峰值 RAM   ~16.2 KB
    推論延遲   57.98 ms / 窗口
```

---

## 實驗結果

| 模型 | 參數量 | LOSO MAE | 備註 |
|------|--------|----------|------|
| NoAttentionCNN_Large（離線）| 353,601 | **7.58 BPM** | 33-fold LOSO，5 種姿勢 |
| PaperCNN_Reliability（部署）| 16,057 | 7.81 BPM | Wilcoxon p = 0.72，無顯著差異 |

兩個模型共用完全相同的後處理管線。部署模型體積縮小 **22 倍**，精度無統計顯著差異。

---

## 倉庫結構

```
smart-mattress-edge-hr/
├── firmware/          ESP32-S3 ESP-IDF 資料收集韌體
├── receiver/          PC 端 Python 串列接收器
├── pipeline/          ML 訓練 + LOSO 驗證腳本
├── deploy/            預訓練權重（ONNX / Int8）
└── docs/              硬體與方法詳細文件
```

---

## 快速上手

### 1. 燒錄韌體

```bash
cd firmware
idf.py build flash monitor
```

需要 ESP-IDF v5.x。接線方式與組建細節見 [firmware/README.md](firmware/README.md)。

### 2. 收集資料

```bash
pip install pyserial matplotlib numpy
python receiver/PPG_read.py
```

每個姿勢存成一個 `.txt` 檔，每行 35 欄：timestamp、ch0–ch31 ADC 值、PPG、esp32_us。

### 3. 訓練模型

```bash
cd pipeline
pip install -r requirements.txt
python run_exp1_main.py          # 主模型 33-fold LOSO
python run_exp5_lightweight.py   # 部署模型比較
```

### 4. 匯出至邊緣端

```bash
python pipeline/run_edge_deploy.py
# 輸出：papercnn_16k.onnx, papercnn_16k_int8.onnx, EDGE_DEPLOY_REPORT.md
```

### 5. Python 推論範例

```python
from pipeline.common import PaperCNN_Reliability
import torch

model = PaperCNN_Reliability()
model.load_state_dict(torch.load('deploy/weights/papercnn_16k.pth'))
model.eval()

# x: (batch, 32, 128) — 32 通道，每窗口 128 個樣本
x = torch.randn(1, 32, 128)
reliability = model(x)   # (batch, 32) 每通道可靠度分數
```

---

## 硬體規格

| 元件 | 規格 |
|------|------|
| MCU | ESP32-S3 (Xtensa LX7, 240 MHz, 512 KB SRAM) |
| 壓力感測 | FSR × 32，16 列 × 2 行網格，ADS1115 × 2 + MUX × 4 |
| PPG（參考用）| MAX30105 指夾，Green LED，800 SPS |
| 取樣率 | 13.5 Hz（32ch）/ 27.03 Hz（16ch 版本）|
| ADC | ADS1115 16-bit，860 SPS，單端量測 |

> **ADC 極性**：數值越高 = 壓力越小；越低 = 壓力越大。  
> BCG 訊號 = 疊加在靜態直流偏壓上的微小動態波動（心跳造成體重位移）。

詳細硬體說明見 [docs/data_description_zh.md](docs/data_description_zh.md)。

---

## 資料集

資料集共 33 名健康受試者（18–35 歲），涵蓋 5 種姿勢：
`Front`（仰臥）、`Back`（俯臥）、`LeftSide`（左側臥）、`RightSide`（右側臥）、`Leave`（離床）。

原始資料因檔案過大（預處理後約 270 MB），不直接放入本倉庫。  
**下載連結**：https://huggingface.co/datasets/m46012002/smart-mattress-bcg

完整收集協議與檔案格式說明見 [docs/data_description_zh.md](docs/data_description_zh.md)。

---

## 預訓練權重

預訓練權重存放於 `deploy/weights/`：

| 檔案 | 大小 | 說明 |
|------|------|------|
| `papercnn_16k.pth` | 86.9 KB | 部署模型 FP32（PyTorch）|
| `papercnn_16k.onnx` | 65.7 KB | ONNX FP32 |
| `papercnn_16k_int8.onnx` | 36.7 KB | ONNX Int8 量化（邊緣部署目標）|
| `main_353k.pth` | 1.4 MB | 完整離線模型 FP32 |

---

## 引用

若本研究對您有幫助，請引用：

```bibtex
@mastersthesis{wu2026smartmattress,
  author  = {吳介能},
  title   = {抗姿勢干擾之輕量化智慧床墊邊緣心率監測},
  school  = {國立陽明交通大學},
  year    = {2026}
}
```

---

## 授權條款

[MIT License](LICENSE) — © 2026 吳介能，國立陽明交通大學
