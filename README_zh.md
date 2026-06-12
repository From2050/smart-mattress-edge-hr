# 智慧床墊邊緣心率監測平台

**抗姿勢干擾之輕量化智慧床墊邊緣心率監測 — 低成本 32 通道壓力感測陣列**

> 國立陽明交通大學 碩士論文，2026  
> [English Version](README.md)

---

## 專案概述

本倉庫為一個**完整的端對端平台**，實現無穿戴式裝置的床墊心率監測，涵蓋資料收集韌體、Streamlit 研究平台、機器學習訓練管線，以及邊緣部署成品。

系統使用 **32 個 FSR（力敏電阻）通道**，以 13.5 Hz 取樣率記錄體重微小震動。16K 參數輕量 CNN 對每個通道的訊號品質評分，再透過「Top-3 峰值投票 + Viterbi HMM」後處理管線融合可靠通道，輸出最終心率——**推論時無需任何穿戴式感測器**。

```
ESP32-S3 (FreeRTOS)                         ← firmware/
  32ch FSR + MAX30105 PPG（僅作 Ground Truth）
         │ Serial 921600 baud
         ▼
  PC 接收軟體 (receiver/PPG_read.py)         ← receiver/
         │  .txt 資料檔，每行 35 欄
         ▼
  Streamlit 研究平台  (app/)                 ← app/   ★ 新增
    收集 · 觀測 · 訓練 · 部署
         │
         ▼
  ML 管線 (pipeline/)                        ← pipeline/
    Stage 1：PPG FFT → 每窗口 HR 標籤
    Stage 2：CNN → 每通道可靠度分數
    Stage 3：Top-3 投票 + Viterbi → 最終 HR
         │
         ▼
  邊緣部署 (deploy/)                         ← deploy/
    Int8 ONNX  36.7 KB
    峰值 RAM   ~16.2 KB
    推論延遲   57.98 ms / 窗口
```

---

## 實驗結果

| 模型 | 參數量 | LOSO MAE | Acc@5 | 備註 |
|------|--------|----------|-------|------|
| NoAttentionCNN_Large（離線）| 353,601 | **7.58 BPM** | 66.7% | 33-fold LOSO，5 種姿勢 |
| PaperCNN_Reliability（部署）| 16,057 | 7.81 BPM | 63.3% | Wilcoxon p = 0.72，無顯著差異 |

兩模型共用完全相同的後處理管線。部署模型體積縮小 **22 倍**，精度無統計顯著差異。

---

## 倉庫結構

```
smart-mattress-edge-hr/
├── app/                   Streamlit 研究平台（5 頁）
│   ├── dashboard.py       入口 — st.navigation 路由
│   ├── lib.py             共用模型、訊號處理、圖表建構
│   └── views/
│       ├── overview.py    管線總覽與設計理念
│       ├── collect.py     韌體接線 + 互動式設定產生器
│       ├── observe.py     互動式 BCG 儀表板（播放 / 拖曳）
│       ├── train.py       架構、標籤規則、LOSO、即時分數檢查
│       └── deploy.py      匯出路徑、成品大小、MCU 資源預算
├── firmware/              ESP32-S3 ESP-IDF 資料收集韌體
├── receiver/              PC 端 Python 串列接收器
├── pipeline/              ML 訓練 + LOSO 驗證腳本
├── deploy/                預訓練權重（PyTorch / ONNX / Int8）
└── docs/                  硬體與方法詳細文件
```

---

## Streamlit 研究平台

瀏覽整個系統最快速的方式。

### 安裝與啟動

```bash
# 在專案根目錄執行
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt

# 下載資料集（觀測與訓練頁面需要）
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

開啟瀏覽器至 **http://localhost:8501**。

### 各頁面功能

| 頁面 | 功能說明 |
|------|---------|
| 🏠 **總覽** | 兩分鐘內理解四階段管線與每通道獨立投票的設計理念。 |
| 📡 **資料收集** | 查看硬體接線圖。使用**互動式設定產生器**選擇通道數與每通道取樣時間，即時預覽取樣率和窗口長度，並下載可直接貼入韌體的 `sensor_config.h`。 |
| 🔬 **即時觀測** | 選擇受試者與姿勢，按 **▶ 播放**或拖曳滑桿逐窗口觀看：(1) 壓力熱圖（體重 DC 分佈），(2) BCG 振幅熱圖（心跳 AC 訊號強度），(3) 可靠通道的 BCG 波形（每個振盪 ≈ 一次心跳），(4) 加權共識頻譜與投票 BPM，(5) 每通道 CNN 可靠度長條圖，(6) 全 session HR 估計對比 PPG ground truth（含即時游標）。 |
| 🧠 **訓練** | 查看 11 層全卷積架構。使用**互動式標籤模擬器**——輸入通道 FFT 峰值、PPG 參考與 SNR，即時判斷通道是否被標記為可靠。執行訓練指令。在任意 session 上查看即時 **CNN 分數分佈直方圖**。 |
| 🚀 **邊緣部署** | 追蹤 PyTorch → ONNX → Int8 → TFLite 的完整轉換路徑。查看**實際磁碟上的成品大小**。透過時序預算圖確認 32 通道最壞情況推論時間遠低於窗口週期。 |

---

## 快速上手 — 各元件獨立使用

### 1. 燒錄韌體

```bash
cd firmware
idf.py set-target esp32s3
idf.py build flash monitor
```

需要 ESP-IDF v5.x。接線方式與組建細節見 [firmware/README.md](firmware/README.md)。

### 2. 收集資料

```bash
pip install pyserial matplotlib numpy
python receiver/PPG_read.py
# 按 S 開始記錄姿勢，按 E 結束並儲存
```

每個 `.txt` 檔每行 35 欄：`timestamp, ch0–ch31, ppg, esp32_us`。

### 3. 訓練模型

```bash
cd pipeline
pip install -r requirements.txt
python run_exp1_main.py          # 主模型 33-fold LOSO（GPU ~2h / CPU ~8h）
python run_exp5_lightweight.py   # 部署模型比較
python run_edge_deploy.py        # ONNX + Int8 匯出 + 延遲分析
```

### 4. Python 推論範例

```python
from pipeline.common import PaperCNN_Reliability
import torch

model = PaperCNN_Reliability()
model.load_state_dict(torch.load('deploy/weights/papercnn_16k.pth'))
model.eval()

# x: (1, 32, 128) — batch=1, 32 通道，每窗口 128 樣本
x = torch.randn(1, 32, 128)
reliability = model(x)   # (1, 32) 每通道可靠度分數，範圍 [0, 1]
```

---

## 硬體規格

| 元件 | 規格 |
|------|------|
| MCU | ESP32-S3 (Xtensa LX7, 240 MHz, 512 KB SRAM) |
| 壓力感測 | FSR × 32，16 列 × 2 行網格，ADS1115 × 2 + 4-to-1 MUX |
| PPG（參考用）| MAX30105 指夾，Green LED，800 SPS |
| 取樣率 | 13.5 Hz（32ch）/ 27.03 Hz（16ch 版本）|
| ADC | ADS1115 16-bit，860 SPS，單端量測 |

> **ADC 極性**：數值越高 = 壓力越小；越低 = 壓力越大。  
> BCG 訊號 = 疊加在靜態直流偏壓上的微小動態波動（心跳造成體重位移 ~100–1000 ADC counts）。

詳細硬體說明見 [docs/data_description_zh.md](docs/data_description_zh.md)。

---

## 資料集

33 名健康受試者（18–35 歲）· 每人 5 種姿勢 · 預處理後約 270 MB

| 姿勢 | 說明 |
|------|------|
| Front | 仰臥（面朝上）|
| Back | 俯臥（面朝下）|
| LeftSide / RightSide | 側臥 |
| Leave | 仰臥 → 離床 → 返回 |

**下載連結**：[huggingface.co/datasets/m46012002/smart-mattress-bcg](https://huggingface.co/datasets/m46012002/smart-mattress-bcg)

---

## 預訓練權重

| 檔案 | 大小 | 說明 |
|------|------|------|
| `deploy/weights/papercnn_16k.pth` | 86.9 KB | 部署模型 FP32（PyTorch）|
| `deploy/weights/papercnn_16k.onnx` | 65.7 KB | ONNX FP32 |
| `deploy/weights/papercnn_16k_int8.onnx` | 36.7 KB | **ONNX Int8 — 邊緣部署目標** |
| `deploy/weights/main_353k.pth` | 1.4 MB | 離線主模型 FP32（精度較高，非 MCU 用）|

---

## 方法摘要

**為什麼採用每通道獨立評分？**  
實驗顯示，在所有姿勢下，BCG 各通道間幾乎無法互相提供有效訊息——即使相位校正後，一個通道也無法改善另一個的估計。因此系統對每個通道獨立評分（共享權重 CNN），再透過加權頻譜投票融合可靠通道。

**訓練標籤**  
通道窗口被標記為「可靠」的條件：BCG FFT 峰值與 PPG 參考 HR 誤差 < 5 BPM，**且**頻譜 SNR > 3。296,448 個通道窗口中只有 13.1% 為可靠，屬嚴重不平衡的二元分類任務。

**部署策略**  
全卷積架構無 FC 層、無跨通道運算，可在 MCU 上逐通道串流推論：每通道峰值啟動 512 bytes，~16 KB 權重常駐 RAM。32 通道全部推論共需約 1.85 秒，遠低於 9.48 秒的窗口週期。

詳細方法說明見 [docs/method_zh.md](docs/method_zh.md)。

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
