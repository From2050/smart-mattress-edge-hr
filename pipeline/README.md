# Pipeline — ML Training & Evaluation

> [繁體中文](#中文說明) | [English](#english-guide)

---

## English Guide

### Setup

```bash
cd pipeline
pip install -r requirements.txt
```

CUDA is recommended but not required. CPU training works but is significantly slower.

### Scripts

| Script | Description |
|--------|-------------|
| `common.py` | All model definitions, training loop, LOSO evaluation, SpectrumFuser post-processor |
| `run_exp1_main.py` | 33-fold LOSO for the offline model (NoAttentionCNN_Large, 353K params) |
| `run_exp3_ablation.py` | Ablation study: model variants and post-processing components |
| `run_exp5_lightweight.py` | LOSO comparison: deployment model vs offline model |
| `run_exp6_analysis.py` | Per-subject / per-posture error analysis |
| `run_edge_deploy.py` | ONNX export + Int8 quantization + latency/RAM profiling |

### Running LOSO Experiments

```bash
# Step 1: main model (takes ~2h on GPU, ~8h on CPU)
python run_exp1_main.py

# Step 2: deployment model comparison (needs exp1 results)
python run_exp5_lightweight.py

# Step 3: export to ONNX + Int8
python run_edge_deploy.py
```

All outputs go to `outputs/` (created automatically).

### Data Path

By default, the pipeline expects data at `../data/ESP32_recored/`.  
Edit `OUTPUT_ROOT` and `DATA_ROOT` at the top of `common.py` if your layout differs.

### Inference Example

```python
from common import PaperCNN_Reliability, make_fuser
import torch

model = PaperCNN_Reliability()
model.load_state_dict(torch.load('../deploy/weights/papercnn_16k.pth'))
model.eval()

# window: (1, 32, 128) — batch=1, 32 channels, 128 samples
window = torch.randn(1, 32, 128)
reliability = model(window)        # (1, 32)

fuser = make_fuser()
hr_bpm = fuser(window.numpy(), reliability.detach().numpy())
print(f"Estimated HR: {hr_bpm:.1f} BPM")
```

---

## 中文說明

### 安裝依賴

```bash
cd pipeline
pip install -r requirements.txt
```

建議使用 CUDA，但 CPU 也可執行（訓練較慢）。

### 主要腳本說明

| 腳本 | 功能 |
|------|------|
| `common.py` | 所有模型定義、訓練迴圈、LOSO 評估、SpectrumFuser 後處理 |
| `run_exp1_main.py` | 主模型 33-fold LOSO（NoAttentionCNN_Large，353K 參數）|
| `run_exp3_ablation.py` | 消融實驗：模型變體與後處理元件 |
| `run_exp5_lightweight.py` | LOSO 比較：部署模型 vs 主模型 |
| `run_exp6_analysis.py` | 每受試者 / 每姿勢誤差分析 |
| `run_edge_deploy.py` | ONNX 匯出 + Int8 量化 + 延遲 / RAM 分析 |
