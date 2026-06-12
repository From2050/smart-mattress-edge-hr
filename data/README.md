---
license: mit
language:
  - zh
  - en
task_categories:
  - other
tags:
  - bcg
  - ballistocardiography
  - heart-rate-monitoring
  - pressure-sensor
  - smart-mattress
  - esp32
  - edge-ai
  - physiological-signals
  - wearable-free
  - posture-robust
pretty_name: Smart Mattress BCG Heart Rate Dataset
size_categories:
  - 1M<n<10M
---

# Smart Mattress BCG Heart Rate Dataset

**Posture-Robust Lightweight Smart Mattress Edge Heart Rate Monitoring on Low-Cost Pressure Sensor Arrays**

> Wu Jie-Neng -- National Yang Ming Chiao Tung University, Master's Thesis, 2026
> Code & models: https://github.com/m46012002/smart-mattress-edge-hr

---

## Dataset Summary

This dataset contains ballistocardiography (BCG) signals collected from a 32-channel FSR (force-sensitive resistor) pressure sensor array embedded in a latex mattress. Simultaneously, a fingertip MAX30105 PPG sensor provides heart rate ground truth.

33 healthy subjects performed 5 body postures (supine, prone, left lateral, right lateral, and on/off-bed transitions) while lying on the mattress. The dataset was used to train and evaluate a lightweight CNN-based channel reliability model for wearable-free, posture-robust heart rate monitoring at the edge.

**Key statistics:**
- 33 subjects (age 18-35, mixed sex)
- 5 postures per subject
- 13.5 Hz sample rate (32ch) / 27.03 Hz (16ch variant, subject 224)
- ~225 raw recording files, ~270 MB preprocessed

---

## Dataset Structure

```
ESP32_recored/
|-- 100/                   # test/calibration subject (excluded from analysis)
|-- 101/ ... 110/          # Group 1: 10 subjects, 2-min per posture
|   |-- Front/             # Supine (face up)
|   |-- Back/              # Prone (face down)
|   |-- LeftSide/
|   |-- RightSide/
|   |-- UFront/            # Post-exercise variant (stair climbing before recording)
|   |-- UBack/
|   |-- ULeftSide/
|   `-- URightSide/
|-- 200/                   # test/calibration subject (excluded from analysis)
|-- 201/ ... 223/          # Group 2: 23 subjects, 5-min per posture
|   |-- Front/
|   |-- Back/
|   |-- LeftSide/
|   |-- RightSide/
|   `-- Leave/             # on-bed -> off-bed -> on-bed transition
`-- 224_16ch/              # Group 3: 1 subject, 16ch firmware (27.03 Hz)
    |-- Front/
    |-- Back/
    |-- LeftSide/
    |-- RightSide/
    `-- Leave/
```

Each posture folder contains one file: `ppg_data_YYYYMMDD_HHMMSS.txt`

---

## Data Fields

Each file is comma-separated with **no header row**, 35 columns per line:

| Column | Name | Type | Description |
|--------|------|------|-------------|
| 0 | `timestamp` | string | PC wall-clock time `"YYYY-MM-DD HH:MM:SS.mmm"` |
| 1-32 | `ch0`-`ch31` | int16 | FSR pressure ADC (higher = less pressure, lower = more pressure) |
| 33 | `ppg` | int32 | MAX30105 green-LED reading (ground truth reference only) |
| 34 | `esp32_us` | int64 | `esp_timer_get_time()` microsecond timestamp on ESP32 |

**Example row:**
```
2025-05-20 13:14:43.343,21879,21750,20843,...,5092,3267,...,21888,99840,991080717
```

**ADC polarity:** higher ADC value means less pressure (no contact ~21800-22000); lower means more pressure (under body weight ~1900-11000).
BCG signal = tiny dynamic fluctuation (~100-1000 ADC counts) riding on a large static DC bias.

**PPG note:** the PPG column is the raw finger-clip signal used only for ground-truth HR labelling. It is **not used** as a model input at inference time.

---

## Collection Protocol

### Group 1 -- Subjects 101-110

- Sample rate: 13.5 Hz (32 channels)
- Duration per posture: **2 minutes** (approx. 1,640 rows)
- Postures: Front, Back, RightSide, LeftSide (+ U-variants after stair climbing)
- **U-variants** (UFront, UBack, ULeftSide, URightSide): subject climbed stairs immediately before recording to create an elevated + recovering heart rate scenario. Merged with base posture in downstream processing.

### Group 2 -- Subjects 201-223

- Sample rate: 13.5 Hz (32 channels)
- Duration per posture: **5 minutes** (approx. 4,100 rows)
- Posture order: Front -> RightSide -> Back -> Leave -> LeftSide
- Front/RightSide/Back/LeftSide: subject performs 10-15 squats to elevate HR, then lies still for 5 minutes
- **Leave**: supine 2 min -> fully leave mattress 30 sec -> return supine 2.5 min (continuous recording)

### Group 3 -- Subject 224

- Firmware variant: 16-channel only (ch0-ch15 active; ch16-ch31 are all zero)
- Sample rate: **27.03 Hz** (halved scan channels -> doubled sample rate)
- Same 5 postures as Group 2

---

## Subject Demographics

| Group | Subjects | N | Weight range (kg) | Sex |
|-------|----------|---|-------------------|-----|
| 1 | 101-110 | 10 | 44.5-105.0 | 7M, 3F |
| 2 | 201-223 | 23 | 42.1-113.0 | 12M, 11F |
| 3 | 224 | 1 | -- | -- |

Subjects 100 and 200 are test/calibration runs, excluded from all analyses (`EXCLUDE_SUBJECTS = [100, 200]`).

---

## Hardware

| Component | Specification |
|-----------|---------------|
| MCU | ESP32-S3 (Xtensa LX7, 240 MHz, 512 KB SRAM) |
| Pressure sensors | FSR x32, 16-column x 2-row grid on latex mattress |
| ADC | ADS1115 16-bit @ 860 SPS x 2 chips, via 4-channel MUX |
| PPG (reference) | MAX30105 fingertip clip, Green LED, 800 SPS |
| Firmware | ESP-IDF FreeRTOS; sequential polling ch0->ch31, ~2.28 ms/channel |

**Channel mapping:**
```
ch = mux_idx x 8 + adc_idx
mux_idx in {0,1,2,3}   # MUX select pins GPIO35/36
adc_idx in {0..7}       # 0-3 -> ADS1115 #1 (addr 0x48), 4-7 -> ADS1115 #2 (addr 0x49)
```

Inter-channel sampling delay: ch0 is first, ch31 is last; max delay ~72 ms (0.97 samples at 13.5 Hz).

---

## Usage

### Load a single recording

```python
import pandas as pd

cols = ['timestamp'] + [f'ch{i}' for i in range(32)] + ['ppg', 'esp32_us']
df = pd.read_csv('ESP32_recored/201/Front/ppg_data_20250601_100000.txt',
                 header=None, names=cols)

print(df.shape)           # (approx. 4100, 35)
print(df['ch0'].describe())
```

### Load all subjects for ML training

The `load_all_data()` function in [pipeline/common.py](https://github.com/m46012002/smart-mattress-edge-hr/blob/main/pipeline/common.py) handles sliding-window extraction, PPG FFT ground-truth labelling, and per-channel reliability labelling.

```python
from pipeline.common import load_all_data
windows, labels = load_all_data()
# windows: (N, 32, 128) float32  -- N windows, 32 channels, 128 samples each
# labels:  (N, 32) binary        -- 1 = reliable channel, 0 = unreliable
```

### Ground-truth labelling parameters

| Parameter | Value |
|-----------|-------|
| Window size | 128 samples |
| Overlap | 50% |
| Window function | Hanning |
| HR frequency range | 0.75-3.0 Hz (45-180 BPM) |
| Channel reliable if | bpm_error < 5 AND FFT SNR > 3 |
| Outlier correction | Forward fill if adjacent HR jump > 20 BPM |

---

## Download

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='m46012002/smart-mattress-bcg',
    repo_type='dataset',
    local_dir='data',
)
```

After download, place the `ESP32_recored/` folder at `data/ESP32_recored/` relative to the code repo root.

---

## Related Resources

- **Code & pre-trained models**: https://github.com/m46012002/smart-mattress-edge-hr
- **Method details**: `docs/method.md` in the code repo
- **Hardware & data format**: `docs/data_description.md` in the code repo

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

[MIT License](https://opensource.org/licenses/MIT) -- Copyright 2026 Wu Jie-Neng, NYCU
