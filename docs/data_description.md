# Dataset Description

> [繁體中文版](data_description_zh.md)

**Data root**: `data/ESP32_recored/`

---

## 1. Hardware Architecture

```
ESP32-S3 (FreeRTOS)
  ├── I2C1 @ 400kHz ──┬── ADS1115_1 (0x48) ── 4-ch single-ended
  │                    └── ADS1115_2 (0x49) ── 4-ch single-ended
  │                         × 4 MUX settings = 32 pressure channels
  └── I2C0 ── MAX30105 PPG (fingertip clip, Green LED)
```

### Pressure Sensors (FSR × 32)

- **Physical layout**: 16-column × 2-row rectangular grid placed on top of a latex mattress (Figure 3-2 in thesis)
  - Bottom row: channels 1–16 (left → right), top row: channels 17–32 (right → left, reversed)
- **Sensing principle**: FSR voltage-divider circuit. **Higher ADC value ≈ less pressure; lower ≈ more pressure**
  - No contact: FSR high impedance → divider node ≈ full-scale → ADC ≈ 21,800–22,000
  - Contact: FSR low impedance → divider node drops → ADC ≈ 1,900–11,000
  - BCG signal = tiny dynamic fluctuation riding on static DC bias (body-weight displacement from heartbeat)
- **ADC**: ADS1115 16-bit, DR = 860 SPS, single-ended
  - Error codes: `-2` (read failure), `-3` (init failure), `-4` (invalid device code)
- **Channel numbering** (firmware scan order):
  ```
  ch = mux_idx × 8 + adc_idx
  mux_idx ∈ {0,1,2,3}    # MUX switch (GPIO35/36)
  adc_idx ∈ {0..7}        # 0–3 → ADS_1, 4–7 → ADS_2
  ```

### PPG Reference (MAX30105)

- Fingertip clip, Green LED mode, sampleRate = 800, pulseWidth = 215, adcRange = 16384
- Used only as heart rate **ground truth** — **not an input at deployment time**

### Sample Rate

The firmware polls ch0 → ch31 sequentially in a FreeRTOS task; each I2C read takes ≈ 2.28 ms:

| Firmware variant | Channels scanned | T_cycle | Sample rate |
|-----------------|-----------------|---------|-------------|
| 32ch (subjects 101–223) | ch0–31 | ≈ 74,000 µs | **13.5 Hz** |
| 16ch (subject 224) | ch0–15 | ≈ 37,000 µs | **27.03 Hz** |

> ch0 is sampled first, ch31 last; maximum inter-channel delay ≈ 0.97 samples (72 ms).

---

## 2. Raw File Format

One file per posture session: `ppg_data_YYYYMMDD_HHMMSS.txt`, comma-separated, no header.

| Column | Name | Type | Description |
|--------|------|------|-------------|
| 0 | timestamp | string | PC wall-clock time `"YYYY-MM-DD HH:MM:SS.mmm"` |
| 1–32 | ch0–ch31 | int16 | FSR pressure ADC (high = no pressure, low = pressure) |
| 33 | ppg | int32 | MAX30105 PPG reading |
| 34 | esp32_us | int64 | `esp_timer_get_time()` microsecond timestamp |

Example row:
```
2025-05-20 13:14:43.343,21879,21879,...,5092,3267,...,21888,99840,991080717
```

---

## 3. Subjects and Collection Protocol

33 healthy subjects (18–35 years old, mixed sex), collected in three groups.

### Group 1 — Subjects 101–110 (10 subjects)

| Item | Details |
|------|---------|
| Sample rate | 13.5 Hz |
| Duration per posture | **2 minutes** (≈ 1,640 rows) |
| Postures | Front, Back, RightSide, LeftSide + U-variants of each (8 total) |

**Protocol**: Each posture: simultaneous pressure mat + fingertip PPG recording for 2 minutes.  
**U-prefix** (UFront, UBack, …) = subject climbed stairs before recording the same posture, creating an elevated / recovering heart rate scenario. Merged into the corresponding posture during post-processing.

| Subject | Weight (kg) | Sex |
|---------|------------|-----|
| 101 | 105 | M |
| 102 | 80 | M |
| 103 | 47.6 | F |
| 104 | 66 | M |
| 105 | 61 | M |
| 106 | 52 | M |
| 107 | 66 | M |
| 108 | 62 | M |
| 109 | 62 | M |
| 110 | 44.5 | F |

### Group 2 — Subjects 201–223 (23 subjects)

| Item | Details |
|------|---------|
| Sample rate | 13.5 Hz |
| Duration per posture | **5 minutes** (≈ 4,100 rows) |
| Posture order | Front → RightSide → Back → Leave → LeftSide |

**Protocol**:
- **Front / RightSide / Back / LeftSide**: 10–15 squats to elevate heart rate → lie still for 5 minutes
- **Leave** (no squats): supine 2 min → sit up and fully leave mattress 30 sec → return supine 2.5 min

| Subject | Weight (kg) | Sex | Age |
|---------|------------|-----|-----|
| 201 | 72.4 | M | 22 |
| 202 | 66.3 | M | 23 |
| 203 | 79.9 | M | 23 |
| 204 | 55.7 | F | 26 |
| 205 | 55.9 | F | 26 |
| 206 | 47.8 | F | 25 |
| 207 | 50.5 | F | 18 |
| 208 | 58.5 | M | 26 |
| 209 | 54.6 | F | 22 |
| 210 | 64.1 | M | 24 |
| 211 | 53.8 | F | 18 |
| 212 | 52.3 | F | — |
| 213 | 57.2 | F | — |
| 214 | 85.8 | M | — |
| 215 | 82.4 | M | — |
| 216 | 68.7 | M | — |
| 217 | 75.3 | M | — |
| 218 | 42.1 | F | — |
| 219 | 51.0 | F | — |
| 220 | 82.5 | M | — |
| 221 | 62.1 | M | — |
| 222 | 113.0 | M | — |
| 223 | 50.8 | F | — |

### Group 3 — Subject 224 (1 subject, 16ch firmware)

| Item | Details |
|------|---------|
| Sample rate | 27.03 Hz (half channels scanned → double sample rate) |
| Postures | Same as Group 2 (5 postures) |
| Note | **ch16–ch31 are all 0** (firmware does not enable them) |

---

## 4. Posture Reference

| Folder name | Posture | Description |
|-------------|---------|-------------|
| `Front` | Supine | Face up, back against mattress |
| `Back` | Prone | Face down, chest against mattress |
| `RightSide` | Right lateral | — |
| `LeftSide` | Left lateral | — |
| `Leave` | On-bed / off-bed | Supine → leave → return, recorded continuously |
| `UFront` etc. | U-variants | Post-exercise version of each posture (Group 1 only) |

---

## 5. Exclusion Rules

| Excluded | Reason |
|----------|--------|
| Subjects 100, 200 | Test/calibration runs — `EXCLUDE_SUBJECTS = [100, 200]` |
| LeftSide / ULeftSide (16ch experiments only) | Left-lateral chest pressure falls on ch16–31, incompatible with 16ch firmware |

---

## 6. Supplementary Files

- **`subject_weights.csv`**: Weight (kg) and Sex (M/F) for subjects 101–223
- **`Utility.txt`** (per-subject folder): subjects 101–110 have Weight/Height; 201–211 have Weight/Age/Sex; 212–223 do not have this file

---

## 7. Downstream Processing Entry Points

```
Raw .txt files
  ↓ generate_ground_truth.py       → PPG FFT → per-window ground-truth HR
  ↓ stage1/step1_generate_annotations.py → per-channel reliability labels
  ↓ pipeline/common.py: load_all_data()  → load training data
```

**Ground-truth parameters**: 128-pt window @ 13.5 Hz (9.48 s), 50% overlap, Hanning FFT, peak in 0.75–3.0 Hz × 60 = BPM. HR jumps > 20 BPM between adjacent windows are corrected with forward fill.
