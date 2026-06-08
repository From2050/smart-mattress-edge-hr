# Firmware — ESP32-S3 Data Collector

> [繁體中文](#中文說明) | [English](#english-guide)

---

## English Guide

### Hardware Wiring

```
ESP32-S3
  I2C1 (SDA=GPIO8, SCL=GPIO9) @ 400kHz
    ├── ADS1115 #1  addr=0x48  → ch0–ch7   (via MUX)
    └── ADS1115 #2  addr=0x49  → ch8–ch15  (via MUX)
        × 4 MUX settings (GPIO35/36) = 32 total FSR channels

  I2C0 (SDA=GPIO5, SCL=GPIO6)
    └── MAX30105  (fingertip PPG, ground truth only)

  MUX control: GPIO35 (A0), GPIO36 (A1)
  UART0 (TX=GPIO43): 921600 baud → PC receiver
```

### Build & Flash (ESP-IDF v5.x)

```bash
# Install ESP-IDF v5.x first: https://docs.espressif.com/projects/esp-idf/
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

### Output Format

One line per sample, 35 comma-separated values, no header:

```
<timestamp>,<ch0>,<ch1>,...,<ch31>,<ppg>,<esp32_us>
```

- `timestamp`: PC wall-clock `"YYYY-MM-DD HH:MM:SS.mmm"` (added by receiver)
- `ch0–ch31`: 16-bit FSR ADC values (high = no pressure, low = pressure)
- `ppg`: 32-bit MAX30105 green-LED reading
- `esp32_us`: 64-bit `esp_timer_get_time()` microsecond counter

### Sample Rates

| Variant | Channels | Cycle time | Sample rate |
|---------|----------|------------|-------------|
| Default (32ch) | ch0–ch31 | ≈ 74,000 µs | **13.5 Hz** |
| 16ch (`#define CHANNELS_16`) | ch0–ch15 | ≈ 37,000 µs | **27.03 Hz** |

### Key Source Files

| File | Description |
|------|-------------|
| `main/hello_world_main.cpp` | Main FreeRTOS task: sensor polling loop, serial output |
| `main/MAX30105.cpp/.h` | MAX30105 PPG sensor driver |
| `main/heartRate.cpp/.h` | Heart rate peak detection utility |
| `main/Wire.cpp/.h` | Arduino-style I2C wrapper for ESP-IDF |
| `main/spo2_algorithm.cpp/.h` | SpO2 / HR algorithm (Maxim reference) |

---

## 中文說明

### 硬體接線

```
ESP32-S3
  I2C1 (SDA=GPIO8, SCL=GPIO9) @ 400kHz
    ├── ADS1115 #1  addr=0x48  → ch0–ch7   （透過 MUX）
    └── ADS1115 #2  addr=0x49  → ch8–ch15  （透過 MUX）
        × 4 組 MUX 設定 (GPIO35/36) = 共 32 個 FSR 通道

  I2C0 (SDA=GPIO5, SCL=GPIO6)
    └── MAX30105  （指夾式 PPG，僅作 Ground Truth）

  MUX 控制：GPIO35 (A0), GPIO36 (A1)
  UART0 (TX=GPIO43)：921600 baud → PC 接收端
```

### 組建與燒錄（ESP-IDF v5.x）

```bash
# 先安裝 ESP-IDF v5.x：https://docs.espressif.com/projects/esp-idf/
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

### 輸出格式

每個樣本一行，35 個逗號分隔欄位，無標題行：

```
<timestamp>,<ch0>,<ch1>,...,<ch31>,<ppg>,<esp32_us>
```

- `timestamp`：PC 時間 `"YYYY-MM-DD HH:MM:SS.mmm"`（由接收端加入）
- `ch0–ch31`：16-bit FSR ADC 值（高 = 無壓力，低 = 有壓力）
- `ppg`：32-bit MAX30105 綠光 LED 讀值
- `esp32_us`：64-bit `esp_timer_get_time()` 微秒時間戳
