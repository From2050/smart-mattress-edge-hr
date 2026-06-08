# Receiver — PC-Side Serial Data Logger

Reads the ESP32-S3 serial stream and saves one `.txt` file per posture session.

## Requirements

```bash
pip install pyserial matplotlib numpy
```

## Usage

```bash
python PPG_read.py
```

- Connect ESP32-S3 via USB; the script auto-detects the COM/tty port.
- A live waveform window shows real-time pressure and PPG signals.
- Press `S` to **start** recording, `E` to **end** and save the current posture file.
- Files are saved as `ppg_data_YYYYMMDD_HHMMSS.txt` in the current directory.

## Output File Format

35 comma-separated columns per row, no header:

```
<timestamp>,<ch0>,<ch1>,...,<ch31>,<ppg>,<esp32_us>
```

See [docs/data_description.md](../docs/data_description.md) for the full format specification.
