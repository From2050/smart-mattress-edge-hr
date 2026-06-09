# Data

Raw sensor data is hosted on HuggingFace (too large for git):

**https://huggingface.co/datasets/m46012002/smart-mattress-bcg**

---

## Download & Setup

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='m46012002/smart-mattress-bcg',
    repo_type='dataset',
    local_dir='data',
)
"
```

After download, your layout should be:

```
data/
└── ESP32_recored/
    ├── 100/
    ├── 101/ … 110/   # Group 1: 2-min per posture
    ├── 201/ … 223/   # Group 2: 5-min per posture
    └── 224_16ch/     # Group 3: 16ch, 27 Hz
```

This is the path `pipeline/common.py` expects by default.

---

## Format

One `.txt` file per posture session, 35 comma-separated columns, no header:

```
timestamp, ch0–ch31 (FSR ADC), ppg (MAX30105), esp32_us
```

See [docs/data_description.md](../docs/data_description.md) for the full hardware and format specification.
