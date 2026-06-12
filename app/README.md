# Smart Mattress BCG Platform (Streamlit)

A single app that ties the whole pipeline together — **Collect → Observe →
Train → Deploy** — so a researcher, clinician, or agent can understand and
operate the system without reading the source.

## Quick Start

```bash
# From project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt

# Make sure data is downloaded first (see data/README.md)

streamlit run app/dashboard.py
```

Then open http://localhost:8501.

## Pages

| Page | What it does |
|------|--------------|
| 🏠 **Overview** | The four-stage pipeline, headline numbers, and the design rationale. |
| 📡 **Data Collection** | Firmware wiring, an interactive `#define` config generator (channels / sample rate), and the PC receiver workflow. |
| 🔬 **Live Observation** | Scrub or **play** through a recording: pressure map, BCG amplitude map, the heartbeat waveform of the most reliable channels, per-channel AI trust scores, the voted spectrum, and full-session HR vs PPG. |
| 🧠 **Training** | Model architecture, the reliable-channel labelling rule (with a live simulator), LOSO protocol, reproduction commands, and a live look at the score distribution on real data. |
| 🚀 **Edge Deployment** | The PyTorch → ONNX → Int8 → TFLite path, real on-disk artifact sizes, and the MCU timing/RAM budget. |

## Code layout

```
app/
├── dashboard.py      # entry point: st.navigation wiring + page config
├── lib.py            # model, signal processing, cached loaders, figure builders
└── views/
    ├── overview.py
    ├── collect.py
    ├── observe.py    # the interactive dashboard (play/scrub)
    ├── train.py
    └── deploy.py
```

## Notes

- The shipped 16 K model and signal constants assume the **32-channel @ 13.5 Hz**
  layout, so the 16-channel variant (`224_16ch`) is hidden from the subject list.
- The fusion algorithm uses **weight > 0.01** as the voting cutoff (not an
  arbitrary threshold) — this is shown faithfully in the Observation page.
