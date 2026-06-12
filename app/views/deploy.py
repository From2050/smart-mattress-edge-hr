"""Edge Deployment page — export path, real on-disk model sizes, and the
MCU resource budget."""

import os
import streamlit as st
import plotly.graph_objects as go

import lib


def _kb(path):
    try:
        return os.path.getsize(path) / 1024
    except OSError:
        return None


def render():
    st.subheader('🚀 Edge Deployment')
    st.caption('From a PyTorch checkpoint to a quantized model that fits in '
               '~16 KB of RAM on a $5 microcontroller.')

    tab_path, tab_budget, tab_run = st.tabs(
        ['Export path', 'Resource budget', 'Run inference'])

    # ── Export path ──────────────────────────────────────────────────────
    with tab_path:
        st.code(
            'papercnn_16k.pth          (FP32 PyTorch)\n'
            '   │  torch.onnx.export (opset 17)\n'
            'papercnn_16k.onnx         (FP32 ONNX)\n'
            '   │  onnxruntime quantize_dynamic\n'
            'papercnn_16k_int8.onnx    (Int8 ONNX  ← edge target)\n'
            '   │  onnx2tf → tflite_converter\n'
            'model.tflite  (~37 KB)\n'
            '   │  xxd -i\n'
            'model_data.h  → embed in ESP32-S3 Flash, run with TFLite Micro',
            language='text')

        st.markdown('**Actual artifact sizes on disk** (`deploy/weights/`):')
        files = [
            ('papercnn_16k.pth',       'FP32 PyTorch state dict'),
            ('papercnn_16k.onnx',      'FP32 ONNX'),
            ('papercnn_16k_int8.onnx', 'Int8 ONNX (edge target)'),
            ('main_353k.pth',          'Offline model (not for MCU)'),
        ]
        rows_name, rows_desc, rows_size = [], [], []
        for fn, desc in files:
            kb = _kb(os.path.join(lib.WEIGHTS_DIR, fn))
            rows_name.append(fn)
            rows_desc.append(desc)
            rows_size.append(f'{kb:.1f} KB' if kb is not None else 'missing')
        st.table({'File': rows_name, 'Description': rows_desc, 'Size': rows_size})
        st.caption('Int8 quantization shrinks the model ~2× versus FP32 ONNX with '
                   'a mean output difference of only 0.0026 — negligible for the '
                   'voting stage.')
        st.warning('TFLite Micro firmware integration is not yet complete; the '
                   'Int8 ONNX model runs correctly on PC with ONNX Runtime today.',
                   icon='🚧')

    # ── Resource budget ──────────────────────────────────────────────────
    with tab_budget:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Int8 model', f'{lib.EDGE_SIZE_KB} KB')
        m2.metric('Peak RAM', f'{lib.EDGE_RAM_KB} KB',
                  help='512 B activation + ~16 KB Int8 weights, streamed per channel.')
        m3.metric('Latency / channel', f'{lib.EDGE_LATENCY_MS} ms')
        m4.metric('Window period', f'{lib.WINDOW_PERIOD_S:.2f} s')

        st.markdown('**Timing headroom** — all 32 channels must finish within one '
                    'window period:')
        total_ms = lib.EDGE_LATENCY_MS * lib.N_CH
        budget_ms = lib.WINDOW_PERIOD_S * 1000
        fig = go.Figure()
        fig.add_trace(go.Bar(y=['compute'], x=[total_ms], orientation='h',
                             marker_color=lib.CLR_PRIMARY,
                             name=f'32 ch × {lib.EDGE_LATENCY_MS} ms = {total_ms:.0f} ms'))
        fig.add_vline(x=budget_ms, line_dash='dash', line_color=lib.CLR_ACCENT,
                      annotation_text=f' window budget {budget_ms:.0f} ms')
        fig.update_layout(**lib._LAYOUT_BASE, height=160,
                          xaxis_title='milliseconds', showlegend=True,
                          legend=dict(orientation='h', y=1.4))
        st.plotly_chart(fig, width='stretch')
        st.success(f'Worst-case compute is {total_ms:.0f} ms — about '
                   f'**{budget_ms/total_ms:.0f}×** under the {budget_ms:.0f} ms '
                   'window budget. Plenty of margin for the fusion + Viterbi C code.',
                   icon='✅')

        st.markdown('**Target platform**')
        st.table({
            'Item': ['MCU', 'Clock', 'SRAM', 'Flash', 'Framework'],
            'Spec': ['ESP32-S3 (Xtensa LX7)', '240 MHz', '512 KB', '4–16 MB',
                     'TFLite Micro / ESP-NN'],
        })

    # ── Run inference ────────────────────────────────────────────────────
    with tab_run:
        st.markdown('**ONNX Runtime inference (PC)** — same Int8 model as the MCU:')
        st.code(
            "import onnxruntime as ort\n"
            "import numpy as np\n\n"
            "sess = ort.InferenceSession('deploy/weights/papercnn_16k_int8.onnx')\n"
            "# one channel at a time: (1, 128) float32\n"
            "x = np.random.randn(1, 128).astype(np.float32)\n"
            "reliability = sess.run(None, {'input': x})[0]   # (1,) in [0,1]\n"
            "# repeat for all 32 channels → feed scores to the SpectrumFuser",
            language='python')
        st.markdown('**Streaming strategy on the MCU**')
        st.code(
            'for ch in 0..31:\n'
            '    reliability[ch] = AllConv_11layer(adc_window[ch])  # 512 B peak\n'
            '→ SpectrumFuser + Viterbi  (~4 KB extra RAM, C implementation)',
            language='text')
        st.caption('No Transformer, no FC, no cross-channel interaction — so '
                   'channels stream one by one and only 512 B of activation memory '
                   'is ever live.')
