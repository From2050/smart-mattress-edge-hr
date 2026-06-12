"""Overview / hub landing page."""

import streamlit as st

import lib


def render():
    st.subheader('🛏️ Smart Mattress — Edge AI Heart-Rate Platform')
    st.markdown(
        'A complete, reproducible pipeline for **wearable-free heart-rate '
        'monitoring** from a low-cost 32-channel pressure mattress — from the '
        'data-collection firmware all the way to a 36.7 KB model running on an '
        'ESP32-S3. Every stage below is openable from the left sidebar.')

    st.divider()

    # ── The four-stage pipeline ──────────────────────────────────────────
    st.markdown('#### The pipeline')
    c1, c2, c3, c4 = st.columns(4)
    for col, (icon, title, body) in zip(
        (c1, c2, c3, c4),
        [('📡', 'Collect',
          'ESP32-S3 polls 32 FSR channels + a fingertip PPG reference and '
          'streams 35-column rows to the PC receiver. Tune channels & rate.'),
         ('🔬', 'Observe',
          'Load any recording and scrub/play through it. Watch pressure, the '
          'BCG heartbeat signal, and the per-channel AI trust scores live.'),
         ('🧠', 'Train',
          'Per-channel reliability CNN trained on the rule "FFT peak within '
          '±5 BPM of PPG and SNR > 3", validated with 33-fold LOSO.'),
         ('🚀', 'Deploy',
          'Export → Int8 ONNX → TFLite Micro. 16 K params, ~16 KB RAM, '
          '58 ms/window on a 240 MHz MCU.')]):
        with col:
            st.markdown(f"<div style='font-size:2rem'>{icon}</div>",
                        unsafe_allow_html=True)
            st.markdown(f'**{title}**')
            st.caption(body)

    st.divider()

    # ── Headline numbers ─────────────────────────────────────────────────
    st.markdown('#### At a glance')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('LOSO MAE (16 K model)', '7.81 BPM',
              help='33-fold leave-one-subject-out cross-validation.')
    m2.metric('Edge model size', f'{lib.EDGE_SIZE_KB} KB', help='Int8 ONNX.')
    m3.metric('Peak inference RAM', f'{lib.EDGE_RAM_KB} KB')
    m4.metric('Dataset', '33 subjects', help='5 postures each, ~270 MB.')

    st.divider()

    # ── How the method works (one paragraph, plain language) ─────────────
    st.markdown('#### Why this design')
    st.markdown(
        'Across channels the BCG signal is **mutually uninformative** — even '
        'after phase alignment, one sensor cannot help another. So instead of '
        'cross-channel fusion, each channel is judged **independently**: a tiny '
        'CNN scores how trustworthy each channel\'s 9.5 s window is, and the '
        'trusted channels then **vote** on the heart rate via their frequency '
        'spectra. A Viterbi pass smooths the result over time. This keeps the '
        'model small enough for an MCU while staying robust to posture.')

    st.info('**New here?** Open **🔬 Live Observation** in the sidebar to see a '
            'real recording play back, then **🧠 Training** and **🚀 Edge '
            'Deployment** for how the model is built and shipped.', icon='👈')

    st.caption('Model: PaperCNN_Reliability (16 K params, 11-layer All-Conv) · '
               'Post-processing: Top-3 Spectral Peak Voting + Viterbi HMM · '
               'Dataset: NYCU BCG 33-subject 2025 · MIT License')
