"""Training page — architecture, labelling rule, LOSO protocol, and a live
look at the reliability distribution the model produces on real data."""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

import lib


def render():
    st.subheader('🧠 Training the Reliability Model')
    st.caption('Instead of regressing heart rate end-to-end, the model learns '
               '*which channels to trust*. The trusted channels then vote.')

    tab_arch, tab_label, tab_run = st.tabs(
        ['Architecture', 'Labelling & LOSO', 'Run it / live check'])

    # ── Architecture ─────────────────────────────────────────────────────
    with tab_arch:
        c1, c2 = st.columns(2)
        c1.metric('Deployment model', '16,057 params',
                  help='PaperCNN_Reliability — 11-layer all-convolutional.')
        c2.metric('Offline model', '353,601 params',
                  help='NoAttentionCNN_Large — higher accuracy, not for MCU.')
        st.markdown(
            'Both take a single channel `(1, 128)` time-domain window and output '
            'one sigmoid reliability score in `[0, 1]`. The 32 channels share '
            'weights and are fully independent — no cross-channel layers, which '
            'is exactly what makes per-channel streaming inference on an MCU possible.')
        st.code(
            '(1,128) → Conv(1→8,  k7,s2) → BN → LReLU  (8,64)\n'
            '        → Conv(8→8,  k5,s1) → BN → LReLU  (8,64)\n'
            '        → Conv(8→16, k5,s2) → BN → LReLU  (16,32)\n'
            '        → Conv(16→16,k3,s1) → BN → LReLU  (16,32)\n'
            '        → Conv(16→16,k3,s2) → BN → LReLU  (16,16)\n'
            '        → Conv(16→32,k3,s1) → BN → LReLU  (32,16)\n'
            '        → Conv(32→32,k3,s2) → BN → LReLU  (32,8)\n'
            '        → Conv(32→32,k3,s2) → BN → LReLU  (32,4)\n'
            '        → Conv(32→32,k3,s2) → BN → LReLU  (32,2)\n'
            '        → Conv(32→32,k2,s2) → BN → LReLU  (32,1)\n'
            '        → Conv(32→1,  k1)   → sigmoid      (1,)', language='text')
        st.caption('All downsampling uses stride (no pooling, no FC) → fully '
                   'supported by TFLite Micro with no operator decomposition.')

        st.markdown('**Accuracy (33-fold LOSO)**')
        st.table({
            'Metric':           ['MAE (BPM)', 'Acc@5 (%)'],
            'Offline (353 K)':  ['7.58', '66.7'],
            'Deployment (16 K)':['7.81', '63.3'],
        })

    # ── Labelling & LOSO ─────────────────────────────────────────────────
    with tab_label:
        st.markdown('#### How a channel gets its training label')
        st.markdown(
            'For each 128-sample window, every channel\'s BCG FFT peak is compared '
            'against the PPG ground-truth HR. A channel is labelled **reliable** if:')
        st.latex(r'|\text{BPM}_{\text{ch}} - \text{BPM}_{\text{PPG}}| < 5'
                 r'\quad\textbf{and}\quad \text{SNR} > 3')
        st.caption('Only **13.1%** of all 296,448 channel-windows are reliable — '
                   'a heavily imbalanced target the CNN must learn.')

        st.markdown('**Label simulator** — try the rule yourself:')
        c1, c2, c3 = st.columns(3)
        bpm_ch  = c1.number_input('Channel FFT peak (BPM)', 30, 200, 74)
        bpm_ppg = c2.number_input('PPG reference (BPM)', 30, 200, 76)
        snr     = c3.number_input('Channel FFT SNR', 0.0, 20.0, 4.5, step=0.5)
        ok = abs(bpm_ch - bpm_ppg) < 5 and snr > 3
        err = abs(bpm_ch - bpm_ppg)
        if ok:
            st.success(f'**RELIABLE** — error {err} BPM (< 5) and SNR {snr} (> 3). '
                       'This channel votes.', icon='✅')
        else:
            reasons = []
            if err >= 5: reasons.append(f'error {err} BPM ≥ 5')
            if snr <= 3: reasons.append(f'SNR {snr} ≤ 3')
            st.error(f'**UNRELIABLE** — {", ".join(reasons)}. Excluded.', icon='🚫')

        st.divider()
        st.markdown('#### LOSO — leave-one-subject-out')
        st.markdown(
            'With 33 subjects, the model trains on 32 and tests on the held-out '
            'one, rotated 33 times. No subject ever appears in both train and test, '
            'so the reported MAE reflects performance on a **genuinely new person** '
            '— the realistic deployment scenario.')

    # ── Run it / live check ──────────────────────────────────────────────
    with tab_run:
        st.markdown('**Reproduce the training pipeline**')
        st.code('cd pipeline\n'
                'pip install -r requirements.txt\n'
                'python run_exp1_main.py        # main model, 33-fold LOSO\n'
                'python run_exp5_lightweight.py # deployment model comparison\n'
                'python run_edge_deploy.py      # ONNX + Int8 export', language='bash')
        st.caption('Outputs land in `pipeline/outputs/`. Edit `DATA_ROOT` in '
                   '`pipeline/common.py` if your data lives elsewhere.')

        st.divider()
        st.markdown('**Live check** — what the trained model outputs on a real '
                    'session (using the shipped 16 K weights):')
        subjects = lib.list_subjects()
        if not subjects:
            st.info('Download the dataset to enable this live check.')
            return
        c1, c2 = st.columns(2)
        subj = c1.selectbox('Subject', subjects, format_func=lambda x: f'Subject {x}',
                            key='train_subj')
        post = c2.selectbox('Posture', lib.list_postures(subj), key='train_post')

        with st.spinner('Scoring all channels across all windows…'):
            result = lib.run_inference(subj, post)
        if result is None:
            st.warning('No data for that selection.')
            return

        scores = np.concatenate([w for w in result['weights_all']])
        contrib_per_win = [int((w > lib.VOTE_FLOOR).sum())
                           for w in result['weights_all']]

        m1, m2, m3 = st.columns(3)
        m1.metric('Reliable share (score > 0.5)', f'{(scores > 0.5).mean()*100:.1f}%')
        m2.metric('Contributing share (> 0.01)', f'{(scores > lib.VOTE_FLOOR).mean()*100:.1f}%')
        m3.metric('Median channels voting / window',
                  f'{int(np.median(contrib_per_win))} / {lib.N_CH}')

        fig = go.Figure(go.Histogram(x=scores, nbinsx=40,
                                     marker_color=lib.CLR_PRIMARY))
        fig.add_vline(x=lib.VOTE_FLOOR, line_dash='dot', line_color='#555',
                      annotation_text=' voting floor')
        fig.update_layout(**lib._LAYOUT_BASE, height=280,
                          xaxis_title='CNN reliability score',
                          yaxis_title='Channel-windows',
                          title='Score distribution for this session')
        st.plotly_chart(fig, width='stretch')
        st.caption('The heavy mass near zero is the 13% / 87% imbalance the model '
                   'learned: most channels, most of the time, carry no usable BCG.')
