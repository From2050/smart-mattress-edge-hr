"""Live Observation page — the interactive BCG dashboard."""

import time
import numpy as np
import streamlit as st

import lib

# Human-readable posture labels
_POSTURE_LABEL = {
    'Front':     'Front (supine)',
    'Back':      'Back (prone)',
    'LeftSide':  'Left side',
    'RightSide': 'Right side',
    'Leave':     'On/off bed',
    'UFront':    'Front (post-exercise)',
    'UBack':     'Back (post-exercise)',
    'ULeftSide': 'Left side (post-exercise)',
    'URightSide':'Right side (post-exercise)',
}


def render():
    st.subheader('🔬 Live Observation')
    st.caption(
        'Scrub or play through a recorded session window by window. '
        'Watch the pressure distribution, the raw BCG heartbeat signal, '
        'and the AI reliability scores update in real time.')

    # ── Session picker ───────────────────────────────────────────────────
    subjects = lib.list_subjects()
    if not subjects:
        st.error(
            f'No data found in `{lib.DATA_DIR}`.\n\n'
            'Download the dataset first — see the **📡 Data Collection** page.')
        st.stop()

    csub, cpos, cinfo = st.columns([1, 1, 2])
    subject_dir = csub.selectbox('Subject', subjects,
                                 format_func=lambda x: f'Subject {x}')
    postures = lib.list_postures(subject_dir)
    posture  = cpos.selectbox(
        'Posture', postures,
        format_func=lambda p: _POSTURE_LABEL.get(p, p))

    # ── Load & infer ─────────────────────────────────────────────────────
    with st.spinner('Running CNN inference on all windows…'):
        result = lib.run_inference(subject_dir, posture)
    if result is None:
        st.error(f'No `.txt` data found for Subject {subject_dir} / {posture}.')
        st.stop()

    n_wins    = len(result['t_axis'])
    dur_s     = result['t_axis'][-1]
    win_s     = lib.WINDOW_PERIOD_S

    # ── Quick session info ───────────────────────────────────────────────
    errs_all = [abs(v - p) for v, p in
                zip(result['vit_hrs'], result['ppg_clean'])
                if 40 < v < 160 and 40 < p < 160]
    mae_str  = f'{np.mean(errs_all):.1f} BPM' if errs_all else '—'
    acc5_str = (f'{sum(e < 5 for e in errs_all)/len(errs_all)*100:.0f}%'
                if errs_all else '—')
    cinfo.markdown(
        f'**{n_wins} windows** · {dur_s:.0f} s · '
        f'{win_s:.1f} s / window &nbsp;|&nbsp; '
        f'Session MAE **{mae_str}** · Acc@5 **{acc5_str}**')

    st.divider()

    # ── Playback state ───────────────────────────────────────────────────
    # '_sl' is the keyed slider value. Only modify it BEFORE the slider
    # widget renders. '_adv' flags "increment on next rerun".
    sess_key = f'{subject_dir}_{posture}'
    if st.session_state.get('_sess') != sess_key:
        st.session_state['_sess']    = sess_key
        st.session_state['_sl']      = 0
        st.session_state['playing']  = False
        st.session_state['_adv']     = False

    if st.session_state.get('_adv', False):
        st.session_state['_adv'] = False
        if st.session_state.get('playing', False):
            nxt = st.session_state.get('_sl', 0) + 1
            if nxt >= n_wins:
                st.session_state['playing'] = False
            else:
                st.session_state['_sl'] = nxt

    def _prev():
        st.session_state['_sl']     = max(0, st.session_state.get('_sl', 0) - 1)
        st.session_state['playing'] = False
        st.session_state['_adv']    = False

    def _next():
        st.session_state['_sl']     = st.session_state.get('_sl', 0) + 1
        st.session_state['playing'] = False
        st.session_state['_adv']    = False

    def _toggle():
        st.session_state['playing'] = not st.session_state.get('playing', False)
        st.session_state['_adv']    = False

    cur = st.session_state.get('_sl', 0)
    c_prev, c_play, c_next, c_sl, c_t = st.columns([1, 1.4, 1, 14, 2])
    c_prev.button('⏮', on_click=_prev, disabled=cur == 0,          width='stretch')
    c_play.button('⏸ Pause' if st.session_state.get('playing')
                  else '▶ Play', on_click=_toggle,                  width='stretch')
    c_next.button('⏭', on_click=_next, disabled=cur >= n_wins - 1, width='stretch')

    with c_sl:
        current_win = st.slider(
            f'Window  (each = {win_s:.1f} s · session = {dur_s:.0f} s)',
            0, n_wins - 1, key='_sl')
    c_t.metric('Time', f'{result["t_axis"][current_win]:.1f} s')

    # ── Current-window data ──────────────────────────────────────────────
    s         = result['starts'][current_win]
    adc_win   = result['adc'][s:s + lib.WIN]
    weights   = result['weights_all'][current_win]
    rel_raw   = result['rel_all'][current_win]
    vit_hr    = result['vit_hrs'][current_win] if result['vit_hrs'] else 0.0
    ppg_hr    = result['ppg_hrs'][current_win]
    n_contrib = int((weights > lib.VOTE_FLOOR).sum())
    err       = abs(vit_hr - ppg_hr) if 40 < ppg_hr < 160 else None
    spectra   = result['spectra_all'][current_win]
    consensus = result['cons_all'][current_win]
    grid      = result['grid']

    # colour the window-error delta
    err_delta = None
    if err is not None:
        err_delta = '✓ within 5 BPM' if err < 5 else f'+{err - 5:.1f} over'

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric('Estimated HR',    f'{vit_hr:.0f} BPM')
    m2.metric('PPG reference',   f'{ppg_hr:.0f} BPM' if 40 < ppg_hr < 160 else '—')
    m3.metric('Window error',    f'{err:.1f} BPM' if err is not None else '—',
              delta=err_delta, delta_color='normal' if err_delta else 'off')
    m4.metric('Contributing ch', f'{n_contrib} / {lib.N_CH}',
              help=f'Channels with CNN score > {lib.VOTE_FLOOR} cast weighted votes '
                   'into the consensus spectrum.')
    m5.metric('CNN top score',   f'{rel_raw.max():.2f}')

    st.divider()

    # ── Row A: spatial heatmaps + HR time series ─────────────────────────
    col_a1, col_a2 = st.columns([1, 1.6])
    with col_a1:
        st.plotly_chart(lib.fig_heatmap(adc_win, weights),
                        width='stretch', key='heatmap')
    with col_a2:
        st.markdown('**Heart rate — full session**')
        st.plotly_chart(
            lib.fig_hr_timeseries(result['t_axis'], result['vit_hrs'],
                                  result['ppg_hrs'], current_win),
            width='stretch', key='hr_ts')

    st.divider()

    # ── Row B: BCG waveform + frequency spectrum ─────────────────────────
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown('**BCG waveform — top reliable channels**')
        st.caption('Bandpass 0.75–3 Hz · each oscillation ≈ one heartbeat · '
                   'traces normalized and offset for readability')
        st.plotly_chart(lib.fig_bcg_waveform(adc_win, weights),
                        width='stretch', key='bcg_wave')
    with col_b2:
        st.markdown('**Frequency spectrum — current window**')
        st.caption('Faint = top-3 channel spectra · '
                   'bold fill = weighted consensus · dashed = voted BPM')
        st.plotly_chart(lib.fig_spectrum(spectra, weights, grid, consensus),
                        width='stretch', key='spectrum')

    # ── Expanders ────────────────────────────────────────────────────────
    with st.expander('Per-channel CNN reliability scores'):
        st.caption(
            f'Green = score > {lib.VOTE_FLOOR} → contributes to voting. '
            'Grey = below voting floor, excluded. '
            'EMA-smoothed across windows (α = 0.8).')
        st.plotly_chart(lib.fig_reliability_bar(weights),
                        width='stretch', key='rel_bar')

    with st.expander('Session summary statistics'):
        valid_vit = [h for h in result['vit_hrs'] if 40 < h < 160]
        valid_ppg = [h for h in result['ppg_clean'] if 40 < h < 160]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Mean estimated HR',
                  f'{np.mean(valid_vit):.1f} BPM' if valid_vit else '—')
        c2.metric('Mean PPG HR',
                  f'{np.mean(valid_ppg):.1f} BPM' if valid_ppg else '—')
        c3.metric('Session MAE',
                  f'{np.mean(errs_all):.2f} BPM' if errs_all else '—')
        c4.metric('Acc@5 BPM',
                  f'{sum(e < 5 for e in errs_all)/len(errs_all)*100:.0f}%'
                  if errs_all else '—')

    # ── Auto-advance ─────────────────────────────────────────────────────
    # Sleep and rerun happen AFTER all charts render so each frame is visible.
    if st.session_state.get('playing', False):
        time.sleep(0.8)
        st.session_state['_adv'] = True
        st.rerun()
