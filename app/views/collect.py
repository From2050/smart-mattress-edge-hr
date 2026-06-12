"""Data Collection page — firmware wiring, an interactive config generator,
and the PC receiver workflow."""

import streamlit as st


WIRING = """ESP32-S3
  I2C1 (ADS1115 bus)  SCL=GPIO38  SDA=GPIO37  @ 400 kHz
    ├── ADS1115 #1  addr 0x48  ──┐
    └── ADS1115 #2  addr 0x49  ──┤  8 ADC ch each
        × 4 MUX settings (S0=GPIO35, S1=GPIO36) = 32 FSR channels

  I2C0 (PPG bus)      SCL=GPIO7   SDA=GPIO8
    └── MAX30105  (fingertip PPG — ground-truth reference only)

  UART0 (TX=GPIO43)   921600 baud  →  PC receiver"""


def render():
    st.subheader('📡 Data Collection')
    st.caption('The firmware streams raw sensor rows; the PC receiver timestamps '
               'and saves them. Adjust the basics below before flashing.')

    tab_fw, tab_cfg, tab_rx = st.tabs(
        ['Firmware & wiring', '⚙️ Config generator', 'PC receiver'])

    # ── Firmware & wiring ────────────────────────────────────────────────
    with tab_fw:
        st.markdown('**Hardware wiring**')
        st.code(WIRING, language='text')
        st.markdown('**Build & flash** (ESP-IDF v5.x)')
        st.code('cd firmware\n'
                'idf.py set-target esp32s3\n'
                'idf.py build\n'
                'idf.py flash monitor', language='bash')
        st.markdown('**Output row** — 35 comma-separated values, no header:')
        st.code('<timestamp>,<ch0>,<ch1>,…,<ch31>,<ppg>,<esp32_us>', language='text')
        st.caption('ch0–ch31: 16-bit FSR ADC (high = no pressure, low = pressure) · '
                   'ppg: MAX30105 green LED · esp32_us: device microsecond clock.')

    # ── Interactive config generator ─────────────────────────────────────
    with tab_cfg:
        st.markdown('Pick the channel layout and ADC timing; the resulting '
                    'sample rate and a ready-to-paste `#define` block update live.')

        c1, c2 = st.columns(2)
        with c1:
            n_ch = st.radio('Active channels', [32, 16], horizontal=True,
                            help='16ch firmware scans half the array → roughly '
                                 'double the sample rate (subject 224 variant).')
            per_ch_us = st.slider('Per-channel acquisition time (µs)',
                                  1000, 4000, 2280, step=20,
                                  help='ADS1115 conversion + MUX settle + I2C. '
                                       'Default firmware ≈ 2.28 ms/channel.')
        with c2:
            duration_min = st.slider('Recording length per posture (min)',
                                     1, 10, 5)
            st.write('')  # spacer

        cycle_us = n_ch * per_ch_us
        fs = 1e6 / cycle_us
        window_s = 128 / fs
        n_rows = int(fs * duration_min * 60)

        m1, m2, m3 = st.columns(3)
        m1.metric('Sample rate', f'{fs:.2f} Hz')
        m2.metric('128-sample window', f'{window_s:.2f} s')
        m3.metric('Rows / recording', f'{n_rows:,}')

        if abs(fs - 13.5) > 0.5 and n_ch == 32:
            st.warning('The shipped 16 K model was trained at **13.5 Hz** (32ch). '
                       'If you change the rate, retrain or resample so the window '
                       'period matches — see the **Training** page.', icon='⚠️')

        n_mux = n_ch // 8
        defines = (
            '// ── Generated sensor config ──────────────────────────\n'
            f'#define NUM_ADC_CHANNELS         8     // ADC ch per MUX setting\n'
            f'#define NUM_MUX_SETTINGS         {n_mux}     // 4-to-1 MUX settings\n'
            f'#define TOTAL_PRESSURE_CHANNELS  {n_ch}    // = NUM_ADC_CHANNELS * NUM_MUX_SETTINGS\n'
            f'#define PER_CHANNEL_ACQ_US       {per_ch_us}  // acquisition budget per channel\n'
            '// I2C / MUX / UART pins (unchanged):\n'
            '#define ADS1115_I2C_MASTER_SCL_IO  GPIO_NUM_38\n'
            '#define ADS1115_I2C_MASTER_SDA_IO  GPIO_NUM_37\n'
            '#define ADS1115_I2C_MASTER_FREQ_HZ 400000\n'
            '#define MUX_S0_PIN                 GPIO_NUM_35\n'
            '#define MUX_S1_PIN                 GPIO_NUM_36\n'
            f'// Resulting sample rate ≈ {fs:.2f} Hz  ·  window ≈ {window_s:.2f} s')
        st.markdown('**Generated `#define` block** — paste into '
                    '`firmware/main/hello_world_main.cpp`:')
        st.code(defines, language='cpp')
        st.download_button('⬇ Download sensor_config.h', defines,
                           file_name='sensor_config.h', mime='text/plain')

    # ── PC receiver ──────────────────────────────────────────────────────
    with tab_rx:
        st.markdown('**PC-side serial logger** — `receiver/PPG_read.py`')
        st.code('pip install pyserial matplotlib numpy\n'
                'python receiver/PPG_read.py', language='bash')
        st.markdown(
            '- Auto-detects the ESP32-S3 USB port and opens a live waveform window.\n'
            '- Press **S** to start a posture recording, **E** to end and save.\n'
            '- Saves `ppg_data_YYYYMMDD_HHMMSS.txt` (the exact format the rest of '
            'this platform reads).')
        st.info('Organize recordings as `data/ESP32_recored/<subject>/<posture>/` '
                'so the Observation and Training pages can find them.', icon='🗂️')
