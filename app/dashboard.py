#!/usr/bin/env python3
"""
Smart Mattress BCG Platform — Edge AI Heart-Rate Monitoring (NYCU 2026)

A single entry point that ties the whole pipeline together:
  Collect → Observe → Train → Deploy

Run with:  streamlit run app/dashboard.py
"""

import os
import sys

import streamlit as st

# Make `import lib` and `from views import ...` work regardless of CWD.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from views import overview, collect, observe, train, deploy   # noqa: E402

st.set_page_config(
    page_title='Smart Mattress BCG Platform',
    page_icon='🛏️',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #666; }
div.stPlotlyChart { border-radius: 8px; }
section[data-testid="stSidebar"] h1 { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### 🛏️ Smart Mattress')
    st.caption('Edge AI Heart-Rate Platform · NYCU 2026')

nav = st.navigation([
    st.Page(overview.render, title='Overview',        icon='🏠', url_path='overview', default=True),
    st.Page(collect.render,  title='Data Collection', icon='📡', url_path='collect'),
    st.Page(observe.render,  title='Live Observation', icon='🔬', url_path='observe'),
    st.Page(train.render,    title='Training',         icon='🧠', url_path='train'),
    st.Page(deploy.render,   title='Edge Deployment',  icon='🚀', url_path='deploy'),
])

with st.sidebar:
    st.divider()
    st.markdown('**⚡ Edge specs**')
    st.caption('16 K params · 36.7 KB Int8 · ~16 KB RAM · 58 ms/window\n\n'
               'Target: ESP32-S3 · 240 MHz')
    st.divider()
    st.caption('[GitHub](https://github.com/m46012002/smart-mattress-edge-hr) · '
               'MIT License')

nav.run()
