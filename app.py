# -*- coding: utf-8 -*-

import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

# =========================
# ASETUKSET
# =========================

st.set_page_config(
    page_title="TreidiMestari AI Cloud",
    page_icon="📈",
    layout="wide"
)

# =========================
# SUOMEN AIKA
# =========================

HELSINKI_TZ = ZoneInfo("Europe/Helsinki")

def now_fi():
    return datetime.now(HELSINKI_TZ)

# =========================
# SESSION STATE
# =========================

def init_state():
    defaults = {
        "logged_in": False,
        "ai_notes": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================
# LOGIN
# =========================

def login():
    st.title("🔐 TreidiMestari AI Cloud")

    password = st.text_input("Salasana", type="password")

    correct = st.secrets.get("APP_PASSWORD", "1234")

    if st.button("Kirjaudu sisään"):
        if password == correct:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Väärä salasana")

if not st.session_state.logged_in:
    login()
    st.stop()

# =========================
# PÄÄ SOVELLUS
# =========================

st.title("📈 TreidiMestari AI Cloud")

st.caption(f"Aika: {now_fi().strftime('%H:%M:%S')} (Suomi)")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ Asetukset")

symbol = st.sidebar.selectbox(
    "Valitse kohde",
    ["BTC", "ETH", "SOL", "XRP", "DOGE"]
)

tf = st.sidebar.selectbox(
    "Aikaväli",
    ["1m", "5m", "15m", "1h"]
)

st.sidebar.divider()

st.sidebar.subheader("🤖 ChatGPT")

api_key = st.sidebar.text_input("API-avain", type="password")

# =========================
# DEMO DATA
# =========================

import pandas as pd
import numpy as np

def demo_data():
    price = 50000
    rows = []

    for i in range(100):
        o = price
        price *= 1 + np.random.normal(0, 0.002)
        c = price
        h = max(o, c) * 1.002
        l = min(o, c) * 0.998

        rows.append([o, h, l, c])

    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])

df = demo_data()

# =========================
# YKSINKERTAINEN SIGNALI
# =========================

last = df.iloc[-1]
prev = df.iloc[-2]

if last["Close"] > prev["Close"]:
    signal = "OSTA"
    color = "green"
elif last["Close"] < prev["Close"]:
    signal = "MYY"
    color = "red"
else:
    signal = "ODOTA"
    color = "orange"

# =========================
# UI
# =========================

st.subheader("📊 Signaali")

st.markdown(
    f"<h1 style='color:{color}'>{signal}</h1>",
    unsafe_allow_html=True
)

# =========================
# AI MUISTI
# =========================

if st.button("Tallenna signaali muistiin"):
    st.session_state.ai_notes.append({
        "time": now_fi().strftime("%H:%M:%S"),
        "symbol": symbol,
        "signal": signal
    })

if st.session_state.ai_notes:
    st.subheader("🧠 AI Muisti")

    for note in st.session_state.ai_notes:
        st.write(note)
