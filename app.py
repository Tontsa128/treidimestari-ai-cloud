# -*- coding: utf-8 -*-

import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

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
# LOGIN
# =========================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 TreidiMestari AI Cloud")
    password = st.text_input("Salasana", type="password")

    if st.button("Kirjaudu"):
        if password == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Väärä salasana")

if not st.session_state.logged_in:
    login()
    st.stop()

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ Asetukset")

symbol = st.sidebar.selectbox(
    "Valitse kohde",
    {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "XRP": "XRPUSDT",
        "DOGE": "DOGEUSDT"
    }
)

tf = st.sidebar.selectbox(
    "Aikaväli",
    ["1m", "5m", "15m", "1h"]
)

limit = st.sidebar.slider("Kynttilät", 50, 300, 120)

# =========================
# DATA BINANCE
# =========================

def get_data(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "_1","_2","_3","_4","_5","_6"
    ])

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df

df = get_data(symbol, tf, limit)

# =========================
# INDIKAATTORIT
# =========================

df["EMA9"] = df["close"].ewm(span=9).mean()
df["EMA21"] = df["close"].ewm(span=21).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df["RSI"] = rsi(df["close"])

# =========================
# SIGNALI
# =========================

last = df.iloc[-1]

if last["EMA9"] > last["EMA21"] and last["RSI"] < 70:
    signal = "OSTA"
    color = "green"

elif last["EMA9"] < last["EMA21"] and last["RSI"] > 30:
    signal = "MYY"
    color = "red"

else:
    signal = "ODOTA"
    color = "orange"

# =========================
# UI
# =========================

st.title("📈 TreidiMestari AI Cloud")
st.caption(f"Aika: {now_fi().strftime('%H:%M:%S')}")

st.subheader("📊 Signaali")

st.markdown(
    f"<h1 style='color:{color}'>{signal}</h1>",
    unsafe_allow_html=True
)

# =========================
# KYNNTILÄKAAVIO
# =========================

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["time"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="Kynttilät"
))

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["EMA9"],
    line=dict(color="blue"),
    name="EMA9"
))

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["EMA21"],
    line=dict(color="red"),
    name="EMA21"
))

fig.update_layout(
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# RSI
# =========================

st.subheader("📉 RSI")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df["time"],
    y=df["RSI"],
    name="RSI"
))

fig2.add_hline(y=70)
fig2.add_hline(y=30)

fig2.update_layout(
    template="plotly_dark",
    height=300
)

st.plotly_chart(fig2, use_container_width=True)
