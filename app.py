# -*- coding: utf-8 -*-

import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from openai import OpenAI

# =========================
# SIVUN ASETUKSET
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

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "ai_memory" not in st.session_state:
    st.session_state.ai_memory = []

# =========================
# LOGIN
# =========================

def login():
    st.title("🔐 TreidiMestari AI Cloud")
    st.write("Kirjaudu sisään jatkaaksesi.")

    password = st.text_input("Salasana", type="password")

    correct_password = st.secrets.get("APP_PASSWORD", "1234")

    if st.button("Kirjaudu"):
        if password == correct_password:
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

symbols = {
    "BTC / USDT": "BTCUSDT",
    "ETH / USDT": "ETHUSDT",
    "SOL / USDT": "SOLUSDT",
    "XRP / USDT": "XRPUSDT",
    "DOGE / USDT": "DOGEUSDT",
    "BNB / USDT": "BNBUSDT"
}

selected_name = st.sidebar.selectbox("Valitse kohde", list(symbols.keys()))
symbol = symbols[selected_name]

tf = st.sidebar.selectbox("Aikaväli", ["1m", "3m", "5m", "15m", "1h"], index=0)
limit = st.sidebar.slider("Kynttilöitä", 80, 300, 160)

st.sidebar.divider()
st.sidebar.subheader("🤖 ChatGPT")

chatgpt_on = st.sidebar.toggle("ChatGPT-analyysi päällä", value=True)
api_key_input = st.sidebar.text_input("OpenAI API-avain", type="password")

# =========================
# DATA BINANCE
# =========================

@st.cache_data(ttl=3, show_spinner=False)
def get_data(symbol_code, interval, limit_count):
    url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": symbol_code,
        "interval": interval,
        "limit": limit_count
    }

    r = requests.get(url, params=params, timeout=10)

    if r.status_code != 200:
        raise RuntimeError(f"Binance virhe: {r.status_code}")

    data = r.json()

    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "_1", "_2", "_3", "_4", "_5", "_6"
    ])

    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["time"] = df["time"].dt.tz_convert("Europe/Helsinki")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df

# =========================
# INDIKAATTORIT
# =========================

def add_indicators(df):
    df = df.copy()

    df["EMA9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA100"] = df["close"].ewm(span=100, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    df["VOL_MA"] = df["volume"].rolling(20).mean().fillna(df["volume"])

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()

    df["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df["ATR"] = df["ATR"].fillna(tr1.mean())

    return df

# =========================
# SIGNAALILOGIIKKA
# =========================

def calculate_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    reasons = []

    price = last["close"]

    if last["EMA9"] > last["EMA21"]:
        score += 15
        reasons.append("EMA9 on EMA21 yläpuolella.")

    if last["EMA9"] < last["EMA21"]:
        score -= 15
        reasons.append("EMA9 on EMA21 alapuolella.")

    if price > last["EMA50"]:
        score += 10
        reasons.append("Hinta on EMA50 yläpuolella.")

    if price < last["EMA50"]:
        score -= 10
        reasons.append("Hinta on EMA50 alapuolella.")

    if last["EMA50"] > last["EMA100"]:
        score += 10
        reasons.append("EMA50 on EMA100 yläpuolella.")

    if last["EMA50"] < last["EMA100"]:
        score -= 10
        reasons.append("EMA50 on EMA100 alapuolella.")

    if last["RSI"] < 30:
        score += 12
        reasons.append("RSI on ylimyyty.")

    elif last["RSI"] > 70:
        score -= 12
        reasons.append("RSI on yliostettu.")

    else:
        reasons.append(f"RSI on neutraali: {last['RSI']:.1f}.")

    if last["MACD"] > last["MACD_SIGNAL"]:
        score += 12
        reasons.append("MACD on noususuuntainen.")

    if last["MACD"] < last["MACD_SIGNAL"]:
        score -= 12
        reasons.append("MACD on laskusuuntainen.")

    if last["volume"] > last["VOL_MA"] * 1.2:
        reasons.append("Volyymi on normaalia suurempi.")
        if last["close"] > last["open"]:
            score += 8
        else:
            score -= 8

    body = abs(last["close"] - last["open"])
    candle_range = max(last["high"] - last["low"], 0.0000001)

    body_ratio = body / candle_range

    if last["close"] > last["open"] and body_ratio > 0.55:
        score += 10
        reasons.append("Viimeinen kynttilä on vahva vihreä.")

    if last["close"] < last["open"] and body_ratio > 0.55:
        score -= 10
        reasons.append("Viimeinen kynttilä on vahva punainen.")

    if score >= 45:
        signal = "VAHVA OSTA"
        color = "#22c55e"
    elif score >= 25:
        signal = "OSTA"
        color = "#86efac"
    elif score <= -45:
        signal = "VAHVA MYY"
        color = "#ef4444"
    elif score <= -25:
        signal = "MYY"
        color = "#fca5a5"
    else:
        signal = "ODOTA"
        color = "#facc15"

    confidence = int(min(95, max(45, 50 + abs(score) * 0.7)))

    atr = max(last["ATR"], price * 0.001)

    if "OSTA" in signal:
        stop = price - atr * 1.4
        target = price + atr * 2.2
    elif "MYY" in signal:
        stop = price + atr * 1.4
        target = price - atr * 2.2
    else:
        stop = None
        target = None

    return {
        "signal": signal,
        "score": int(score),
        "confidence": confidence,
        "color": color,
        "price": price,
        "stop": stop,
        "target": target,
        "reasons": reasons
    }

# =========================
# CHATGPT ANALYYSI
# =========================

def chatgpt_analysis(df, ai):
    if not chatgpt_on:
        return "ChatGPT-analyysi ei ole päällä."

    api_key = api_key_input or st.secrets.get("OPENAI_API_KEY", "")

    if not api_key:
        return "Lisää OpenAI API-avain sivupalkkiin tai secrets.toml-tiedostoon."

    try:
        client = OpenAI(api_key=api_key)

        last = df.iloc[-1]

        prompt = f"""
Olet selkeä treidausopettaja. Tämä on opetustyökalu, ei sijoitusneuvo.

Analysoi tilanne suomeksi lyhyesti.

Kohde: {selected_name}
Aikaväli: {tf}
Hinta: {ai['price']}
Signaali: {ai['signal']}
AI-score: {ai['score']}
Varmuus: {ai['confidence']} %
RSI: {last['RSI']:.2f}
MACD: {last['MACD']:.6f}
MACD signal: {last['MACD_SIGNAL']:.6f}
EMA9: {last['EMA9']:.6f}
EMA21: {last['EMA21']:.6f}
EMA50: {last['EMA50']:.6f}
EMA100: {last['EMA100']:.6f}

Selitä:
1. Mitä kaaviossa tapahtuu nyt
2. Miksi signaali on tämä
3. Mitä aloittelijan pitää varoa
4. Mikä voisi vahvistaa signaalin
5. Mikä voisi kumota signaalin

Vastaa napakasti ja käytännöllisesti.
"""

        response = client.responses.create(
            model="gpt-5.5",
            input=prompt
        )

        return response.output_text

    except Exception as e:
        return f"ChatGPT virhe: {e}"

# =========================
# UI
# =========================

st.title("📈 TreidiMestari AI Cloud")
st.caption(f"Suomen aika: {now_fi().strftime('%H:%M:%S')}")

try:
    df = get_data(symbol, tf, limit)
    df = add_indicators(df)
    ai = calculate_signal(df)

except Exception as e:
    st.error(f"Dataa ei saatu: {e}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Kohde", selected_name)
m2.metric("Hinta", f"{ai['price']:,.4f}")
m3.metric("AI-score", f"{ai['score']} / 100")
m4.metric("Varmuus", f"{ai['confidence']} %")

st.markdown(
    f"""
    <div style="
        background:{ai['color']};
        padding:25px;
        border-radius:20px;
        color:#111;
        text-align:center;
        font-size:36px;
        font-weight:900;
        margin-top:15px;
        margin-bottom:20px;">
        {ai['signal']}<br>
        {ai['confidence']} %
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# KAAVIO
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
    name="EMA9",
    mode="lines"
))

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["EMA21"],
    name="EMA21",
    mode="lines"
))

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["EMA50"],
    name="EMA50",
    mode="lines"
))

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["EMA100"],
    name="EMA100",
    mode="lines"
))

if ai["stop"] is not None:
    fig.add_hline(y=ai["stop"], line_dash="dash")
    fig.add_hline(y=ai["target"], line_dash="dash")

fig.update_layout(
    title=f"{selected_name} — {tf}",
    template="plotly_dark",
    height=650,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# LISÄKAAVIOT
# =========================

left, right = st.columns(2)

with left:
    st.subheader("📉 RSI")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["RSI"],
        name="RSI"
    ))
    rsi_fig.add_hline(y=70, line_dash="dot")
    rsi_fig.add_hline(y=30, line_dash="dot")
    rsi_fig.update_layout(template="plotly_dark", height=300)

    st.plotly_chart(rsi_fig, use_container_width=True)

with right:
    st.subheader("📊 MACD")

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["MACD"],
        name="MACD"
    ))
    macd_fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["MACD_SIGNAL"],
        name="Signal"
    ))
    macd_fig.add_trace(go.Bar(
        x=df["time"],
        y=df["MACD_HIST"],
        name="Histogram"
    ))
    macd_fig.update_layout(template="plotly_dark", height=300)

    st.plotly_chart(macd_fig, use_container_width=True)

# =========================
# STOP / TARGET
# =========================

st.subheader("🎯 Entry / Stop / Target")

if ai["stop"] is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Entry", f"{ai['price']:,.4f}")
    c2.metric("Stop", f"{ai['stop']:,.4f}")
    c3.metric("Target", f"{ai['target']:,.4f}")
else:
    st.info("Ei vielä selkeää entryä. Odota vahvempaa signaalia.")

# =========================
# MIKSI SIGNAAALI
# =========================

st.subheader("🧠 Miksi signaali on tämä?")

for reason in ai["reasons"]:
    st.write("• " + reason)

# =========================
# CHATGPT
# =========================

st.subheader("🤖 ChatGPT-treidiopettaja")

if st.button("Analysoi ChatGPT:llä"):
    result = chatgpt_analysis(df, ai)

    st.session_state.ai_memory.append({
        "time": now_fi().strftime("%H:%M:%S"),
        "symbol": selected_name,
        "tf": tf,
        "signal": ai["signal"],
        "score": ai["score"],
        "analysis": result
    })

    st.write(result)

# =========================
# AI MUISTI
# =========================

with st.expander("🧠 AI-muisti"):
    if st.session_state.ai_memory:
        for item in st.session_state.ai_memory[-5:]:
            st.markdown(
                f"**{item['time']} — {item['symbol']} — {item['tf']} — {item['signal']} — score {item['score']}**"
            )
            st.write(item["analysis"])
    else:
        st.info("Ei vielä tallennettuja AI-analyysejä.")

st.warning("Opetustyökalu. Tämä ei ole sijoitusneuvo eikä tee oikeita kauppoja.")
