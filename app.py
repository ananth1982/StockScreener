"""
Intraday Momentum Screener — deployable version
Free hosting: push this to a public GitHub repo, then deploy at
https://share.streamlit.io (Streamlit Community Cloud, free tier).

Telegram alerts (free, no sandbox limits):
    1. Message @BotFather on Telegram, send /newbot, follow prompts -> get a bot token
    2. Message your new bot anything once (so it can message you back)
    3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates -> find your "chat":{"id": ...}
    4. Paste the token and chat id into the sidebar below, or set them as
       Streamlit secrets (Settings -> Secrets on share.streamlit.io):
           TELEGRAM_BOT_TOKEN = "123456:ABC-your-token"
           TELEGRAM_CHAT_ID = "987654321"

Local run:
    pip install streamlit yfinance pandas numpy requests
    streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

st.set_page_config(page_title="Ticker Watch", layout="wide")
# requirements.txt needs: streamlit, yfinance, pandas, numpy, requests

# --- Watchlist ---
# Preferred: set in Streamlit Cloud under Settings -> Secrets:
#     DEFAULT_TICKERS = "RXT, EVTL, HUT, HOOD, MRVL, AMD, HIMS, ACHV, LYFT, APLD, RGTI, KEY, ASTS, WRBY, AMPX, MP, APH, UBER, BBAI, NVDA"
# Also works as a local env var (e.g. in a .env file or shell export) of the same name.
# Falls back to the hardcoded list below if neither is set.
_FALLBACK_TICKERS = ["RXT", "EVTL", "HUT", "HOOD", "MRVL", "AMD", "HIMS", "ACHV",
                      "LYFT", "APLD", "RGTI", "KEY", "ASTS", "WRBY", "AMPX",
                      "MP", "APH", "UBER", "BBAI", "NVDA"]
_tickers_raw = st.secrets.get("DEFAULT_TICKERS") or os.environ.get("DEFAULT_TICKERS")
DEFAULT_TICKERS = (
    [t.strip().upper() for t in _tickers_raw.split(",") if t.strip()]
    if _tickers_raw else _FALLBACK_TICKERS
)

st.title("Intraday Momentum Screener")

with st.sidebar:
    st.header("Filters")
    tickers_input = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS))
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    min_rvol = st.number_input("Min RVOL", value=0.5, step=0.1)
    max_price = st.number_input("Max price ($)", value=100.0, step=1.0)
    max_float_m = st.number_input("Max float (millions, 0 = no cap)", value=0.0, step=10.0)
    alert_rvol = st.number_input("Alert RVOL threshold", value=1.0, step=0.5)
    rising_only = st.checkbox("Only show volume rising every 30min", value=False)
    refresh = st.button("Refresh data")

    st.divider()
    st.header("Telegram alerts")
    bot_token = st.text_input("Bot token", value=st.secrets.get("TELEGRAM_BOT_TOKEN", ""), type="password")
    chat_id = st.text_input("Chat ID", value=st.secrets.get("TELEGRAM_CHAT_ID", ""))

    st.divider()
    st.header("WhatsApp alerts (Twilio sandbox)")
    st.caption("Join your Twilio sandbox first by messaging the join code to the sandbox number from WhatsApp.")
    twilio_sid = st.text_input("Twilio Account SID", value=st.secrets.get("TWILIO_ACCOUNT_SID", ""))
    twilio_auth = st.text_input("Twilio Auth Token", value=st.secrets.get("TWILIO_AUTH_TOKEN", ""), type="password")
    twilio_from = st.text_input("Twilio WhatsApp number", value=st.secrets.get("TWILIO_FROM", "whatsapp:+14155238886"))
    twilio_to = st.text_input("Your WhatsApp number", value=st.secrets.get("TWILIO_TO", ""), placeholder="whatsapp:+919876543210")

    alerts_on = st.checkbox("Send alerts on this refresh", value=False)


def send_telegram(token, chat_id, text):
    if not token or not chat_id:
        return False, "Missing bot token or chat ID"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        return r.ok, r.text
    except Exception as e:
        return False, str(e)


def send_whatsapp(account_sid, auth_token, from_number, to_number, text):
    if not account_sid or not auth_token or not to_number:
        return False, "Missing Twilio SID, auth token, or destination number"
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
            auth=(account_sid, auth_token),
            data={"From": from_number, "To": to_number, "Body": text},
            timeout=10,
        )
        return r.ok, r.text
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=60)
def fetch(ticker):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="20d", interval="1d")
        intraday = tk.history(period="1d", interval="1m")
    except Exception as e:
        return {"Ticker": ticker, "_error": f"fetch failed: {e}"}

    if hist.empty:
        return {"Ticker": ticker, "_error": "no daily history (bad ticker or Yahoo blocked the request)"}
    if intraday.empty:
        return {"Ticker": ticker, "_error": "no intraday 1m data (market closed, or too far outside trading hours)"}

    avg_vol_20d = hist["Volume"][:-1].mean()
    today_vol = intraday["Volume"].sum()
    rvol = today_vol / avg_vol_20d if avg_vol_20d else np.nan

    px = intraday["Close"].iloc[-1]
    prev_close = hist["Close"].iloc[-2]
    chg_pct = (px - prev_close) / prev_close * 100

    typical_price = (intraday["High"] + intraday["Low"] + intraday["Close"]) / 3
    vwap = (typical_price * intraday["Volume"]).cumsum() / intraday["Volume"].cumsum()
    vwap_last = vwap.iloc[-1]

    close_daily = hist["Close"]
    ema9 = close_daily.ewm(span=9).mean().iloc[-1]
    ema20 = close_daily.ewm(span=20).mean().iloc[-1]

    try:
        float_shares = tk.get_info().get("floatShares")
        float_m = round(float_shares / 1e6, 1) if float_shares else np.nan
    except Exception:
        float_m = np.nan

    # 30-minute volume trend: resample today's 1-min bars into 30-min buckets
    # and check whether volume has risen bucket-over-bucket (accumulation pattern).
    vol_30m = intraday["Volume"].resample("30min").sum()
    vol_30m = vol_30m[vol_30m > 0]  # drop empty trailing/leading buckets
    increasing = bool(len(vol_30m) >= 3 and vol_30m.is_monotonic_increasing)
    streak = 0
    for i in range(len(vol_30m) - 1, 0, -1):
        if vol_30m.iloc[i] > vol_30m.iloc[i - 1]:
            streak += 1
        else:
            break

    return {
        "Ticker": ticker, "Price": round(px, 2), "Chg %": round(chg_pct, 2),
        "RVOL": round(rvol, 2), "VWAP": round(vwap_last, 2),
        "vs VWAP %": round((px - vwap_last) / vwap_last * 100, 2),
        "EMA9": round(ema9, 2), "EMA20": round(ema20, 2),
        "Trend": "up" if ema9 > ema20 else "down",
        "Volume": int(today_vol), "Float (M)": float_m,
        "Vol Rising (30m)": increasing, "Rising Streak": streak,
    }


rows = [fetch(t) for t in tickers]
errors = [r for r in rows if r and "_error" in r]
good_rows = [r for r in rows if r and "_error" not in r]
df = pd.DataFrame(good_rows)

if errors:
    with st.expander(f"⚠️ {len(errors)} ticker(s) returned no data — click for details", expanded=True):
        for e in errors:
            st.write(f"**{e['Ticker']}**: {e['_error']}")

if not df.empty:
    df = df[(df["RVOL"] >= min_rvol) & (df["Price"] <= max_price)]
    if max_float_m > 0:
        df = df[df["Float (M)"].fillna(np.inf) <= max_float_m]
    if rising_only:
        df = df[df["Vol Rising (30m)"]]
    df["Alert"] = np.where(df["RVOL"] >= alert_rvol, "⚠️ RVOL", "")
    df = df.sort_values("RVOL", ascending=False)

st.caption(f"{len(df)} of {len(tickers)} tickers match filters · data cached 60s")

alerts = df[df["Alert"] != ""] if not df.empty else df
if not alerts.empty:
    st.warning(f"RVOL alert ({alert_rvol}x+): " + ", ".join(alerts["Ticker"]))

rising = df[df["Vol Rising (30m)"]] if not df.empty else df
if not rising.empty:
    st.info("📈 Volume rising every 30min: " + ", ".join(rising["Ticker"]))

st.dataframe(df, use_container_width=True, hide_index=True)

if alerts_on and (not alerts.empty or not rising.empty):
    lines = ["Ticker Watch alert"]
    if not alerts.empty:
        lines.append(f"RVOL {alert_rvol}x+: " + ", ".join(alerts["Ticker"]))
    if not rising.empty:
        lines.append("Vol rising every 30min: " + ", ".join(rising["Ticker"]))
    message = "\n".join(lines)

    if bot_token and chat_id:
        ok, info = send_telegram(bot_token, chat_id, message)
        if ok:
            st.success("Telegram alert sent")
        else:
            st.error(f"Telegram send failed: {info}")

    if twilio_sid and twilio_auth and twilio_to:
        ok, info = send_whatsapp(twilio_sid, twilio_auth, twilio_from, twilio_to, message)
        if ok:
            st.success("WhatsApp alert sent")
        else:
            st.error(f"WhatsApp send failed: {info}")
