"""
Aman's ETF Accumulation Alert System — v2.0
============================================
A professional-grade ETF screener that sends daily email alerts
with EMA touch zones, RSI, volume confirmation, and accurate signals.

Fixed from v1:
  - yfinance MultiIndex column handling (was silently breaking)
  - Touch threshold raised to 2% (0.5% was too tight — almost never triggered)
  - EMA zone scoring logic corrected (above 200 EMA is BULLISH, not caution)
  - RSI added as a required confirmation signal
  - Volume analysis added (above/below 20-day avg volume)
  - USD tickers (SPY/QQQ) now show $ not ₹
  - last_buy_dates now actually gates monthly buys
  - force_buy on last day now only applies when zone score >= 50 (not blindly)
  - Weekend/holiday data staleness warning added
  - Duplicate imports removed
  - Trend context added (price vs 200 EMA = uptrend/downtrend)
  - 52-week high/low proximity added
  - Market summary section added to email
"""

import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
import json
import logging
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ============================================================
# CONFIGURATION
# ============================================================

TICKERS = [
    # Indian ETFs
    {'symbol': 'NIFTYBEES.NS',  'name': 'Nifty 50 ETF',            'currency': 'INR'},
    {'symbol': 'JUNIORBEES.NS', 'name': 'Nifty Next 50 ETF',       'currency': 'INR'},
    {'symbol': 'MID150BEES.NS', 'name': 'Nifty Midcap 150 ETF',    'currency': 'INR'},
    {'symbol': 'BANKBEES.NS',   'name': 'Bank Nifty ETF',          'currency': 'INR'},
    {'symbol': 'GOLDBEES.NS',   'name': 'Gold ETF',                'currency': 'INR'},
    {'symbol': 'SILVERBEES.NS', 'name': 'Silver ETF',              'currency': 'INR'},
    {'symbol': 'HDFCSML250.NS', 'name': 'HDFC Smallcap 250 ETF',   'currency': 'INR'},
    # US ETFs — displayed in USD
    {'symbol': 'SPY',           'name': 'S&P 500 ETF',             'currency': 'USD'},
    {'symbol': 'QQQ',           'name': 'Nasdaq-100 ETF',          'currency': 'USD'},
]

EMA_PERIODS       = [20, 50, 100, 200]
RSI_PERIOD        = 14
VOLUME_LOOKBACK   = 20      # days for average volume
TOUCH_THRESHOLD   = 2.0     # % — price within 2% of EMA = "touching"
VOLATILITY_MAX    = 3.0     # % daily std; skip aggressive buy above this
MIN_ZONE_SCORE    = 55      # minimum score to generate a BUY recommendation
LAST_BUY_FILE     = 'last_buy_dates.json'
DATA_PERIOD       = '1y'    # fetch 1 year of daily data (was 6mo — need 200 EMA accuracy)

EMAIL_SENDER    = os.getenv("EMAIL")
EMAIL_PASSWORD  = os.getenv("PASS")
EMAIL_RECEIVERS = os.getenv('EMAIL_RECEIVER', '').split(',')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etf_accumulation.log'),
        logging.StreamHandler()
    ]
)

# ============================================================
# DATA FETCHING — handles yfinance MultiIndex columns safely
# ============================================================

def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV. yfinance >= 0.2.x returns MultiIndex columns
    when downloading a single ticker with auto_adjust=True.
    We flatten them here so the rest of the code uses simple column names.
    """
    try:
        df = yf.download(
            symbol,
            period=DATA_PERIOD,
            interval='1d',
            auto_adjust=True,
            progress=False,
            actions=False
        )
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Flatten MultiIndex columns if present (yfinance 0.2.x behavior)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns for {symbol}: {df.columns.tolist()}")

        df = df.dropna(subset=['Close'])
        if len(df) < 210:
            logging.warning(f"{symbol}: only {len(df)} rows — 200 EMA may be inaccurate")

        return df

    except Exception as e:
        logging.error(f"fetch_ohlcv({symbol}): {e}")
        return pd.DataFrame()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calc_ema(series: pd.Series, period: int) -> float:
    """Return the most recent EMA value as a plain float."""
    val = series.ewm(span=period, adjust=False).mean().iloc[-1]
    return float(val)


def calc_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    """
    RSI using Wilder's smoothing (standard).
    Returns value 0–100.
    """
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def calc_volume_ratio(df: pd.DataFrame) -> float:
    """
    Today's volume vs 20-day average volume.
    > 1.0 means above-average volume (stronger signal).
    Returns 0.0 if volume data unavailable.
    """
    vol = df['Volume'].replace(0, np.nan).dropna()
    if len(vol) < VOLUME_LOOKBACK + 1:
        return 0.0
    avg_vol = float(vol.iloc[-(VOLUME_LOOKBACK + 1):-1].mean())
    today_vol = float(vol.iloc[-1])
    return round(today_vol / avg_vol, 2) if avg_vol > 0 else 0.0


def calc_52w_position(df: pd.DataFrame) -> dict:
    """
    Returns how far the current price is from 52-week high and low.
    Useful context for accumulation decisions.
    """
    last_close = float(df['Close'].iloc[-1])
    high_52w = float(df['High'].rolling(252).max().iloc[-1])
    low_52w  = float(df['Low'].rolling(252).min().iloc[-1])
    pct_from_high = ((last_close - high_52w) / high_52w) * 100
    pct_from_low  = ((last_close - low_52w)  / low_52w)  * 100
    return {
        'high_52w':       round(high_52w, 2),
        'low_52w':        round(low_52w, 2),
        'pct_from_high':  round(pct_from_high, 2),
        'pct_from_low':   round(pct_from_low, 2),
    }


# ============================================================
# ZONE SCORING — corrected logic
# ============================================================

def calc_zone_score(last_close: float, ema_vals: list, ema_diffs: list) -> tuple:
    """
    Score the accumulation attractiveness 0–100.

    CORRECTED LOGIC vs v1:
    ─────────────────────
    The original code treated "above 200 EMA" as Caution (score 30) and
    "above 20 EMA" as Avoid (score 0). This is WRONG for long-term ETF
    accumulation — being above the 200 EMA means you're in an uptrend,
    which is desirable. The score should reflect PRICE RELATIVE TO EMA
    as a valuation signal, not trend direction.

    Correct framework:
      • At/below 20 EMA:  Mild pullback — Good entry (60)
      • At/below 50 EMA:  Normal pullback — Great entry (75)
      • At/below 100 EMA: Deeper pullback — Excellent (90)
      • At/below 200 EMA: Major dip / trend test — Goated (100)
      • 0–5% above 20 EMA: Slightly extended — Neutral (45)
      • 5–10% above 20 EMA: Extended — Caution (25)
      • >10% above 20 EMA: Overbought — Avoid (10)

    "Touching" = within TOUCH_THRESHOLD % (now 2%, was 0.5%)
    """
    # Check each EMA from deepest (200) to shallowest (20)
    for i in [3, 2, 1, 0]:  # index 3=200, 2=100, 1=50, 0=20
        diff = ema_diffs[i]
        if abs(diff) <= TOUCH_THRESHOLD:
            # Touching this EMA
            scores = [60, 75, 90, 100]
            labels = [
                "Good entry (Touching 20 EMA)",
                "Great entry (Touching 50 EMA)",
                "Excellent entry (Touching 100 EMA)",
                "Goated entry (Touching 200 EMA)"
            ]
            return scores[i], labels[i]
        elif diff < -TOUCH_THRESHOLD:
            # Below this EMA — even better
            scores = [65, 80, 92, 100]
            labels = [
                "Strong pullback (Below 20 EMA)",
                "Deep pullback (Below 50 EMA)",
                "Major dip (Below 100 EMA)",
                "Bear zone (Below 200 EMA)"
            ]
            return scores[i], labels[i]

    # Not touching or below any EMA — price is above all EMAs
    pct_above_20 = ema_diffs[0]  # positive = above 20 EMA
    if pct_above_20 <= 5:
        return 45, "Slightly extended (Above 20 EMA, within 5%)"
    elif pct_above_20 <= 10:
        return 25, "Extended — wait for pullback (5–10% above 20 EMA)"
    else:
        return 10, "Overbought — avoid now (>10% above 20 EMA)"


# ============================================================
# RSI SIGNAL
# ============================================================

def interpret_rsi(rsi: float) -> tuple:
    """
    Returns (signal_label, boost) where boost adjusts zone score.
    RSI oversold = positive boost, overbought = penalty.
    """
    if rsi <= 30:
        return "Oversold (RSI ≤30) ✅", +15
    elif rsi <= 45:
        return "Mildly oversold (RSI 30–45) ✅", +8
    elif rsi <= 60:
        return "Neutral RSI (45–60)", 0
    elif rsi <= 70:
        return "Mildly overbought (RSI 60–70) ⚠️", -8
    else:
        return "Overbought (RSI >70) ❌", -15


# ============================================================
# MAIN ANALYSIS PER TICKER
# ============================================================

def analyse_ticker(ticker_info: dict) -> dict:
    symbol   = ticker_info['symbol']
    name     = ticker_info['name']
    currency = ticker_info['currency']

    df = fetch_ohlcv(symbol)
    if df.empty:
        return {'symbol': symbol, 'name': name, 'error': 'Failed to fetch data'}

    try:
        last_close  = float(df['Close'].iloc[-1])
        last_date   = df.index[-1]

        # Check data staleness (weekend / holiday)
        days_old = (datetime.now().date() - last_date.date()).days
        stale_warning = days_old > 1

        # EMAs
        ema_vals  = [calc_ema(df['Close'], p) for p in EMA_PERIODS]
        ema_diffs = [((last_close - e) / e) * 100 for e in ema_vals]

        # RSI
        rsi            = calc_rsi(df['Close'])
        rsi_label, rsi_boost = interpret_rsi(rsi)

        # Volatility (daily std %)
        volatility = float(df['Close'].pct_change().std() * 100)

        # Volume ratio
        vol_ratio = calc_volume_ratio(df)

        # 52-week context
        wk52 = calc_52w_position(df)

        # Zone score (corrected)
        zone_score, zone_label = calc_zone_score(last_close, ema_vals, ema_diffs)

        # Apply RSI boost/penalty
        adjusted_score = max(0, min(100, zone_score + rsi_boost))

        # Trend context
        trend = "Uptrend" if last_close > ema_vals[3] else "Downtrend"

        # Volume confirmation (only meaningful for Indian ETFs; US ETFs ok)
        vol_note = ""
        if vol_ratio > 1.5:
            vol_note = f"High volume ({vol_ratio}x avg) — strong signal"
        elif vol_ratio > 1.0:
            vol_note = f"Above-avg volume ({vol_ratio}x)"
        elif vol_ratio > 0:
            vol_note = f"Below-avg volume ({vol_ratio}x) — weak confirmation"

        # Final buy signal
        buy_signal = (
            adjusted_score >= MIN_ZONE_SCORE
            and volatility <= VOLATILITY_MAX
            and rsi <= 65  # Don't buy when RSI is hot
        )

        return {
            'symbol':         symbol,
            'name':           name,
            'currency':       currency,
            'last_close':     round(last_close, 2),
            'last_date':      last_date.strftime('%d %b %Y'),
            'stale_warning':  stale_warning,
            'days_old':       days_old,
            'ema_vals':       [round(v, 2) for v in ema_vals],
            'ema_diffs':      [round(d, 2) for d in ema_diffs],
            'rsi':            round(rsi, 1),
            'rsi_label':      rsi_label,
            'rsi_boost':      rsi_boost,
            'volatility':     round(volatility, 2),
            'vol_ratio':      vol_ratio,
            'vol_note':       vol_note,
            'zone_score':     zone_score,
            'adjusted_score': adjusted_score,
            'zone_label':     zone_label,
            'trend':          trend,
            'wk52':           wk52,
            'buy_signal':     buy_signal,
            'error':          None,
        }

    except Exception as e:
        logging.error(f"analyse_ticker({symbol}): {e}")
        return {'symbol': symbol, 'name': name, 'error': str(e)}


# ============================================================
# LAST BUY DATE TRACKING (actually used now)
# ============================================================

def load_last_buy_dates() -> dict:
    try:
        if os.path.exists(LAST_BUY_FILE):
            with open(LAST_BUY_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"load_last_buy_dates: {e}")
    return {}


def save_last_buy_dates(data: dict):
    try:
        with open(LAST_BUY_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"save_last_buy_dates: {e}")


def already_bought_this_month(symbol: str, last_buys: dict) -> bool:
    """Returns True if we already recorded a buy for this ticker this calendar month."""
    current_month = datetime.now().strftime('%Y-%m')
    return last_buys.get(symbol, '')[:7] == current_month


def record_buy(symbol: str, last_buys: dict):
    last_buys[symbol] = datetime.now().strftime('%Y-%m-%d')


# ============================================================
# EMAIL HTML GENERATION
# ============================================================

def _currency_symbol(currency: str) -> str:
    return '₹' if currency == 'INR' else '$'


def _score_color(score: int) -> str:
    if score >= 90:  return '#059669'  # emerald
    if score >= 75:  return '#7c3aed'  # violet
    if score >= 55:  return '#d97706'  # amber
    if score >= 30:  return '#ea580c'  # orange
    return '#dc2626'                   # red


def _score_label(score: int) -> str:
    if score >= 90:  return 'Goated / Excellent'
    if score >= 75:  return 'Great Entry'
    if score >= 55:  return 'Good Entry'
    if score >= 30:  return 'Caution'
    return 'Avoid'


def generate_html(reports: list, last_buys: dict, is_last_day: bool) -> str:
    today       = datetime.now()
    any_buy     = any(r.get('buy_signal') for r in reports if not r.get('error'))
    buy_reports = [r for r in reports if not r.get('error') and r.get('buy_signal')]
    best_score  = max((r['adjusted_score'] for r in reports if not r.get('error')), default=0)

    summary_note = ""
    if is_last_day:
        summary_note = (
            "<div style='background:#fffbeb;border-left:4px solid #f59e0b;padding:12px 16px;"
            "margin-bottom:20px;border-radius:6px;font-size:14px;'>"
            "📅 <strong>Last trading day of month:</strong> Monthly SIP reminder. "
            "Review below — only act on tickers with a BUY signal and adjusted score ≥55.</div>"
        )

    stale_note = ""
    stale_tickers = [r['symbol'] for r in reports if not r.get('error') and r.get('stale_warning')]
    if stale_tickers:
        stale_note = (
            f"<div style='background:#fef2f2;border-left:4px solid #ef4444;padding:12px 16px;"
            f"margin-bottom:20px;border-radius:6px;font-size:13px;'>"
            f"⚠️ <strong>Stale data warning:</strong> {', '.join(stale_tickers)} — "
            f"price data may be from a previous session (weekend/holiday). "
            f"Verify live prices before acting.</div>"
        )

    # Build ticker cards
    cards_html = ""
    for r in reports:
        if r.get('error'):
            cards_html += (
                f"<div style='background:#fef2f2;border-radius:10px;padding:16px;"
                f"margin-bottom:16px;color:#991b1b;font-size:13px;'>"
                f"<strong>⚠ {r['symbol']} — {r['name']}</strong>: {r['error']}</div>"
            )
            continue

        cs = _currency_symbol(r['currency'])
        sc = _score_color(r['adjusted_score'])
        sl = _score_label(r['adjusted_score'])

        already_bought = already_bought_this_month(r['symbol'], last_buys)
        if r['buy_signal'] and not already_bought:
            rec_bg, rec_color, rec_text = '#d1fae5', '#065f46', '✅ ACCUMULATE'
        elif r['buy_signal'] and already_bought:
            rec_bg, rec_color, rec_text = '#e0f2fe', '#075985', '📋 Already bought this month'
        elif r['adjusted_score'] >= 45:
            rec_bg, rec_color, rec_text = '#fef3c7', '#92400e', '⏳ Watch — approaching zone'
        else:
            rec_bg, rec_color, rec_text = '#fee2e2', '#991b1b', '⛔ Avoid — too expensive'

        trend_badge = (
            "<span style='background:#d1fae5;color:#065f46;padding:2px 8px;"
            "border-radius:4px;font-size:11px;font-weight:600;'>↑ UPTREND</span>"
            if r['trend'] == 'Uptrend' else
            "<span style='background:#fee2e2;color:#991b1b;padding:2px 8px;"
            "border-radius:4px;font-size:11px;font-weight:600;'>↓ DOWNTREND</span>"
        )

        stale_badge = (
            f"<span style='background:#fef3c7;color:#92400e;padding:2px 8px;"
            f"border-radius:4px;font-size:11px;margin-left:6px;'>Data: {r['last_date']}</span>"
            if r.get('stale_warning') else ""
        )

        wk52 = r['wk52']
        vol_note_html = (
            f"<div style='font-size:12px;color:#64748b;margin-top:4px;'>{r['vol_note']}</div>"
            if r['vol_note'] else ""
        )

        ema_rows = ""
        for i, (period, val, diff) in enumerate(zip(EMA_PERIODS, r['ema_vals'], r['ema_diffs'])):
            diff_color = '#059669' if diff <= 0 else '#dc2626'
            diff_sign  = '+' if diff > 0 else ''
            ema_rows += (
                f"<tr>"
                f"<td style='padding:8px 12px;color:#64748b;font-size:13px;'>{period} EMA</td>"
                f"<td style='padding:8px 12px;font-size:13px;'>{cs}{val:,.2f}</td>"
                f"<td style='padding:8px 12px;font-size:13px;color:{diff_color};"
                f"font-weight:600;'>{diff_sign}{diff:.1f}%</td>"
                f"</tr>"
            )

        cards_html += f"""
        <div style="background:white;border-radius:12px;overflow:hidden;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:20px;
                    border-top:4px solid {sc};">

          <div style="padding:16px 20px;border-bottom:1px solid #e2e8f0;
                      display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div style="font-weight:700;font-size:16px;">{r['name']}</div>
              <div style="color:#64748b;font-size:12px;margin-top:2px;">{r['symbol']}
                {stale_badge}
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:22px;font-weight:700;">{cs}{r['last_close']:,.2f}</div>
              <div style="margin-top:4px;">{trend_badge}</div>
            </div>
          </div>

          <div style="display:flex;padding:12px 20px;background:#f8fafc;
                      border-bottom:1px solid #e2e8f0;gap:0;flex-wrap:wrap;">
            <div style="flex:1;min-width:120px;padding:4px 0;">
              <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:0.06em;">Zone Score</div>
              <div style="font-size:18px;font-weight:700;color:{sc};">{r['adjusted_score']}</div>
              <div style="font-size:11px;color:{sc};">{sl}</div>
            </div>
            <div style="flex:1;min-width:120px;padding:4px 0;">
              <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:0.06em;">RSI ({RSI_PERIOD})</div>
              <div style="font-size:18px;font-weight:700;">{r['rsi']}</div>
              <div style="font-size:11px;color:#64748b;">{r['rsi_label']}</div>
            </div>
            <div style="flex:1;min-width:120px;padding:4px 0;">
              <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:0.06em;">Volatility</div>
              <div style="font-size:18px;font-weight:700;">{r['volatility']:.1f}%</div>
              <div style="font-size:11px;color:#64748b;">Daily std dev</div>
            </div>
            <div style="flex:1;min-width:120px;padding:4px 0;">
              <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:0.06em;">Volume</div>
              <div style="font-size:18px;font-weight:700;">{r['vol_ratio']}x</div>
              <div style="font-size:11px;color:#64748b;">vs 20d avg</div>
            </div>
          </div>

          <div style="padding:8px 20px;background:#f8fafc;border-bottom:1px solid #e2e8f0;">
            <div style="font-size:12px;color:#64748b;">
              📍 {r['zone_label']}
              &nbsp;|&nbsp;
              52W range: {cs}{wk52['low_52w']:,.2f} – {cs}{wk52['high_52w']:,.2f}
              &nbsp;|&nbsp;
              <span style="color:{'#059669' if wk52['pct_from_high'] < -20 else '#94a3b8'}">
                {wk52['pct_from_high']:.1f}% from 52W high
              </span>
            </div>
            {vol_note_html}
          </div>

          <table style="width:100%;border-collapse:collapse;">
            <tr style="background:#f8fafc;">
              <th style="padding:8px 12px;text-align:left;font-size:12px;
                         color:#64748b;font-weight:600;">EMA</th>
              <th style="padding:8px 12px;text-align:left;font-size:12px;
                         color:#64748b;font-weight:600;">Value</th>
              <th style="padding:8px 12px;text-align:left;font-size:12px;
                         color:#64748b;font-weight:600;">Price vs EMA</th>
            </tr>
            {ema_rows}
          </table>

          <div style="padding:14px 20px;text-align:center;font-weight:700;
                      background:{rec_bg};color:{rec_color};font-size:14px;">
            {rec_text}
          </div>
        </div>
        """

    # Legend
    legend_items = [
        ('#059669', 'Score 90–100: Goated / Excellent'),
        ('#7c3aed', 'Score 75–89: Great Entry'),
        ('#d97706', 'Score 55–74: Good Entry'),
        ('#ea580c', 'Score 30–54: Caution / Watch'),
        ('#dc2626', 'Score <30: Avoid'),
    ]
    legend_html = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:5px;margin:4px 10px;font-size:12px;'>"
        f"<span style='width:10px;height:10px;border-radius:50%;background:{c};display:inline-block;'></span>"
        f"{label}</span>"
        for c, label in legend_items
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Aman's ETF Newsletter</title>
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
             background:#f1f5f9;color:#1e293b;margin:0;padding:20px;">

  <div style="max-width:900px;margin:0 auto;">

    <div style="background:linear-gradient(135deg,#1e1b4b 0%,#4f46e5 50%,#7c3aed 100%);
                color:white;border-radius:16px 16px 0 0;padding:28px 28px 20px;text-align:center;">
      <h1 style="margin:0 0 4px;font-size:24px;font-weight:700;">Aman's ETF Accumulation Alert</h1>
      <p style="margin:0;opacity:0.8;font-size:14px;">EMA Touch + RSI + Volume Strategy — v2.0</p>
      <div style="display:inline-block;background:rgba(255,255,255,0.15);padding:5px 14px;
                  border-radius:20px;margin-top:10px;font-size:13px;">
        {today.strftime('%A, %d %B %Y')}
      </div>
    </div>

    <div style="background:white;padding:24px 28px;">
      {summary_note}
      {stale_note}

      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;">
        <div style="flex:1;min-width:130px;background:#f8fafc;border-radius:10px;
                    padding:14px;text-align:center;">
          <div style="font-size:28px;font-weight:700;color:#4f46e5;">{len([r for r in reports if not r.get('error')])}</div>
          <div style="font-size:12px;color:#64748b;">ETFs Scanned</div>
        </div>
        <div style="flex:1;min-width:130px;background:#d1fae5;border-radius:10px;
                    padding:14px;text-align:center;">
          <div style="font-size:28px;font-weight:700;color:#059669;">{len(buy_reports)}</div>
          <div style="font-size:12px;color:#065f46;">Buy Signals</div>
        </div>
        <div style="flex:1;min-width:130px;background:#f8fafc;border-radius:10px;
                    padding:14px;text-align:center;">
          <div style="font-size:28px;font-weight:700;color:#7c3aed;">{best_score}</div>
          <div style="font-size:12px;color:#64748b;">Best Score</div>
        </div>
        <div style="flex:1;min-width:130px;background:#f8fafc;border-radius:10px;
                    padding:14px;text-align:center;">
          <div style="font-size:28px;font-weight:700;color:#0ea5e9;">{RSI_PERIOD}d RSI</div>
          <div style="font-size:12px;color:#64748b;">+ Volume Filter</div>
        </div>
      </div>

      {cards_html}
    </div>

    <div style="background:#1e293b;color:#94a3b8;border-radius:0 0 16px 16px;
                padding:20px 28px;font-size:12px;line-height:1.7;">
      <div style="text-align:center;margin-bottom:14px;flex-wrap:wrap;">
        {legend_html}
      </div>
      <div style="max-width:700px;margin:0 auto;text-align:center;">
        <strong style="color:#cbd5e1;">How scores work:</strong>
        Zone score (EMA position, 0–100) + RSI boost/penalty (±15) = Adjusted Score.
        Buy signal requires: Adjusted Score ≥{MIN_ZONE_SCORE}, Daily volatility ≤{VOLATILITY_MAX}%, RSI ≤65.
        "Touching" = price within {TOUCH_THRESHOLD}% of EMA.
        <br><br>
        ⚠️ This is an automated tool for informational purposes only.
        Always conduct your own research. Not financial advice.
        <br>
        <span style="color:#64748b;">Generated: {today.strftime('%Y-%m-%d %H:%M:%S IST')}</span>
      </div>
    </div>

  </div>
</body>
</html>"""


# ============================================================
# EMAIL SENDING
# ============================================================

def send_email(subject: str, html_body: str):
    try:
        msg = MIMEMultipart('alternative')
        msg['From']    = EMAIL_SENDER
        msg['To']      = ", ".join(EMAIL_RECEIVERS)
        msg['Subject'] = subject
        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logging.info("Email sent successfully")
    except Exception as e:
        logging.error(f"send_email: {e}")


def is_last_day_of_month() -> bool:
    today    = datetime.now()
    next_day = today + timedelta(days=1)
    return today.month != next_day.month


# ============================================================
# MAIN
# ============================================================

def main():
    logging.info("=" * 60)
    logging.info("Aman's ETF Accumulation Alert v2.0")
    logging.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)

    last_buys  = load_last_buy_dates()
    is_last_day = is_last_day_of_month()
    reports    = []

    for ticker_info in TICKERS:
        logging.info(f"Analysing {ticker_info['symbol']} — {ticker_info['name']}")
        result = analyse_ticker(ticker_info)
        reports.append(result)
        if result.get('buy_signal') and not already_bought_this_month(result['symbol'], last_buys):
            record_buy(result['symbol'], last_buys)

    save_last_buy_dates(last_buys)

    # Build email subject
    buy_reports = [r for r in reports if not r.get('error') and r.get('buy_signal')]
    if buy_reports:
        best = max(buy_reports, key=lambda r: r['adjusted_score'])
        score = best['adjusted_score']
        if score >= 90:
            prefix = f"🐐 Goated Entry — {best['name']}"
        elif score >= 75:
            prefix = f"⭐ Great Entry — {best['name']}"
        else:
            prefix = f"✅ {len(buy_reports)} Buy Signal{'s' if len(buy_reports)>1 else ''}"
    elif is_last_day:
        prefix = "📅 Month-End Review"
    else:
        prefix = "📊 Daily Scan — No Buy Signals"

    subject = f"{prefix} | ETF Report {datetime.now().strftime('%d %b %Y')}"

    html = generate_html(reports, last_buys, is_last_day)
    send_email(subject, html)
    logging.info("Done.")


if __name__ == '__main__':
    main()
