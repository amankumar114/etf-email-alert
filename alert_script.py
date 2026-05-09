"""
Aman's ETF Accumulation Alert System — v2.1
============================================
Fixes vs v2.0:
  - 52W high/low NaN bug: rolling(252) needs >252 rows to return a value.
    Fixed with rolling(window, min_periods=1) — also bumped DATA_PERIOD to
    '2y' so the window is always fully populated regardless.
  - Mobile-first email layout: replaced display:flex stat bar with HTML
    tables (renders correctly in Gmail/Apple Mail on every screen size).
  - Zone label, 52W range, volume note now each on their own line — no
    horizontal overflow on narrow mobile screens.
  - All v2.0 fixes retained (RSI, volume, corrected scoring, USD symbols,
    monthly buy gate, staleness warning, MultiIndex column handling, etc.)
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
    {'symbol': 'NIFTYBEES.NS',  'name': 'Nifty 50 ETF',          'currency': 'INR'},
    {'symbol': 'JUNIORBEES.NS', 'name': 'Nifty Next 50 ETF',     'currency': 'INR'},
    {'symbol': 'MID150BEES.NS', 'name': 'Nifty Midcap 150 ETF',  'currency': 'INR'},
    {'symbol': 'BANKBEES.NS',   'name': 'Bank Nifty ETF',        'currency': 'INR'},
    {'symbol': 'GOLDBEES.NS',   'name': 'Gold ETF',              'currency': 'INR'},
    {'symbol': 'SILVERBEES.NS', 'name': 'Silver ETF',            'currency': 'INR'},
    {'symbol': 'HDFCSML250.NS', 'name': 'HDFC Smallcap 250 ETF', 'currency': 'INR'},
    {'symbol': 'SPY',           'name': 'S&P 500 ETF',           'currency': 'USD'},
    {'symbol': 'QQQ',           'name': 'Nasdaq-100 ETF',        'currency': 'USD'},
]

EMA_PERIODS     = [20, 50, 100, 200]
RSI_PERIOD      = 14
VOLUME_LOOKBACK = 20
TOUCH_THRESHOLD = 2.0    # % — price within 2% of EMA counts as "touching"
VOLATILITY_MAX  = 3.0    # % daily std dev
MIN_ZONE_SCORE  = 55     # adjusted score needed for a BUY
LAST_BUY_FILE   = 'last_buy_dates.json'
DATA_PERIOD     = '2y'   # 2y ensures rolling 52W never hits NaN

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
# DATA FETCHING
# ============================================================

def fetch_ohlcv(symbol):
    try:
        df = yf.download(
            symbol, period=DATA_PERIOD, interval='1d',
            auto_adjust=True, progress=False, actions=False
        )
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {df.columns.tolist()}")
        df = df.dropna(subset=['Close'])
        if len(df) < 210:
            logging.warning(f"{symbol}: only {len(df)} rows — 200 EMA may be less accurate")
        return df
    except Exception as e:
        logging.error(f"fetch_ohlcv({symbol}): {e}")
        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def calc_ema(series, period):
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def calc_rsi(series, period=RSI_PERIOD):
    delta    = series.diff().dropna()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


def calc_volume_ratio(df):
    vol = df['Volume'].replace(0, np.nan).dropna()
    if len(vol) < VOLUME_LOOKBACK + 1:
        return 0.0
    avg = float(vol.iloc[-(VOLUME_LOOKBACK+1):-1].mean())
    return round(float(vol.iloc[-1]) / avg, 2) if avg > 0 else 0.0


def calc_52w_position(df):
    """
    NaN fix: rolling(252) needs 252 complete rows before it emits.
    rolling(252, min_periods=1) works on any data length >= 1.
    DATA_PERIOD='2y' gives ~500 rows so the 252-row window is always full,
    but min_periods=1 is kept as a defensive safety net.
    """
    last = float(df['Close'].iloc[-1])
    h52  = float(df['High'].rolling(252, min_periods=1).max().iloc[-1])
    l52  = float(df['Low'].rolling(252,  min_periods=1).min().iloc[-1])
    return {
        'high_52w':      round(h52, 2),
        'low_52w':       round(l52, 2),
        'pct_from_high': round(((last - h52) / h52) * 100, 2),
        'pct_from_low':  round(((last - l52) / l52) * 100, 2),
    }


# ============================================================
# SCORING
# ============================================================

def calc_zone_score(last_close, ema_vals, ema_diffs):
    for i in [3, 2, 1, 0]:
        d = ema_diffs[i]
        if abs(d) <= TOUCH_THRESHOLD:
            return [60,75,90,100][i], [
                "Good entry — touching 20 EMA",
                "Great entry — touching 50 EMA",
                "Excellent entry — touching 100 EMA",
                "Goated entry — touching 200 EMA",
            ][i]
        elif d < -TOUCH_THRESHOLD:
            return [65,80,92,100][i], [
                "Strong pullback — below 20 EMA",
                "Deep pullback — below 50 EMA",
                "Major dip — below 100 EMA",
                "Bear zone — below 200 EMA",
            ][i]
    pct = ema_diffs[0]
    if pct <= 5:   return 45, "Slightly extended — within 5% above 20 EMA"
    if pct <= 10:  return 25, "Extended — 5-10% above 20 EMA, wait for pullback"
    return 10, "Overbought — more than 10% above 20 EMA"


def interpret_rsi(rsi):
    if rsi <= 30:  return "Oversold (RSI <=30) \u2705", +15
    if rsi <= 45:  return "Mildly oversold (RSI 30-45) \u2705", +8
    if rsi <= 60:  return "Neutral (RSI 45-60)", 0
    if rsi <= 70:  return "Mildly overbought (RSI 60-70) \u26a0\ufe0f", -8
    return "Overbought (RSI >70) \u274c", -15


# ============================================================
# ANALYSIS
# ============================================================

def analyse_ticker(ticker_info):
    symbol   = ticker_info['symbol']
    name     = ticker_info['name']
    currency = ticker_info['currency']

    df = fetch_ohlcv(symbol)
    if df.empty:
        return {'symbol': symbol, 'name': name, 'currency': currency,
                'error': 'Failed to fetch price data'}
    try:
        last  = float(df['Close'].iloc[-1])
        ldate = df.index[-1]
        days  = (datetime.now().date() - ldate.date()).days

        ema_vals  = [calc_ema(df['Close'], p) for p in EMA_PERIODS]
        ema_diffs = [((last - e) / e) * 100 for e in ema_vals]
        rsi               = calc_rsi(df['Close'])
        rsi_label, boost  = interpret_rsi(rsi)
        vol               = float(df['Close'].pct_change().std() * 100)
        vol_ratio         = calc_volume_ratio(df)
        wk52              = calc_52w_position(df)
        zone_score, zlabel = calc_zone_score(last, ema_vals, ema_diffs)
        adj               = max(0, min(100, zone_score + boost))
        trend             = "Uptrend" if last > ema_vals[3] else "Downtrend"

        if vol_ratio >= 1.5:   vnote = f"High volume ({vol_ratio}x avg) — strong confirmation"
        elif vol_ratio >= 1.0: vnote = f"Above-average volume ({vol_ratio}x avg)"
        elif vol_ratio > 0:    vnote = f"Below-average volume ({vol_ratio}x avg) — weak confirmation"
        else:                  vnote = "Volume data unavailable"

        return {
            'symbol': symbol, 'name': name, 'currency': currency,
            'last_close': round(last, 2),
            'last_date':  ldate.strftime('%d %b %Y'),
            'stale_warning': days > 1, 'days_old': days,
            'ema_vals':  [round(v, 2) for v in ema_vals],
            'ema_diffs': [round(d, 2) for d in ema_diffs],
            'rsi': round(rsi, 1), 'rsi_label': rsi_label, 'rsi_boost': boost,
            'volatility': round(vol, 2),
            'vol_ratio': vol_ratio, 'vol_note': vnote,
            'zone_score': zone_score, 'adjusted_score': adj,
            'zone_label': zlabel, 'trend': trend, 'wk52': wk52,
            'buy_signal': adj >= MIN_ZONE_SCORE and vol <= VOLATILITY_MAX and rsi <= 65,
            'error': None,
        }
    except Exception as e:
        logging.error(f"analyse_ticker({symbol}): {e}")
        return {'symbol': symbol, 'name': name, 'currency': currency, 'error': str(e)}


# ============================================================
# BUY DATE TRACKING
# ============================================================

def load_last_buy_dates():
    try:
        if os.path.exists(LAST_BUY_FILE):
            with open(LAST_BUY_FILE) as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"load_last_buy_dates: {e}")
    return {}


def save_last_buy_dates(data):
    try:
        with open(LAST_BUY_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"save_last_buy_dates: {e}")


def already_bought_this_month(symbol, last_buys):
    return last_buys.get(symbol, '')[:7] == datetime.now().strftime('%Y-%m')


def record_buy(symbol, last_buys):
    last_buys[symbol] = datetime.now().strftime('%Y-%m-%d')


# ============================================================
# HTML EMAIL — mobile-first, table-based layout
# ============================================================

def _cs(currency):
    return '&#8377;' if currency == 'INR' else '$'


def _score_color(s):
    if s >= 90: return '#059669'
    if s >= 75: return '#7c3aed'
    if s >= 55: return '#d97706'
    if s >= 30: return '#ea580c'
    return '#dc2626'


def _score_label(s):
    if s >= 90: return 'Goated / Excellent'
    if s >= 75: return 'Great Entry'
    if s >= 55: return 'Good Entry'
    if s >= 30: return 'Caution / Watch'
    return 'Avoid'


def generate_html(reports, last_buys, is_last_day):
    today      = datetime.now()
    good       = [r for r in reports if not r.get('error')]
    buy_rpts   = [r for r in good if r.get('buy_signal')]
    best_score = max((r['adjusted_score'] for r in good), default=0)

    # banners
    banners = ''
    if is_last_day:
        banners += """<tr><td style="padding:0 0 12px;">
          <div style="background:#fffbeb;border-left:4px solid #f59e0b;
                      padding:12px 14px;border-radius:6px;font-size:14px;
                      line-height:1.5;color:#78350f;">
            &#128197; <strong>Last trading day of the month.</strong>
            Monthly SIP reminder &mdash; only act on tickers showing
            <strong>BUY</strong> with adjusted score &ge;55.
          </div></td></tr>"""

    stale = [r['symbol'] for r in good if r.get('stale_warning')]
    if stale:
        banners += f"""<tr><td style="padding:0 0 12px;">
          <div style="background:#fef2f2;border-left:4px solid #ef4444;
                      padding:12px 14px;border-radius:6px;font-size:13px;
                      line-height:1.5;color:#7f1d1d;">
            &#9888;&#65039; <strong>Stale data:</strong>
            {', '.join(stale)} &mdash; price is from a previous session.
            Verify live prices before acting.
          </div></td></tr>"""

    # summary bar — 4-cell table (works in all email clients)
    summary = f"""
    <table width="100%" cellpadding="0" cellspacing="4"
           style="margin-bottom:20px;">
      <tr>
        <td width="25%">
          <div style="background:#f0f4ff;border-radius:10px;
                      padding:12px 6px;text-align:center;">
            <div style="font-size:24px;font-weight:800;color:#4f46e5;">
              {len(good)}</div>
            <div style="font-size:11px;color:#6366f1;margin-top:2px;">
              ETFs Scanned</div>
          </div>
        </td>
        <td width="25%">
          <div style="background:#d1fae5;border-radius:10px;
                      padding:12px 6px;text-align:center;">
            <div style="font-size:24px;font-weight:800;color:#059669;">
              {len(buy_rpts)}</div>
            <div style="font-size:11px;color:#065f46;margin-top:2px;">
              Buy Signals</div>
          </div>
        </td>
        <td width="25%">
          <div style="background:#f5f3ff;border-radius:10px;
                      padding:12px 6px;text-align:center;">
            <div style="font-size:24px;font-weight:800;color:#7c3aed;">
              {best_score}</div>
            <div style="font-size:11px;color:#7c3aed;margin-top:2px;">
              Best Score</div>
          </div>
        </td>
        <td width="25%">
          <div style="background:#f0f9ff;border-radius:10px;
                      padding:12px 6px;text-align:center;">
            <div style="font-size:24px;font-weight:800;color:#0ea5e9;">
              {RSI_PERIOD}d</div>
            <div style="font-size:11px;color:#0284c7;margin-top:2px;">
              RSI Filter</div>
          </div>
        </td>
      </tr>
    </table>"""

    # cards
    cards = ''
    for r in reports:
        if r.get('error'):
            cards += f"""
            <div style="background:#fef2f2;border-radius:10px;padding:14px;
                        margin-bottom:16px;color:#991b1b;font-size:13px;">
              <strong>&#9888; {r['symbol']} &mdash; {r.get('name','')}</strong>:
              {r['error']}
            </div>"""
            continue

        cs      = _cs(r['currency'])
        sc      = _score_color(r['adjusted_score'])
        sl      = _score_label(r['adjusted_score'])
        wk52    = r['wk52']
        already = already_bought_this_month(r['symbol'], last_buys)

        if r['buy_signal'] and not already:
            rb, rf, rt = '#d1fae5','#065f46','&#9989; ACCUMULATE'
        elif r['buy_signal'] and already:
            rb, rf, rt = '#e0f2fe','#075985','&#128203; Already bought this month'
        elif r['adjusted_score'] >= 45:
            rb, rf, rt = '#fef3c7','#92400e','&#9203; Watch &mdash; approaching zone'
        else:
            rb, rf, rt = '#fee2e2','#991b1b','&#128683; Avoid &mdash; price too high'

        tbg = '#d1fae5' if r['trend']=='Uptrend' else '#fee2e2'
        tfg = '#065f46' if r['trend']=='Uptrend' else '#991b1b'
        ts  = '&#8679;' if r['trend']=='Uptrend' else '&#8681;'

        stag = (f" &nbsp;<span style='background:#fef3c7;color:#92400e;"
                f"padding:2px 6px;border-radius:4px;font-size:10px;'>"
                f"Data: {r['last_date']}</span>"
                if r.get('stale_warning') else '')

        hcolor = '#059669' if wk52['pct_from_high'] < -20 else '#64748b'

        ema_rows = ''
        for period, val, diff in zip(EMA_PERIODS, r['ema_vals'], r['ema_diffs']):
            dc   = '#059669' if diff <= 0 else '#dc2626'
            sign = '+' if diff > 0 else ''
            ema_rows += f"""
            <tr style="border-bottom:1px solid #f1f5f9;">
              <td style="padding:9px 12px;font-size:13px;color:#475569;
                         white-space:nowrap;">{period}&nbsp;EMA</td>
              <td style="padding:9px 12px;font-size:13px;
                         white-space:nowrap;">{cs}{val:,.2f}</td>
              <td style="padding:9px 12px;font-size:13px;font-weight:700;
                         color:{dc};white-space:nowrap;">
                {sign}{diff:.1f}%</td>
            </tr>"""

        cards += f"""
        <div style="background:#ffffff;border-radius:12px;overflow:hidden;
                    margin-bottom:20px;border:1px solid #e2e8f0;
                    border-top:4px solid {sc};">

          <!-- header -->
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="padding:14px 14px 12px;">
            <tr>
              <td style="vertical-align:top;">
                <div style="font-weight:800;font-size:16px;color:#0f172a;">
                  {r['name']}</div>
                <div style="font-size:11px;color:#94a3b8;margin-top:3px;">
                  {r['symbol']}{stag}</div>
              </td>
              <td style="text-align:right;vertical-align:top;
                         white-space:nowrap;padding-left:8px;">
                <div style="font-size:20px;font-weight:800;color:#0f172a;">
                  {cs}{r['last_close']:,.2f}</div>
                <div style="margin-top:4px;">
                  <span style="background:{tbg};color:{tfg};padding:2px 8px;
                               border-radius:4px;font-size:10px;font-weight:700;">
                    {ts} {r['trend'].upper()}</span>
                </div>
              </td>
            </tr>
          </table>
          <div style="border-top:1px solid #f1f5f9;"></div>

          <!-- 4-stat bar — table layout -->
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="background:#f8fafc;">
            <tr>
              <td width="25%" style="padding:10px 6px 10px 12px;
                                     border-right:1px solid #e2e8f0;
                                     vertical-align:top;">
                <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;
                             letter-spacing:0.07em;font-weight:700;">Zone</div>
                <div style="font-size:20px;font-weight:800;color:{sc};
                             line-height:1.1;margin-top:3px;">{r['adjusted_score']}</div>
                <div style="font-size:10px;color:{sc};font-weight:600;
                             margin-top:2px;">{sl}</div>
              </td>
              <td width="25%" style="padding:10px 6px;
                                     border-right:1px solid #e2e8f0;
                                     vertical-align:top;">
                <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;
                             letter-spacing:0.07em;font-weight:700;">RSI&nbsp;({RSI_PERIOD})</div>
                <div style="font-size:20px;font-weight:800;line-height:1.1;
                             margin-top:3px;">{r['rsi']}</div>
                <div style="font-size:10px;color:#64748b;margin-top:2px;
                             word-break:break-word;">{r['rsi_label']}</div>
              </td>
              <td width="25%" style="padding:10px 6px;
                                     border-right:1px solid #e2e8f0;
                                     vertical-align:top;">
                <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;
                             letter-spacing:0.07em;font-weight:700;">Volatility</div>
                <div style="font-size:20px;font-weight:800;line-height:1.1;
                             margin-top:3px;">{r['volatility']:.1f}%</div>
                <div style="font-size:10px;color:#64748b;margin-top:2px;">
                  daily std</div>
              </td>
              <td width="25%" style="padding:10px 6px;
                                     vertical-align:top;">
                <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;
                             letter-spacing:0.07em;font-weight:700;">Volume</div>
                <div style="font-size:20px;font-weight:800;line-height:1.1;
                             margin-top:3px;">{r['vol_ratio']}x</div>
                <div style="font-size:10px;color:#64748b;margin-top:2px;">
                  vs 20d avg</div>
              </td>
            </tr>
          </table>

          <!-- context (each fact on its own line — no mobile overflow) -->
          <div style="padding:10px 14px;background:#f8fafc;
                      border-top:1px solid #f1f5f9;
                      border-bottom:1px solid #f1f5f9;
                      font-size:12px;color:#475569;line-height:1.8;">
            <div>&#128205; <strong>Zone:</strong> {r['zone_label']}</div>
            <div>&#128202; <strong>52W range:</strong>
              {cs}{wk52['low_52w']:,.2f} &ndash; {cs}{wk52['high_52w']:,.2f}
              &nbsp;
              <span style="color:{hcolor};font-weight:600;">
                ({wk52['pct_from_high']:.1f}% from 52W high)
              </span>
            </div>
            <div>&#128200; <strong>Volume:</strong> {r['vol_note']}</div>
          </div>

          <!-- EMA table -->
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="border-collapse:collapse;">
            <tr style="background:#f8fafc;">
              <th style="padding:7px 12px;text-align:left;font-size:10px;
                         color:#94a3b8;font-weight:700;text-transform:uppercase;
                         letter-spacing:0.07em;
                         border-bottom:1px solid #e2e8f0;">EMA</th>
              <th style="padding:7px 12px;text-align:left;font-size:10px;
                         color:#94a3b8;font-weight:700;text-transform:uppercase;
                         letter-spacing:0.07em;
                         border-bottom:1px solid #e2e8f0;">Value</th>
              <th style="padding:7px 12px;text-align:left;font-size:10px;
                         color:#94a3b8;font-weight:700;text-transform:uppercase;
                         letter-spacing:0.07em;
                         border-bottom:1px solid #e2e8f0;">Price vs EMA</th>
            </tr>
            {ema_rows}
          </table>

          <!-- recommendation -->
          <div style="padding:13px 14px;text-align:center;font-weight:700;
                      font-size:15px;background:{rb};color:{rf};">
            {rt}
          </div>

        </div>"""  # end card

    # legend row
    legend_data = [
        ('#059669','90-100: Goated'),
        ('#7c3aed','75-89: Great'),
        ('#d97706','55-74: Good'),
        ('#ea580c','30-54: Caution'),
        ('#dc2626','<30: Avoid'),
    ]
    legend_cells = ''.join(
        f"<td style='padding:3px 5px;text-align:center;white-space:nowrap;"
        f"font-size:10px;color:#cbd5e1;'>"
        f"<span style='display:inline-block;width:8px;height:8px;border-radius:50%;"
        f"background:{c};vertical-align:middle;margin-right:3px;'></span>{lb}</td>"
        for c, lb in legend_data
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Aman's ETF Alert</title>
</head>
<body style="margin:0;padding:10px;background:#f1f5f9;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
             color:#1e293b;">
<div style="max-width:640px;margin:0 auto;">

  <!-- HEADER -->
  <div style="background:#1e1b4b;border-radius:14px 14px 0 0;
              padding:22px 18px 16px;text-align:center;">
    <div style="font-size:20px;font-weight:800;color:#ffffff;">
      Aman's ETF Accumulation Alert</div>
    <div style="font-size:12px;color:#a5b4fc;margin-top:4px;">
      EMA Touch + RSI + Volume &#183; v2.1</div>
    <div style="display:inline-block;background:rgba(255,255,255,0.12);
                padding:4px 14px;border-radius:20px;margin-top:8px;
                font-size:11px;color:#e0e7ff;">
      {today.strftime('%A, %d %B %Y')}
    </div>
  </div>

  <!-- BODY -->
  <div style="background:#ffffff;padding:16px 14px;">
    <table width="100%" cellpadding="0" cellspacing="0">
      {banners}
      <tr><td>{summary}</td></tr>
      <tr><td>{cards}</td></tr>
    </table>
  </div>

  <!-- FOOTER -->
  <div style="background:#1e293b;border-radius:0 0 14px 14px;
              padding:14px 14px;color:#94a3b8;font-size:11px;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>{legend_cells}</tr>
    </table>
    <div style="margin-top:10px;text-align:center;line-height:1.7;color:#64748b;">
      <strong style="color:#cbd5e1;">
        Score = EMA zone (0&ndash;100) + RSI adjustment (&plusmn;15)
      </strong><br>
      Buy: score &ge;{MIN_ZONE_SCORE} &amp; vol &le;{VOLATILITY_MAX}% &amp; RSI &le;65
      &nbsp;&#183;&nbsp; "Touching" = within {TOUCH_THRESHOLD}% of EMA<br>
      &#9888;&#65039; Automated tool &mdash; not financial advice. DYOR.<br>
      <span style="color:#475569;">
        Generated {today.strftime('%Y-%m-%d %H:%M')} IST
      </span>
    </div>
  </div>

</div>
</body>
</html>"""


# ============================================================
# EMAIL
# ============================================================

def send_email(subject, html_body):
    try:
        msg            = MIMEMultipart('alternative')
        msg['From']    = EMAIL_SENDER
        msg['To']      = ', '.join(EMAIL_RECEIVERS)
        msg['Subject'] = subject
        msg.attach(MIMEText(html_body, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as srv:
            srv.login(EMAIL_SENDER, EMAIL_PASSWORD)
            srv.send_message(msg)
        logging.info("Email sent successfully")
    except Exception as e:
        logging.error(f"send_email: {e}")


def is_last_day_of_month():
    today = datetime.now()
    return today.month != (today + timedelta(days=1)).month


# ============================================================
# MAIN
# ============================================================

def main():
    logging.info("=" * 60)
    logging.info(f"Aman's ETF Alert v2.1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)

    last_buys   = load_last_buy_dates()
    is_last_day = is_last_day_of_month()
    reports     = []

    for t in TICKERS:
        logging.info(f"Analysing {t['symbol']} — {t['name']}")
        result = analyse_ticker(t)
        reports.append(result)
        if result.get('buy_signal') and not already_bought_this_month(result['symbol'], last_buys):
            record_buy(result['symbol'], last_buys)

    save_last_buy_dates(last_buys)

    buy_rpts = [r for r in reports if not r.get('error') and r.get('buy_signal')]
    if buy_rpts:
        best = max(buy_rpts, key=lambda r: r['adjusted_score'])
        s    = best['adjusted_score']
        if s >= 90:   prefix = f"Goated Entry — {best['name']}"
        elif s >= 75: prefix = f"Great Entry — {best['name']}"
        else:         prefix = f"{len(buy_rpts)} Buy Signal{'s' if len(buy_rpts)>1 else ''}"
    elif is_last_day:
        prefix = "Month-End Review"
    else:
        prefix = "Daily Scan — No Buy Signals"

    subject = f"{prefix} | ETF Report {datetime.now().strftime('%d %b %Y')}"
    html    = generate_html(reports, last_buys, is_last_day)
    send_email(subject, html)
    logging.info("Done.")


if __name__ == '__main__':
    main()
