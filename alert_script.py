"""
Aman's ETF Accumulation Alert — v3.0
======================================
Goal: Accumulate ₹20,000/month across ETFs at the BEST price corrections
      for aggressive long-term wealth building.

FIXES vs v2.1:
──────────────
1. "Already bought" GitHub Actions bug FIXED:
   - GitHub Actions runner filesystem is EPHEMERAL — last_buy_dates.json
     was wiped every run. Fixed by committing the JSON back to the repo
     via git at the end of each run (see README section below).
   - Also fixed in-run ordering bug: record_buy() now called AFTER
     generate_html() so the current run's signals show ACCUMULATE,
     not "Already bought."

2. MACD added as momentum confirmation (12/26/9 standard).
   - Bullish MACD crossover = +10 score boost (momentum turning up).
   - Bearish crossover = -10 penalty.
   - Strong positive histogram = +5, strong negative = -5.

3. ₹20K smart allocation table added to email:
   - Every email now shows exactly how to split ₹20,000 across
     qualifying ETFs weighted by their adjusted score.
   - Rounded to nearest ₹500 for practical order sizing.

4. Score explanation added per ticker — "why this score" in plain English.

5. Downtrend + Goated zone paradox explained in email copy.

6. "Watch" tickers show % gap to nearest buy zone so you know
   exactly how far the price needs to fall before acting.

GITHUB ACTIONS SETUP — READ THIS:
──────────────────────────────────
For "already bought" tracking to persist between daily runs, add this
to your workflow YAML after the python step:

    - name: Save buy dates
      run: |
        git config user.email "actions@github.com"
        git config user.name "GitHub Actions"
        git add last_buy_dates.json
        git diff --staged --quiet || git commit -m "chore: update buy dates [skip ci]"
        git push
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

Also ensure your workflow checks out with:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
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
# CONFIGURATION — edit only this section
# ============================================================

TICKERS = [
    {'symbol': 'NIFTYBEES.NS',  'name': 'Nifty 50',          'currency': 'INR', 'category': 'largecap'},
    {'symbol': 'JUNIORBEES.NS', 'name': 'Nifty Next 50',     'currency': 'INR', 'category': 'largecap'},
    {'symbol': 'MID150BEES.NS', 'name': 'Nifty Midcap 150',  'currency': 'INR', 'category': 'midcap'},
    {'symbol': 'BANKBEES.NS',   'name': 'Bank Nifty',        'currency': 'INR', 'category': 'sectoral'},
    {'symbol': 'GOLDBEES.NS',   'name': 'Gold',              'currency': 'INR', 'category': 'commodity'},
    {'symbol': 'SILVERBEES.NS', 'name': 'Silver',            'currency': 'INR', 'category': 'commodity'},
    {'symbol': 'HDFCSML250.NS', 'name': 'Smallcap 250',      'currency': 'INR', 'category': 'smallcap'},
    {'symbol': 'SPY',           'name': 'S&P 500',           'currency': 'USD', 'category': 'us'},
    {'symbol': 'QQQ',           'name': 'Nasdaq-100',        'currency': 'USD', 'category': 'us'},
]

# ──────────────────────────────────────────────────────────────
# CATEGORY ALLOCATION CAPS  (as % of monthly budget)
# ──────────────────────────────────────────────────────────────
# Rationale for aggressive wealth building at age 27:
#   • largecap / midcap / smallcap / us = full equity — no cap
#   • sectoral (Bank Nifty) = higher concentration risk — 20% max
#   • commodity (Gold, Silver) = hedge only, NOT a compounder —
#     Gold CAGR ~9% vs equity ~15%. At 27 with a 5-10yr horizon,
#     over-allocating to Gold is the single biggest drag on returns.
#     Hard cap at 10% of budget = max ₹2,000 on a ₹20K budget.
#     Silver is even more volatile with no income — same 10% shared cap.
#
# The scoring system is pure technicals — it will always score Gold
# highly when it's near an EMA. These caps OVERRIDE the score so that
# financial logic governs allocation, not just price momentum.
# ──────────────────────────────────────────────────────────────
CATEGORY_MAX_PCT = {
    'largecap':  1.00,   # up to 100% — core equity, no limit
    'midcap':    1.00,   # up to 100% — aggressive growth, no limit
    'smallcap':  0.40,   # max 40% — high risk, high reward but volatile
    'sectoral':  0.20,   # max 20% — concentrated sector risk
    'commodity': 0.10,   # max 10% — hedge only, NOT a wealth compounder
    'us':        0.30,   # max 30% — great diversification, currency risk
}

MONTHLY_BUDGET    = 20000   # ₹ — your monthly ETF accumulation budget
EMA_PERIODS       = [20, 50, 100, 200]
RSI_PERIOD        = 14
VOLUME_LOOKBACK   = 20
TOUCH_THRESHOLD   = 2.0    # % — within 2% of EMA = "touching"
VOLATILITY_MAX    = 3.0    # % — skip buy recommendation above this
MIN_ZONE_SCORE    = 55     # minimum adjusted score to get a BUY signal
LAST_BUY_FILE     = 'last_buy_dates.json'
DATA_PERIOD       = '2y'   # 2y = ~500 rows, ensures 200 EMA + 52W are accurate

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
        # yfinance >= 0.2.x returns MultiIndex columns for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {df.columns.tolist()}")
        df = df.dropna(subset=['Close'])
        if len(df) < 210:
            logging.warning(f"{symbol}: only {len(df)} rows — indicators may be less accurate")
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
    """Wilder smoothing RSI — standard implementation."""
    delta    = series.diff().dropna()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


def calc_macd(series):
    """
    Standard MACD (12, 26, 9).
    Returns dict with macd_val, signal_val, histogram, prev_histogram,
    trend (positive/negative), and crossover status.
    """
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal

    macd_v    = float(macd.iloc[-1])
    signal_v  = float(signal.iloc[-1])
    hist_v    = float(hist.iloc[-1])
    prev_hist = float(hist.iloc[-2])

    # Price-normalise histogram for comparison across ETFs
    price     = float(series.iloc[-1])
    hist_pct  = (hist_v / price) * 100  # histogram as % of price

    bullish_cross = hist_v > 0 and prev_hist <= 0   # just crossed up
    bearish_cross = hist_v < 0 and prev_hist >= 0   # just crossed down
    momentum      = "positive" if hist_v > 0 else "negative"

    # Score adjustment
    if bullish_cross:
        macd_boost = +10
        macd_note  = "MACD bullish crossover \u2705 (+10)"
    elif bearish_cross:
        macd_boost = -10
        macd_note  = "MACD bearish crossover \u274c (-10)"
    elif hist_pct > 0.05:
        macd_boost = +5
        macd_note  = "MACD momentum positive \u2197\ufe0f (+5)"
    elif hist_pct < -0.05:
        macd_boost = -5
        macd_note  = "MACD momentum negative \u2198\ufe0f (-5)"
    else:
        macd_boost = 0
        macd_note  = "MACD neutral (\u00b10)"

    return {
        'macd':          round(macd_v, 3),
        'signal':        round(signal_v, 3),
        'histogram':     round(hist_v, 3),
        'histogram_pct': round(hist_pct, 4),
        'momentum':      momentum,
        'bullish_cross': bullish_cross,
        'bearish_cross': bearish_cross,
        'boost':         macd_boost,
        'note':          macd_note,
    }


def calc_volume_ratio(df):
    """Today's volume vs 20-day average. Returns 0.0 if insufficient data."""
    vol = df['Volume'].replace(0, np.nan).dropna()
    if len(vol) < VOLUME_LOOKBACK + 1:
        return 0.0
    avg = float(vol.iloc[-(VOLUME_LOOKBACK+1):-1].mean())
    return round(float(vol.iloc[-1]) / avg, 2) if avg > 0 else 0.0


def calc_52w_position(df):
    """
    52W high/low — uses min_periods=1 so it works on any data length.
    With DATA_PERIOD='2y' (~500 rows) the 252-bar window is always full,
    but min_periods=1 is retained as a defensive guard.
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

def calc_zone_score(ema_diffs):
    """
    EMA-based accumulation score 0–100.
    Deeper below key EMAs = better price = higher score.
    Above all EMAs = extended/overbought = low score.
    """
    for i in [3, 2, 1, 0]:          # check 200, 100, 50, 20 EMA in order
        d = ema_diffs[i]
        if abs(d) <= TOUCH_THRESHOLD:
            scores = [60, 75, 90, 100]
            labels = [
                "Touching 20 EMA — good pullback entry",
                "Touching 50 EMA — great correction entry",
                "Touching 100 EMA — excellent dip entry",
                "Touching 200 EMA — rare, goated entry",
            ]
            return scores[i], labels[i]
        elif d < -TOUCH_THRESHOLD:
            scores = [65, 80, 92, 100]
            labels = [
                "Below 20 EMA — solid pullback",
                "Below 50 EMA — deep correction",
                "Below 100 EMA — major dip",
                "Below 200 EMA — bear zone / max opportunity",
            ]
            return scores[i], labels[i]

    # Above all EMAs
    pct = ema_diffs[0]   # % above 20 EMA
    if pct <= 5:   return 45, f"Extended +{pct:.1f}% above 20 EMA — wait for pullback"
    if pct <= 10:  return 25, f"Overbought +{pct:.1f}% above 20 EMA — patience needed"
    return 10, f"Very overbought +{pct:.1f}% above 20 EMA — do not buy"


def interpret_rsi(rsi):
    if rsi <= 30:  return "Oversold \u2705 (RSI \u226430)", +15
    if rsi <= 45:  return "Mildly oversold \u2705 (RSI 30\u201345)", +8
    if rsi <= 60:  return "Neutral (RSI 45\u201360)", 0
    if rsi <= 70:  return "Mildly overbought \u26a0\ufe0f (RSI 60\u201370)", -8
    return "Overbought \u274c (RSI >70)", -15


def gap_to_buy_zone(ema_diffs, last_close, ema_vals):
    """
    For tickers NOT yet in a buy zone, calculate how far (in %) the price
    needs to fall to reach the nearest qualifying EMA (50 or 200).
    """
    # Find nearest EMA below current price
    gaps = []
    for i, (diff, ema) in enumerate(zip(ema_diffs, ema_vals)):
        if diff > TOUCH_THRESHOLD:   # price is above this EMA
            gap_pct = diff           # positive = % above
            gaps.append((gap_pct, EMA_PERIODS[i], ema))
    if not gaps:
        return None
    gaps.sort()
    pct, period, ema = gaps[0]
    # Price needs to drop 'pct'% to reach this EMA
    return {'pct': round(pct, 1), 'ema': period, 'price': round(ema, 2)}


# ============================================================
# FULL TICKER ANALYSIS
# ============================================================

def analyse_ticker(ticker_info):
    symbol   = ticker_info['symbol']
    name     = ticker_info['name']
    currency = ticker_info['currency']
    category = ticker_info.get('category', '')

    df = fetch_ohlcv(symbol)
    if df.empty:
        return {'symbol': symbol, 'name': name, 'currency': currency,
                'category': category, 'error': 'Failed to fetch price data'}
    try:
        last  = float(df['Close'].iloc[-1])
        ldate = df.index[-1]
        days  = (datetime.now().date() - ldate.date()).days

        ema_vals  = [calc_ema(df['Close'], p) for p in EMA_PERIODS]
        ema_diffs = [((last - e) / e) * 100 for e in ema_vals]

        rsi              = calc_rsi(df['Close'])
        rsi_label, rb    = interpret_rsi(rsi)
        macd_data        = calc_macd(df['Close'])
        volatility       = float(df['Close'].pct_change().std() * 100)
        vol_ratio        = calc_volume_ratio(df)
        wk52             = calc_52w_position(df)
        zone_score, zlbl = calc_zone_score(ema_diffs)
        trend            = "Uptrend" if last > ema_vals[3] else "Downtrend"

        # Composite adjusted score: zone + RSI + MACD
        adj = max(0, min(100, zone_score + rb + macd_data['boost']))

        buy_signal = (
            adj >= MIN_ZONE_SCORE
            and volatility <= VOLATILITY_MAX
            and rsi <= 65
        )

        # Volume note
        if vol_ratio >= 1.5:   vnote = f"High volume ({vol_ratio}x avg) \u2014 strong confirmation"
        elif vol_ratio >= 1.0: vnote = f"Above-average ({vol_ratio}x avg)"
        elif vol_ratio > 0:    vnote = f"Below-average ({vol_ratio}x avg) \u2014 weak confirmation"
        else:                  vnote = "Volume data unavailable"

        # Gap to buy zone (for non-buy tickers)
        gap = gap_to_buy_zone(ema_diffs, last, ema_vals) if not buy_signal else None

        # Human-readable score explanation
        score_why = []
        score_why.append(f"EMA zone: {zone_score}")
        score_why.append(f"RSI adj: {'+' if rb>=0 else ''}{rb}")
        score_why.append(f"MACD adj: {'+' if macd_data['boost']>=0 else ''}{macd_data['boost']}")
        score_why.append(f"= {adj} total")

        return {
            'symbol': symbol, 'name': name, 'currency': currency, 'category': category,
            'last_close':     round(last, 2),
            'last_date':      ldate.strftime('%d %b %Y'),
            'stale_warning':  days > 1, 'days_old': days,
            'ema_vals':       [round(v, 2) for v in ema_vals],
            'ema_diffs':      [round(d, 2) for d in ema_diffs],
            'rsi':            round(rsi, 1), 'rsi_label': rsi_label, 'rsi_boost': rb,
            'macd':           macd_data,
            'volatility':     round(volatility, 2),
            'vol_ratio':      vol_ratio, 'vol_note': vnote,
            'zone_score':     zone_score, 'adjusted_score': adj,
            'zone_label':     zlbl, 'trend': trend, 'wk52': wk52,
            'score_why':      ' | '.join(score_why),
            'gap_to_buy':     gap,
            'buy_signal':     buy_signal,
            'error':          None,
        }
    except Exception as e:
        logging.error(f"analyse_ticker({symbol}): {e}")
        return {'symbol': symbol, 'name': name, 'currency': currency,
                'category': category, 'error': str(e)}


# ============================================================
# ₹20K ALLOCATION ENGINE
# ============================================================

def calc_allocation(reports, budget=MONTHLY_BUDGET):
    """
    Split the monthly budget across qualifying BUY tickers.

    Step 1 — Score-weighted raw allocation (same as before).
    Step 2 — Apply CATEGORY_MAX_PCT caps so commodities / sectorals
             can never crowd out core equity regardless of their EMA score.
             Example: Gold scores 100 near its 200 EMA, but commodity cap
             is 10%, so Gold gets max ₹2,000 on a ₹20K budget.
    Step 3 — Freed-up budget is redistributed to uncapped equity ETFs
             proportionally, so ₹20K is always fully deployed.
    Step 4 — Round to nearest ₹500. Remainder goes to top equity ticker.

    Category caps (CATEGORY_MAX_PCT) reflect long-term return profiles:
      Commodity (Gold/Silver) ~9% CAGR vs equity ~15% CAGR over 20 years.
      At 27, over-weighting Gold is the single biggest drag on final corpus.
    """
    buys = [r for r in reports if not r.get('error') and r.get('buy_signal')]
    if not buys:
        return {}

    # ── build a lookup: symbol → category ───────────────────
    cat_map = {t['symbol']: t.get('category', 'largecap') for t in TICKERS}

    # ── step 1: raw score-weighted amounts ───────────────────
    total_score = sum(r['adjusted_score'] for r in buys)
    raw = {r['symbol']: (r['adjusted_score'] / total_score) * budget for r in buys}

    # ── step 2: apply per-category caps ──────────────────────
    capped   = {}
    overflow = 0.0          # budget freed from capped tickers
    for sym, amt in raw.items():
        cat     = cat_map.get(sym, 'largecap')
        max_amt = CATEGORY_MAX_PCT.get(cat, 1.0) * budget
        if amt > max_amt:
            overflow += amt - max_amt
            capped[sym] = max_amt
        else:
            capped[sym] = amt

    # ── step 3: redistribute overflow to uncapped equity ETFs ─
    # "uncapped" = tickers whose category max is 100% (core equity)
    # If no uncapped equity exists, overflow stays undeployed — it is
    # financially correct NOT to force-buy commodities over their cap.
    if overflow > 0:
        uncapped_syms = [
            sym for sym, amt in capped.items()
            if CATEGORY_MAX_PCT.get(cat_map.get(sym, 'largecap'), 1.0) >= 1.0
        ]
        if uncapped_syms:
            uncapped_score_total = sum(
                r['adjusted_score'] for r in buys if r['symbol'] in uncapped_syms
            )
            for r in buys:
                if r['symbol'] in uncapped_syms and uncapped_score_total > 0:
                    extra = (r['adjusted_score'] / uncapped_score_total) * overflow
                    capped[r['symbol']] += extra
        # else: no equity ETFs in buy zone — keep overflow as cash this month

    # ── step 4: round to nearest ₹500, fix rounding gap ──────
    alloc = {sym: max(500, round(amt / 500) * 500) for sym, amt in capped.items()}
    # Only redistribute rounding gap if there are uncapped equity ETFs.
    # If only commodity/sectoral ETFs qualified, do NOT inflate their allocation
    # to fill the budget — it is correct to deploy less than ₹20K this month.
    equity_buys = [r for r in buys
                   if CATEGORY_MAX_PCT.get(cat_map.get(r['symbol'], 'largecap'), 1.0) >= 1.0]
    if equity_buys:
        diff = budget - sum(alloc.values())
        if diff != 0:
            top = max(equity_buys, key=lambda r: r['adjusted_score'])['symbol']
            alloc[top] = max(500, alloc[top] + diff)

    return alloc


# ============================================================
# BUY DATE TRACKING
# ============================================================

def load_last_buy_dates():
    """
    Load from last_buy_dates.json in the repo.
    In GitHub Actions this file must be committed back at end of run
    (see workflow YAML in module docstring) for persistence between runs.
    """
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
        logging.info(f"Saved {LAST_BUY_FILE}")
    except Exception as e:
        logging.error(f"save_last_buy_dates: {e}")


def already_bought_this_month(symbol, last_buys):
    return last_buys.get(symbol, '')[:7] == datetime.now().strftime('%Y-%m')


def record_buy(symbol, last_buys):
    last_buys[symbol] = datetime.now().strftime('%Y-%m-%d')


# ============================================================
# HTML HELPERS
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
    if s >= 90: return 'Goated'
    if s >= 75: return 'Great Entry'
    if s >= 55: return 'Good Entry'
    if s >= 30: return 'Caution'
    return 'Avoid'


# ============================================================
# HTML EMAIL — mobile-first, table layout, allocation table
# ============================================================

def generate_html(reports, last_buys, is_last_day, allocation):
    """
    FIX: last_buys passed in is the state BEFORE record_buy() runs.
    So this correctly shows ACCUMULATE on the day of detection,
    and "Already bought" only on subsequent days in the same month.
    """
    today      = datetime.now()
    good       = [r for r in reports if not r.get('error')]
    buy_rpts   = [r for r in good if r.get('buy_signal')]
    best_score = max((r['adjusted_score'] for r in good), default=0)

    # ── banners ──────────────────────────────────────────────
    banners = ''
    if is_last_day:
        banners += """<tr><td style="padding:0 0 12px 0;">
          <div style="background:#fffbeb;border-left:4px solid #f59e0b;padding:12px 14px;
                      border-radius:6px;font-size:14px;line-height:1.5;color:#78350f;">
            &#128197; <strong>Last trading day of the month.</strong>
            If you haven&rsquo;t accumulated yet this month, review
            <strong>BUY</strong> signals below (score &ge;55).
          </div></td></tr>"""

    stale = [r['symbol'] for r in good if r.get('stale_warning')]
    if stale:
        banners += f"""<tr><td style="padding:0 0 12px 0;">
          <div style="background:#fef2f2;border-left:4px solid #ef4444;padding:12px 14px;
                      border-radius:6px;font-size:13px;line-height:1.5;color:#7f1d1d;">
            &#9888;&#65039; <strong>Weekend/holiday data:</strong>
            {', '.join(stale)} prices are from a previous session.
            Confirm live prices on NSE/BSE before placing orders.
          </div></td></tr>"""

    # ── summary bar ──────────────────────────────────────────
    summary = f"""
    <table width="100%" cellpadding="0" cellspacing="4" style="margin-bottom:18px;">
      <tr>
        <td width="25%"><div style="background:#f0f4ff;border-radius:10px;
            padding:12px 6px;text-align:center;">
          <div style="font-size:24px;font-weight:800;color:#4f46e5;">{len(good)}</div>
          <div style="font-size:11px;color:#6366f1;margin-top:2px;">Scanned</div>
        </div></td>
        <td width="25%"><div style="background:#d1fae5;border-radius:10px;
            padding:12px 6px;text-align:center;">
          <div style="font-size:24px;font-weight:800;color:#059669;">{len(buy_rpts)}</div>
          <div style="font-size:11px;color:#065f46;margin-top:2px;">Buy Signals</div>
        </div></td>
        <td width="25%"><div style="background:#f5f3ff;border-radius:10px;
            padding:12px 6px;text-align:center;">
          <div style="font-size:24px;font-weight:800;color:#7c3aed;">{best_score}</div>
          <div style="font-size:11px;color:#7c3aed;margin-top:2px;">Best Score</div>
        </div></td>
        <td width="25%"><div style="background:#f0f9ff;border-radius:10px;
            padding:12px 6px;text-align:center;">
          <div style="font-size:16px;font-weight:800;color:#0ea5e9;">
            &#8377;{MONTHLY_BUDGET//1000}K</div>
          <div style="font-size:11px;color:#0284c7;margin-top:2px;">Budget</div>
        </div></td>
      </tr>
    </table>"""

    # ── ₹20K allocation table ────────────────────────────────
    alloc_section = ''
    if allocation:
        rows = ''
        for r in buy_rpts:
            amt = allocation.get(r['symbol'], 0)
            sc  = _score_color(r['adjusted_score'])
            cs  = _cs(r['currency'])
            rows += f"""
            <tr style="border-bottom:1px solid #f1f5f9;">
              <td style="padding:10px 12px;font-size:13px;font-weight:600;
                         color:#0f172a;">{r['name']}</td>
              <td style="padding:10px 12px;font-size:12px;color:#64748b;">
                {r['symbol']}</td>
              <td style="padding:10px 12px;text-align:center;">
                <span style="background:{sc}18;color:{sc};padding:2px 8px;
                             border-radius:4px;font-size:12px;font-weight:700;">
                  {r['adjusted_score']}</span></td>
              <td style="padding:10px 12px;font-size:14px;font-weight:800;
                         color:#059669;text-align:right;">
                &#8377;{amt:,}</td>
            </tr>"""

        alloc_section = f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;
                    overflow:hidden;margin-bottom:20px;">
          <div style="background:#059669;padding:12px 16px;">
            <div style="font-size:15px;font-weight:800;color:#ffffff;">
              &#128181; This Month&rsquo;s &#8377;{MONTHLY_BUDGET:,} Allocation Plan
            </div>
            <div style="font-size:11px;color:#a7f3d0;margin-top:2px;">
              Weighted by adjusted score &mdash; deploy at current market price
            </div>
          </div>
          <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
            <tr style="background:#f0fdf4;">
              <th style="padding:8px 12px;text-align:left;font-size:11px;
                         color:#64748b;font-weight:700;text-transform:uppercase;
                         letter-spacing:0.06em;border-bottom:1px solid #bbf7d0;">ETF</th>
              <th style="padding:8px 12px;text-align:left;font-size:11px;
                         color:#64748b;font-weight:700;text-transform:uppercase;
                         letter-spacing:0.06em;border-bottom:1px solid #bbf7d0;">Ticker</th>
              <th style="padding:8px 12px;text-align:center;font-size:11px;
                         color:#64748b;font-weight:700;text-transform:uppercase;
                         letter-spacing:0.06em;border-bottom:1px solid #bbf7d0;">Score</th>
              <th style="padding:8px 12px;text-align:right;font-size:11px;
                         color:#64748b;font-weight:700;text-transform:uppercase;
                         letter-spacing:0.06em;border-bottom:1px solid #bbf7d0;">Allocate</th>
            </tr>
            {rows}
          </table>
          <div style="padding:10px 14px;background:#ecfdf5;font-size:12px;color:#065f46;">
            &#128161; Place a single lump-sum order per ETF via Dhan/Zerodha.
            If price moves &gt;2% before you order, re-check the score first.
          </div>
        </div>"""
    else:
        alloc_section = """
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:16px;margin-bottom:20px;text-align:center;color:#64748b;
                    font-size:13px;">
          &#9203; No ETFs in a buy zone today. Hold cash, watch for pullbacks.
          Scroll down to see how far each ETF is from its buy zone.
        </div>"""

    # ── ticker cards ─────────────────────────────────────────
    cards = ''
    for r in reports:
        if r.get('error'):
            cards += f"""
            <div style="background:#fef2f2;border-radius:10px;padding:14px;
                        margin-bottom:16px;color:#991b1b;font-size:13px;">
              <strong>&#9888; {r['symbol']}</strong>: {r['error']}
            </div>"""
            continue

        cs      = _cs(r['currency'])
        sc      = _score_color(r['adjusted_score'])
        sl      = _score_label(r['adjusted_score'])
        wk52    = r['wk52']
        macd    = r['macd']
        already = already_bought_this_month(r['symbol'], last_buys)

        # Recommendation — note: already uses last_buys from BEFORE record_buy()
        if r['buy_signal'] and not already:
            rb2, rf, rt = '#d1fae5','#065f46','&#9989; ACCUMULATE NOW'
        elif r['buy_signal'] and already:
            rb2, rf, rt = '#e0f2fe','#075985','&#128203; Already accumulated this month'
        elif r['adjusted_score'] >= 45:
            gap   = r.get('gap_to_buy')
            gtext = f" (needs -{gap['pct']:.1f}% to reach {gap['ema']} EMA)" if gap else ""
            rb2, rf, rt = '#fef3c7','#92400e', f'&#9203; Watch{gtext}'
        else:
            gap   = r.get('gap_to_buy')
            gtext = f" (needs -{gap['pct']:.1f}% drop)" if gap else ""
            rb2, rf, rt = '#fee2e2','#991b1b', f'&#128683; Avoid{gtext}'

        tbg = '#d1fae5' if r['trend']=='Uptrend' else '#fee2e2'
        tfg = '#065f46' if r['trend']=='Uptrend' else '#991b1b'
        ts  = '&#8679;' if r['trend']=='Uptrend' else '&#8681;'

        # Trend + score paradox explanation
        trend_note = ''
        if r['trend'] == 'Downtrend' and r['adjusted_score'] >= 90:
            trend_note = (
                '<div style="background:#fef3c7;border-radius:5px;padding:6px 10px;'
                'font-size:11px;color:#92400e;margin-top:6px;line-height:1.5;">'
                '&#128161; Price is in a downtrend (below 200 EMA) AND near the 200 EMA '
                '&mdash; this is the classic maximum-opportunity zone. '
                'The downtrend is the reason the score is high. Accumulate with a '
                'long-term view; consider splitting across 2&ndash;3 months.</div>'
            )

        stag = (f"&nbsp;<span style='background:#fef3c7;color:#92400e;"
                f"padding:1px 6px;border-radius:4px;font-size:10px;'>"
                f"Data: {r['last_date']}</span>"
                if r.get('stale_warning') else '')

        hcolor = '#059669' if wk52['pct_from_high'] < -20 else '#64748b'

        # MACD badge
        macd_color = '#059669' if macd['boost'] > 0 else ('#dc2626' if macd['boost'] < 0 else '#64748b')
        macd_bg    = '#ecfdf5' if macd['boost'] > 0 else ('#fef2f2' if macd['boost'] < 0 else '#f8fafc')

        ema_rows = ''
        for period, val, diff in zip(EMA_PERIODS, r['ema_vals'], r['ema_diffs']):
            dc   = '#059669' if diff <= 0 else '#dc2626'
            sign = '+' if diff > 0 else ''
            ema_rows += f"""
            <tr style="border-bottom:1px solid #f8fafc;">
              <td style="padding:8px 12px;font-size:13px;color:#475569;
                         white-space:nowrap;">{period}&nbsp;EMA</td>
              <td style="padding:8px 12px;font-size:13px;white-space:nowrap;">
                {cs}{val:,.2f}</td>
              <td style="padding:8px 12px;font-size:13px;font-weight:700;
                         color:{dc};white-space:nowrap;">{sign}{diff:.1f}%</td>
            </tr>"""

        cards += f"""
        <div style="background:#ffffff;border-radius:12px;overflow:hidden;
                    margin-bottom:18px;border:1px solid #e2e8f0;
                    border-top:4px solid {sc};">

          <!-- HEADER -->
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="padding:13px 14px 10px;">
            <tr>
              <td style="vertical-align:top;">
                <div style="font-weight:800;font-size:16px;color:#0f172a;">
                  {r['name']}</div>
                <div style="font-size:11px;color:#94a3b8;margin-top:2px;">
                  {r['symbol']}{stag}</div>
                {trend_note}
              </td>
              <td style="text-align:right;vertical-align:top;
                         padding-left:8px;white-space:nowrap;">
                <div style="font-size:20px;font-weight:800;color:#0f172a;">
                  {cs}{r['last_close']:,.2f}</div>
                <div style="margin-top:4px;">
                  <span style="background:{tbg};color:{tfg};padding:2px 8px;
                               border-radius:4px;font-size:10px;font-weight:700;">
                    {ts}&nbsp;{r['trend'].upper()}</span>
                </div>
              </td>
            </tr>
          </table>
          <div style="border-top:1px solid #f1f5f9;"></div>

          <!-- STAT BAR -->
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="background:#f8fafc;">
            <tr>
              <td width="25%" style="padding:10px 4px 10px 12px;
                                     border-right:1px solid #e8edf2;vertical-align:top;">
                <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;
                             letter-spacing:0.07em;font-weight:700;">Score</div>
                <div style="font-size:20px;font-weight:800;color:{sc};
                             line-height:1.1;margin-top:3px;">{r['adjusted_score']}</div>
                <div style="font-size:9px;color:{sc};font-weight:600;margin-top:2px;">
                  {sl}</div>
                <div style="font-size:9px;color:#94a3b8;margin-top:3px;
                             word-break:break-word;">{r['score_why']}</div>
              </td>
              <td width="25%" style="padding:10px 4px;
                                     border-right:1px solid #e8edf2;vertical-align:top;">
                <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;
                             letter-spacing:0.07em;font-weight:700;">RSI&nbsp;(14)</div>
                <div style="font-size:20px;font-weight:800;line-height:1.1;
                             margin-top:3px;">{r['rsi']}</div>
                <div style="font-size:9px;color:#64748b;margin-top:2px;
                             word-break:break-word;">{r['rsi_label']}</div>
              </td>
              <td width="25%" style="padding:10px 4px;
                                     border-right:1px solid #e8edf2;vertical-align:top;">
                <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;
                             letter-spacing:0.07em;font-weight:700;">Volatility</div>
                <div style="font-size:20px;font-weight:800;line-height:1.1;
                             margin-top:3px;">{r['volatility']:.1f}%</div>
                <div style="font-size:9px;color:#64748b;margin-top:2px;">daily std</div>
              </td>
              <td width="25%" style="padding:10px 4px;vertical-align:top;">
                <div style="font-size:9px;color:#94a3b8;text-transform:uppercase;
                             letter-spacing:0.07em;font-weight:700;">Volume</div>
                <div style="font-size:20px;font-weight:800;line-height:1.1;
                             margin-top:3px;">{r['vol_ratio']}x</div>
                <div style="font-size:9px;color:#64748b;margin-top:2px;">vs 20d avg</div>
              </td>
            </tr>
          </table>

          <!-- MACD row -->
          <div style="padding:8px 14px;background:{macd_bg};
                      border-top:1px solid #f1f5f9;border-bottom:1px solid #f1f5f9;
                      font-size:12px;color:{macd_color};font-weight:600;">
            &#128200; MACD: {macd['note']}
            &nbsp;&nbsp;
            <span style="font-weight:400;color:#64748b;">
              Histogram: {'+' if macd['histogram']>0 else ''}{macd['histogram']:.3f}
              &nbsp;|&nbsp; Momentum: {macd['momentum']}
            </span>
          </div>

          <!-- CONTEXT -->
          <div style="padding:10px 14px;background:#f8fafc;
                      border-bottom:1px solid #f1f5f9;
                      font-size:12px;color:#475569;line-height:1.8;">
            <div>&#128205; <strong>Zone:</strong> {r['zone_label']}</div>
            <div>&#128202; <strong>52W range:</strong>
              {cs}{wk52['low_52w']:,.2f}&ndash;{cs}{wk52['high_52w']:,.2f}
              &nbsp;<span style="color:{hcolor};font-weight:600;">
                ({wk52['pct_from_high']:.1f}% from 52W high)
              </span>
            </div>
            <div>&#128346; <strong>Volume:</strong> {r['vol_note']}</div>
          </div>

          <!-- EMA TABLE -->
          <table width="100%" cellpadding="0" cellspacing="0"
                 style="border-collapse:collapse;">
            <tr style="background:#f8fafc;">
              <th style="padding:7px 12px;text-align:left;font-size:10px;
                         color:#94a3b8;font-weight:700;text-transform:uppercase;
                         border-bottom:1px solid #e2e8f0;">EMA</th>
              <th style="padding:7px 12px;text-align:left;font-size:10px;
                         color:#94a3b8;font-weight:700;text-transform:uppercase;
                         border-bottom:1px solid #e2e8f0;">Value</th>
              <th style="padding:7px 12px;text-align:left;font-size:10px;
                         color:#94a3b8;font-weight:700;text-transform:uppercase;
                         border-bottom:1px solid #e2e8f0;">Price vs EMA</th>
            </tr>
            {ema_rows}
          </table>

          <!-- RECOMMENDATION -->
          <div style="padding:13px 14px;text-align:center;font-weight:700;
                      font-size:15px;background:{rb2};color:{rf};">
            {rt}
          </div>

        </div>"""  # end card

    # ── legend ───────────────────────────────────────────────
    legend_data = [
        ('#059669','90-100: Goated'),
        ('#7c3aed','75-89: Great'),
        ('#d97706','55-74: Good'),
        ('#ea580c','30-54: Watch'),
        ('#dc2626','<30: Avoid'),
    ]
    legend_cells = ''.join(
        f"<td style='padding:3px 4px;text-align:center;font-size:10px;"
        f"color:#cbd5e1;white-space:nowrap;'>"
        f"<span style='display:inline-block;width:8px;height:8px;border-radius:50%;"
        f"background:{c};vertical-align:middle;margin-right:3px;'></span>{lb}</td>"
        for c, lb in legend_data
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Aman&rsquo;s ETF Alert</title>
</head>
<body style="margin:0;padding:10px;background:#f1f5f9;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
             color:#1e293b;">
<div style="max-width:640px;margin:0 auto;">

  <div style="background:#1e1b4b;border-radius:14px 14px 0 0;
              padding:22px 18px 16px;text-align:center;">
    <div style="font-size:20px;font-weight:800;color:#fff;">
      Aman&rsquo;s ETF Accumulation Alert</div>
    <div style="font-size:12px;color:#a5b4fc;margin-top:4px;">
      EMA + RSI + MACD + Volume &nbsp;&#183;&nbsp; v3.0</div>
    <div style="display:inline-block;background:rgba(255,255,255,0.12);
                padding:4px 14px;border-radius:20px;margin-top:8px;
                font-size:11px;color:#e0e7ff;">
      {today.strftime('%A, %d %B %Y')}
    </div>
  </div>

  <div style="background:#ffffff;padding:16px 14px;">
    <table width="100%" cellpadding="0" cellspacing="0">
      {banners}
      <tr><td>{summary}</td></tr>
      <tr><td>{alloc_section}</td></tr>
      <tr><td>{cards}</td></tr>
    </table>
  </div>

  <div style="background:#1e293b;border-radius:0 0 14px 14px;
              padding:14px;color:#94a3b8;font-size:11px;">
    <table width="100%" cellpadding="0" cellspacing="2">
      <tr>{legend_cells}</tr>
    </table>
    <div style="margin-top:10px;text-align:center;line-height:1.8;color:#64748b;">
      <strong style="color:#cbd5e1;">
        Score = EMA zone (0&ndash;100) + RSI (&plusmn;15) + MACD (&plusmn;10)
      </strong><br>
      BUY requires: score &ge;{MIN_ZONE_SCORE} &amp;&amp; volatility &le;{VOLATILITY_MAX}% &amp;&amp; RSI &le;65<br>
      &ldquo;Touching&rdquo; = price within {TOUCH_THRESHOLD}% of EMA &nbsp;&#183;&nbsp; Data: 2y daily<br>
      &#9888;&#65039; Automated tool &mdash; not financial advice. Always DYOR.<br>
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
    logging.info(f"ETF Alert v3.0 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)

    # Load BEFORE analysis — this is the persistent state from the last commit
    last_buys   = load_last_buy_dates()
    is_last_day = is_last_day_of_month()
    reports     = []

    for t in TICKERS:
        logging.info(f"  {t['symbol']} — {t['name']}")
        result = analyse_ticker(t)
        reports.append(result)

    # Calculate allocation BEFORE updating buy dates
    allocation = calc_allocation(reports)

    # Generate HTML BEFORE record_buy() so "already bought" shows correctly:
    # - Tickers bought in previous days this month → "Already accumulated"
    # - Tickers with buy signal today (first detection) → "ACCUMULATE NOW"
    html = generate_html(reports, last_buys, is_last_day, allocation)

    # NOW update buy dates (AFTER html generation)
    for r in reports:
        if r.get('buy_signal') and not already_bought_this_month(r['symbol'], last_buys):
            record_buy(r['symbol'], last_buys)
    save_last_buy_dates(last_buys)
    # NOTE: For persistence across GitHub Actions runs, the workflow YAML must
    # commit last_buy_dates.json back to the repo after this script runs.
    # See the docstring at the top of this file for the exact workflow snippet.

    # Build subject
    buy_rpts = [r for r in reports if not r.get('error') and r.get('buy_signal')]
    if buy_rpts:
        best = max(buy_rpts, key=lambda r: r['adjusted_score'])
        s    = best['adjusted_score']
        if s >= 90:   prefix = f"Goated Entry \u2014 {best['name']}"
        elif s >= 75: prefix = f"Great Entry \u2014 {best['name']}"
        else:         prefix = f"{len(buy_rpts)} Buy Signal{'s' if len(buy_rpts)>1 else ''}"
        prefix += f" | Deploy \u20b9{sum(allocation.values()):,}"
    elif is_last_day:
        prefix = "Month-End Review \u2014 No Buy Zones"
    else:
        prefix = "Daily Scan \u2014 No Buy Signals Today"

    subject = f"{prefix} | ETF Report {datetime.now().strftime('%d %b %Y')}"
    send_email(subject, html)
    logging.info("Done.")


if __name__ == '__main__':
    main()
