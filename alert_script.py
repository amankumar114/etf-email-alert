import yfinance as yf
import pandas as pd
import smtplib
import os
import json
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# === CONFIG ===
TICKERS = [
    'NIFTYBEES.NS',        # Nifty 50 ETF
    'BANKBEES.NS',         # Banking Sector ETF
    'GOLDBEES.NS',         # Gold ETF
    'AUTOBEES.NS',         # Nippon India Nifty Auto ETF
    'ITBEES.NS',           # Nippon India Nifty IT ETF
    'JUNIORBEES.NS',       # Nippon India Nifty FMCG ETF
    'PHARMABEES.NS',       # Nippon India Nifty Pharma ETF
    'SPY',                 # S&P 500 ETF
    'QQQ'                  # NASDAQ-100 ETF
]
EMA_DAYS = [20, 50, 100, 200]  # Key EMAs to track
VOLATILITY_PERIOD = 21          # 1-month volatility
VOLATILITY_THRESHOLD = 2.5      # %
LAST_BUY_FILE = 'last_buy_dates.json'

EMAIL_SENDER = os.getenv("EMAIL")
EMAIL_PASSWORD = os.getenv("PASS")
EMAIL_RECEIVERS = os.getenv('EMAIL_RECEIVER', '').split(',')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etf_accumulation.log'),
        logging.StreamHandler()
    ]
)

def fetch_data(ticker):
    """Fetch data with sufficient history for EMA calculations"""
    try:
        # Calculate required days (200 EMA + buffer)
        required_days = int(EMA_DAYS[-1] * 1.5)
        df = yf.download(ticker, period=f'{required_days}d', interval='1d', auto_adjust=True, progress=False)
        
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Validate data length
        if len(df) < EMA_DAYS[-1]:
            raise ValueError(f"Insufficient data ({len(df)} points) for EMA calculations")
            
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def safe_float_conversion(value):
    """Safely convert values to float handling NaNs and infinities"""
    try:
        if pd.isna(value) or np.isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def calculate_percentage_diff(current, ma):
    """Calculate percentage difference with error handling"""
    if ma == 0 or math.isnan(ma) or math.isinf(ma):
        return 0.0
    return ((current - ma) / ma) * 100

def calculate_zone_score(ema_diffs):
    """
    Calculate accumulation zone score with stricter criteria:
    - Touching 200 EMA: Goated (Score 100)
    - Below 100 EMA: Excellent (Score 90)
    - Below 50 EMA: Great (Score 75)
    - Below 20 EMA: Good (Score 60)
    - Above EMAs get progressively lower scores
    """
    # Score based on position relative to EMAs
    if ema_diffs[3] < 0:   # Below 200 EMA
        if ema_diffs[2] < 0:  # Below 100 EMA
            if ema_diffs[1] < 0:  # Below 50 EMA
                if ema_diffs[0] < 0:  # Below 20 EMA
                    return 60, "➖ Good Price (Below 20 EMA)"
                return 75, "👍 Great Price (Below 50 EMA)"
            return 90, "⭐ Excellent Price (Below 100 EMA)"
        return 100, "🐐 Goated Price (Below 200 EMA)"
    
    # Above EMAs - less favorable
    if ema_diffs[0] > 0:   # Above 20 EMA
        if ema_diffs[1] > 0:  # Above 50 EMA
            if ema_diffs[2] > 0:  # Above 100 EMA
                return 0, "⛔ Very Expensive (Above all EMAs)"
            return 10, "⛔ Expensive (Above 50 EMA)"
        return 20, "⚠️ High Zone (Above 20 EMA)"
    return 30, "⚠️ Caution Zone (Near 20 EMA)"

def calculate_volatility(df):
    """Calculate realistic volatility using log returns"""
    try:
        returns = np.log(df['Close'] / df['Close'].shift(1))
        return returns.std() * np.sqrt(252) * 100  # Annualized percentage
    except:
        return 0.0

def calculate_signal(daily_df):
    try:
        if daily_df.empty:
            raise ValueError("Empty DataFrame received")
        
        # Calculate all EMAs
        ema_values = []
        for ema in EMA_DAYS:
            # Ensure enough data points for EMA calculation
            if len(daily_df) >= ema:
                ema_val = daily_df['Close'].ewm(span=ema, adjust=False).mean().iloc[-1]
            else:
                ema_val = daily_df['Close'].mean()  # Fallback
            ema_values.append(safe_float_conversion(ema_val))
        
        last_close = safe_float_conversion(daily_df['Close'].iloc[-1])
        
        # Calculate percentage differences from EMAs
        ema_diffs = [
            calculate_percentage_diff(last_close, ema)
            for ema in ema_values
        ]
        
        # Calculate volatility using realistic method
        volatility = calculate_volatility(daily_df.tail(VOLATILITY_PERIOD * 3))  # 3 months data
        
        # Calculate zone score and classification
        zone_score, zone_class = calculate_zone_score(ema_diffs)

        # Buy signal conditions (stricter criteria)
        buy_signal = (
            zone_score >= 75 and  # At least "Great" zone
            volatility <= VOLATILITY_THRESHOLD and
            last_close < min(ema_values)  # Price below all key EMAs
        )

        return {
            'buy_signal': buy_signal,
            'last_close': last_close,
            'ema_values': ema_values,
            'ema_diffs': ema_diffs,
            'volatility': volatility,
            'zone_score': zone_score,
            'zone_class': zone_class,
            'error': None
        }
    except Exception as e:
        logging.error(f"Error in calculate_signal: {str(e)}")
        return {'error': str(e)}

# Rest of the functions remain the same with minor improvements...

def load_last_buy_dates():
    try:
        if not os.path.exists(LAST_BUY_FILE):
            return {}
        with open(LAST_BUY_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading last buy dates: {str(e)}")
        return {}

def save_last_buy_dates(data):
    try:
        with open(LAST_BUY_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving last buy dates: {str(e)}")

def send_email(subject, html_body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg["To"] = ", ".join(EMAIL_RECEIVERS)
        msg['Subject'] = subject
        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logging.info("Email sent successfully")
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")

def is_last_day_of_month():
    today = datetime.now()
    next_day = today + timedelta(days=1)
    return today.month != next_day.month

# HTML generation remains mostly the same...

def main():
    logging.info("=" * 60)
    logging.info("Enhanced ETF Accumulation Report")
    logging.info("=" * 60)

    last_buys = load_last_buy_dates()
    today = datetime.now()
    current_month = today.strftime('%Y-%m')
    reports = []
    force_buy = is_last_day_of_month()

    for ticker in TICKERS:
        logging.info(f"Analyzing: {ticker}")
        
        # Fetch data with sufficient history
        daily = fetch_data(ticker)

        if daily.empty:
            reports.append({
                'ticker': ticker,
                'error': 'Failed to fetch data',
                'buy_signal': False
            })
            continue

        result = calculate_signal(daily)
        if result.get('error'):
            reports.append({
                'ticker': ticker,
                'error': result['error'],
                'buy_signal': False
            })
            continue

        reports.append({
            'ticker': ticker,
            'last_close': result['last_close'],
            'ema_values': result['ema_values'],
            'ema_diffs': result['ema_diffs'],
            'volatility': result['volatility'],
            'zone_score': result['zone_score'],
            'zone_class': result['zone_class'],
            'buy_signal': result['buy_signal'],
            'error': None
        })

    # Email subject logic remains...
    
    # Generate and send email
    subject = f"📊 Enhanced ETF Report - {today.strftime('%d %b %Y')}"
    if force_buy:
        subject = f"🚨 Monthly Reminder: {subject}"
    elif any(r.get('buy_signal', False) for r in reports):
        best_zone = max(r['zone_score'] for r in reports if not r.get('error'))
        if best_zone >= 90:
            subject = f"🐐 Goated Price Alert: {subject}"
        elif best_zone >= 75:
            subject = f"👍 Great Accumulation Zone: {subject}"
    
    html = generate_html(reports, force_buy)
    send_email(subject, html)

if __name__ == '__main__':
    main()
