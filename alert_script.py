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
    Calculate accumulation zone score with strict criteria:
    - Below 200 EMA: Goated (Score 100)
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
        if len(df) < 2:
            return 0.0
            
        # Use last 3 months data for volatility calculation
        volatility_data = df.tail(VOLATILITY_PERIOD * 3)
        if len(volatility_data) < 2:
            return 0.0
            
        returns = np.log(volatility_data['Close'] / volatility_data['Close'].shift(1))
        return returns.std() * np.sqrt(252) * 100  # Annualized percentage
    except Exception as e:
        logging.error(f"Volatility calculation error: {str(e)}")
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
                # Fallback to SMA if not enough data
                ema_val = daily_df['Close'].iloc[-ema:].mean() if len(daily_df) >= 5 else daily_df['Close'].iloc[-1]
            ema_values.append(safe_float_conversion(ema_val))
        
        last_close = safe_float_conversion(daily_df['Close'].iloc[-1])
        
        # Calculate percentage differences from EMAs
        ema_diffs = [
            calculate_percentage_diff(last_close, ema)
            for ema in ema_values
        ]
        
        # Calculate volatility using realistic method
        volatility = calculate_volatility(daily_df)

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

def generate_html(reports, force_buy=False):
    today = datetime.now()
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Aman's ETF Newsletter</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background-color: #f8fafc;
                color: #1e293b;
                line-height: 1.6;
                padding: 20px;
                max-width: 100%;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }}
            .header {{
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                text-align: center;
                padding: 25px 20px;
                border-radius: 12px 12px 0 0;
            }}
            .header h1 {{
                margin: 0;
                font-size: 1.8rem;
            }}
            .header p {{
                margin: 5px 0 0;
                opacity: 0.9;
            }}
            .date-badge {{
                display: inline-block;
                background: rgba(255,255,255,0.15);
                padding: 5px 12px;
                border-radius: 20px;
                margin-top: 12px;
                font-size: 0.9rem;
            }}
            .force-buy-notice {{
                background: #fffbeb;
                padding: 15px;
                text-align: center;
                border-bottom: 1px solid #fde68a;
            }}
            .cards-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                padding: 25px;
            }}
            .card {{
                background: white;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                border-top: 4px solid;
            }}
            .card-header {{
                padding: 15px 20px;
                border-bottom: 1px solid #e2e8f0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .card-title {{
                font-weight: 600;
                font-size: 1.1rem;
            }}
            .zone-class {{
                font-weight: 600;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 0.85rem;
            }}
            .price-container {{
                display: flex;
                padding: 15px 20px;
                border-bottom: 1px solid #e2e8f0;
            }}
            .price-box {{
                flex: 1;
            }}
            .price-label {{
                font-size: 0.9rem;
                color: #64748b;
                margin-bottom: 5px;
            }}
            .price-value {{
                font-weight: 700;
                font-size: 1.4rem;
            }}
            .ma-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9rem;
            }}
            .ma-table th, .ma-table td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #e2e8f0;
            }}
            .ma-table th {{
                font-weight: 600;
                color: #64748b;
                background-color: #f8fafc;
            }}
            .diff-down {{
                color: #10b981;
                font-weight: 600;
            }}
            .diff-up {{
                color: #ef4444;
                font-weight: 600;
            }}
            .recommendation {{
                padding: 15px;
                text-align: center;
                font-weight: 700;
                border-top: 1px solid #e2e8f0;
            }}
            .buy {{
                background: #d1fae5;
                color: #065f46;
            }}
            .hold {{
                background: #fef3c7;
                color: #92400e;
            }}
            .avoid {{
                background: #fee2e2;
                color: #991b1b;
            }}
            .footer {{
                background: #f1f5f9;
                padding: 25px;
                text-align: center;
                border-top: 1px solid #e2e8f0;
            }}
            .legend {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 15px;
                margin-bottom: 20px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 0.85rem;
            }}
            .legend-color {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
            }}
            .disclaimer {{
                font-size: 0.8rem;
                color: #64748b;
                line-height: 1.5;
                max-width: 700px;
                margin: 0 auto;
            }}
            .error-card {{
                background: #fee2e2;
                border-radius: 8px;
                padding: 20px;
                color: #991b1b;
                font-weight: 500;
            }}
            
            /* Responsive adjustments */
            @media (max-width: 768px) {{
                .cards-container {{
                    grid-template-columns: 1fr;
                    padding: 15px;
                }}
                .header {{
                    padding: 20px 15px;
                }}
                .header h1 {{
                    font-size: 1.5rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Aman's ETF Newsletter</h1>
                <p>Enhanced EMA Strategy</p>
                <div class="date-badge">{today.strftime('%d %B %Y')}</div>
            </div>
            
            {f'<div class="force-buy-notice">📅 Monthly Reminder: Today is the last trading day of the month. Recommended to accumulate if not done already.</div>' if force_buy else ''}
            
            <div class="cards-container">
    """
    
    for r in reports:
        if r.get('error'):
            html += f"""
                <div class="error-card">
                    <strong>Error analyzing {r['ticker']}:</strong> {r['error']}
                </div>
            """
            continue
            
        # Determine card class based on zone score
        if r['zone_score'] >= 90:
            card_class = "excellent"
            zone_color = "#0ea5e9"
        elif r['zone_score'] >= 75:
            card_class = "great"
            zone_color = "#8b5cf6"
        elif r['zone_score'] >= 60:
            card_class = "good"
            zone_color = "#f59e0b"
        elif r['zone_score'] >= 30:
            card_class = "caution"
            zone_color = "#f97316"
        else:
            card_class = "avoid"
            zone_color = "#ef4444"
        
        # Special case for Goated price
        if "Goated" in r['zone_class']:
            card_class = "goated"
            zone_color = "#10b981"
        
        # Determine recommendation
        if force_buy or r['buy_signal']:
            rec_class = "buy"
            if "Goated" in r['zone_class']:
                recommendation = "🐐 GOATED PRICE - STRONG ACCUMULATE"
            elif "Excellent" in r['zone_class']:
                recommendation = "⭐ EXCELLENT PRICE - ACCUMULATE"
            elif "Great" in r['zone_class']:
                recommendation = "👍 GREAT PRICE - ACCUMULATE"
            else:
                recommendation = "🔼 GOOD PRICE - ACCUMULATE"
        elif r['zone_score'] >= 60:
            rec_class = "hold"
            recommendation = "⏳ WAIT - Approaching good price"
        else:
            rec_class = "avoid"
            recommendation = "⛔ AVOID - Price too high"
        
        # Ensure all values are floats for formatting
        last_close = float(r['last_close'])
        volatility = float(r['volatility'])
        ema_values = [float(x) for x in r['ema_values']]
        ema_diffs = [float(x) for x in r['ema_diffs']]
        
        html += f"""
                <div class="card" style="border-top-color: {zone_color}">
                    <div class="card-header">
                        <div class="card-title">{r['ticker']}</div>
                        <div class="zone-class" style="background: {zone_color}22; color: {zone_color}">
                            {r['zone_class']}
                        </div>
                    </div>
                    
                    <div class="price-container">
                        <div class="price-box">
                            <div class="price-label">CURRENT PRICE</div>
                            <div class="price-value">₹{last_close:.2f}</div>
                        </div>
                        <div class="price-box">
                            <div class="price-label">VOLATILITY</div>
                            <div class="price-value">{volatility:.1f}%</div>
                        </div>
                    </div>
                    
                    <table class="ma-table">
                        <tr>
                            <th>EMA</th>
                            <th>Value</th>
                            <th>Difference</th>
                        </tr>
                        <tr>
                            <td>20 EMA</td>
                            <td>₹{ema_values[0]:.2f}</td>
                            <td class="{'diff-down' if ema_diffs[0] < 0 else 'diff-up'}">{ema_diffs[0]:+.1f}%</td>
                        </tr>
                        <tr>
                            <td>50 EMA</td>
                            <td>₹{ema_values[1]:.2f}</td>
                            <td class="{'diff-down' if ema_diffs[1] < 0 else 'diff-up'}">{ema_diffs[1]:+.1f}%</td>
                        </tr>
                        <tr>
                            <td>100 EMA</td>
                            <td>₹{ema_values[2]:.2f}</td>
                            <td class="{'diff-down' if ema_diffs[2] < 0 else 'diff-up'}">{ema_diffs[2]:+.1f}%</td>
                        </tr>
                        <tr>
                            <td>200 EMA</td>
                            <td>₹{ema_values[3]:.2f}</td>
                            <td class="{'diff-down' if ema_diffs[3] < 0 else 'diff-up'}">{ema_diffs[3]:+.1f}%</td>
                        </tr>
                    </table>
                    
                    <div class="recommendation {rec_class}">
                        {recommendation}
                    </div>
                </div>
        """
    
    html += f"""
            </div>
            
            <div class="footer">
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #10b981;"></div>
                        <div>Goated Zone (100)</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #0ea5e9;"></div>
                        <div>Excellent Zone (90)</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #8b5cf6;"></div>
                        <div>Great Zone (75)</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #f59e0b;"></div>
                        <div>Good Zone (60)</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #f97316;"></div>
                        <div>Caution Zone (30-59)</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ef4444;"></div>
                        <div>Avoid Zone (&lt;30)</div>
                    </div>
                </div>
                
                <div class="disclaimer">
                    Enhanced accumulation strategy. Buy signals trigger only when:
                    <ul style="text-align: left; max-width: 500px; margin: 10px auto;">
                        <li>Price is below all key EMAs</li>
                        <li>In Great or better accumulation zone</li>
                        <li>Volatility &lt; {VOLATILITY_THRESHOLD}%</li>
                    </ul>
                    Volatility is annualized. Always conduct your own research.
                    <br><br>
                    Generated on {today.strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

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

    # Email subject logic
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
