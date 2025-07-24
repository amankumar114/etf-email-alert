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
import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# === CONFIG ===
# Update the TICKERS list to include the new ETFs
TICKERS = [
    'NIFTYBEES.NS',        # Nifty 50 ETF
    'BANKBEES.NS',         # Banking Sector ETF
    'GOLDBEES.NS',         # Gold ETF         # Nippon India Nifty Auto ETF
    'HDFCSML250.NS',           # Nippon India Nifty IT ETF
    'JUNIORBEES.NS',         # Nippon India Nifty FMCG ETF
    'MID150BEES.NS',       # Nippon India Nifty Pharma ETF
    'SPY',                 # S&P 500 ETF
    'QQQ'                  # NASDAQ-100 ETF
]
# Rest of the code remains exactly the same...
EMA_DAYS = [20, 50, 100, 200]  # Key EMAs to track
VOLATILITY_THRESHOLD = 2.5  # %
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

def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def safe_float_conversion(value):
    """Safely convert any numeric value to float without warnings"""
    if isinstance(value, (np.ndarray, pd.Series)):
        if value.size == 0:
            return 0.0
        value = value.item()  # Convert numpy array to Python scalar
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def calculate_percentage_diff(current, ma):
    """Calculate percentage difference between current price and MA"""
    if ma == 0:  # Avoid division by zero
        return 0.0
    return ((current - ma) / ma) * 100

def calculate_zone_score(ema_diffs):
    """
    Calculate accumulation zone score based on which EMAs the price is touching:
    - Touching 20 EMA: Good (Score 60)
    - Touching 50 EMA: Great (Score 75)
    - Touching 100 EMA: Excellent (Score 90)
    - Touching 200 EMA: Goated (Score 100)
    """
    # Tolerance for "touching" (percentage difference threshold)
    TOUCH_THRESHOLD = 0.5  # %
    
    # Check which EMAs the price is touching (within threshold)
    touching = [
        abs(diff) <= TOUCH_THRESHOLD
        for diff in ema_diffs
    ]
    
    # Determine the highest EMA being touched
    if touching[3]:  # 200 EMA
        return 100, "🐐 Goated Price (Touching 200 EMA)"
    elif touching[2]:  # 100 EMA
        return 90, "⭐ Excellent Price (Touching 100 EMA)"
    elif touching[1]:  # 50 EMA
        return 75, "👍 Great Price (Touching 50 EMA)"
    elif touching[0]:  # 20 EMA
        return 60, "➖ Good Price (Touching 20 EMA)"
    else:
        # If not touching any EMA, score based on how close we are to key EMAs
        closest_ema_index = np.argmin([abs(d) for d in ema_diffs])
        closest_diff = ema_diffs[closest_ema_index]
        
        if closest_diff < 0:  # Below EMA
            if closest_ema_index == 3:  # Closest to 200 EMA
                return 85, "Near Goated Zone (Approaching 200 EMA)"
            elif closest_ema_index == 2:  # Closest to 100 EMA
                return 70, "Near Excellent Zone (Approaching 100 EMA)"
            elif closest_ema_index == 1:  # Closest to 50 EMA
                return 55, "Near Great Zone (Approaching 50 EMA)"
            else:  # Closest to 20 EMA
                return 40, "Near Good Zone (Approaching 20 EMA)"
        else:  # Above EMA
            if closest_ema_index == 3:  # Above 200 EMA
                return 30, "⚠️ Caution Zone (Above 200 EMA)"
            elif closest_ema_index == 2:  # Above 100 EMA
                return 20, "⚠️ High Zone (Above 100 EMA)"
            elif closest_ema_index == 1:  # Above 50 EMA
                return 10, "⛔ Expensive Zone (Above 50 EMA)"
            else:  # Above 20 EMA
                return 0, "⛔ Very Expensive Zone (Above 20 EMA)"

def calculate_signal(daily_df):
    try:
        if daily_df.empty:
            raise ValueError("Empty DataFrame received")
            
        # Calculate all EMAs
        ema_values = [
            safe_float_conversion(daily_df['Close'].ewm(span=ema, adjust=False).mean().iloc[-1])
            for ema in EMA_DAYS
        ]
        
        last_close = safe_float_conversion(daily_df['Close'].iloc[-1])
        volatility = safe_float_conversion(daily_df['Close'].pct_change().std() * 100)

        # Calculate percentage differences from EMAs
        ema_diffs = [
            calculate_percentage_diff(last_close, ema)
            for ema in ema_values
        ]
        
        # Calculate zone score and classification
        zone_score, zone_class = calculate_zone_score(ema_diffs)

        # Buy signal conditions
        buy_signal = (
            zone_score >= 60 and  # At least "Good" zone
            volatility <= VOLATILITY_THRESHOLD
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
                <p>EMA Touch Strategy</p>
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
        
        html += f"""
                <div class="card" style="border-top-color: {zone_color}">
                    <div class="card-header">
                        <div class="card-title">{r['ticker']}</div>
                        <div class="zone-class" style="background: {zone_color}22; color: {zone_color}">
                            {r['zone_class'].split('(')[0].strip()}
                        </div>
                    </div>
                    
                    <div class="price-container">
                        <div class="price-box">
                            <div class="price-label">CURRENT PRICE</div>
                            <div class="price-value">₹{r['last_close']:.2f}</div>
                        </div>
                        <div class="price-box">
                            <div class="price-label">VOLATILITY</div>
                            <div class="price-value">{r['volatility']:.1f}%</div>
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
                            <td>₹{r['ema_values'][0]:.2f}</td>
                            <td class="{'diff-down' if r['ema_diffs'][0] < 0 else 'diff-up'}">{r['ema_diffs'][0]:+.1f}%</td>
                        </tr>
                        <tr>
                            <td>50 EMA</td>
                            <td>₹{r['ema_values'][1]:.2f}</td>
                            <td class="{'diff-down' if r['ema_diffs'][1] < 0 else 'diff-up'}">{r['ema_diffs'][1]:+.1f}%</td>
                        </tr>
                        <tr>
                            <td>100 EMA</td>
                            <td>₹{r['ema_values'][2]:.2f}</td>
                            <td class="{'diff-down' if r['ema_diffs'][2] < 0 else 'diff-up'}">{r['ema_diffs'][2]:+.1f}%</td>
                        </tr>
                        <tr>
                            <td>200 EMA</td>
                            <td>₹{r['ema_values'][3]:.2f}</td>
                            <td class="{'diff-down' if r['ema_diffs'][3] < 0 else 'diff-up'}">{r['ema_diffs'][3]:+.1f}%</td>
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
                    This is an automated report. "Touching" means price is within 0.5% of the EMA value.
                    Volatility Threshold: Prefer accumulation when volatility &lt; {VOLATILITY_THRESHOLD}%.
                    Always conduct your own research before making investment decisions.
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
    logging.info("ETF EMA Touch Accumulation Report")
    logging.info("=" * 60)

    last_buys = load_last_buy_dates()
    today = datetime.now()
    current_month = today.strftime('%Y-%m')
    reports = []
    force_buy = is_last_day_of_month()

    for ticker in TICKERS:
        logging.info(f"Analyzing: {ticker}")
        
        # Fetch daily data
        daily = fetch_data(ticker, '6mo', '1d')

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

    subject = f"📊 Aman's ETF Report - {today.strftime('%d %b %Y')}"
    if force_buy:
        subject = f"🚨 Monthly Reminder: {subject}"
    elif any(r.get('buy_signal', False) for r in reports):
        best_zone = max(r['zone_score'] for r in reports if not r.get('error'))
        if best_zone >= 90:
            subject = f"🐐 Goated Price Alert: {subject}"
        elif best_zone >= 75:
            subject = f"⭐ Excellent Price Alert: {subject}"
        else:
            subject = f"✅ Good Accumulation: {subject}"
    
    html = generate_html(reports, force_buy)
    send_email(subject, html)

if __name__ == '__main__':
    main()
