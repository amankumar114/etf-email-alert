# etf-email-alert

📈 Daily ETF Signal Email Automation

This project automates the process of sending daily email alerts for selected ETFs based on technical indicators (e.g., EMA crossovers). It fetches market data, performs signal calculations, and emails a beautifully formatted summary — all scheduled to run daily at 2:30 PM IST using GitHub Actions.

⸻

🚀 Features
	•	🔁 Daily Scheduled Execution (via GitHub Actions)
	•	📩 Automated Email Alerts to one or more recipients
	•	📊 EMA-based Strategy: Customizable signal logic
	•	🔐 Secure Credentials using GitHub Secrets
	•	💻 No Local Machine Needed (runs entirely on GitHub)

⸻

🛠️ Tech Stack
	•	Python 3.10
	•	yfinance for market data
	•	pandas, numpy for analysis
	•	smtplib + email.mime for email delivery
	•	GitHub Actions for scheduling
