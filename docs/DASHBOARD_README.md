# BenBot Homepage

This is the main dashboard application for the trading platform. It features:

- Portfolio management
- Sentiment-categorized news (positive/neutral/negative)
- Backtesting functionality
- Paper trading
- Strategy management
- News predictions

## How to Run

```bash
# Activate the virtual environment
source trading_env/bin/activate

# Run the dashboard
streamlit run app.py
```

The main file is `app.py`. A backup is kept at `app_sentiment_news_dashboard.py`.

## Sentiment-Categorized News Feature

The dashboard includes a special news section that categorizes financial news into:
- 📈 Positive News
- 📊 Neutral News
- 📉 Negative News

Each news item includes trading impact analysis and strategic recommendations. 