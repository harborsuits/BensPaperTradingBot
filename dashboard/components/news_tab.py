"""
News Tab Component

This module renders the News/Predictions tab of the trading platform, showing symbol-specific
news with impact assessments and action plans.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import random
from plotly.subplots import make_subplots

# Import UI styles
from dashboard.ui_styles import (
    ThemeMode, UIColors, UIEffects, UITypography, UISpacing,
    create_card, create_metric_card, format_currency, format_percentage,
    theme_plotly_chart
)

def create_symbol_selector():
    """Creates the symbol selection and market data section"""
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Symbol selection with typeahead
        symbols = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "NVDA", "META", "AMD", "INTC", "XOM"]
        selected_symbol = st.selectbox("Select Symbol", symbols)
    
    with col2:
        # Market data
        symbol_data = {
            "AAPL": {"price": 202.34, "change": 1.89, "volume": "32.4M", "sentiment": "Bullish"},
            "MSFT": {"price": 415.56, "change": 2.31, "volume": "24.8M", "sentiment": "Bullish"},
            "TSLA": {"price": 175.45, "change": -2.67, "volume": "41.2M", "sentiment": "Neutral"},
            "AMZN": {"price": 179.78, "change": 0.45, "volume": "28.7M", "sentiment": "Bullish"},
            "GOOGL": {"price": 171.32, "change": 1.23, "volume": "18.3M", "sentiment": "Bullish"},
            "NVDA": {"price": 924.67, "change": 3.78, "volume": "36.9M", "sentiment": "Very Bullish"},
            "META": {"price": 481.23, "change": 2.15, "volume": "20.1M", "sentiment": "Bullish"},
            "AMD": {"price": 158.34, "change": 1.56, "volume": "15.8M", "sentiment": "Bullish"},
            "INTC": {"price": 30.45, "change": -1.23, "volume": "22.3M", "sentiment": "Bearish"},
            "XOM": {"price": 108.92, "change": 0.87, "volume": "10.5M", "sentiment": "Neutral"}
        }
        
        data = symbol_data.get(selected_symbol, {"price": 0, "change": 0, "volume": "0", "sentiment": "Unknown"})
        
        # Price and change
        price_change_color = UIColors.Dark.PROFIT if data["change"] >= 0 else UIColors.Dark.LOSS
        change_sign = "+" if data["change"] >= 0 else ""
        
        price_html = f"""
        <div style="text-align: center; margin-top: 20px;">
            <div style="font-size: 24px; font-weight: bold;">
                ${data["price"]}
            </div>
            <div style="font-size: 16px; color: {price_change_color}; margin-top: 4px;">
                {change_sign}{data["change"]}%
            </div>
        </div>
        """
        st.markdown(price_html, unsafe_allow_html=True)
    
    with col3:
        # Volume and sentiment
        sentiment_color = {
            "Very Bullish": UIColors.Dark.SUCCESS,
            "Bullish": UIColors.Dark.PROFIT,
            "Neutral": UIColors.Dark.NEUTRAL,
            "Bearish": UIColors.Dark.LOSS,
            "Very Bearish": UIColors.Dark.ERROR
        }.get(data["sentiment"], UIColors.Dark.NEUTRAL)
        
        volume_sentiment_html = f"""
        <div style="text-align: center; margin-top: 20px;">
            <div style="font-size: 18px; font-weight: bold;">
                {data["volume"]}
            </div>
            <div style="font-size: 16px; color: {sentiment_color}; margin-top: 4px;">
                {data["sentiment"]}
            </div>
        </div>
        """
        st.markdown(volume_sentiment_html, unsafe_allow_html=True)
    
    # Mini price chart
    # Create sample price data
    np.random.seed(42)  # For reproducibility
    n_days = 30
    date_range = pd.date_range(end=datetime.datetime.now(), periods=n_days)
    
    # Create realistic price movement based on final price and change percentage
    final_price = symbol_data[selected_symbol]["price"]
    change_pct = symbol_data[selected_symbol]["change"] / 100
    
    # Reverse engineer the starting price
    start_price = final_price / (1 + change_pct)
    
    # Create a brownian motion with drift to match the change percentage
    # This creates more realistic looking price movement
    noise = np.random.normal(0, 0.01, n_days)
    daily_returns = np.exp(noise)
    price_series = [start_price]
    
    for i in range(1, n_days):
        # Add some randomness with an overall drift toward the final price
        drift = (final_price / price_series[0]) ** (1/n_days) - 1
        price_series.append(price_series[-1] * (1 + drift + noise[i]))
    
    # Ensure the last price matches our target
    price_series[-1] = final_price
    
    # Create the dataframe
    df = pd.DataFrame({
        'Date': date_range,
        'Price': price_series
    })
    
    # Create the figure
    fig = px.line(df, x='Date', y='Price', title=f"{selected_symbol} - 30 Day Price")
    
    # Apply styling
    colors = UIColors.Dark if st.session_state.theme_mode == ThemeMode.DARK else UIColors.Light
    fig = theme_plotly_chart(fig, st.session_state.theme_mode)
    
    # Add color to the line based on overall trend
    line_color = colors.PROFIT if data["change"] >= 0 else colors.LOSS
    fig.update_traces(line_color=line_color)
    
    # Remove title since we'll use streamlit's header system
    fig.update_layout(
        title=None,
        height=250,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_symbol_news_section(ai_coordinator=None, symbol="AAPL"):
    """Creates the symbol-specific news section with impact assessments"""
    
    st.markdown("<h2>Symbol-Specific News & Analysis</h2>", unsafe_allow_html=True)
    
    # Generate symbol-specific news items with sources, timestamps, and assessments
    if symbol == "AAPL":
        news_items = [
            {
                "title": "Apple to Accelerate AI Integration Across Product Line",
                "source": "Bloomberg",
                "logo": "https://via.placeholder.com/40x20?text=BL",
                "timestamp": "2025-05-04 11:30",
                "impact": "high",
                "content": "Apple is reportedly accelerating the integration of AI features across its entire product lineup, with significant announcements expected at the upcoming WWDC in June.",
                "assessment": "This move positions Apple to compete more effectively with Google and Microsoft in the AI space, potentially driving increased product adoption cycles.",
                "action": "Consider increasing allocation to AAPL in tech-focused strategies. Long-term outlook improved.",
                "url": "https://bloomberg.com/news/articles/2025-05-04/apple-to-accelerate-ai-integration"
            },
            {
                "title": "Apple Services Revenue Growth Exceeds Expectations",
                "source": "CNBC",
                "logo": "https://via.placeholder.com/40x20?text=CNBC",
                "timestamp": "2025-05-04 09:15",
                "impact": "medium",
                "content": "Apple's services segment, including App Store, Apple Music, and Apple TV+, has shown stronger-than-expected revenue growth in Q1, potentially offsetting hardware sales slowdown.",
                "assessment": "The shift toward services provides a more stable revenue stream with higher margins, reducing dependence on hardware upgrade cycles.",
                "action": "Monitor services segment growth in upcoming earnings. May require adjustments to revenue projection models.",
                "url": "https://cnbc.com/2025/05/04/apple-services-revenue-growth"
            },
            {
                "title": "Supply Chain Report: iPhone Production Stabilizing After Disruptions",
                "source": "Reuters",
                "logo": "https://via.placeholder.com/40x20?text=RT",
                "timestamp": "2025-05-03 22:45",
                "impact": "medium",
                "content": "Apple's supply chain partners report that iPhone production is stabilizing after recent component shortages and manufacturing disruptions in Southeast Asia.",
                "assessment": "Production stabilization reduces risk of significant shipment delays for the upcoming product cycle.",
                "action": "No immediate action needed, but improves confidence in Q3-Q4 revenue projections.",
                "url": "https://reuters.com/technology/2025/05/03/apple-supply-chain-stabilizing"
            },
            {
                "title": "Apple Faces Regulatory Scrutiny in EU Over App Store Policies",
                "source": "Financial Times",
                "logo": "https://via.placeholder.com/40x20?text=FT",
                "timestamp": "2025-05-03 08:20",
                "impact": "medium",
                "content": "European regulators are increasing scrutiny of Apple's App Store policies, particularly regarding fees charged to developers and restrictions on alternative payment methods.",
                "assessment": "Potential regulatory action could impact App Store revenue, which is a high-margin component of Apple's services business.",
                "action": "Monitor regulatory developments. Consider scenario analysis for potential App Store policy changes.",
                "url": "https://ft.com/content/2025/05/03/apple-eu-app-store-scrutiny"
            }
        ]
    elif symbol == "TSLA":
        news_items = [
            {
                "title": "Tesla to Expand Gigafactory Capacity Amid Growing EV Demand",
                "source": "Reuters",
                "logo": "https://via.placeholder.com/40x20?text=RT",
                "timestamp": "2025-05-04 10:45",
                "impact": "high",
                "content": "Tesla announces plans to expand production capacity at its Berlin and Texas gigafactories by 25% over the next 18 months to meet growing global EV demand.",
                "assessment": "Expansion indicates confidence in demand forecasts and should help address delivery timeline issues that have affected customer satisfaction.",
                "action": "Positive for long-term production capacity. Monitor capex increases in upcoming quarters.",
                "url": "https://reuters.com/business/autos/2025/05/04/tesla-expands-gigafactory-capacity"
            },
            {
                "title": "Tesla's FSD Version 12.3 Shows Significant Improvement in Testing",
                "source": "Electrek",
                "logo": "https://via.placeholder.com/40x20?text=ELEC",
                "timestamp": "2025-05-04 08:30",
                "impact": "medium",
                "content": "Early testing of Tesla's Full Self-Driving version 12.3 shows substantial improvements in urban navigation and complex traffic scenarios compared to previous versions.",
                "assessment": "FSD improvements strengthen Tesla's technological lead in autonomous driving, a key differentiator from traditional automakers.",
                "action": "Positive catalyst for stock. Reinforces long-term AI and autonomous driving thesis.",
                "url": "https://electrek.co/2025/05/04/tesla-fsd-12-3-improvements"
            },
            {
                "title": "Tesla Energy Storage Deployments Double Year-Over-Year",
                "source": "Bloomberg",
                "logo": "https://via.placeholder.com/40x20?text=BL",
                "timestamp": "2025-05-03 14:15",
                "impact": "medium",
                "content": "Tesla's energy storage business has seen deployments double compared to the same period last year, driven by utility-scale projects and growing residential demand.",
                "assessment": "Energy business becoming a more significant growth driver, diversifying revenue streams beyond vehicle sales.",
                "action": "Adjust revenue models to account for accelerating energy storage growth.",
                "url": "https://bloomberg.com/news/articles/2025-05-03/tesla-energy-storage-growth"
            }
        ]
    else:
        # Generic news items for other symbols
        news_items = [
            {
                "title": f"{symbol} Quarterly Earnings Exceed Analyst Expectations",
                "source": "Bloomberg",
                "logo": "https://via.placeholder.com/40x20?text=BL",
                "timestamp": "2025-05-04 09:30",
                "impact": "high",
                "content": f"{symbol} reported quarterly earnings of $1.45 per share, beating analyst expectations of $1.32, driven by strong product demand and improved margins.",
                "assessment": "Strong earnings performance demonstrates operational efficiency and pricing power in current market conditions.",
                "action": "Consider increasing position size in momentum strategies. Sentiment shift likely positive.",
                "url": f"https://bloomberg.com/news/articles/2025-05-04/{symbol.lower()}-earnings"
            },
            {
                "title": f"Analyst Upgrades {symbol} to 'Buy' on Growth Prospects",
                "source": "CNBC",
                "logo": "https://via.placeholder.com/40x20?text=CNBC",
                "timestamp": "2025-05-03 14:45",
                "impact": "medium",
                "content": f"Leading analysts have upgraded {symbol} to 'Buy' from 'Hold', citing improved growth prospects and market share gains in key segments.",
                "assessment": "Analyst upgrade likely to drive institutional buying. Technical picture improving with potential breakout pattern forming.",
                "action": "Technical indicators support entry point. Consider options positions to capitalize on momentum.",
                "url": f"https://cnbc.com/2025/05/03/{symbol.lower()}-analyst-upgrade"
            },
            {
                "title": f"{symbol} Announces Share Repurchase Program",
                "source": "Reuters",
                "logo": "https://via.placeholder.com/40x20?text=RT",
                "timestamp": "2025-05-02 16:20",
                "impact": "medium",
                "content": f"{symbol} has announced a $5 billion share repurchase program, signaling confidence in its financial position and future prospects.",
                "assessment": "Repurchase program should provide price support and indicates management believes shares are undervalued at current levels.",
                "action": "Positive for short-term price action. Long-term implications depend on capital allocation strategy.",
                "url": f"https://reuters.com/business/2025/05/02/{symbol.lower()}-share-repurchase"
            }
        ]
    
    # Display news cards with professional styling
    for news in news_items:
        impact_color = {
            "high": UIColors.Dark.ERROR,
            "medium": UIColors.Dark.WARNING,
            "low": UIColors.Dark.SUCCESS
        }.get(news["impact"], UIColors.Dark.INFO)
        
        news_card = f"""
        <div class="card" style="background-color: {UIColors.Dark.BG_TERTIARY}; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <h3 style="margin: 0 0 8px 0;">{news["title"]}</h3>
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <img src="{news["logo"]}" style="height: 20px; margin-right: 8px;" />
                        <span style="color: {UIColors.Dark.TEXT_SECONDARY}; margin-right: 12px;">{news["source"]}</span>
                        <span style="color: {UIColors.Dark.TEXT_TERTIARY};">{news["timestamp"]}</span>
                    </div>
                </div>
                <div>
                    <span style="background-color: {impact_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">
                        {news["impact"].upper()} IMPACT
                    </span>
                </div>
            </div>
            
            <p style="margin: 12px 0;">{news["content"]}</p>
            
            <div style="background-color: {UIColors.Dark.BG_SECONDARY}; padding: 12px; border-radius: 4px; margin: 12px 0;">
                <div style="margin-bottom: 8px;"><strong>Impact Assessment:</strong> {news["assessment"]}</div>
                <div><strong>Suggested Action:</strong> {news["action"]}</div>
            </div>
            
            <div style="display: flex; justify-content: flex-end;">
                <a href="{news["url"]}" target="_blank" style="color: {UIColors.Dark.ACCENT_PRIMARY}; text-decoration: none; font-size: 14px;">
                    Read More <i class="fas fa-external-link-alt" style="font-size: 12px; margin-left: 4px;"></i>
                </a>
            </div>
        </div>
        """
        st.markdown(news_card, unsafe_allow_html=True)

def get_market_sentiment(ai_coordinator=None):
    """Get market sentiment data either from AI coordinator or sample data"""
    if ai_coordinator and hasattr(ai_coordinator, 'get_market_sentiment'):
        try:
            sentiment = ai_coordinator.get_market_sentiment()
            # Ensure proper format for chart
            return {
                "categories": sentiment.get("categories", ["Social Media", "News", "Analyst Reports", "Earnings", "Technical"]),
                "values": sentiment.get("values", [70, 65, 75, 72, 68])
            }
        except Exception as e:
            # Fallback to sample data if AI call fails
            print(f"Error getting market sentiment: {e}")
    
    # Sample sentiment data
    return {
        "categories": ["Social Media", "News", "Analyst Reports", "Earnings", "Technical"],
        "values": [70, 65, 75, 72, 68]
    }

def create_economic_calendar_section(ai_coordinator=None):
    """Creates the economic calendar section"""
    st.header("Economic Calendar")
    
    # Create sample economic calendar data
    today = datetime.datetime.now().date()
    calendar_data = [
        {"date": today, "time": "8:30 AM", "event": "Non-Farm Payrolls", "actual": "285K", "forecast": "270K", "previous": "256K", "impact": "High"},
        {"date": today, "time": "9:45 AM", "event": "ISM Manufacturing PMI", "actual": "52.8", "forecast": "51.9", "previous": "51.5", "impact": "Medium"},
        {"date": today + datetime.timedelta(days=1), "time": "10:00 AM", "event": "Fed Chair Speech", "actual": "", "forecast": "", "previous": "", "impact": "High"},
        {"date": today + datetime.timedelta(days=2), "time": "8:30 AM", "event": "Initial Jobless Claims", "actual": "", "forecast": "240K", "previous": "238K", "impact": "Medium"},
        {"date": today + datetime.timedelta(days=3), "time": "8:30 AM", "event": "CPI m/m", "actual": "", "forecast": "0.3%", "previous": "0.2%", "impact": "High"},
    ]
    
    # Display in a table format
    calendar_df = pd.DataFrame(calendar_data)
    
    # Format the table
    st.dataframe(
        calendar_df,
        column_config={
            "date": "Date",
            "time": "Time",
            "event": "Event",
            "actual": "Actual",
            "forecast": "Forecast",
            "previous": "Previous",
            "impact": st.column_config.Column(
                "Impact",
                help="Potential market impact",
                width="small",
            )
        },
        hide_index=True,
        use_container_width=True
    )

def create_portfolio_impact_section(ai_coordinator=None):
    """Creates the portfolio impact section"""
    st.header("Portfolio Impact Analysis")
    
    if ai_coordinator and hasattr(ai_coordinator, 'analyze_portfolio_impact'):
        try:
            impact_analysis = ai_coordinator.analyze_portfolio_impact()
            # Use the AI-generated analysis
        except Exception as e:
            st.warning(f"Could not retrieve AI portfolio analysis: {e}")
            # Fall back to sample data
    
    # Sample portfolio impact cards
    impact_items = [
        {
            "title": "Fed Rate Decision Impact",
            "content": "The Federal Reserve's recent 25bp hike is likely to impact your fixed income positions. Consider rebalancing your bond exposure to shorter durations.",
            "severity": "medium",
            "actions": ["Review bond allocations", "Consider floating rate instruments"]
        },
        {
            "title": "Technology Sector Earnings",
            "content": "Major tech earnings this week may cause volatility in your portfolio. Your current tech exposure is 32%, which is overweight compared to your target allocation.",
            "severity": "high",
            "actions": ["Hedge tech exposure", "Review stop-loss levels"]
        },
        {
            "title": "Upcoming Economic Data",
            "content": "CPI data release on Thursday may affect market sentiment. Your inflation-sensitive assets (TIPS, commodities) are currently underweight at 8% vs. target of 12%.",
            "severity": "low",
            "actions": ["Consider increasing inflation hedges", "Monitor commodity positions"]
        }
    ]
    
    # Display impact cards
    for item in impact_items:
        severity_color = {
            "high": UIColors.Dark.ERROR,
            "medium": UIColors.Dark.WARNING,
            "low": UIColors.Dark.SUCCESS
        }.get(item["severity"].lower(), UIColors.Dark.INFO)
        
        impact_card = f"""
        <div style="border-left: 4px solid {severity_color}; padding: 16px; margin-bottom: 16px; background-color: {UIColors.Dark.BG_TERTIARY}; border-radius: 4px;">
            <h4 style="margin-top: 0; margin-bottom: 8px;">{item['title']}</h4>
            <p style="margin-bottom: 12px;">{item['content']}</p>
            
            <div style="margin-top: 8px;">
                <strong>Recommended Actions:</strong>
                <ul style="margin-top: 4px; margin-bottom: 0;">
        """
        
        for action in item["actions"]:
            impact_card += f"<li>{action}</li>"
        
        impact_card += """
                </ul>
            </div>
        </div>
        """
        
        st.markdown(impact_card, unsafe_allow_html=True)

def create_market_overview_section(ai_coordinator=None):
    """Creates the market overview section"""
    st.header("Market Overview")
    
    # Placeholder for market overview content
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("S&P 500", "4,927.11", "+0.68%")
    
    with cols[1]:
        st.metric("Nasdaq", "15,451.93", "+1.12%")
        
    with cols[2]:
        st.metric("VIX", "14.21", "-1.31%", delta_color="inverse")
    
    # Market sentiment analysis
    st.subheader("Market Sentiment Analysis")
    
    # Use AI coordinator if available
    sentiment_data = get_market_sentiment(ai_coordinator)
    
    # Create radar chart for sentiment
    colors = UIColors.Dark if st.session_state.get('theme_mode', ThemeMode.DARK) == ThemeMode.DARK else UIColors.Light
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=sentiment_data["values"],
        theta=sentiment_data["categories"],
        fill='toself',
        line_color=colors.ACCENT_PRIMARY,
        fillcolor=f'rgba({int(colors.ACCENT_PRIMARY[1:3], 16)}, {int(colors.ACCENT_PRIMARY[3:5], 16)}, {int(colors.ACCENT_PRIMARY[5:7], 16)}, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False
    )
    
    fig = theme_plotly_chart(fig, st.session_state.theme_mode)
    
    # Custom title and explanation
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 10px;">
        <strong>Overall Sentiment Score: {sum(sentiment_data["values"]) / len(sentiment_data["values"]):.1f}/100</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation
    sentiment_avg = sum(sentiment_data["values"]) / len(sentiment_data["values"])
    sentiment_text = "bearish" if sentiment_avg < 40 else "neutral" if sentiment_avg < 60 else "bullish"
    sentiment_color = UIColors.Dark.ERROR if sentiment_avg < 40 else UIColors.Dark.NEUTRAL if sentiment_avg < 60 else UIColors.Dark.SUCCESS
    
    st.markdown(f"""
    <div style="background-color: {colors.BG_SECONDARY}; padding: 12px; border-radius: 4px; margin-top: 4px;">
        <p style="margin: 0; font-size: 13px;">
            <strong>Market Sentiment Analysis:</strong> The overall market sentiment is trending <span style="color: {sentiment_color};"><strong>{sentiment_text}</strong></span>.
            Social media sentiment is {"negative" if sentiment_data["values"][0] < 50 else "positive"}, while analyst reports are {"cautious" if sentiment_data["values"][2] < 60 else "optimistic"}.
            This indicates that the market may {"face headwinds" if sentiment_avg < 50 else "continue its current trend" if sentiment_avg < 70 else "see further upside"} in the near term.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with col2:
        # AI Price Prediction
        st.markdown("<h3>Price Prediction Analysis</h3>", unsafe_allow_html=True)
        
        # Create price prediction data
        current_price = {
            "AAPL": 202.34,
            "TSLA": 175.45
        }.get(symbol, 150.00)
        
        prediction_data = {
            "AAPL": {
                "prediction": current_price * 1.12,
                "lower_bound": current_price * 1.05,
                "upper_bound": current_price * 1.18,
                "confidence": 76
            },
            "TSLA": {
                "prediction": current_price * 1.08,
                "lower_bound": current_price * 0.95,
                "upper_bound": current_price * 1.22,
                "confidence": 68
            }
        }.get(symbol, {
            "prediction": current_price * 1.06,
            "lower_bound": current_price * 0.98,
            "upper_bound": current_price * 1.14,
            "confidence": 72
        })
        
        # Create future dates for prediction
        dates = pd.date_range(start=datetime.datetime.now(), periods=31, freq='D')
        
        # Generate smooth prediction curve
        base_price = current_price
        target_price = prediction_data["prediction"]
        
        # Create a smooth S-curve between current price and predicted price
        t = np.linspace(0, 1, len(dates))
        s_curve = base_price + (target_price - base_price) * (1 / (1 + np.exp(-10 * (t - 0.5))))
        
        # Add some random noise to make it look realistic
        np.random.seed(42)
        noise = np.random.normal(0, (target_price - base_price) * 0.03, len(dates))
        predicted_prices = s_curve + noise
        
        # Create lower and upper bound curves
        lower_bound = prediction_data["lower_bound"] * (predicted_prices / predicted_prices[-1])
        upper_bound = prediction_data["upper_bound"] * (predicted_prices / predicted_prices[-1])
        
        # Create dataframe
        df = pd.DataFrame({
            'Date': dates,
            'Predicted': predicted_prices,
            'Lower': lower_bound,
            'Upper': upper_bound
        })
        
        # Create the figure
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Predicted'],
            line=dict(color=colors.ACCENT_PRIMARY, width=2),
            name='AI Prediction'
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Upper'],
            line=dict(color='rgba(0,0,0,0)'),
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Lower'],
            fill='tonexty',
            fillcolor=f'rgba({int(colors.ACCENT_PRIMARY[1:3], 16)}, {int(colors.ACCENT_PRIMARY[3:5], 16)}, {int(colors.ACCENT_PRIMARY[5:7], 16)}, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Lower Bound'
        ))
        
        # Add a point for current price
        fig.add_trace(go.Scatter(
            x=[dates[0]],
            y=[current_price],
            mode='markers',
            marker=dict(size=8, color=colors.NEUTRAL),
            name='Current Price'
        ))
        
        # Add a point for predicted price
        fig.add_trace(go.Scatter(
            x=[dates[-1]],
            y=[predicted_prices[-1]],
            mode='markers',
            marker=dict(size=8, color=colors.PROFIT if predicted_prices[-1] > current_price else colors.LOSS),
            name='Target Price'
        ))
        
        # Apply styling
        fig = theme_plotly_chart(fig, st.session_state.theme_mode)
        fig.update_layout(
            title=None,
            height=270,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        # Custom title with prediction details
        price_change = ((prediction_data["prediction"] / current_price) - 1) * 100
        price_color = colors.PROFIT if price_change > 0 else colors.LOSS
        
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 10px;">
            <strong>30-Day Price Target: <span style="color: {price_color};">${prediction_data["prediction"]:.2f} ({price_change:+.2f}%)</span></strong>
            <div style="font-size: 13px; color: {colors.TEXT_SECONDARY};">Confidence: {prediction_data["confidence"]}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        days_to_target = 30
        st.markdown(f"""
        <div style="background-color: {colors.BG_SECONDARY}; padding: 10px; border-radius: 4px; margin-top: 0;">
            <p style="margin: 0; font-size: 13px;">
                <strong>AI Analysis:</strong> Based on technical patterns, sentiment analysis, and historical performance, 
                our AI model projects a {days_to_target}-day price target of <span style="color: {price_color};">${prediction_data["prediction"]:.2f}</span> with 
                a confidence interval of ${prediction_data["lower_bound"]:.2f} to ${prediction_data["upper_bound"]:.2f}. 
                Key factors include {'positive sentiment momentum' if price_change > 0 else 'cautious market sentiment'} and 
                {'favorable technical indicators' if price_change > 5 else 'mixed technical signals'}.
            </p>
        </div>
        """, unsafe_allow_html=True)

def create_news_tabs(ai_coordinator=None):
    """Creates the tabbed interface for news content with AI integration"""
    
    tabs = st.tabs(["Market Overview", "Symbol News", "Economic Calendar", "Portfolio Impact"])
    
    with tabs[0]:
        create_market_overview_section(ai_coordinator)
    
    with tabs[1]:
        create_symbol_news_section(ai_coordinator)
    
    with tabs[2]:
        create_economic_calendar_section(ai_coordinator)
    
    with tabs[3]:
        create_portfolio_impact_section(ai_coordinator)

def render_news_tab(ai_coordinator=None):
    """Renders the complete News/Predictions tab with AI integration"""
    
    st.title("Market News & Predictions")
    
    # Watchlist symbols selection in the sidebar
    with st.sidebar:
        st.subheader("News Watchlist")
        watchlist = st.multiselect(
            "Select Symbols",
            ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL", "META", "BTC-USD"],
            ["SPY", "AAPL", "TSLA"]
        )
        
        # Store watchlist in session state for access across re-renders
        st.session_state.news_watchlist = watchlist
        
        st.divider()
        st.subheader("News Settings")
        
        impact_threshold = st.slider("Impact Threshold", min_value=1, max_value=10, value=3, 
                               help="Only show news with impact above this threshold")
        st.session_state.impact_threshold = impact_threshold
        
        news_sources = st.multiselect(
            "News Sources",
            ["Bloomberg", "Reuters", "CNBC", "WSJ", "Financial Times", "MarketWatch", "Seeking Alpha", "Twitter"],
            ["Bloomberg", "Reuters", "CNBC", "WSJ"]
        )
        st.session_state.news_sources = news_sources
        
        news_categories = st.multiselect(
            "Categories",
            ["Market News", "Economic Data", "Earnings", "Analyst Reports", "SEC Filings", "Central Banks"],
            ["Market News", "Economic Data", "Earnings"]
        )
        st.session_state.news_categories = news_categories
        
        st.divider()
        timeframe = st.radio("Timeframe", options=["Today", "Last 3 Days", "Last Week", "Last Month"])
        st.session_state.news_timeframe = timeframe
    
    # Use the tabbed interface to display all news sections
    create_news_tabs(ai_coordinator)
