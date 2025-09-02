"""
Market Context Component for BensBot Dashboard
Displays market sentiment, news, VIX, and heatmap visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random

# Get market sentiment data
def get_market_sentiment(db):
    """Get market sentiment indicators from MongoDB"""
    if db is None:
        return generate_mock_sentiment_data()
    
    try:
        # Try to get sentiment data from MongoDB
        sentiment_doc = db.market_context.find_one({"type": "sentiment"})
        
        if sentiment_doc:
            return sentiment_doc
        else:
            return generate_mock_sentiment_data()
    except Exception as e:
        st.error(f"Error retrieving market sentiment data: {e}")
        return generate_mock_sentiment_data()

# Generate mock market sentiment data
def generate_mock_sentiment_data():
    """Generate synthetic market sentiment data for development and testing"""
    return {
        "type": "sentiment",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fear_greed_index": random.randint(30, 70),
        "market_sentiment": random.choice(["bearish", "neutral", "bullish"]),
        "institutional_sentiment": random.uniform(-0.8, 0.8),
        "retail_sentiment": random.uniform(-0.8, 0.8),
        "social_media_sentiment": random.uniform(-0.7, 0.7),
        "put_call_ratio": random.uniform(0.7, 1.3),
        "vix": random.uniform(15.0, 30.0)
    }

# Get market news data
def get_market_news(db):
    """Get financial news items from MongoDB"""
    if db is None:
        return generate_mock_news_data()
    
    try:
        # Try to get news data from MongoDB
        news_docs = list(db.market_news.find({}).sort("timestamp", -1).limit(10))
        
        if news_docs:
            # Convert to DataFrame
            df = pd.DataFrame(news_docs)
            return df
        else:
            return generate_mock_news_data()
    except Exception as e:
        st.error(f"Error retrieving market news data: {e}")
        return generate_mock_news_data()

# Generate mock market news data
def generate_mock_news_data():
    """Generate synthetic financial news data for development and testing"""
    headlines = [
        "Fed signals potential rate cut in upcoming meeting",
        "Tech stocks rally on strong earnings reports",
        "Oil prices surge amid Middle East tensions",
        "Major bank reports better-than-expected quarterly profit",
        "Inflation data comes in below market expectations",
        "Consumer confidence index shows unexpected improvement",
        "Manufacturing data suggests economic slowdown",
        "S&P 500 reaches new all-time high",
        "Treasury yields fall following central bank comments",
        "Market volatility increases amid geopolitical concerns",
        "Retail sales data shows resilient consumer spending",
        "Housing market shows signs of cooling as mortgage rates rise"
    ]
    
    sources = ["Bloomberg", "CNBC", "Reuters", "WSJ", "Financial Times", "MarketWatch", "Yahoo Finance"]
    
    categories = ["economy", "stocks", "bonds", "commodities", "forex", "interest_rates", "macro"]
    
    # Generate 10 news items with random timestamps in the past 24 hours
    news_items = []
    now = datetime.datetime.now()
    
    for i in range(10):
        # Random headline and source
        headline = random.choice(headlines)
        source = random.choice(sources)
        category = random.choice(categories)
        
        # Random timestamp in the past 24 hours
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        timestamp = now - datetime.timedelta(hours=hours_ago, minutes=minutes_ago)
        
        # Random impact and sentiment
        impact = random.uniform(-1.0, 1.0)
        sentiment = "positive" if impact > 0.3 else "negative" if impact < -0.3 else "neutral"
        
        # Fake URL
        url = f"https://example.com/{source.lower().replace(' ', '')}/news/{i}"
        
        # Create news item
        news_item = {
            "headline": headline,
            "source": source,
            "category": category,
            "timestamp": timestamp,
            "impact": round(impact, 2),
            "sentiment": sentiment,
            "url": url
        }
        
        news_items.append(news_item)
    
    # Sort by timestamp (newest first)
    news_items = sorted(news_items, key=lambda x: x["timestamp"], reverse=True)
    
    return pd.DataFrame(news_items)

# Get VIX and fear/greed historical data
def get_vix_history(db):
    """Get VIX and fear/greed index history from MongoDB"""
    if db is None:
        return generate_mock_vix_history()
    
    try:
        # Try to get VIX history from MongoDB
        vix_docs = list(db.market_vix.find({}).sort("date", 1).limit(60))  # Last 60 days
        
        if vix_docs:
            # Convert to DataFrame
            df = pd.DataFrame(vix_docs)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            return generate_mock_vix_history()
    except Exception as e:
        st.error(f"Error retrieving VIX history data: {e}")
        return generate_mock_vix_history()

# Generate mock VIX history data
def generate_mock_vix_history():
    """Generate synthetic VIX and fear/greed index history for development and testing"""
    days = 60
    now = datetime.datetime.now()
    dates = [now - datetime.timedelta(days=i) for i in range(days, 0, -1)]
    
    # Start with base values
    vix_base = 20.0
    fear_greed_base = 50
    
    # Generate data with random walk but some correlation between VIX and fear/greed
    vix_values = []
    fear_greed_values = []
    
    for i in range(days):
        # VIX tends to mean-revert around 20
        vix_change = random.normalvariate(0, 1.0) + (20 - vix_base) * 0.05
        vix_base = max(10, min(40, vix_base + vix_change))
        vix_values.append(round(vix_base, 2))
        
        # Fear/greed is inversely related to VIX but has its own dynamics
        fear_greed_change = random.normalvariate(0, 2.0) - vix_change * 2
        fear_greed_base = max(5, min(95, fear_greed_base + fear_greed_change))
        fear_greed_values.append(round(fear_greed_base))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'vix': vix_values,
        'fear_greed_index': fear_greed_values
    })
    
    return df

# Get sector performance data
def get_sector_performance(db):
    """Get sector performance data from MongoDB"""
    if db is None:
        return generate_mock_sector_data()
    
    try:
        # Try to get sector data from MongoDB
        sector_doc = db.market_sectors.find_one({"type": "performance"})
        
        if sector_doc:
            return sector_doc
        else:
            return generate_mock_sector_data()
    except Exception as e:
        st.error(f"Error retrieving sector performance data: {e}")
        return generate_mock_sector_data()

# Generate mock sector performance data
def generate_mock_sector_data():
    """Generate synthetic sector performance data for development and testing"""
    sectors = [
        "Technology", "Healthcare", "Financials", "Consumer Discretionary", 
        "Communication Services", "Industrials", "Consumer Staples", 
        "Energy", "Utilities", "Materials", "Real Estate"
    ]
    
    # Generate daily, weekly, monthly performance
    performance = {}
    
    # Daily (more volatile)
    performance["daily"] = {
        sector: round(random.normalvariate(0, 0.8), 2) for sector in sectors
    }
    
    # Weekly (less volatile than daily)
    performance["weekly"] = {
        sector: round(random.normalvariate(0, 1.5), 2) for sector in sectors
    }
    
    # Monthly (can have larger moves)
    performance["monthly"] = {
        sector: round(random.normalvariate(1, 3.0), 2) for sector in sectors
    }
    
    # YTD (largest variation)
    performance["ytd"] = {
        sector: round(random.normalvariate(5, 10.0), 2) for sector in sectors
    }
    
    # Create result
    result = {
        "type": "performance",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sectors": sectors,
        "performance": performance
    }
    
    return result

# Render market sentiment indicators
def render_sentiment_indicators(sentiment_data):
    """Render market sentiment visualization"""
    st.subheader("Market Sentiment & VIX")
    
    # Create 2-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Fear & Greed Index gauge
        fear_greed = sentiment_data.get("fear_greed_index", 50)
        
        # Determine the fear/greed category and color
        if fear_greed <= 20:
            category = "Extreme Fear"
            color = "darkred"
        elif fear_greed <= 40:
            category = "Fear"
            color = "red"
        elif fear_greed <= 60:
            category = "Neutral"
            color = "yellow"
        elif fear_greed <= 80:
            category = "Greed"
            color = "lightgreen"
        else:
            category = "Extreme Greed"
            color = "green"
        
        # Create gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fear_greed,
            title={"text": "Fear & Greed Index", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 20], "color": "rgba(200, 0, 0, 0.3)"},
                    {"range": [20, 40], "color": "rgba(250, 150, 0, 0.3)"},
                    {"range": [40, 60], "color": "rgba(200, 200, 0, 0.3)"},
                    {"range": [60, 80], "color": "rgba(150, 200, 0, 0.3)"},
                    {"range": [80, 100], "color": "rgba(0, 200, 0, 0.3)"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": fear_greed
                }
            },
            domain={"x": [0, 1], "y": [0, 1]}
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={"color": "black", "family": "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the category
        st.markdown(f"""
        <div style="text-align: center; margin-top: -30px;">
            <p style="font-size: 1.2rem; font-weight: bold; color: {color};">{category}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # VIX indicator
        vix = sentiment_data.get("vix", 20.0)
        
        # Determine VIX category and color
        if vix <= 15:
            vix_category = "Low Volatility"
            vix_color = "green"
        elif vix <= 25:
            vix_category = "Normal Volatility"
            vix_color = "blue"
        elif vix <= 35:
            vix_category = "High Volatility"
            vix_color = "orange"
        else:
            vix_category = "Extreme Volatility"
            vix_color = "red"
        
        # Create VIX gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=vix,
            title={"text": "VIX (Volatility Index)", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 50], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": vix_color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 15], "color": "rgba(0, 200, 0, 0.3)"},
                    {"range": [15, 25], "color": "rgba(0, 0, 200, 0.3)"},
                    {"range": [25, 35], "color": "rgba(250, 150, 0, 0.3)"},
                    {"range": [35, 50], "color": "rgba(200, 0, 0, 0.3)"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": vix
                }
            },
            domain={"x": [0, 1], "y": [0, 1]}
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={"color": "black", "family": "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the category
        st.markdown(f"""
        <div style="text-align: center; margin-top: -30px;">
            <p style="font-size: 1.2rem; font-weight: bold; color: {vix_color};">{vix_category}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional sentiment indicators in a 2x2 grid
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    
    # Function to create a sentiment indicator
    def create_sentiment_indicator(container, title, value, min_val=-1, max_val=1, is_ratio=False):
        # Determine color based on value
        if is_ratio:
            # For ratios like put/call
            if value < 0.8:
                color = "green"  # bullish
                sentiment = "Bullish"
            elif value < 1.0:
                color = "lightgreen"  # slightly bullish
                sentiment = "Slightly Bullish"
            elif value < 1.2:
                color = "orange"  # slightly bearish
                sentiment = "Slightly Bearish"
            else:
                color = "red"  # bearish
                sentiment = "Bearish"
        else:
            # For normalized sentiment scores
            if value < -0.5:
                color = "red"
                sentiment = "Bearish"
            elif value < 0:
                color = "orange"
                sentiment = "Slightly Bearish"
            elif value < 0.5:
                color = "lightgreen"
                sentiment = "Slightly Bullish"
            else:
                color = "green"
                sentiment = "Bullish"
        
        # Create the indicator
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [min_val, max_val], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [min_val, (max_val - min_val) * 0.25 + min_val], "color": "rgba(200, 0, 0, 0.3)"},
                    {"range": [(max_val - min_val) * 0.25 + min_val, (max_val - min_val) * 0.5 + min_val], "color": "rgba(250, 150, 0, 0.3)"},
                    {"range": [(max_val - min_val) * 0.5 + min_val, (max_val - min_val) * 0.75 + min_val], "color": "rgba(150, 200, 0, 0.3)"},
                    {"range": [(max_val - min_val) * 0.75 + min_val, max_val], "color": "rgba(0, 200, 0, 0.3)"}
                ] if not is_ratio else [
                    {"range": [0, 0.8], "color": "rgba(0, 200, 0, 0.3)"},
                    {"range": [0.8, 1.0], "color": "rgba(150, 200, 0, 0.3)"},
                    {"range": [1.0, 1.2], "color": "rgba(250, 150, 0, 0.3)"},
                    {"range": [1.2, 2.0], "color": "rgba(200, 0, 0, 0.3)"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": value
                }
            },
            number={"valueformat": ".2f"}
        ))
        
        fig.update_layout(
            height=150,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="white",
            font={"color": "black", "family": "Arial"}
        )
        
        container.plotly_chart(fig, use_container_width=True)
        
        # Display sentiment label
        container.markdown(f"""
        <div style="text-align: center; margin-top: -20px;">
            <p style="font-size: 0.9rem; font-weight: bold; color: {color};">{sentiment}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Institutional sentiment
    create_sentiment_indicator(
        col3,
        "Institutional Sentiment",
        sentiment_data.get("institutional_sentiment", 0)
    )
    
    # Retail sentiment
    create_sentiment_indicator(
        col4,
        "Retail Sentiment",
        sentiment_data.get("retail_sentiment", 0)
    )
    
    # Social media sentiment
    create_sentiment_indicator(
        col5,
        "Social Media Sentiment",
        sentiment_data.get("social_media_sentiment", 0)
    )
    
    # Put/Call ratio
    create_sentiment_indicator(
        col6,
        "Put/Call Ratio",
        sentiment_data.get("put_call_ratio", 1.0),
        min_val=0,
        max_val=2.0,
        is_ratio=True
    )

# Render financial news feed
def render_news_feed(news_df):
    """Render financial news feed with sentiment highlighting"""
    st.subheader("Financial News")
    
    # Display each news item
    for _, news in news_df.iterrows():
        # Determine sentiment color
        sentiment = news.get("sentiment", "neutral")
        if sentiment == "positive":
            sentiment_color = "rgba(0, 200, 0, 0.1)"
            sentiment_text_color = "green"
        elif sentiment == "negative":
            sentiment_color = "rgba(200, 0, 0, 0.1)"
            sentiment_text_color = "red"
        else:
            sentiment_color = "rgba(200, 200, 200, 0.1)"
            sentiment_text_color = "gray"
        
        # Format timestamp
        timestamp = news.get("timestamp", datetime.datetime.now())
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except:
                pass
        
        if isinstance(timestamp, datetime.datetime):
            time_ago = datetime.datetime.now() - timestamp
            if time_ago.days > 0:
                time_display = f"{time_ago.days}d ago"
            elif time_ago.seconds >= 3600:
                time_display = f"{time_ago.seconds // 3600}h ago"
            else:
                time_display = f"{time_ago.seconds // 60}m ago"
        else:
            time_display = "Unknown"
        
        # Display news item
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: {sentiment_color};">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-size: 0.8rem; color: #666;">{news.get('source', 'Unknown Source')}</span>
                <span style="font-size: 0.8rem; color: #666;">{time_display}</span>
            </div>
            <p style="font-size: 1.0rem; font-weight: 600; margin-bottom: 5px;">{news.get('headline', 'No headline')}</p>
            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                <span style="font-size: 0.8rem; text-transform: capitalize; color: #666;">{news.get('category', 'general').replace('_', ' ')}</span>
                <span style="font-size: 0.8rem; color: {sentiment_text_color}; font-weight: 600;">{sentiment.title()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Render VIX and fear/greed history chart
def render_vix_history(vix_df):
    """Render VIX and fear/greed index history chart"""
    st.subheader("VIX & Fear/Greed Index History")
    
    # Create the subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add VIX trace
    fig.add_trace(
        go.Scatter(
            x=vix_df["date"],
            y=vix_df["vix"],
            mode="lines",
            name="VIX",
            line=dict(width=2, color="#d62728")
        ),
        secondary_y=False
    )
    
    # Add Fear/Greed Index trace
    fig.add_trace(
        go.Scatter(
            x=vix_df["date"],
            y=vix_df["fear_greed_index"],
            mode="lines",
            name="Fear/Greed Index",
            line=dict(width=2, color="#2ca02c")
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="white"
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(
        title_text="VIX",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)",
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Fear/Greed Index",
        showgrid=False,
        secondary_y=True,
        range=[0, 100]
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Render sector performance heatmap
def render_sector_heatmap(sector_data):
    """Render sector performance heatmap"""
    st.subheader("Sector Performance")
    
    # Get sectors and performance data
    sectors = sector_data.get("sectors", [])
    performance = sector_data.get("performance", {})
    
    # Time periods for the tabs
    periods = ["daily", "weekly", "monthly", "ytd"]
    period_labels = {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly", "ytd": "YTD"}
    
    # Create tabs for different time periods
    period_tabs = st.tabs([period_labels[p] for p in periods])
    
    # Function to create a sector heatmap for a specific time period
    def create_sector_heatmap(period, container):
        # Get performance data for this period
        period_data = performance.get(period, {})
        
        if not period_data:
            container.info(f"No {period} sector data available")
            return
        
        # Create DataFrame for the heatmap
        data = []
        for sector in sectors:
            value = period_data.get(sector, 0)
            data.append({
                "Sector": sector,
                "Performance": value
            })
        
        df = pd.DataFrame(data)
        
        # Sort by performance
        df = df.sort_values("Performance", ascending=False)
        
        # Create the bar chart
        fig = px.bar(
            df,
            x="Performance",
            y="Sector",
            orientation="h",
            color="Performance",
            color_continuous_scale=["red", "white", "green"],
            color_continuous_midpoint=0,
            range_color=[-max(abs(df["Performance"].min()), abs(df["Performance"].max())), 
                         max(abs(df["Performance"].min()), abs(df["Performance"].max()))]
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(
                title="",
                autorange="reversed"
            ),
            xaxis=dict(
                title="Performance (%)"
            ),
            coloraxis_showscale=True,
            plot_bgcolor="white"
        )
        
        # Add percentage signs to the values
        fig.update_traces(
            texttemplate="%{x:.1f}%",
            textposition="outside"
        )
        
        container.plotly_chart(fig, use_container_width=True)
    
    # Create heatmap for each time period
    for i, period in enumerate(periods):
        with period_tabs[i]:
            create_sector_heatmap(period, period_tabs[i])

# Main render function for this component
def render(db):
    """Main render function for the Market Context section"""
    
    # Get data from MongoDB (or mock data if unavailable)
    sentiment_data = get_market_sentiment(db)
    news_df = get_market_news(db)
    vix_df = get_vix_history(db)
    sector_data = get_sector_performance(db)
    
    # Create a 2-column layout for the top section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Render sentiment indicators
        render_sentiment_indicators(sentiment_data)
    
    with col2:
        # Render news feed
        render_news_feed(news_df)
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render VIX and fear/greed history chart
    render_vix_history(vix_df)
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render sector performance heatmap
    render_sector_heatmap(sector_data)
