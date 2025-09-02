"""AI Chat Widget for trading dashboard.

Provides a draggable, persistent chat widget that can be
triggered from anywhere in the application."""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
import logging

# Import BenBotAssistant
try:
    from trading_bot.assistant.benbot_assistant import BenBotAssistant
    BENBOT_AVAILABLE = True
except ImportError:
    BENBOT_AVAILABLE = False
    logging.warning("BenBotAssistant not available. Falling back to mock implementation.")

def render_ai_chat_widget(assistant=None):
    """Render a draggable, collapsible AI chat widget with context awareness.
    
    Args:
        assistant: Optional BenBotAssistant instance. If provided, will use this
                  instance instead of creating a new one.
    """
    # Setup initial state if not present
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'chat_widget_position' not in st.session_state:
        # Default position is bottom right corner
        st.session_state.chat_widget_position = {'top': 'auto', 'right': '20px', 'bottom': '20px', 'left': 'auto'}
    
    # Use the passed assistant if provided, otherwise check session state or initialize
    if assistant is not None:
        # Use the passed assistant
        st.session_state.benbot_assistant = assistant
        st.session_state.benbot_initialized = True
        logging.info("Using BenBotAssistant passed from main app")
    elif BENBOT_AVAILABLE and 'benbot_assistant' not in st.session_state:
        try:
            # Only initialize if not already in session state
            st.session_state.benbot_assistant = BenBotAssistant()
            st.session_state.benbot_initialized = True
            logging.info("BenBotAssistant initialized in widget")
        except Exception as e:
            logging.error(f"Error initializing BenBotAssistant: {e}")
            st.session_state.benbot_initialized = False
    
    # Chat widget CSS
    st.markdown("""
    <style>
    .chat-widget {
        position: fixed;
        width: 350px;
        height: 450px;
        background-color: rgba(17, 25, 40, 0.95);
        border: 1px solid #2196F3;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        z-index: 1000;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .chat-header {
        padding: 10px 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: move;
        background-color: rgba(33, 150, 243, 0.2);
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }
    
    .chat-header h3 {
        margin: 0;
        color: #ffffff;
        font-size: 16px;
        font-weight: 600;
    }
    
    .chat-controls {
        display: flex;
        gap: 10px;
    }
    
    .chat-control-btn {
        background: none;
        border: none;
        color: rgba(255, 255, 255, 0.7);
        cursor: pointer;
        font-size: 18px;
        padding: 0;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
    }
    
    .chat-control-btn:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 15px;
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    
    .message {
        max-width: 80%;
        padding: 10px 15px;
        border-radius: 18px;
        line-height: 1.4;
        word-wrap: break-word;
        font-size: 14px;
    }
    
    .user-message {
        align-self: flex-end;
        background-color: rgba(33, 150, 243, 0.15);
        border: 1px solid rgba(33, 150, 243, 0.3);
        color: #ffffff;
        border-bottom-right-radius: 5px;
    }
    
    .assistant-message {
        align-self: flex-start;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border-bottom-left-radius: 5px;
    }
    
    .message-time {
        font-size: 10px;
        color: rgba(255, 255, 255, 0.5);
        margin-top: 5px;
        text-align: right;
    }
    
    .chat-input-container {
        padding: 10px 15px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        gap: 10px;
    }
    
    .chat-input {
        flex: 1;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 10px 15px;
        color: #ffffff;
        outline: none;
        resize: none;
        font-family: inherit;
        font-size: 14px;
    }
    
    .chat-input:focus {
        border-color: rgba(33, 150, 243, 0.5);
    }
    
    .chat-send-btn {
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .chat-send-btn:hover {
        background-color: #1976D2;
    }
    
    .context-indicator {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.6);
        padding: 5px 15px;
        background-color: rgba(0, 0, 0, 0.2);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Make scrollbar more subtle */
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 3px;
    }
    
    /* Trading context styling */
    .trading-context {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 10px;
        color: rgba(255, 255, 255, 0.7);
        background-color: rgba(33, 150, 243, 0.1);
        padding: 4px 15px;
        border-bottom: 1px solid rgba(33, 150, 243, 0.2);
    }
    
    .context-label {
        font-weight: 600;
    }
    
    .context-value {
        color: #66BB6A;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Get current trading context
    current_context = get_current_trading_context()
    
    # Format timestamp
    current_time = datetime.now().strftime("%H:%M")
    
    # Start widget HTML
    position_style = '; '.join([f"{k}: {v}" for k, v in st.session_state.chat_widget_position.items()])
    
    st.markdown(f"""
    <div class="chat-widget" id="aiChatWidget" style="{position_style}">
        <div class="chat-header" id="chatHeader">
            <h3>🤖 Trading Assistant</h3>
            <div class="chat-controls">
                <button class="chat-control-btn" id="minimizeChat">−</button>
                <button class="chat-control-btn" id="closeChat">×</button>
            </div>
        </div>
        
        <div class="trading-context">
            <span><span class="context-label">Symbol:</span> <span class="context-value">{current_context['symbol']}</span></span>
            <span><span class="context-label">Tab:</span> <span class="context-value">{current_context['current_tab']}</span></span>
        </div>
        
        <div class="chat-messages" id="chatMessages">
    """, unsafe_allow_html=True)
    
    # Render chat messages
    for msg in st.session_state.chat_history:
        msg_type = "user-message" if msg["role"] == "user" else "assistant-message"
        st.markdown(f"""
        <div class="message {msg_type}">
            {msg["content"]}
            <div class="message-time">{msg.get("time", current_time)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Close messages div and add input form
    st.markdown("""
        </div>
        <div class="chat-input-container">
            <textarea class="chat-input" id="chatInput" placeholder="Ask about your trading data..." rows="1"></textarea>
            <button class="chat-send-btn" id="sendMessage">➤</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # JavaScript for widget functionality
    st.components.v1.html("""
    <script>
        // Wait for the DOM to be fully loaded
        document.addEventListener('DOMContentLoaded', function() {
            setupChatWidget();
        });
        
        // Setup may execute after DOMContentLoaded if Streamlit loads this later
        setupChatWidget();
        
        function setupChatWidget() {
            const chatWidget = document.getElementById('aiChatWidget');
            const chatHeader = document.getElementById('chatHeader');
            const minimizeBtn = document.getElementById('minimizeChat');
            const closeBtn = document.getElementById('closeChat');
            const chatInput = document.getElementById('chatInput');
            const sendBtn = document.getElementById('sendMessage');
            
            if (!chatWidget || !chatHeader) return; // Elements not loaded yet
            
            // Make widget draggable
            let isDragging = false;
            let dragOffsetX = 0;
            let dragOffsetY = 0;
            
            chatHeader.addEventListener('mousedown', function(e) {
                isDragging = true;
                dragOffsetX = e.clientX - chatWidget.getBoundingClientRect().left;
                dragOffsetY = e.clientY - chatWidget.getBoundingClientRect().top;
            });
            
            document.addEventListener('mousemove', function(e) {
                if (isDragging) {
                    const left = e.clientX - dragOffsetX;
                    const top = e.clientY - dragOffsetY;
                    
                    chatWidget.style.left = left + 'px';
                    chatWidget.style.top = top + 'px';
                    chatWidget.style.right = 'auto';
                    chatWidget.style.bottom = 'auto';
                    
                    // Send position to Streamlit
                    updateWidgetPosition(left, top);
                }
            });
            
            document.addEventListener('mouseup', function() {
                isDragging = false;
            });
            
            // Handle minimize button
            minimizeBtn.addEventListener('click', function() {
                const isMinimized = chatWidget.classList.toggle('minimized');
                if (isMinimized) {
                    chatWidget.style.height = '40px';
                    minimizeBtn.textContent = '□';
                } else {
                    chatWidget.style.height = '450px';
                    minimizeBtn.textContent = '−';
                }
            });
            
            // Handle close button
            closeBtn.addEventListener('click', function() {
                window.parent.postMessage({type: 'closeAIChat'}, '*');
            });
            
            // Handle send message
            function sendMessage() {
                const message = chatInput.value.trim();
                if (message) {
                    window.parent.postMessage({
                        type: 'sendAIChatMessage',
                        message: message
                    }, '*');
                    chatInput.value = '';
                }
            }
            
            sendBtn.addEventListener('click', sendMessage);
            chatInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            // Auto-resize textarea
            chatInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });
            
            // Function to update widget position in Streamlit
            function updateWidgetPosition(left, top) {
                window.parent.postMessage({
                    type: 'updateChatPosition',
                    position: {
                        left: left + 'px',
                        top: top + 'px',
                        right: 'auto',
                        bottom: 'auto'
                    }
                }, '*');
            }
            
            // Scroll to bottom of chat
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
    </script>
    """, height=0)
    
    # Handle incoming JS events
    handle_chat_events()

def get_current_trading_context():
    """Get current trading context for the AI assistant."""
    # Get current app state from session state
    active_tab = st.session_state.get("active_tab", "Dashboard")
    active_symbol = st.session_state.get("active_symbol", st.session_state.get("selected_symbol", "SPY"))
    last_run = st.session_state.get("last_pipeline_run", "Never")
    pipeline_status = st.session_state.get("pipeline_status", "ok")
    
    # Build a comprehensive context object
    context = {
        "symbol": active_symbol,
        "current_tab": active_tab,
        "last_pipeline_run": last_run,
        "pipeline_status": pipeline_status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "available_strategies": [],
        "opportunities": []
    }
    
    # Add code context for deeper awareness
    code_context = {
        "strategy_types": [
            "stock_momentum", "stock_mean_reversion", "stock_breakout", "stock_trend", 
            "crypto_momentum", "crypto_range", "crypto_arbitrage", "crypto_sentiment",
            "forex_trend", "forex_carry", "forex_interest_rate", "forex_momentum",
            "options_covered_call", "options_iron_condor", "options_bull_call_spread", "options_calendar_spread"
        ],
        "asset_classes": ["stock", "crypto", "forex", "options", "futures"],
        "indicators": ["RSI", "MACD", "Bollinger Bands", "ATR", "MFI", "OBV", "Ichimoku", "ADX"],
        "data_sources": ["Alpha Vantage", "Finnhub", "Tradier", "Alpaca", "NewsAPI", "Marketaux", "NewsData.io"]
    }
    context.update({"code_context": code_context})
    
    # Add more context based on the active tab
    if active_tab == "Autonomous":
        # Get opportunities if available
        opportunities = []
        if "pipeline_results" in st.session_state:
            opportunities = st.session_state.pipeline_results
        else:
            try:
                # Try to get from app_new
                from app_new import get_mock_approved_opportunities
                opportunities = get_mock_approved_opportunities()
            except:
                pass
                
        if opportunities:
            # Extract key opportunity info for context
            opp_summaries = []
            for opp in opportunities[:3]:  # Limit to top 3 for brevity
                if isinstance(opp, dict):
                    symbol = opp.get('symbol', 'Unknown')
                    asset_type = opp.get('asset_type', 'stock')
                    strategy = opp.get('strategy', f"{asset_type}_strategy")
                    expected_return = opp.get('expected_return', 0.0)
                    confidence = opp.get('confidence', 0.0)
                    opp_summaries.append({
                        "symbol": symbol,
                        "strategy": strategy,
                        "asset_type": asset_type,
                        "expected_return": f"{expected_return:+.1%}" if isinstance(expected_return, (int, float)) else expected_return,
                        "confidence": f"{confidence:.0%}" if isinstance(confidence, (int, float)) else confidence
                    })
            context["opportunities"] = opp_summaries
    
    # Add context for dashboard tab
    if active_tab == "Dashboard":
        context["market_regime"] = st.session_state.get("market_regime", "Bullish")
        context["sector_performance"] = st.session_state.get("sector_performance", {"Technology": "+2.3%", "Energy": "-0.5%"})
    
    return context

def handle_chat_events():
    """Handle events from the JS chat widget."""
    # Create a container for the event listener
    event_listener = st.empty()
    
    # Listen for message events from JS
    event_listener.markdown("""
    <script>
        window.addEventListener('message', function(event) {
            const data = event.data;
            
            // Handle different event types
            if (data.type === 'closeAIChat') {
                window.parent.postMessage({type: 'streamlit:setComponentValue', value: {action: 'close_chat'}}, '*');
            } else if (data.type === 'sendAIChatMessage') {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue', 
                    value: {
                        action: 'new_message',
                        message: data.message
                    }
                }, '*');
            } else if (data.type === 'updateChatPosition') {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue', 
                    value: {
                        action: 'update_position',
                        position: data.position
                    }
                }, '*');
            }
        });
    </script>
    """, unsafe_allow_html=True)
    
    # Custom component to receive the message
    component_value = st.components.v1.html(
        '<div id="eventReceiver"></div>',
        height=0
    )
    
    # Process component value if present
    if component_value:
        if isinstance(component_value, dict):
            action = component_value.get('action')
            
            if action == 'close_chat':
                st.session_state.chat_shown = False
                st.experimental_rerun()
                
            elif action == 'new_message':
                message = component_value.get('message')
                if message:
                    # Add user message to history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": message,
                        "time": datetime.now().strftime("%H:%M")
                    })
                    
                    # Get AI response (now using BenBotAssistant if available)
                    ai_response = generate_ai_response(message, get_current_trading_context())
                    
                    # Add AI response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": ai_response,
                        "time": datetime.now().strftime("%H:%M")
                    })
                    
                    st.experimental_rerun()
                    
            elif action == 'update_position':
                position = component_value.get('position')
                if position:
                    st.session_state.chat_widget_position = position

def generate_ai_response(user_message, context):
    """Generate AI response based on user message and trading context.
    
    Now uses BenBotAssistant if available, otherwise falls back to the mock implementation.
    """
    # Use BenBotAssistant if available
    if BENBOT_AVAILABLE and 'benbot_assistant' in st.session_state and st.session_state.benbot_assistant is not None:
        try:
            # Convert context dict to the format expected by BenBotAssistant
            assistant_context = {
                "current_backtest": context.get("symbol"),
                "current_model": None,
                "current_topic": context.get("current_tab", "dashboard").lower()
            }
            
            # Add additional context
            for key, value in context.items():
                if key not in assistant_context:
                    assistant_context[key] = value
            
            # Get the assistant from session state
            assistant = st.session_state.benbot_assistant
            
            # Log that we're using the BenBot assistant
            logging.info(f"Using BenBot assistant for response to: {user_message[:50]}...")
            
            # Process the message with BenBotAssistant
            response = assistant.process_message(user_message, context=context.get("current_tab", "dashboard").lower())
            
            # Handle response that might be a dict with text and data
            if isinstance(response, dict) and "text" in response:
                # Store any visualizations or data for later use
                if "data" in response:
                    st.session_state.last_assistant_data = response["data"]
                return response["text"]
            
            return response
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.error(f"Error getting response from BenBotAssistant: {e}\n{error_details}")
            # Fall back to mock implementation
            return _generate_mock_response(user_message, context)
    else:
        # Log why we're falling back
        if not BENBOT_AVAILABLE:
            logging.warning("BenBot assistant not available (import failed). Using mock responses.")
        elif 'benbot_assistant' not in st.session_state:
            logging.warning("BenBot assistant not in session state. Using mock responses.")
        elif st.session_state.benbot_assistant is None:
            logging.warning("BenBot assistant is None in session state. Using mock responses.")
            
        # Fall back to mock implementation
        return _generate_mock_response(user_message, context)

def _generate_mock_response(user_message, context):
    """Placeholder response logic - would be replaced with actual LLM call."""
    # Same as the original implementation for backward compatibility
    symbol = context['symbol']
    current_tab = context['current_tab']
    code_context = context.get('code_context', {})
    asset_classes = code_context.get('asset_classes', [])
    opportunities = context.get('opportunities', [])
    
    # More specific responses based on user query and current context
    if "code" in user_message.lower() or "codebase" in user_message.lower() or "how does" in user_message.lower():
        return f"The trading system is built with multiple asset-class support for {', '.join(asset_classes)}. Each asset class has specialized strategies (e.g., {code_context.get('strategy_types', [])[0:3]}). The system uses the AutonomousOrchestrator to detect asset types, select appropriate strategies, and evaluate opportunities through backtesting and risk assessment."
    
    elif "explain" in user_message.lower() and "autonomous" in user_message.lower():
        return f"The autonomous pipeline works in several stages: 1) Opportunity discovery across all asset classes, 2) Asset detection to classify symbols, 3) Strategy selection based on asset type and market conditions, 4) Parameter optimization through machine learning, 5) Backtesting against historical data, and 6) Risk assessment. You're currently in the {current_tab} tab where you can {'view and approve opportunities' if current_tab == 'Autonomous' else 'see an overview of the system'}."    
    
    elif "performance" in user_message.lower() or "metrics" in user_message.lower():
        asset_specific = "" if current_tab != "Autonomous" else f"\n\nCurrently available {'opportunities' if opportunities else 'strategies'} include: {', '.join([f"{o.get('symbol')} ({o.get('strategy')}, {o.get('expected_return')})" for o in opportunities[:2]])}" if opportunities else ""
        
        return f"For {symbol}, I'm analyzing performance metrics based on your {current_tab} view. The key metrics show a momentum pattern with increasing volume. The RSI is currently showing potential divergence, while the ATR indicates moderate volatility.{asset_specific}"
    
    elif "strategy" in user_message.lower() or "approach" in user_message.lower():
        strategy_suggestions = []
        for asset_type in asset_classes:
            if asset_type == "stock":
                strategy_suggestions.append(f"For stocks like {symbol}: momentum or mean-reversion")
            elif asset_type == "crypto":
                strategy_suggestions.append("For crypto: range trading in sideways markets")
            elif asset_type == "forex":
                strategy_suggestions.append("For forex: interest rate differential strategies")
            elif asset_type == "options":
                strategy_suggestions.append("For options: vertical spreads in the current IV environment")
        
        return f"Based on current market conditions for {symbol}, I'd recommend strategies aligned with your current asset focus in the {current_tab} tab.\n\n{' '.join(strategy_suggestions[:3])}"
    
    elif "risk" in user_message.lower() or "position size" in user_message.lower():
        return f"The current risk assessment for {symbol} indicates moderate volatility (ATR: 2.3%). For optimal position sizing:\n\n1. Stock positions: 2% account risk per trade\n2. Crypto: smaller sizing (1-1.5%) due to higher volatility\n3. Forex: standard 1-2% with proper stop placement\n4. Options: limit premium to 0.5-1% of account\n\nIn the {current_tab} tab, stop losses should be placed at 2-2.5× ATR from entry."
    
    elif "news" in user_message.lower() or "sentiment" in user_message.lower():
        return f"Recent news sentiment for {symbol} is positive (0.78), with 5 significant articles in the past 24 hours. Most notable is the quarterly earnings report which exceeded analyst expectations by 12%. Based on your focus in the {current_tab} tab, I'd pay special attention to sector-specific news that could affect your open opportunities."
    
    elif "what can you do" in user_message.lower() or "help me" in user_message.lower():
        return f"As your trading assistant, I can:\n\n1. Explain any aspect of the trading system's code and architecture\n2. Analyze current market conditions for {symbol} and other assets\n3. Suggest asset-specific strategies for {', '.join(asset_classes)}\n4. Provide risk assessment and position sizing guidance\n5. Interpret news sentiment and technical indicators\n6. Help troubleshoot the autonomous pipeline\n\nI have full awareness of your current context in the {current_tab} tab."
    
    else:
        opp_context = ""
        if opportunities and len(opportunities) > 0:
            opp = opportunities[0]
            opp_context = f" I see your top opportunity is {opp.get('symbol')} using {opp.get('strategy')} with {opp.get('expected_return')} expected return."
            
        return f"I've analyzed your trading dashboard data for {symbol}.{opp_context} From your current position in the {current_tab} tab, I can see that market conditions are favorable for tactical entries. Would you like insights about technical indicators, fundamentals, or recent news catalysts?"
