"""
ML Dashboard Integration

This module integrates the ML pipeline with the Streamlit dashboard,
providing UI components for model management, backtesting, and autonomous trading.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import threading
import time
import os
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Import ML pipeline components
try:
    from trading_bot.ml_pipeline.autonomous_integration import AutonomousIntegration
    from trading_bot.ml_pipeline.feature_engineering import FeatureEngineeringFramework
    from trading_bot.ml_pipeline.model_trainer import ModelTrainer
    from trading_bot.ml_pipeline.signal_generator import SignalGenerator
    ML_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML pipeline components not available: {e}")
    ML_COMPONENTS_AVAILABLE = False


class MLDashboardIntegration:
    """Integrates ML pipeline with the Streamlit dashboard"""
    
    def __init__(self, config=None):
        """Initialize the ML dashboard integration"""
        self.config = config or {}
        
        # Initialize the autonomous integration system
        if ML_COMPONENTS_AVAILABLE:
            try:
                self.autonomous_system = AutonomousIntegration(config=self.config)
                logger.info("Autonomous trading system initialized")
            except Exception as e:
                logger.error(f"Error initializing autonomous system: {e}")
                self.autonomous_system = None
        else:
            self.autonomous_system = None
            
        # Initialize session state variables if not present
        if 'autonomous_enabled' not in st.session_state:
            st.session_state.autonomous_enabled = False
        if 'ml_insights' not in st.session_state:
            st.session_state.ml_insights = {}
        if 'last_backtest_results' not in st.session_state:
            st.session_state.last_backtest_results = {}
        if 'backtesting_status' not in st.session_state:
            st.session_state.backtesting_status = "idle"
        if 'autonomous_thread' not in st.session_state:
            st.session_state.autonomous_thread = None
        if 'autonomous_status' not in st.session_state:
            st.session_state.autonomous_status = "idle"
        if 'ml_signals' not in st.session_state:
            st.session_state.ml_signals = []
    
    def render_ml_controls(self):
        """Render ML system controls in the sidebar"""
        st.sidebar.markdown("## 🧠 ML Trading System")
        
        if not ML_COMPONENTS_AVAILABLE or self.autonomous_system is None:
            st.sidebar.error("ML trading system is not available")
            return
        
        # ML system status
        system_status = "Active" if st.session_state.autonomous_enabled else "Inactive"
        st.sidebar.markdown(f"**System Status:** {system_status}")
        
        # Toggle autonomous trading
        if st.sidebar.button("Toggle Autonomous Trading"):
            st.session_state.autonomous_enabled = not st.session_state.autonomous_enabled
            if st.session_state.autonomous_enabled:
                self._start_autonomous_thread()
            else:
                self._stop_autonomous_thread()
            st.rerun()
        
        # Show status indicator
        status_color = "green" if st.session_state.autonomous_enabled else "red"
        st.sidebar.markdown(f"<div style='background-color: {status_color}; height: 10px; width: 100%; border-radius: 5px;'></div>", 
                         unsafe_allow_html=True)
        
        # Run backtesting
        st.sidebar.markdown("### Backtesting")
        if st.sidebar.button("Run Backtest Cycle"):
            self._run_backtest()
        
        # Show backtest status
        st.sidebar.markdown(f"**Backtest Status:** {st.session_state.backtesting_status}")
        
        # Advanced settings
        with st.sidebar.expander("Advanced Settings"):
            confidence_threshold = st.slider(
                "Signal Confidence Threshold", 
                min_value=0.5, 
                max_value=0.9, 
                value=0.65, 
                step=0.05
            )
            
            if confidence_threshold != self.config.get('confidence_threshold', 0.65):
                self.config['confidence_threshold'] = confidence_threshold
                if self.autonomous_system and hasattr(self.autonomous_system, 'signal_generator'):
                    self.autonomous_system.signal_generator.confidence_threshold = confidence_threshold
    
    def render_ml_insights_tab(self):
        """Render ML insights in the dashboard tab"""
        st.markdown("## 🧠 ML Trading Insights")
        
        if not ML_COMPONENTS_AVAILABLE or self.autonomous_system is None:
            st.error("ML trading system is not available")
            return
        
        # Get latest insights
        with st.spinner("Updating ML insights..."):
            self._update_ml_insights()
        
        # ML System Status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### System Status")
            status = "🟢 Active" if st.session_state.autonomous_enabled else "🔴 Inactive"
            st.markdown(f"**Status:** {status}")
            st.markdown(f"**Models Loaded:** {len(st.session_state.ml_insights.get('models', []))}")
            st.markdown(f"**Signal Confidence Threshold:** {self.config.get('confidence_threshold', 0.65)}")
            
            # System controls
            if st.button("Toggle Autonomous Trading", key="toggle_main"):
                st.session_state.autonomous_enabled = not st.session_state.autonomous_enabled
                if st.session_state.autonomous_enabled:
                    self._start_autonomous_thread()
                else:
                    self._stop_autonomous_thread()
                st.rerun()
        
        with col2:
            st.markdown("### Signal Performance")
            perf = st.session_state.ml_insights.get('signal_performance', {})
            
            if perf:
                st.markdown(f"**Total Signals:** {perf.get('total_signals', 0)}")
                st.markdown(f"**Buy Signals:** {perf.get('buy_signals', 0)} ({perf.get('signal_types', {}).get('buy_pct', 0):.1%})")
                st.markdown(f"**Sell Signals:** {perf.get('sell_signals', 0)} ({perf.get('signal_types', {}).get('sell_pct', 0):.1%})")
                st.markdown(f"**Average Confidence:** {perf.get('avg_confidence', 0):.2f}")
            else:
                st.markdown("No signal performance data available yet")
        
        # Recent Signals
        st.markdown("### Recent ML Trading Signals")
        signals = st.session_state.ml_signals
        
        if signals:
            # Convert to DataFrame for display
            signals_df = pd.DataFrame([
                {
                    'Ticker': s.get('ticker', ''),
                    'Time': s.get('timestamp', datetime.now()),
                    'Signal': s.get('signal', 'neutral'),
                    'Confidence': s.get('confidence', 0.0),
                    'Position Size': s.get('position_size', 0.0)
                }
                for s in signals
            ])
            
            signals_df['Time'] = pd.to_datetime(signals_df['Time'])
            signals_df = signals_df.sort_values('Time', ascending=False)
            
            # Style the dataframe
            def color_signal(val):
                if val == 'buy':
                    return 'background-color: #d4f7dc'
                elif val == 'sell':
                    return 'background-color: #f7dad4'
                else:
                    return ''
            
            styled_df = signals_df.style.applymap(color_signal, subset=['Signal'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Visualize signal confidence distribution
            if len(signals_df) > 1:
                signal_viz = px.histogram(
                    signals_df, 
                    x="Confidence", 
                    color="Signal", 
                    barmode="group",
                    title="Signal Confidence Distribution",
                    color_discrete_map={'buy': '#22c55e', 'sell': '#ef4444', 'neutral': '#a3a3a3'}
                )
                st.plotly_chart(signal_viz, use_container_width=True)
        else:
            st.info("No recent ML signals available")
        
        # ML Models
        st.markdown("### ML Models")
        models = st.session_state.ml_insights.get('models', [])
        
        if models:
            # Convert to DataFrame for display
            models_df = pd.DataFrame(models)
            st.dataframe(models_df, use_container_width=True)
            
            # Show model performance chart
            if 'accuracy' in models_df.columns:
                model_viz = px.bar(
                    models_df,
                    x='name',
                    y='accuracy',
                    color='type',
                    title="Model Accuracy",
                    labels={'name': 'Model Name', 'accuracy': 'Accuracy', 'type': 'Model Type'}
                )
                model_viz.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(model_viz, use_container_width=True)
        else:
            st.warning("No ML models loaded")
            
            # Button to train initial model
            if st.button("Train Initial Model"):
                self._train_initial_model()
    
    def render_backtester_tab(self):
        """Render ML backtester in the backtester tab"""
        st.markdown("## 🧮 ML Strategy Backtester")
        
        if not ML_COMPONENTS_AVAILABLE or self.autonomous_system is None:
            st.error("ML backtesting system is not available")
            return
        
        # Backtesting controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tickers = st.multiselect(
                "Select Tickers",
                options=['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'NVDA', 'TSLA'],
                default=['SPY', 'QQQ', 'AAPL']
            )
        
        with col2:
            timeframes = st.multiselect(
                "Select Timeframes",
                options=['1d', '1h', '15m'],
                default=['1d']
            )
        
        with col3:
            optimize_models = st.checkbox("Optimize ML Models", value=True)
            
            if st.button("Run Backtest"):
                self._run_backtest(tickers, timeframes, optimize_models)
        
        # Show backtest status
        st.markdown(f"**Backtest Status:** {st.session_state.backtesting_status}")
        
        # Show backtest results
        if st.session_state.last_backtest_results:
            st.markdown("### Backtest Results")
            
            results = st.session_state.last_backtest_results
            insights = results.get('insights', {})
            
            # Winning vs Losing strategies
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Winning Strategies")
                winning = results.get('results', {}).get('winning_strategies', [])
                if winning:
                    win_df = pd.DataFrame([
                        {
                            'Strategy': w.get('strategy', {}).get('template', ''),
                            'Return': w.get('aggregate_performance', {}).get('return', 0),
                            'Sharpe': w.get('aggregate_performance', {}).get('sharpe_ratio', 0),
                            'Max DD': w.get('aggregate_performance', {}).get('max_drawdown', 0),
                            'Win Rate': w.get('aggregate_performance', {}).get('win_rate', 0)
                        }
                        for w in winning
                    ])
                    st.dataframe(win_df, use_container_width=True)
                else:
                    st.info("No winning strategies found")
            
            with col2:
                st.markdown("#### Losing Strategies")
                losing = results.get('results', {}).get('losing_strategies', [])
                if losing:
                    lose_df = pd.DataFrame([
                        {
                            'Strategy': l.get('strategy', {}).get('template', ''),
                            'Return': l.get('aggregate_performance', {}).get('return', 0),
                            'Sharpe': l.get('aggregate_performance', {}).get('sharpe_ratio', 0),
                            'Max DD': l.get('aggregate_performance', {}).get('max_drawdown', 0),
                            'Win Rate': l.get('aggregate_performance', {}).get('win_rate', 0)
                        }
                        for l in losing
                    ])
                    st.dataframe(lose_df, use_container_width=True)
                else:
                    st.info("No losing strategies found")
            
            # Strategy Insights
            st.markdown("### Strategy Insights")
            
            if insights:
                # Winning patterns
                with st.expander("Winning Strategy Patterns", expanded=True):
                    patterns = insights.get('winning_patterns', [])
                    if patterns:
                        for i, pattern in enumerate(patterns):
                            st.markdown(f"{i+1}. {pattern}")
                    else:
                        st.info("No clear winning patterns identified")
                
                # Parameter insights
                with st.expander("Parameter Insights"):
                    param_insights = insights.get('parameter_insights', {})
                    if param_insights:
                        st.json(param_insights)
                    else:
                        st.info("No parameter insights available")
                
                # Market condition analysis
                with st.expander("Market Condition Analysis"):
                    market_analysis = insights.get('market_condition_analysis', {})
                    if market_analysis:
                        st.json(market_analysis)
                    else:
                        st.info("No market condition analysis available")
            else:
                st.warning("No insights available from backtest")
        else:
            st.info("Run a backtest to see results")
    
    def _update_ml_insights(self):
        """Update ML insights from the autonomous system"""
        if not self.autonomous_system:
            return
            
        try:
            # Generate insights
            insights = self.autonomous_system.generate_dashboard_insights()
            st.session_state.ml_insights = insights
            
            # Update signals
            if self.autonomous_system.signal_generator:
                signals = self.autonomous_system.signal_generator.get_signal_history(days_back=7)
                st.session_state.ml_signals = signals
                
        except Exception as e:
            logger.error(f"Error updating ML insights: {e}")
    
    def _run_backtest(self, tickers=None, timeframes=None, optimize_models=True):
        """Run a backtesting cycle in a separate thread"""
        if st.session_state.backtesting_status == "running":
            st.warning("Backtest already running!")
            return
            
        st.session_state.backtesting_status = "running"
        
        def backtest_thread():
            try:
                # Run backtest cycle
                results = self.autonomous_system.run_backtest_cycle(
                    tickers=tickers,
                    timeframes=timeframes,
                    optimize_models=optimize_models
                )
                
                # Store results
                st.session_state.last_backtest_results = results
                st.session_state.backtesting_status = "completed"
                
                # Update insights
                self._update_ml_insights()
                
            except Exception as e:
                logger.error(f"Error in backtest thread: {e}")
                st.session_state.backtesting_status = "error"
        
        # Start thread
        thread = threading.Thread(target=backtest_thread)
        thread.daemon = True
        thread.start()
    
    def _start_autonomous_thread(self):
        """Start the autonomous trading thread"""
        if st.session_state.autonomous_thread is not None and st.session_state.autonomous_thread.is_alive():
            logger.warning("Autonomous thread already running")
            return
            
        st.session_state.autonomous_status = "running"
        
        def autonomous_thread():
            """Background thread for autonomous trading"""
            while st.session_state.autonomous_enabled:
                try:
                    # Run autonomous cycle
                    self.autonomous_system.run_autonomous_cycle()
                    
                    # Update insights
                    self._update_ml_insights()
                    
                except Exception as e:
                    logger.error(f"Error in autonomous thread: {e}")
                
                # Sleep for interval (default 15 minutes)
                sleep_time = self.config.get('autonomous_interval_minutes', 15) * 60
                time.sleep(sleep_time)
            
            st.session_state.autonomous_status = "stopped"
        
        # Start thread
        thread = threading.Thread(target=autonomous_thread)
        thread.daemon = True
        thread.start()
        
        st.session_state.autonomous_thread = thread
    
    def _stop_autonomous_thread(self):
        """Stop the autonomous trading thread"""
        st.session_state.autonomous_enabled = False
        st.session_state.autonomous_status = "stopping"
        
        # Thread will stop on next cycle
        logger.info("Stopping autonomous trading thread")
    
    def _train_initial_model(self):
        """Train an initial ML model"""
        if not self.autonomous_system or not hasattr(self.autonomous_system, 'model_trainer'):
            st.error("ML training system not available")
            return
            
        with st.spinner("Training initial ML model... This may take a few minutes"):
            try:
                # Get default tickers
                tickers = self.config.get('default_tickers', ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN'])
                
                # Get data for each ticker
                all_data = {}
                for ticker in tickers:
                    data = self.autonomous_system._get_market_data(ticker, '1d', days=365)
                    if data is not None and not data.empty:
                        # Generate features
                        data = self.autonomous_system.feature_engineering.generate_features(data)
                        all_data[ticker] = data
                
                # Generate training labels
                for ticker, data in all_data.items():
                    # Generate directional labels
                    data['label_direction'] = self.autonomous_system.model_trainer.generate_labels(
                        data, label_type='directional', horizon=5, threshold=0.0
                    )
                
                # Combine data from all tickers
                combined_data = pd.concat(all_data.values())
                combined_data = combined_data.dropna()
                
                # Select relevant features
                features = [col for col in combined_data.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume', 'date', 
                                         'label_direction']]
                
                # Train model
                X_train, X_test, y_train, y_test, _ = self.autonomous_system.model_trainer.prepare_training_data(
                    combined_data, 'label_direction', features
                )
                
                model = self.autonomous_system.model_trainer.train_model(
                    X_train, y_train, model_type='random_forest'
                )
                
                # Evaluate model
                metrics = self.autonomous_system.model_trainer.evaluate_model(model, X_test, y_test)
                
                # Save model
                model_name = 'initial_directional_model'
                self.autonomous_system.model_trainer.save_model(
                    model, 
                    model_name, 
                    {
                        'features': features,
                        'metrics': metrics,
                        'training_date': datetime.now().isoformat()
                    }
                )
                
                # Add to signal generator
                if self.autonomous_system.signal_generator:
                    self.autonomous_system.signal_generator.add_model(
                        model_name, 
                        model, 
                        {
                            'features': features,
                            'metrics': metrics,
                            'weight': 1.0
                        }
                    )
                
                st.success(f"Initial model trained with accuracy: {metrics['accuracy']:.4f}")
                
                # Update insights
                self._update_ml_insights()
                
            except Exception as e:
                logger.error(f"Error training initial model: {e}")
                st.error(f"Error training model: {str(e)}")


# Function to get the ML dashboard integration
def get_ml_dashboard_integration(config=None):
    """Get or create the ML dashboard integration singleton"""
    if 'ml_dashboard' not in st.session_state:
        st.session_state.ml_dashboard = MLDashboardIntegration(config)
    return st.session_state.ml_dashboard
