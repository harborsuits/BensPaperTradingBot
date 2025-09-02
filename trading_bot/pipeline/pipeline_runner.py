#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Runner Module for Trading Bot

This module provides the primary orchestration for the end-to-end
machine learning pipeline, from data ingestion to model deployment.
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from trading_bot.utils.feature_engineering import FeatureEngineering
from trading_bot.models.model_trainer import ModelTrainer
from trading_bot.models.model_evaluator import WalkForwardEvaluator

class PipelineRunner:
    """
    Main orchestrator for the trading bot pipeline that manages the end-to-end
    process from raw data to deployable trading models.
    """
    
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize the pipeline runner with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config: Dictionary with configuration (if not using config file)
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")
            
        # Setup run ID and output directories
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.config.get('output_dir', './output')
        self.model_dir = os.path.join(self.output_dir, 'models', self.run_id)
        self.results_dir = os.path.join(self.output_dir, 'results', self.run_id)
        
        # Create directories
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Save configuration for reproducibility
        with open(os.path.join(self.output_dir, f'config_{self.run_id}.yaml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        # Initialize component objects
        self._initialize_components()
        
        # Initialize state dictionary to track pipeline progress
        self.state = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'stages_completed': [],
            'data_stats': {},
            'model_metrics': {},
            'status': 'initialized'
        }
        
    def _initialize_components(self):
        """Initialize the main component objects based on configuration."""
        # Feature engineering component
        fe_config = self.config.get('feature_engineering', {})
        self.feature_engineer = FeatureEngineering(fe_config)
        
        # Model trainer component
        model_config = self.config.get('model', {})
        self.model_trainer = ModelTrainer(model_config)
        
        # Evaluator component
        eval_config = self.config.get('evaluation', {})
        self.evaluator = WalkForwardEvaluator(
            model_trainer=self.model_trainer,
            feature_engineer=self.feature_engineer,
            fee_rate=eval_config.get('transaction_cost', 0.001),
            slippage_rate=eval_config.get('slippage_rate', 0.0005),
            params=eval_config
        )
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline and return final results.
        
        Returns:
            Dictionary with pipeline results
        """
        try:
            print(f"Starting pipeline run {self.run_id}")
            self.state['status'] = 'running'
            
            # Stage 1: Data Ingestion
            market_data, sentiment_data = self._ingest_data()
            self.state['stages_completed'].append('data_ingestion')
            
            # Stage 2: Feature Engineering
            feature_data = self._engineer_features(market_data, sentiment_data)
            self.state['stages_completed'].append('feature_engineering')
            
            # Stage 3: Model Training
            models = self._train_models(feature_data)
            self.state['stages_completed'].append('model_training')
            
            # Stage 4: Model Evaluation
            eval_results = self._evaluate_models(models, feature_data)
            self.state['stages_completed'].append('model_evaluation')
            
            # Stage 5: Save Artifacts
            self._save_artifacts(models, eval_results)
            self.state['stages_completed'].append('save_artifacts')
            
            # Update final state
            self.state['end_time'] = datetime.now().isoformat()
            self.state['status'] = 'completed'
            
            print(f"Pipeline run {self.run_id} completed successfully")
            return self.state
            
        except Exception as e:
            self.state['status'] = 'failed'
            self.state['error'] = str(e)
            print(f"Pipeline run {self.run_id} failed: {str(e)}")
            raise
            
    def _ingest_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Ingest market data and optional sentiment data.
        
        Returns:
            Tuple of (market_data, sentiment_data)
        """
        print("Stage 1: Ingesting data")
        data_config = self.config.get('data', {})
        
        # Get market data
        market_data_path = data_config.get('market_data_path')
        if not market_data_path:
            raise ValueError("Market data path must be specified in config")
            
        market_data = pd.read_csv(market_data_path, parse_dates=['date'], index_col='date')
        
        # Log data stats
        self.state['data_stats']['market_data_rows'] = len(market_data)
        self.state['data_stats']['market_data_start'] = market_data.index.min().isoformat()
        self.state['data_stats']['market_data_end'] = market_data.index.max().isoformat()
        
        # Get sentiment data if available
        sentiment_data = None
        sentiment_path = data_config.get('sentiment_data_path')
        if sentiment_path:
            sentiment_data = pd.read_csv(sentiment_path, parse_dates=['date'], index_col='date')
            self.state['data_stats']['sentiment_data_rows'] = len(sentiment_data)
            
        print(f"Loaded {len(market_data)} market data records")
        return market_data, sentiment_data
    
    def _engineer_features(self, 
                          market_data: pd.DataFrame, 
                          sentiment_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Generate features from market and sentiment data.
        
        Args:
            market_data: DataFrame with OHLCV data
            sentiment_data: Optional DataFrame with sentiment indicators
            
        Returns:
            DataFrame with engineered features
        """
        print("Stage 2: Engineering features")
        
        # Generate technical features from market data
        features_df = self.feature_engineer.generate_features(market_data)
        
        # Add sentiment features if available
        if sentiment_data is not None:
            # Align sentiment data with market data
            aligned_sentiment = sentiment_data.reindex(market_data.index, method='ffill')
            
            # Add sentiment columns to features
            for col in aligned_sentiment.columns:
                features_df[f'sentiment_{col}'] = aligned_sentiment[col]
        
        # Generate target variables (what we want to predict)
        target_config = self.config.get('target', {})
        features_with_targets = self.feature_engineer.add_return_labels(
            features_df,
            future_windows=target_config.get('horizons', [1, 5, 10]),
            thresholds=target_config.get('thresholds', [0.0, 0.005, 0.01])
        )
        
        # Save feature columns for reference
        feature_cols = features_df.columns.tolist()
        with open(os.path.join(self.results_dir, 'feature_columns.json'), 'w') as f:
            json.dump(feature_cols, f, indent=2)
            
        self.state['data_stats']['feature_count'] = len(feature_cols)
        print(f"Generated {len(feature_cols)} features")
        
        return features_with_targets
    
    def _train_models(self, feature_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train models by market regime.
        
        Args:
            feature_data: DataFrame with features and targets
            
        Returns:
            Dictionary with trained models
        """
        print("Stage 3: Training models")
        
        # Get configuration
        model_config = self.config.get('model', {})
        target_config = self.config.get('target', {})
        
        # Determine target variable
        target_horizon = target_config.get('primary_horizon', 5)
        target_threshold = target_config.get('primary_threshold', 0.0)
        
        if target_threshold > 0:
            target_col = f'label_{target_horizon}d_{int(target_threshold*100)}pct'
        else:
            target_col = f'label_{target_horizon}d'
            
        # Check if target exists
        if target_col not in feature_data.columns:
            # Fallback target
            target_col = [col for col in feature_data.columns if col.startswith('label_')][0]
            print(f"Primary target not found, using {target_col} instead")
            
        self.state['model_metrics']['target_variable'] = target_col
        
        # Prepare feature and target datasets
        X, y, meta = self.feature_engineer.to_ml_dataset(
            feature_data, 
            target_col=target_col,
            meta_cols=['open', 'high', 'low', 'close', 'volume', 'market_regime']
        )
        
        # Determine model type based on target
        model_type = model_config.get('type', 'classification')
        if model_type not in ['classification', 'regression']:
            # Auto-detect based on target
            unique_values = y.nunique()
            model_type = 'classification' if unique_values < 10 else 'regression'
            
        # Train model based on regime if available
        regime_column = 'market_regime_numeric' if 'market_regime_numeric' in X.columns else None
        
        if regime_column:
            print(f"Training regime-specific ensemble model with target: {target_col}")
            # Train regime-specific models
            ensemble = self.model_trainer.build_regime_ensemble(
                X=X,
                y=y,
                regime_column=regime_column,
                model_type=model_type,
                base_name='regime',
                include_meta=True
            )
            
            # Record metrics
            self.state['model_metrics']['model_type'] = 'regime_ensemble'
            self.state['model_metrics']['regimes'] = list(ensemble.get('regime_models', {}).keys())
        else:
            print(f"Training single model with target: {target_col}")
            # Train a single model
            model = self.model_trainer.train_model(
                X=X,
                y=y,
                model_type=model_type,
                model_name='primary'
            )
            
            # Record metrics
            self.state['model_metrics']['model_type'] = 'single_model'
        
        # Return models dictionary
        return self.model_trainer.models
    
    def _evaluate_models(self, models: Dict[str, Any], 
                        feature_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate models using walk-forward testing.
        
        Args:
            models: Dictionary of trained models
            feature_data: DataFrame with features and targets
            
        Returns:
            Dictionary with evaluation results
        """
        print("Stage 4: Evaluating models")
        
        # Get configuration
        eval_config = self.config.get('evaluation', {})
        target_config = self.config.get('target', {})
        
        # Determine target and features
        target_horizon = target_config.get('primary_horizon', 5)
        target_threshold = target_config.get('primary_threshold', 0.0)
        
        if target_threshold > 0:
            target_col = f'label_{target_horizon}d_{int(target_threshold*100)}pct'
        else:
            target_col = f'label_{target_horizon}d'
            
        # Fallback target if primary not found
        if target_col not in feature_data.columns:
            target_col = [col for col in feature_data.columns if col.startswith('label_')][0]
        
        # Set up regime column
        regime_col = 'market_regime_numeric' if 'market_regime_numeric' in feature_data.columns else None
        
        # Determine model type
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'classification')
        
        # Run evaluation
        results = self.evaluator.evaluate(
            df=feature_data,
            target_col=target_col,
            regime_col=regime_col,
            model_type=model_type,
            train_size=eval_config.get('train_size', 0.6),
            test_size=eval_config.get('test_size', 0.2),
            step_size=eval_config.get('step_size', 0.1)
        )
        
        # Save evaluation results
        eval_path = self.evaluator.save_results(
            os.path.join(self.results_dir, 'evaluation_results.json')
        )
        
        # Create visualizations
        try:
            for plot_type in ['equity_curve', 'drawdown', 'trade_dist', 'fold_comparison']:
                fig = self.evaluator.plot_results(plot_type=plot_type)
                fig.savefig(os.path.join(self.results_dir, f'{plot_type}.png'))
        except Exception as e:
            print(f"Warning: Could not generate all visualizations: {str(e)}")
            
        # Update metrics
        self.state['model_metrics'].update({
            'total_return': results.get('total_return', 0),
            'sharpe_ratio': results.get('avg_sharpe_ratio', 0),
            'max_drawdown': results.get('avg_max_drawdown', 0),
            'win_rate': results.get('avg_win_rate', 0),
            'trade_count': results.get('total_trades', 0)
        })
        
        print(f"Evaluation complete with {results.get('total_trades', 0)} trades")
        return results
    
    def _save_artifacts(self, models: Dict[str, Any], 
                      eval_results: Dict[str, Any]) -> None:
        """
        Save models and results for deployment and reference.
        
        Args:
            models: Dictionary of trained models
            eval_results: Evaluation metrics and results
        """
        print("Stage 5: Saving artifacts")
        
        # Save each model
        for model_name, model in models.items():
            model_path = self.model_trainer.save_model(
                model_name=model_name, 
                filepath=os.path.join(self.model_dir, f'{model_name}.pkl')
            )
            print(f"Saved model {model_name} to {model_path}")
            
        # Save feature importance data
        for model_name, importance in self.model_trainer.feature_importance.items():
            with open(os.path.join(self.results_dir, f'{model_name}_importance.json'), 'w') as f:
                # Convert to sorted list of (feature, importance) for readability
                sorted_importance = sorted(
                    importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                json.dump(sorted_importance, f, indent=2)
                
        # Save pipeline state
        with open(os.path.join(self.results_dir, 'pipeline_state.json'), 'w') as f:
            json.dump(self.state, f, indent=2)
            
        # Create model metadata for easy loading in prediction service
        model_metadata = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'models': list(models.keys()),
            'metrics': self.state.get('model_metrics', {}),
            'config': self.config
        }
        
        with open(os.path.join(self.model_dir, 'metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=2)
            
        # Create symlink to latest successful run for easy access
        latest_dir = os.path.join(self.output_dir, 'models', 'latest')
        if os.path.exists(latest_dir) and os.path.islink(latest_dir):
            os.unlink(latest_dir)
        os.symlink(self.model_dir, latest_dir, target_is_directory=True)
        
        print(f"All artifacts saved to {self.output_dir}") 