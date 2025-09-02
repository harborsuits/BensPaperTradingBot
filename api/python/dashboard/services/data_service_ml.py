#!/usr/bin/env python3
"""
Data Service Extension for Machine Learning Predictions

Extends the DataService with methods for accessing ML prediction data
for broker performance and health.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Type, Tuple
import random
import logging
import json

# For type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trading_bot.dashboard.services.data_service import DataService

logger = logging.getLogger(__name__)


def extend_data_service_with_ml(DataServiceClass):
    """
    Extend the DataService class with ML prediction functionality
    
    Args:
        DataServiceClass: Original DataService class to extend
        
    Returns:
        Extended DataService class
    """
    class ExtendedDataService(DataServiceClass):
        """Extended DataService with ML prediction support"""
        
        def __init__(self, *args, **kwargs):
            """Initialize the extended data service"""
            # Call the parent constructor
            super().__init__(*args, **kwargs)
            
            # Flag for demo data
            self.use_demo_data = kwargs.get('use_demo_data', True)
            
            # Cache for ML prediction data
            self.ml_prediction_cache = {}
            self.ml_prediction_cache_time = {}
            self.ml_prediction_cache_ttl = 300  # 5 minutes
        
        def get_broker_ml_prediction_data(self, broker_id: str) -> Dict[str, Any]:
            """
            Get ML prediction data for a broker
            
            Args:
                broker_id: ID of the broker
                
            Returns:
                Dictionary with prediction data
            """
            # Check cache
            cache_key = f"ml_pred_{broker_id}"
            if cache_key in self.ml_prediction_cache:
                cache_time = self.ml_prediction_cache_time.get(cache_key, datetime.min)
                if (datetime.now() - cache_time).total_seconds() < self.ml_prediction_cache_ttl:
                    return self.ml_prediction_cache[cache_key]
            
            # If using real data, fetch from API
            if not self.use_demo_data and not self.use_mock_data:
                try:
                    # API path for ML predictions
                    api_path = f"/api/brokers/{broker_id}/ml_predictions"
                    data = self._get_from_api(api_path)
                    
                    if data:
                        # Process dataframes
                        if 'anomaly_data' in data and data['anomaly_data']:
                            data['anomaly_data'] = pd.DataFrame(data['anomaly_data'])
                            data['anomaly_data'].index = pd.to_datetime(data['anomaly_data'].index)
                        
                        if 'failure_data' in data and data['failure_data']:
                            data['failure_data'] = pd.DataFrame(data['failure_data'])
                            data['failure_data'].index = pd.to_datetime(data['failure_data'].index)
                        
                        # Update cache
                        self.ml_prediction_cache[cache_key] = data
                        self.ml_prediction_cache_time[cache_key] = datetime.now()
                        
                        return data
                except Exception as e:
                    logger.error(f"Error fetching ML prediction data: {str(e)}")
            
            # Fall back to demo data
            if self.use_demo_data or self.use_mock_data:
                data = self._generate_demo_ml_prediction_data(broker_id)
                
                # Update cache
                self.ml_prediction_cache[cache_key] = data
                self.ml_prediction_cache_time[cache_key] = datetime.now()
                
                return data
            
            return {}
        
        def get_all_broker_risk_summary(self) -> Dict[str, Dict[str, Any]]:
            """
            Get risk summary for all brokers
            
            Returns:
                Dictionary mapping broker IDs to risk summaries
            """
            # Try to get from API if using real data
            if not self.use_demo_data and not self.use_mock_data:
                try:
                    api_path = "/api/brokers/ml_risk_summary"
                    data = self._get_from_api(api_path)
                    if data:
                        return data
                except Exception as e:
                    logger.error(f"Error fetching broker risk summary: {str(e)}")
            
            # Fall back to demo data
            brokers = self.get_all_brokers()
            
            summary = {}
            for broker in brokers:
                broker_id = broker['broker_id']
                
                # Get prediction data for risk assessment
                pred_data = self.get_broker_ml_prediction_data(broker_id)
                
                # Extract risk assessment
                risk_data = pred_data.get('risk_assessment', {})
                anomaly_pct = pred_data.get('anomaly_pct', 0.0)
                failure_prob = pred_data.get('failure_prob', 0.0)
                
                summary[broker_id] = {
                    'level': risk_data.get('level', 'low'),
                    'action_recommended': risk_data.get('action_recommended', False),
                    'anomaly_pct': anomaly_pct,
                    'failure_prob': failure_prob
                }
            
            return summary
        
        def trigger_build_broker_models(self, broker_id: str) -> bool:
            """
            Trigger building ML models for a broker
            
            Args:
                broker_id: ID of the broker
                
            Returns:
                True if successful, False otherwise
            """
            # If using real data, call API
            if not self.use_demo_data and not self.use_mock_data:
                try:
                    api_path = f"/api/brokers/{broker_id}/build_ml_models"
                    response = self._post_to_api(api_path, {})
                    
                    # Invalidate cache
                    cache_key = f"ml_pred_{broker_id}"
                    if cache_key in self.ml_prediction_cache:
                        del self.ml_prediction_cache[cache_key]
                    
                    return response.get('success', False)
                except Exception as e:
                    logger.error(f"Error triggering model build: {str(e)}")
                    return False
            
            # Demo mode - pretend it worked
            if self.use_demo_data or self.use_mock_data:
                cache_key = f"ml_pred_{broker_id}"
                if cache_key in self.ml_prediction_cache:
                    del self.ml_prediction_cache[cache_key]
                return True
            
            return False
        
        def trigger_rebuild_all_broker_models(self) -> bool:
            """
            Trigger rebuilding ML models for all brokers
            
            Returns:
                True if successful, False otherwise
            """
            # If using real data, call API
            if not self.use_demo_data and not self.use_mock_data:
                try:
                    api_path = "/api/brokers/rebuild_all_ml_models"
                    response = self._post_to_api(api_path, {})
                    
                    # Invalidate all cache
                    self.ml_prediction_cache = {}
                    self.ml_prediction_cache_time = {}
                    
                    return response.get('success', False)
                except Exception as e:
                    logger.error(f"Error triggering models rebuild: {str(e)}")
                    return False
            
            # Demo mode - pretend it worked
            if self.use_demo_data or self.use_mock_data:
                self.ml_prediction_cache = {}
                self.ml_prediction_cache_time = {}
                return True
            
            return False
        
        def _generate_demo_ml_prediction_data(self, broker_id: str) -> Dict[str, Any]:
            """
            Generate demo ML prediction data for a broker
            
            Args:
                broker_id: ID of the broker
                
            Returns:
                Dictionary with prediction data
            """
            # Generate a random seed based on broker_id for consistency
            import hashlib
            seed = int(hashlib.md5(broker_id.encode()).hexdigest(), 16) % 10000
            np.random.seed(seed)
            
            # Get current time
            now = datetime.now()
            
            # Generate time series data for the past week
            end_time = now
            start_time = now - timedelta(days=7)
            
            # Create date range with hourly intervals
            date_range = pd.date_range(start=start_time, end=end_time, freq='H')
            
            # Determine if this broker should be high, medium, or low risk
            broker_risk = np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
            
            # Generate anomaly data
            anomaly_data = pd.DataFrame(index=date_range)
            
            # Add base metrics
            base_latency = 100 + 50 * np.random.random()
            latency_trend = np.random.choice([-0.5, 0, 0.5])  # Decreasing, stable, or increasing
            
            # Generate latency with trend, daily periodicity, and some noise
            hours = np.array([(d.hour + d.minute/60) for d in date_range])
            days = np.array([(d - start_time).total_seconds() / 86400 for d in date_range])
            
            # Daily seasonality - higher latency during market hours
            daily_pattern = 20 * np.sin(hours * 2 * np.pi / 24 - np.pi/2) + 20
            
            # Add trend
            trend = latency_trend * days * 10
            
            # Add noise
            noise = np.random.normal(0, 15, len(date_range))
            
            # Combine components
            latency_values = base_latency + daily_pattern + trend + noise
            
            # Add spikes for anomalies
            if broker_risk == 'high':
                # More anomalies for high risk
                anomaly_count = int(0.15 * len(date_range))
                anomaly_indices = np.random.choice(len(date_range), anomaly_count, replace=False)
                latency_values[anomaly_indices] += np.random.uniform(50, 150, anomaly_count)
            elif broker_risk == 'medium':
                # Some anomalies for medium risk
                anomaly_count = int(0.05 * len(date_range))
                anomaly_indices = np.random.choice(len(date_range), anomaly_count, replace=False)
                latency_values[anomaly_indices] += np.random.uniform(50, 100, anomaly_count)
            else:
                # Few anomalies for low risk
                anomaly_count = int(0.01 * len(date_range))
                anomaly_indices = np.random.choice(len(date_range), anomaly_count, replace=False)
                latency_values[anomaly_indices] += np.random.uniform(30, 70, anomaly_count)
            
            # Add metrics to dataframe
            anomaly_data['latency_mean_ms'] = latency_values
            
            # Generate reliability metrics based on latency
            reliability_errors = np.zeros(len(date_range))
            high_latency_indices = latency_values > (base_latency + 100)
            reliability_errors[high_latency_indices] = np.random.poisson(2, sum(high_latency_indices))
            anomaly_data['reliability_errors'] = reliability_errors
            
            # Calculate reliability percentage
            reliability_pct = 100 - np.minimum(reliability_errors * 5, 100)
            anomaly_data['reliability_pct'] = reliability_pct
            
            # Generate execution quality metrics
            anomaly_data['execution_quality_slippage_bps'] = np.random.normal(5, 2, len(date_range))
            anomaly_data['execution_quality_slippage_bps'][high_latency_indices] += np.random.uniform(5, 15, sum(high_latency_indices))
            
            # Generate cost metrics
            anomaly_data['cost_per_order'] = np.random.normal(1.5, 0.2, len(date_range))
            
            # Generate overall score
            latency_score = 100 - np.clip((latency_values - 50) / 200 * 100, 0, 100)
            reliability_score = reliability_pct
            execution_score = 100 - np.clip(anomaly_data['execution_quality_slippage_bps'] * 5, 0, 100)
            cost_score = 100 - np.clip((anomaly_data['cost_per_order'] - 1) * 50, 0, 100)
            
            anomaly_data['score_latency'] = latency_score
            anomaly_data['score_reliability'] = reliability_score
            anomaly_data['score_execution'] = execution_score
            anomaly_data['score_cost'] = cost_score
            anomaly_data['score_overall'] = (latency_score * 0.3 + reliability_score * 0.3 + 
                                          execution_score * 0.25 + cost_score * 0.15)
            
            # Generate anomaly flags and scores
            anomaly_scores = np.zeros(len(date_range))
            
            # Use latency, reliability, and execution quality to determine anomalies
            z_latency = (latency_values - np.mean(latency_values)) / np.std(latency_values)
            z_reliability = (reliability_pct - np.mean(reliability_pct)) / (np.std(reliability_pct) + 0.001)
            z_execution = (anomaly_data['execution_quality_slippage_bps'] - np.mean(anomaly_data['execution_quality_slippage_bps'])) / np.std(anomaly_data['execution_quality_slippage_bps'])
            
            # Combine z-scores
            combined_z = 0.5 * abs(z_latency) - 0.3 * z_reliability + 0.2 * z_execution
            
            # Normalize to a consistent range
            anomaly_scores = (combined_z - np.min(combined_z)) / (np.max(combined_z) - np.min(combined_z) + 0.001) * 2 - 1
            
            # Threshold for anomalies
            threshold = np.percentile(anomaly_scores, 90)  # Top 10% are anomalies
            if broker_risk == 'high':
                threshold = np.percentile(anomaly_scores, 80)  # Top 20% are anomalies for high risk
            elif broker_risk == 'low':
                threshold = np.percentile(anomaly_scores, 95)  # Top 5% are anomalies for low risk
            
            anomaly_flags = np.ones(len(date_range))
            anomaly_flags[anomaly_scores > threshold] = -1
            
            anomaly_data['anomaly'] = anomaly_flags
            anomaly_data['anomaly_score'] = anomaly_scores
            
            # Generate failure prediction data (similar to anomaly data)
            failure_data = anomaly_data.copy()
            
            # Calculate failure probability based on metrics
            # Higher probabilities for:
            # - High latency
            # - Low reliability
            # - High slippage
            # - Low overall score
            
            # Normalize metrics to 0-1 range
            norm_latency = (latency_values - np.min(latency_values)) / (np.max(latency_values) - np.min(latency_values) + 0.001)
            norm_reliability = (reliability_pct - np.min(reliability_pct)) / (np.max(reliability_pct) - np.min(reliability_pct) + 0.001)
            norm_execution = (anomaly_data['execution_quality_slippage_bps'] - np.min(anomaly_data['execution_quality_slippage_bps'])) / (np.max(anomaly_data['execution_quality_slippage_bps']) - np.min(anomaly_data['execution_quality_slippage_bps']) + 0.001)
            norm_score = (anomaly_data['score_overall'] - np.min(anomaly_data['score_overall'])) / (np.max(anomaly_data['score_overall']) - np.min(anomaly_data['score_overall']) + 0.001)
            
            # Combine metrics with weights
            failure_prob = (0.3 * norm_latency - 0.3 * norm_reliability + 
                          0.2 * norm_execution - 0.2 * norm_score)
            
            # Scale to 0-1 range
            failure_prob = (failure_prob - np.min(failure_prob)) / (np.max(failure_prob) - np.min(failure_prob) + 0.001)
            
            # Adjust based on broker risk
            if broker_risk == 'high':
                # For high risk, increase probabilities
                failure_prob = failure_prob * 0.7 + 0.3
            elif broker_risk == 'medium':
                # For medium risk, moderate adjustment
                failure_prob = failure_prob * 0.8 + 0.1
            else:
                # For low risk, decrease probabilities
                failure_prob = failure_prob * 0.5
            
            # Add some trend toward the end
            if broker_risk == 'high':
                # Increasing trend for high risk
                end_trend = np.linspace(0, 0.3, 24)
                failure_prob[-24:] += end_trend
                failure_prob = np.minimum(failure_prob, 1.0)  # Cap at 1.0
            
            # Add to dataframe
            failure_data['failure_probability'] = failure_prob
            failure_data['failure_predicted'] = (failure_prob > 0.5).astype(int)
            
            # Calculate overall risk metrics
            anomaly_pct = (anomaly_flags == -1).mean()
            recent_failure_prob = failure_prob[-6:].max()  # Max of last 6 hours
            
            # Determine risk level
            risk_level = 'low'
            if anomaly_pct > 0.15 or recent_failure_prob > 0.7:
                risk_level = 'high'
            elif anomaly_pct > 0.05 or recent_failure_prob > 0.3:
                risk_level = 'medium'
            
            # Create model info
            model_info = {
                'anomaly_detection': {
                    'last_updated': (now - timedelta(days=np.random.randint(1, 5))).isoformat(),
                    'lookback_days': 30
                },
                'failure_prediction': {
                    'last_updated': (now - timedelta(days=np.random.randint(1, 5))).isoformat(),
                    'lookback_days': 30,
                    'accuracy': 0.7 + np.random.random() * 0.2,
                    'failure_definition': {
                        'metric': 'reliability_errors',
                        'threshold': 5,
                        'op': 'gt'
                    }
                }
            }
            
            # Compile all data
            prediction_data = {
                'anomaly_data': anomaly_data,
                'anomaly_pct': float(anomaly_pct),
                'failure_data': failure_data,
                'failure_prob': float(recent_failure_prob),
                'prediction_window': 24,
                'risk_assessment': {
                    'level': risk_level,
                    'action_recommended': risk_level == 'high'
                },
                'model_info': model_info
            }
            
            return prediction_data
    
    return ExtendedDataService


# For direct imports
if __name__ == "__main__":
    # For testing
    class MockDataService:
        def __init__(self, *args, **kwargs):
            self.use_mock_data = kwargs.get('use_mock_data', True)
        
        def _get_from_api(self, path):
            return {}
        
        def _post_to_api(self, path, data):
            return {'success': True}
        
        def get_all_brokers(self):
            return [
                {'broker_id': 'broker_a', 'name': 'Broker A'},
                {'broker_id': 'broker_b', 'name': 'Broker B'},
                {'broker_id': 'broker_c', 'name': 'Broker C'}
            ]
    
    # Extend the mock service
    ExtendedService = extend_data_service_with_ml(MockDataService)
    
    # Create instance
    service = ExtendedService(use_mock_data=True, use_demo_data=True)
    
    # Test
    data = service.get_broker_ml_prediction_data('broker_a')
    print(f"Got prediction data with keys: {list(data.keys())}")
    
    summary = service.get_all_broker_risk_summary()
    print(f"Got risk summary for {len(summary)} brokers")
