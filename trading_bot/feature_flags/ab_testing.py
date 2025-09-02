#!/usr/bin/env python3
"""
A/B Testing Framework for Feature Flags

This module provides a structured framework for running A/B tests using feature flags,
managing experiment lifecycles, and analyzing results.
"""

import logging
import json
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
import numpy as np
from enum import Enum

from .metrics_integration import FeatureFlagMetrics

# Setup logging
logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Status of an A/B test experiment"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"

class ABTestExperiment:
    """
    Represents a single A/B test experiment using feature flags.
    
    This class manages the lifecycle of an experiment from creation to analysis,
    including user assignment, metric tracking, and statistical evaluation.
    """
    
    def __init__(self, 
                name: str, 
                flag_name: str,
                description: str = "",
                hypothesis: str = "",
                owner: str = "",
                target_metrics: List[str] = None,
                assignment_ratio: Dict[str, float] = None,
                min_sample_size: int = 1000,
                max_duration_days: int = 30,
                filters: Dict[str, Any] = None,
                metrics_instance: Optional[FeatureFlagMetrics] = None):
        """
        Initialize a new A/B test experiment.
        
        Args:
            name: Name of the experiment
            flag_name: Name of the feature flag to use
            description: Description of what is being tested
            hypothesis: The hypothesis being tested
            owner: Owner/creator of the experiment
            target_metrics: List of metrics to track for this experiment
            assignment_ratio: Assignment ratio for variant groups (e.g. {"control": 0.5, "treatment": 0.5})
            min_sample_size: Minimum sample size before concluding the experiment
            max_duration_days: Maximum duration of the experiment in days
            filters: Filters to apply for user targeting
            metrics_instance: Instance of FeatureFlagMetrics to use
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.flag_name = flag_name
        self.description = description
        self.hypothesis = hypothesis
        self.owner = owner
        self.target_metrics = target_metrics or ["conversion", "revenue"]
        self.min_sample_size = min_sample_size
        self.max_duration_days = max_duration_days
        self.filters = filters or {}
        
        # Set default assignment ratio if not provided
        self.assignment_ratio = assignment_ratio or {"control": 0.5, "treatment": 0.5}
        self._validate_assignment_ratio()
        
        # Experiment metadata
        self.status = ExperimentStatus.DRAFT
        self.created_at = datetime.now().isoformat()
        self.started_at = None
        self.ended_at = None
        
        # Results tracking
        self.results = {}
        self.current_sample_size = {"control": 0, "treatment": 0}
        
        # Connect to metrics system
        self.metrics = metrics_instance
        
        # User assignment cache (user_id -> variant)
        self._user_assignments = {}
        
        logger.info(f"Created experiment '{name}' (ID: {self.id}) for flag '{flag_name}'")
    
    def _validate_assignment_ratio(self):
        """Validate that assignment ratios sum to 1.0"""
        total = sum(self.assignment_ratio.values())
        if abs(total - 1.0) > 0.001:  # Allow for small floating point errors
            logger.warning(f"Assignment ratios sum to {total}, adjusting to 1.0")
            # Normalize to sum to 1.0
            factor = 1.0 / total
            self.assignment_ratio = {k: v * factor for k, v in self.assignment_ratio.items()}
    
    def start(self) -> bool:
        """
        Start the experiment.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.status != ExperimentStatus.DRAFT and self.status != ExperimentStatus.PAUSED:
            logger.warning(f"Cannot start experiment '{self.name}' with status {self.status}")
            return False
            
        self.status = ExperimentStatus.RUNNING
        self.started_at = self.started_at or datetime.now().isoformat()
        
        # Register experiment with metrics system if available
        if self.metrics:
            exp_data = {
                "id": self.id,
                "name": self.name,
                "flag_name": self.flag_name,
                "status": self.status.value,
                "start_date": self.started_at,
                "target_metrics": self.target_metrics,
                "assignment_ratio": self.assignment_ratio
            }
            self.metrics.experiments[self.id] = exp_data
        
        logger.info(f"Started experiment '{self.name}' (ID: {self.id})")
        return True
    
    def pause(self) -> bool:
        """
        Pause the experiment.
        
        Returns:
            bool: True if paused successfully, False otherwise
        """
        if self.status != ExperimentStatus.RUNNING:
            logger.warning(f"Cannot pause experiment '{self.name}' with status {self.status}")
            return False
            
        self.status = ExperimentStatus.PAUSED
        
        # Update status in metrics system if available
        if self.metrics and self.id in self.metrics.experiments:
            self.metrics.experiments[self.id]["status"] = self.status.value
        
        logger.info(f"Paused experiment '{self.name}' (ID: {self.id})")
        return True
    
    def complete(self) -> bool:
        """
        Mark the experiment as completed.
        
        Returns:
            bool: True if completed successfully, False otherwise
        """
        if self.status != ExperimentStatus.RUNNING and self.status != ExperimentStatus.PAUSED:
            logger.warning(f"Cannot complete experiment '{self.name}' with status {self.status}")
            return False
            
        self.status = ExperimentStatus.COMPLETED
        self.ended_at = datetime.now().isoformat()
        
        # Calculate final results
        self._analyze_results()
        
        # Update metrics system if available
        if self.metrics and self.id in self.metrics.experiments:
            self.metrics.experiments[self.id].update({
                "status": self.status.value,
                "end_date": self.ended_at,
                "results": self.results
            })
        
        logger.info(f"Completed experiment '{self.name}' (ID: {self.id})")
        return True
    
    def assign_user(self, user_id: str, context: Dict[str, Any] = None) -> str:
        """
        Assign a user to a variant group based on the experiment configuration.
        
        Args:
            user_id: User identifier
            context: Additional context about the user/request
            
        Returns:
            str: The variant group name ('control' or 'treatment')
        """
        # If experiment is not running, always assign to control
        if self.status != ExperimentStatus.RUNNING:
            return "control"
        
        # Check if user is already assigned
        if user_id in self._user_assignments:
            return self._user_assignments[user_id]
        
        # Apply filters if specified
        if self.filters and context:
            for key, value in self.filters.items():
                if key not in context or context[key] != value:
                    return "control"  # User doesn't match filter criteria
        
        # Assign user based on assignment ratio
        # Use hash of user_id for consistent assignment
        hash_val = hash(user_id) % 1000 / 1000.0  # Value between 0 and 1
        
        # Determine which group based on hash value
        cumulative = 0
        assigned_group = "control"  # Default
        
        for group, ratio in self.assignment_ratio.items():
            cumulative += ratio
            if hash_val <= cumulative:
                assigned_group = group
                break
        
        # Store assignment for consistency
        self._user_assignments[user_id] = assigned_group
        
        # Update sample sizes
        if assigned_group in self.current_sample_size:
            self.current_sample_size[assigned_group] += 1
        
        # Check if we've hit minimum sample size
        if sum(self.current_sample_size.values()) >= self.min_sample_size:
            logger.info(f"Experiment '{self.name}' has reached minimum sample size")
        
        return assigned_group
    
    def record_exposure(self, user_id: str, context: Dict[str, Any] = None) -> bool:
        """
        Record that a user was exposed to the experiment.
        
        Args:
            user_id: User identifier
            context: Additional context about the user/request
            
        Returns:
            bool: True if recorded successfully, False otherwise
        """
        if self.status != ExperimentStatus.RUNNING:
            return False
            
        # Determine which variant the user is assigned to
        variant = self.assign_user(user_id, context)
        is_treatment = variant == "treatment"
        
        # Add experiment context
        full_context = context.copy() if context else {}
        full_context.update({
            "user_id": user_id,
            "experiment_id": self.id,
            "experiment_name": self.name,
            "variant": variant
        })
        
        # Record flag evaluation with metrics system if available
        if self.metrics:
            self.metrics.record_evaluation(
                self.flag_name, 
                is_treatment,  # True for treatment, False for control
                full_context
            )
        
        return True
    
    def record_metric(self, user_id: str, metrics: Dict[str, Any], context: Dict[str, Any] = None) -> bool:
        """
        Record metrics for a user in the experiment.
        
        Args:
            user_id: User identifier
            metrics: Dictionary of metrics to record
            context: Additional context about the user/request
            
        Returns:
            bool: True if recorded successfully, False otherwise
        """
        if self.status != ExperimentStatus.RUNNING:
            return False
            
        # Determine which variant the user is assigned to
        variant = self.assign_user(user_id, context)
        is_treatment = variant == "treatment"
        
        # Filter metrics to only include target metrics
        filtered_metrics = {k: v for k, v in metrics.items() if k in self.target_metrics}
        
        if not filtered_metrics:
            return False
            
        # Add experiment context
        full_context = context.copy() if context else {}
        full_context.update({
            "user_id": user_id,
            "experiment_id": self.id,
            "experiment_name": self.name,
            "variant": variant
        })
        
        # Record metrics with metrics system if available
        if self.metrics:
            self.metrics.associate_metrics(
                self.flag_name,
                filtered_metrics,
                is_treatment,
                full_context
            )
        
        return True
    
    def check_completion_criteria(self) -> bool:
        """
        Check if the experiment should be automatically completed.
        
        Returns:
            bool: True if the experiment should be completed
        """
        if self.status != ExperimentStatus.RUNNING:
            return False
            
        # Check if experiment has exceeded maximum duration
        if self.started_at:
            start_date = datetime.fromisoformat(self.started_at)
            max_end_date = start_date + timedelta(days=self.max_duration_days)
            
            if datetime.now() > max_end_date:
                logger.info(f"Experiment '{self.name}' has exceeded maximum duration")
                return True
                
        # Check if experiment has reached minimum sample size and has statistically significant results
        if sum(self.current_sample_size.values()) >= self.min_sample_size:
            # Analyze current results
            has_significant_results = self._analyze_results(interim=True)
            
            if has_significant_results:
                logger.info(f"Experiment '{self.name}' has statistically significant results")
                return True
                
        return False
    
    def _analyze_results(self, interim: bool = False) -> bool:
        """
        Analyze the experiment results.
        
        Args:
            interim: Whether this is an interim analysis or final
            
        Returns:
            bool: True if there are statistically significant results
        """
        has_significant_results = False
        
        try:
            # If we have a metrics system, use it to analyze results
            if self.metrics:
                analysis = self.metrics.analyze_flag_impact(
                    self.flag_name,
                    self.target_metrics,
                    time_period=timedelta(days=self.max_duration_days)
                )
                
                if analysis['status'] == 'success':
                    self.results = {
                        'sample_size': {
                            'control': analysis['false_sample_count'],
                            'treatment': analysis['true_sample_count'],
                            'total': analysis['sample_count']
                        },
                        'metrics': {}
                    }
                    
                    # Process each metric
                    for metric_name, metric_data in analysis['metrics'].items():
                        if 'status' in metric_data and metric_data['status'] != 'success':
                            continue
                            
                        # Calculate relative improvement
                        relative_improvement = metric_data.get('percent_difference', 0)
                        
                        # Determine if result is statistically significant
                        is_significant = metric_data.get('statistically_significant', False)
                        
                        if is_significant:
                            has_significant_results = True
                            
                        self.results['metrics'][metric_name] = {
                            'control': metric_data['false_mean'],
                            'treatment': metric_data['true_mean'],
                            'lift': relative_improvement,
                            'p_value': metric_data.get('p_value', 1.0),
                            'significant': is_significant
                        }
                        
            # Update experiment status based on results
            if not interim and self.status == ExperimentStatus.COMPLETED:
                if has_significant_results:
                    self.results['conclusion'] = "Experiment yielded statistically significant results."
                else:
                    self.status = ExperimentStatus.INCONCLUSIVE
                    self.results['conclusion'] = "Experiment did not yield statistically significant results."
            
            return has_significant_results
            
        except Exception as e:
            logger.error(f"Error analyzing results for experiment '{self.name}': {str(e)}")
            if not interim:
                self.status = ExperimentStatus.ERROR
                self.results = {
                    'error': str(e)
                }
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert experiment to dictionary for serialization.
        
        Returns:
            Dict: Dictionary representation of the experiment
        """
        return {
            'id': self.id,
            'name': self.name,
            'flag_name': self.flag_name,
            'description': self.description,
            'hypothesis': self.hypothesis,
            'owner': self.owner,
            'target_metrics': self.target_metrics,
            'min_sample_size': self.min_sample_size,
            'max_duration_days': self.max_duration_days,
            'assignment_ratio': self.assignment_ratio,
            'filters': self.filters,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'ended_at': self.ended_at,
            'current_sample_size': self.current_sample_size,
            'results': self.results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], metrics_instance: Optional[FeatureFlagMetrics] = None):
        """
        Create an experiment from a dictionary.
        
        Args:
            data: Dictionary representation of an experiment
            metrics_instance: FeatureFlagMetrics instance
            
        Returns:
            ABTestExperiment: Reconstructed experiment
        """
        instance = cls(
            name=data['name'],
            flag_name=data['flag_name'],
            description=data.get('description', ''),
            hypothesis=data.get('hypothesis', ''),
            owner=data.get('owner', ''),
            target_metrics=data.get('target_metrics', ['conversion', 'revenue']),
            assignment_ratio=data.get('assignment_ratio', {'control': 0.5, 'treatment': 0.5}),
            min_sample_size=data.get('min_sample_size', 1000),
            max_duration_days=data.get('max_duration_days', 30),
            filters=data.get('filters', {}),
            metrics_instance=metrics_instance
        )
        
        # Restore fields that aren't part of init
        instance.id = data['id']
        instance.status = ExperimentStatus(data['status'])
        instance.created_at = data['created_at']
        instance.started_at = data.get('started_at')
        instance.ended_at = data.get('ended_at')
        instance.current_sample_size = data.get('current_sample_size', {'control': 0, 'treatment': 0})
        instance.results = data.get('results', {})
        
        return instance


class ABTestingManager:
    """
    Manages multiple A/B test experiments, providing a centralized interface
    for creating, tracking, and analyzing experiments.
    """
    
    def __init__(self, 
                storage_path: str = "data/feature_flags/experiments",
                metrics_instance: Optional[FeatureFlagMetrics] = None):
        """
        Initialize the A/B testing manager.
        
        Args:
            storage_path: Path to store experiment data
            metrics_instance: FeatureFlagMetrics instance for tracking
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize metrics tracking if not provided
        if metrics_instance:
            self.metrics = metrics_instance
        else:
            from .metrics_integration import FeatureFlagMetrics
            self.metrics = FeatureFlagMetrics()
        
        # Active experiments
        self.experiments: Dict[str, ABTestExperiment] = {}
        
        # Load existing experiments
        self._load_experiments()
        
        logger.info(f"Initialized A/B Testing Manager with {len(self.experiments)} experiments")
    
    def _load_experiments(self):
        """Load existing experiments from storage"""
        try:
            # List all experiment files
            if not os.path.exists(self.storage_path):
                return
                
            for filename in os.listdir(self.storage_path):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(self.storage_path, filename)
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                experiment = ABTestExperiment.from_dict(data, self.metrics)
                self.experiments[experiment.id] = experiment
                
        except Exception as e:
            logger.error(f"Error loading experiments: {str(e)}")
    
    def _save_experiment(self, experiment: ABTestExperiment):
        """Save an experiment to storage"""
        try:
            file_path = os.path.join(self.storage_path, f"{experiment.id}.json")
            
            with open(file_path, 'w') as f:
                json.dump(experiment.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving experiment {experiment.id}: {str(e)}")
    
    def create_experiment(self, **kwargs) -> ABTestExperiment:
        """
        Create a new A/B test experiment.
        
        Args:
            **kwargs: Arguments to pass to ABTestExperiment constructor
            
        Returns:
            ABTestExperiment: The created experiment
        """
        # Create the experiment
        experiment = ABTestExperiment(
            metrics_instance=self.metrics,
            **kwargs
        )
        
        # Register and save
        self.experiments[experiment.id] = experiment
        self._save_experiment(experiment)
        
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[ABTestExperiment]:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Optional[ABTestExperiment]: The experiment, or None if not found
        """
        return self.experiments.get(experiment_id)
    
    def get_experiments_by_flag(self, flag_name: str) -> List[ABTestExperiment]:
        """
        Get all experiments for a specific feature flag.
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            List[ABTestExperiment]: List of experiments
        """
        return [exp for exp in self.experiments.values() if exp.flag_name == flag_name]
    
    def get_running_experiments(self) -> List[ABTestExperiment]:
        """
        Get all currently running experiments.
        
        Returns:
            List[ABTestExperiment]: List of running experiments
        """
        return [exp for exp in self.experiments.values() 
                if exp.status == ExperimentStatus.RUNNING]
    
    def check_experiment_completion(self):
        """Check all running experiments for completion criteria"""
        for experiment in self.get_running_experiments():
            if experiment.check_completion_criteria():
                experiment.complete()
                self._save_experiment(experiment)
                logger.info(f"Automatically completed experiment '{experiment.name}'")
    
    def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
            
        success = experiment.start()
        if success:
            self._save_experiment(experiment)
            
        return success
    
    def pause_experiment(self, experiment_id: str) -> bool:
        """
        Pause an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            bool: True if paused successfully, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
            
        success = experiment.pause()
        if success:
            self._save_experiment(experiment)
            
        return success
    
    def complete_experiment(self, experiment_id: str) -> bool:
        """
        Complete an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            bool: True if completed successfully, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
            
        success = experiment.complete()
        if success:
            self._save_experiment(experiment)
            
        return success
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        if experiment_id not in self.experiments:
            return False
            
        # Remove from memory
        del self.experiments[experiment_id]
        
        # Remove from storage
        try:
            file_path = os.path.join(self.storage_path, f"{experiment_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return True
        except Exception as e:
            logger.error(f"Error deleting experiment {experiment_id}: {str(e)}")
            return False
    
    def generate_experiment_report(self, experiment_id: str, format: str = 'json') -> Union[Dict, str]:
        """
        Generate a report for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            format: Output format ('json' or 'html')
            
        Returns:
            Union[Dict, str]: Report data or HTML report
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
            
        # Force result calculation
        experiment._analyze_results()
        
        # Build report data
        report = experiment.to_dict()
        
        if format == 'html':
            # Generate simple HTML report
            html = [
                '<!DOCTYPE html>',
                '<html>',
                '<head>',
                f'  <title>Experiment Report: {experiment.name}</title>',
                '  <style>',
                '    body { font-family: Arial, sans-serif; margin: 20px; }',
                '    .card { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 4px; }',
                '    .metric-positive { color: green; }',
                '    .metric-negative { color: red; }',
                '    .metric-neutral { color: orange; }',
                '    table { border-collapse: collapse; width: 100%; }',
                '    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
                '    th { background-color: #f2f2f2; }',
                '  </style>',
                '</head>',
                '<body>',
                f'  <h1>Experiment Report: {experiment.name}</h1>',
                '  <div class="card">',
                '    <h2>Overview</h2>',
                f'    <p><strong>Description:</strong> {experiment.description}</p>',
                f'    <p><strong>Hypothesis:</strong> {experiment.hypothesis}</p>',
                f'    <p><strong>Feature Flag:</strong> {experiment.flag_name}</p>',
                f'    <p><strong>Status:</strong> {experiment.status.value}</p>',
                f'    <p><strong>Created:</strong> {experiment.created_at}</p>'
            ]
            
            if experiment.started_at:
                html.append(f'    <p><strong>Started:</strong> {experiment.started_at}</p>')
                
            if experiment.ended_at:
                html.append(f'    <p><strong>Ended:</strong> {experiment.ended_at}</p>')
                
            html.append('  </div>')
            
            # Add sample size info
            html.extend([
                '  <div class="card">',
                '    <h2>Sample Sizes</h2>',
                '    <table>',
                '      <tr><th>Variant</th><th>Count</th></tr>',
                f'      <tr><td>Control</td><td>{experiment.current_sample_size.get("control", 0)}</td></tr>',
                f'      <tr><td>Treatment</td><td>{experiment.current_sample_size.get("treatment", 0)}</td></tr>',
                '    </table>',
                '  </div>'
            ])
            
            # Add results if available
            if 'metrics' in experiment.results:
                html.extend([
                    '  <div class="card">',
                    '    <h2>Results</h2>'
                ])
                
                if 'conclusion' in experiment.results:
                    html.append(f'    <p><strong>Conclusion:</strong> {experiment.results["conclusion"]}</p>')
                    
                html.extend([
                    '    <table>',
                    '      <tr><th>Metric</th><th>Control</th><th>Treatment</th><th>Lift</th><th>Significant</th></tr>'
                ])
                
                for metric, data in experiment.results['metrics'].items():
                    lift = data.get('lift', 0)
                    lift_class = 'metric-neutral'
                    if lift > 1:
                        lift_class = 'metric-positive'
                    elif lift < 0:
                        lift_class = 'metric-negative'
                        
                    significant = "Yes" if data.get('significant', False) else "No"
                    
                    html.extend([
                        '      <tr>',
                        f'        <td>{metric}</td>',
                        f'        <td>{data.get("control", "N/A")}</td>',
                        f'        <td>{data.get("treatment", "N/A")}</td>',
                        f'        <td class="{lift_class}">{lift:.2f}%</td>',
                        f'        <td>{significant}</td>',
                        '      </tr>'
                    ])
                    
                html.extend([
                    '    </table>',
                    '  </div>'
                ])
                
            html.extend([
                '</body>',
                '</html>'
            ])
            
            # Save report to file
            report_path = os.path.join(self.storage_path, f"report_{experiment_id}.html")
            with open(report_path, 'w') as f:
                f.write('\n'.join(html))
                
            return '\n'.join(html)
            
        return report
                
    def assign_user_to_experiment(self, user_id: str, flag_name: str, 
                                context: Dict[str, Any] = None) -> bool:
        """
        Assign a user to an experiment and record the exposure.
        
        Args:
            user_id: User identifier
            flag_name: Feature flag to check for experiments
            context: Additional context about the user/request
            
        Returns:
            bool: Whether the user should see the treatment (True) or control (False)
        """
        # Find active experiments for this flag
        active_experiments = [exp for exp in self.experiments.values() 
                             if exp.flag_name == flag_name and 
                             exp.status == ExperimentStatus.RUNNING]
        
        # If no active experiments, return default (False)
        if not active_experiments:
            return False
            
        # For now, just use the first active experiment
        # More sophisticated logic could be added to handle multiple experiments
        experiment = active_experiments[0]
        
        # Assign user and record exposure
        variant = experiment.assign_user(user_id, context)
        experiment.record_exposure(user_id, context)
        
        # Return True for treatment, False for control or other
        return variant == "treatment"
        
    def record_experiment_metrics(self, user_id: str, metrics: Dict[str, Any], 
                                context: Dict[str, Any] = None):
        """
        Record metrics for all experiments the user is part of.
        
        Args:
            user_id: User identifier
            metrics: Dictionary of metrics to record
            context: Additional context about the user/request
        """
        # Find all running experiments
        running_experiments = self.get_running_experiments()
        
        for experiment in running_experiments:
            # Check if user is assigned to this experiment
            if user_id in experiment._user_assignments:
                experiment.record_metric(user_id, metrics, context)
                
    def periodic_maintenance(self):
        """
        Perform periodic maintenance tasks:
        - Check for experiments that should be completed
        - Save experiment data
        - Generate reports for completed experiments
        """
        # Check for experiments that should be completed
        self.check_experiment_completion()
        
        # Save all experiments
        for experiment in self.experiments.values():
            self._save_experiment(experiment)
            
        logger.debug(f"Performed periodic maintenance for {len(self.experiments)} experiments") 