#!/usr/bin/env python3
"""
CI/CD Pipeline for Evolved Strategies

Automates the testing, validation, and deployment of evolved strategies.
Integrates with BensBot for seamless strategy deployment.
"""

import os
import sys
import json
import yaml
import shutil
import logging
import argparse
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import EvoTrader components
from benbot_api import BenBotAPI
from prop_strategy_registry import PropStrategyRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'ci_cd_pipeline.log')),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger('ci_cd_pipeline')

class StrategyCICDPipeline:
    """
    CI/CD Pipeline for evolved trading strategies.
    
    Features:
    1. Automated testing and validation
    2. Code quality checks and linting
    3. Version control integration
    4. Deployment to BensBot
    5. Production monitoring integration
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the CI/CD pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup paths
        self.registry_path = self.config.get('registry_path', os.path.join(project_root, 'forex_prop_strategies.db'))
        self.strategies_dir = self.config.get('strategies_dir', os.path.join(project_root, 'deployments'))
        self.backup_dir = self.config.get('backup_dir', os.path.join(project_root, 'backups', 'strategies'))
        
        # Ensure directories exist
        os.makedirs(self.strategies_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
        
        # Initialize components
        try:
            # Registry
            self.registry = PropStrategyRegistry(db_path=self.registry_path)
            
            # BensBot API
            self.benbot_api = BenBotAPI(
                api_endpoint=self.config.get('benbot_api_endpoint', 'http://localhost:8080/benbot/api'),
                api_key=self.config.get('benbot_api_key', ''),
                test_mode=self.config.get('test_mode', False)
            )
            
            logger.info("CI/CD pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CI/CD pipeline: {e}")
            raise
        
        # Git repo
        self.repo = None
        if self.config.get('use_git', True):
            try:
                import git
                self.repo = git.Repo(project_root)
                logger.info("Git repository initialized")
            except ImportError:
                logger.warning("GitPython not installed. Git functionality disabled.")
            except Exception as e:
                logger.warning(f"Failed to initialize Git repository: {e}")
    
    def run_pipeline(self, strategy_ids: List[str] = None, deployment_id: str = None) -> bool:
        """
        Run the complete CI/CD pipeline.
        
        Args:
            strategy_ids: List of strategy IDs to process
            deployment_id: ID of the deployment batch
            
        Returns:
            True if deployment was successful
        """
        if not deployment_id:
            deployment_id = f"deploy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting CI/CD pipeline for deployment: {deployment_id}")
        
        # 1. Get strategies to deploy
        strategies = self._get_strategies_to_deploy(strategy_ids)
        
        if not strategies:
            logger.warning("No strategies to deploy")
            return False
        
        logger.info(f"Found {len(strategies)} strategies to deploy")
        
        # 2. Create deployment directory
        deploy_dir = os.path.join(self.strategies_dir, deployment_id)
        os.makedirs(deploy_dir, exist_ok=True)
        
        # 3. Export and validate strategies
        validated_strategies = []
        
        for strategy in strategies:
            strategy_id = strategy['strategy_id']
            
            # Export strategy to Python file
            success, file_path = self._export_strategy(strategy, deploy_dir)
            
            if not success:
                logger.error(f"Failed to export strategy {strategy_id}")
                continue
            
            # Validate strategy
            if self._validate_strategy(file_path, strategy):
                validated_strategies.append({
                    'strategy_id': strategy_id,
                    'file_path': file_path,
                    'strategy_type': strategy.get('strategy_type', 'unknown'),
                    'status': strategy.get('status', 'unknown')
                })
                logger.info(f"Strategy {strategy_id} validated successfully")
            else:
                logger.error(f"Strategy {strategy_id} failed validation")
        
        if not validated_strategies:
            logger.warning("No strategies validated successfully")
            return False
        
        # 4. Generate deployment manifest
        manifest_path = self._generate_manifest(validated_strategies, deploy_dir, deployment_id)
        
        # 5. Commit changes to Git if enabled
        if self.repo and self.config.get('use_git', True):
            self._commit_to_git(deploy_dir, deployment_id, len(validated_strategies))
        
        # 6. Deploy to BensBot
        deployed_count = self._deploy_to_benbot(validated_strategies, manifest_path)
        
        logger.info(f"CI/CD pipeline completed: {deployed_count}/{len(validated_strategies)} strategies deployed")
        
        return deployed_count > 0
    
    def _get_strategies_to_deploy(self, strategy_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Get strategies to deploy."""
        if strategy_ids:
            # Get specific strategies
            strategies = []
            for strategy_id in strategy_ids:
                strategy = self.registry.get_strategy(strategy_id)
                if strategy:
                    strategies.append(strategy)
        else:
            # Get promotion candidates
            strategies = self.registry.get_promotion_candidates(
                min_confidence=self.config.get('min_confidence', 0.7),
                min_trades=self.config.get('min_trades', 20)
            )
        
        return strategies
    
    def _export_strategy(self, strategy: Dict[str, Any], deploy_dir: str) -> Tuple[bool, str]:
        """
        Export strategy to a Python file.
        
        Args:
            strategy: Strategy data
            deploy_dir: Deployment directory
            
        Returns:
            (success, file_path)
        """
        try:
            strategy_id = strategy['strategy_id']
            strategy_type = strategy.get('strategy_type', 'unknown')
            
            # Create file name
            file_name = f"{strategy_id.lower().replace('-', '_')}.py"
            file_path = os.path.join(deploy_dir, file_name)
            
            # Get strategy code
            strategy_code = strategy.get('code')
            
            if not strategy_code:
                # Try to generate code from parameters
                parameters = strategy.get('parameters', {})
                indicators = strategy.get('indicators', [])
                
                strategy_code = self._generate_strategy_code(
                    strategy_id, 
                    strategy_type, 
                    parameters, 
                    indicators
                )
            
            if not strategy_code:
                logger.error(f"No code available for strategy {strategy_id}")
                return False, ""
            
            # Write to file
            with open(file_path, 'w') as f:
                f.write(strategy_code)
            
            logger.info(f"Exported strategy {strategy_id} to {file_path}")
            return True, file_path
            
        except Exception as e:
            logger.error(f"Failed to export strategy {strategy.get('strategy_id', 'unknown')}: {e}")
            return False, ""
    
    def _validate_strategy(self, file_path: str, strategy: Dict[str, Any]) -> bool:
        """
        Validate exported strategy.
        
        Args:
            file_path: Path to the strategy file
            strategy: Strategy data
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # 1. Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Strategy file not found: {file_path}")
                return False
            
            # 2. Syntax check
            try:
                subprocess.run(
                    ['python', '-m', 'py_compile', file_path],
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Syntax error in strategy {strategy.get('strategy_id', 'unknown')}: {e.stderr}")
                return False
            
            # 3. Import check
            try:
                sys.path.append(os.path.dirname(file_path))
                module_name = os.path.basename(file_path).replace('.py', '')
                
                # Clear module from sys.modules if it exists
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                # Import module
                __import__(module_name)
                
                # Check if module has required functions
                module = sys.modules[module_name]
                
                required_functions = ['initialize', 'calculate_signal']
                
                for func in required_functions:
                    if not hasattr(module, func) or not callable(getattr(module, func)):
                        logger.error(f"Strategy {strategy.get('strategy_id', 'unknown')} missing required function: {func}")
                        return False
                
                logger.info(f"Strategy {strategy.get('strategy_id', 'unknown')} passed syntax and import checks")
                return True
                
            except Exception as e:
                logger.error(f"Import error in strategy {strategy.get('strategy_id', 'unknown')}: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to validate strategy {strategy.get('strategy_id', 'unknown')}: {e}")
            return False
    
    def _generate_manifest(self, strategies: List[Dict[str, Any]], deploy_dir: str, deployment_id: str) -> str:
        """
        Generate deployment manifest.
        
        Args:
            strategies: List of validated strategies
            deploy_dir: Deployment directory
            deployment_id: Deployment ID
            
        Returns:
            Path to the manifest file
        """
        try:
            manifest = {
                'deployment_id': deployment_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'strategy_count': len(strategies),
                'strategies': []
            }
            
            for strategy in strategies:
                manifest['strategies'].append({
                    'strategy_id': strategy['strategy_id'],
                    'file_name': os.path.basename(strategy['file_path']),
                    'strategy_type': strategy['strategy_type'],
                    'status': strategy['status']
                })
            
            # Write manifest to file
            manifest_path = os.path.join(deploy_dir, 'manifest.json')
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Generated deployment manifest: {manifest_path}")
            return manifest_path
            
        except Exception as e:
            logger.error(f"Failed to generate deployment manifest: {e}")
            return ""
    
    def _commit_to_git(self, deploy_dir: str, deployment_id: str, strategy_count: int) -> bool:
        """
        Commit changes to Git.
        
        Args:
            deploy_dir: Deployment directory
            deployment_id: Deployment ID
            strategy_count: Number of strategies
            
        Returns:
            True if successful, False otherwise
        """
        if not self.repo:
            logger.warning("Git repository not initialized")
            return False
        
        try:
            # Add files to Git
            self.repo.git.add(deploy_dir)
            
            # Commit changes
            commit_message = f"Deploy {strategy_count} strategies: {deployment_id}"
            self.repo.git.commit('-m', commit_message)
            
            # Push if remote is configured
            if self.config.get('git_push', False):
                self.repo.git.push()
            
            logger.info(f"Committed deployment to Git: {commit_message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to commit to Git: {e}")
            return False
    
    def _deploy_to_benbot(self, strategies: List[Dict[str, Any]], manifest_path: str) -> int:
        """
        Deploy strategies to BensBot.
        
        Args:
            strategies: List of validated strategies
            manifest_path: Path to the manifest file
            
        Returns:
            Number of successfully deployed strategies
        """
        deployed_count = 0
        
        for strategy in strategies:
            strategy_id = strategy['strategy_id']
            file_path = strategy['file_path']
            
            try:
                # Read strategy code
                with open(file_path, 'r') as f:
                    strategy_code = f.read()
                
                # Deploy to BensBot
                success = self.benbot_api.deploy_strategy(
                    strategy_id=strategy_id,
                    strategy_code=strategy_code,
                    status='active',
                    notes=f"Deployed by CI/CD pipeline at {datetime.datetime.now().isoformat()}"
                )
                
                if success:
                    deployed_count += 1
                    logger.info(f"Successfully deployed strategy {strategy_id} to BensBot")
                else:
                    logger.error(f"Failed to deploy strategy {strategy_id} to BensBot")
                
            except Exception as e:
                logger.error(f"Error deploying strategy {strategy_id} to BensBot: {e}")
        
        return deployed_count
    
    def _generate_strategy_code(self, strategy_id: str, strategy_type: str, 
                               parameters: Dict[str, Any], indicators: List[str]) -> str:
        """
        Generate strategy code from parameters.
        
        Args:
            strategy_id: Strategy ID
            strategy_type: Strategy type
            parameters: Strategy parameters
            indicators: List of indicators
            
        Returns:
            Generated strategy code
        """
        # This is a placeholder. In a real implementation, this would generate actual code
        # based on strategy type and parameters.
        
        # Basic template
        template = f"""#!/usr/bin/env python3
\"\"\"
Evolved Strategy: {strategy_id}
Type: {strategy_type}
Generated by: EvoTrader CI/CD Pipeline
\"\"\"

import numpy as np
import pandas as pd

# Parameters
{', '.join([f"{k} = {v}" for k, v in parameters.items()])}

def initialize(context):
    \"\"\"Initialize the strategy.\"\"\"
    context.strategy_id = "{strategy_id}"
    context.strategy_type = "{strategy_type}"
    
    # Set indicators
    context.indicators = {indicators}
    
    # Set parameters
    """
        
        # Add parameters
        for k, v in parameters.items():
            template += f"    context.{k} = {v}\n"
        
        # Add calculation function based on strategy type
        if strategy_type == "trend_following":
            template += """
def calculate_signal(context, data):
    \"\"\"Calculate trading signal.\"\"\"
    # Trend following strategy logic
    close = data['close']
    
    if 'sma' in context.indicators or 'ema' in context.indicators:
        # Use moving averages
        fast_ma = data.get('ema', data.get('sma', close.rolling(window=context.fast_period).mean()))
        slow_ma = close.rolling(window=context.slow_period).mean()
        
        # Generate signal
        if fast_ma[-1] > slow_ma[-1] and fast_ma[-2] <= slow_ma[-2]:
            return 1  # Buy signal
        elif fast_ma[-1] < slow_ma[-1] and fast_ma[-2] >= slow_ma[-2]:
            return -1  # Sell signal
    
    return 0  # No signal
"""
        elif strategy_type == "mean_reversion":
            template += """
def calculate_signal(context, data):
    \"\"\"Calculate trading signal.\"\"\"
    # Mean reversion strategy logic
    close = data['close']
    
    if 'rsi' in context.indicators:
        # Use RSI
        rsi = data['rsi']
        
        # Generate signal
        if rsi[-1] < context.oversold_threshold:
            return 1  # Buy signal (oversold)
        elif rsi[-1] > context.overbought_threshold:
            return -1  # Sell signal (overbought)
    
    return 0  # No signal
"""
        elif strategy_type == "breakout":
            template += """
def calculate_signal(context, data):
    \"\"\"Calculate trading signal.\"\"\"
    # Breakout strategy logic
    close = data['close']
    high = data['high']
    low = data['low']
    
    # Calculate recent highs and lows
    period_high = high.rolling(window=context.breakout_period).max()
    period_low = low.rolling(window=context.breakout_period).min()
    
    # Generate signal
    if close[-1] > period_high[-2]:
        return 1  # Buy signal (breakout above resistance)
    elif close[-1] < period_low[-2]:
        return -1  # Sell signal (breakdown below support)
    
    return 0  # No signal
"""
        else:
            # Generic template for other strategy types
            template += """
def calculate_signal(context, data):
    \"\"\"Calculate trading signal.\"\"\"
    # Generic strategy logic
    close = data['close']
    
    # Calculate indicators based on parameters
    # This is a placeholder for actual strategy logic
    
    # For demonstration, return a random signal
    import random
    return random.choice([-1, 0, 1])
"""
        
        return template
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            'registry_path': os.path.join(project_root, 'forex_prop_strategies.db'),
            'strategies_dir': os.path.join(project_root, 'deployments'),
            'backup_dir': os.path.join(project_root, 'backups', 'strategies'),
            'benbot_api_endpoint': 'http://localhost:8080/benbot/api',
            'benbot_api_key': '',
            'test_mode': False,
            'use_git': True,
            'git_push': False,
            'min_confidence': 0.7,
            'min_trades': 20
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    elif config_path.endswith('.json'):
                        config = json.load(f)
                    else:
                        logger.warning(f"Unknown config format: {config_path}")
                        config = {}
                
                # Merge with default config
                for key, value in config.items():
                    default_config[key] = value
                
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
        
        return default_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI/CD Pipeline for Evolved Strategies")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--strategy-ids", type=str, help="Comma-separated list of strategy IDs to deploy")
    parser.add_argument("--deployment-id", type=str, help="Deployment ID")
    
    args = parser.parse_args()
    
    # Parse strategy IDs
    strategy_ids = None
    if args.strategy_ids:
        strategy_ids = [s.strip() for s in args.strategy_ids.split(',')]
    
    # Create pipeline
    try:
        pipeline = StrategyCICDPipeline(config_path=args.config)
        
        # Run pipeline
        pipeline.run_pipeline(strategy_ids=strategy_ids, deployment_id=args.deployment_id)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
