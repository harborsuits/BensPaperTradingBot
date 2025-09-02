#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Pipeline Script for Trading Bot

This script provides a command-line interface for running the trading bot pipeline.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from trading_bot.pipeline.pipeline_runner import PipelineRunner

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the trading bot pipeline')
    
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to the configuration YAML file')
    
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory for models and results (overrides config setting)')
    
    parser.add_argument('--log-file', '-l', type=str, default=None,
                        help='Path to log file (default: output_dir/logs/pipeline_YYYYMMDD_HHMMSS.log)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def setup_logging(args):
    """Set up logging configuration."""
    import logging
    
    # Determine log file path
    if args.log_file:
        log_file = args.log_file
    else:
        # Create logs directory in output dir
        output_dir = args.output_dir
        if not output_dir:
            # Try to get from config
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                output_dir = config.get('output_dir', './output')
        
        log_dir = os.path.join(output_dir, 'logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create log file name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'pipeline_{timestamp}.log')
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('pipeline')

def run_pipeline(args, logger):
    """Run the trading bot pipeline with the given arguments."""
    try:
        logger.info(f"Starting pipeline with config: {args.config}")
        
        # Prepare config overrides
        config_overrides = {}
        if args.output_dir:
            config_overrides['output_dir'] = args.output_dir
            
        # Create and run pipeline
        pipeline = PipelineRunner(config_path=args.config)
        
        # Apply any overrides
        for key, value in config_overrides.items():
            pipeline.config[key] = value
            
        # Run the pipeline
        results = pipeline.run()
        
        # Log results
        logger.info(f"Pipeline completed successfully with run ID: {results['run_id']}")
        logger.info(f"Models saved to: {pipeline.model_dir}")
        logger.info(f"Results saved to: {pipeline.results_dir}")
        
        if 'model_metrics' in results:
            metrics = results['model_metrics']
            logger.info(f"Model performance:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args)
    
    # Run the pipeline
    exit_code = run_pipeline(args, logger)
    
    # Exit with appropriate code
    sys.exit(exit_code)

if __name__ == '__main__':
    main() 