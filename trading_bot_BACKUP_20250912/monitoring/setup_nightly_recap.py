#!/usr/bin/env python
"""
Setup Nightly Recap

This script sets up the nightly recap system by:
1. Creating necessary directories
2. Creating a sample configuration file
3. Setting up a cron job to run the recap after market close
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
from crontab import CronTab  # pip install python-crontab

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for the nightly recap system"""
    directories = [
        'logs',
        'reports',
        'reports/nightly',
        'reports/nightly/charts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_sample_config(config_path='config/nightly_recap_config.json'):
    """Create a sample configuration file for the nightly recap system"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Default configuration
    config = {
        'email': {
            'enabled': True,
            'server': 'smtp.gmail.com',
            'port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-app-password',  # Use app password for Gmail
            'recipients': ['your-email@gmail.com']
        },
        'benchmarks': ['SPY', 'VIX'],
        'thresholds': {
            'sharpe_ratio': 0.5,
            'win_rate': 45.0,  # percentage
            'max_drawdown': -10.0,  # percentage
            'rolling_windows': [5, 10, 20, 60]  # days
        },
        'optimization': {
            'auto_optimize': False,
            'optimization_threshold': -20.0  # percentage deterioration
        }
    }
    
    # Write configuration to file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Created sample configuration file at {config_path}")
    return config_path

def setup_cron_job(time='0 18 * * 1-5'):
    """
    Set up a cron job to run the nightly recap after market close
    
    Args:
        time: Cron schedule expression (default: 6:00 PM on weekdays)
    """
    try:
        # Get the full path to the script
        script_dir = os.path.abspath(os.path.dirname(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, '../..'))
        script_path = os.path.join(script_dir, 'run_nightly_recap.py')
        
        # Create command to run script
        cmd = f"cd {project_dir} && "
        cmd += f"export PYTHONPATH={project_dir} && "
        cmd += f"python {script_path} --config {os.path.join(project_dir, 'config/nightly_recap_config.json')}"
        
        # Get the current user's crontab
        cron = CronTab(user=True)
        
        # Check if job already exists
        for job in cron:
            if 'run_nightly_recap.py' in str(job):
                logger.warning("Cron job for nightly recap already exists, removing it")
                cron.remove(job)
        
        # Create new job
        job = cron.new(command=cmd, comment="Trading Bot Nightly Recap")
        job.setall(time)
        
        # Write to crontab
        cron.write()
        
        logger.info(f"Cron job set up to run at {time}")
        logger.info(f"Command: {cmd}")
        return True
    
    except Exception as e:
        logger.error(f"Error setting up cron job: {e}")
        return False

def run_test_recap():
    """Run a test recap to verify the setup"""
    try:
        # Get the full path to the script
        script_dir = os.path.abspath(os.path.dirname(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, '../..'))
        script_path = os.path.join(script_dir, 'run_nightly_recap.py')
        
        # Create command to run script
        cmd = [
            sys.executable, 
            script_path,
            '--config', os.path.join(project_dir, 'config/nightly_recap_config.json')
        ]
        
        # Set environment variable
        env = os.environ.copy()
        env['PYTHONPATH'] = project_dir
        
        # Run the command
        logger.info(f"Running test recap: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Log output
        if result.stdout:
            logger.info(f"Test recap output: {result.stdout}")
        if result.stderr:
            logger.error(f"Test recap error: {result.stderr}")
        
        return result.returncode == 0
    
    except Exception as e:
        logger.error(f"Error running test recap: {e}")
        return False

def main():
    """Main function to set up the nightly recap system"""
    parser = argparse.ArgumentParser(description='Set up the nightly recap system')
    
    parser.add_argument(
        '--time',
        type=str,
        default='0 18 * * 1-5',
        help='Cron schedule expression (default: 6:00 PM on weekdays)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/nightly_recap_config.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run a test recap after setup'
    )
    
    parser.add_argument(
        '--no-cron',
        action='store_true',
        help='Skip setting up cron job'
    )
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Create sample configuration file
    config_path = create_sample_config(args.config)
    print()
    print(f"Created sample configuration file at {config_path}")
    print("Please update the configuration with your email settings")
    print()
    
    # Set up cron job if not skipped
    if not args.no_cron:
        if setup_cron_job(args.time):
            print(f"Cron job set up to run at {args.time}")
        else:
            print("Failed to set up cron job")
    else:
        print("Skipping cron job setup")
        print("To manually set up a cron job, add the following line to your crontab:")
        script_dir = os.path.abspath(os.path.dirname(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, '../..'))
        script_path = os.path.join(script_dir, 'run_nightly_recap.py')
        cmd = f"cd {project_dir} && export PYTHONPATH={project_dir} && python {script_path} --config {os.path.join(project_dir, 'config/nightly_recap_config.json')}"
        print(f"{args.time} {cmd}")
    
    # Run test recap if requested
    if args.test:
        print("Running test recap...")
        if run_test_recap():
            print("Test recap completed successfully")
        else:
            print("Test recap failed")
    
    print()
    print("Nightly recap system setup complete")
    print("To run the API server and access the dashboard, run:")
    print("cd /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/api")
    print("export PYTHONPATH=/Users/bendickinson/Desktop/Trading:BenBot")
    print("python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000")
    print()
    print("Then visit the dashboard and use the 'Run New Recap' button to generate your first performance analysis.")

if __name__ == "__main__":
    main()
