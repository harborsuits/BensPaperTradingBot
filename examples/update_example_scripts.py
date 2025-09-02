#!/usr/bin/env python3
"""
Example Script Updater

This utility helps update old BensBot example scripts and notebooks to work with the
new architecture. It demonstrates how to:

1. Convert old script initialization patterns to the new unified system
2. Update imports to match the new package structure
3. Replace deprecated API calls with the new equivalents

Usage:
    python examples/update_example_scripts.py --script path/to/old_script.py --output path/to/new_script.py
    python examples/update_example_scripts.py --notebook path/to/old_notebook.ipynb --output path/to/new_notebook.ipynb
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# Import replacement mappings
IMPORT_MAPPINGS = {
    # Core module restructuring
    r"from trading_bot\.main import TradingBot": "from trading_bot.core.main_orchestrator import MainOrchestrator",
    r"from trading_bot\.config import load_config": "from trading_bot.config.config_loader import load_config",
    r"from trading_bot\.backtest import Backtester": "from trading_bot.simulation.backtest import Backtester",
    
    # Broker modules
    r"from trading_bot\.brokers\.tradier import TradierBroker": "from trading_bot.brokers.tradier.adapter import TradierAdapter",
    r"from trading_bot\.brokers\.alpaca import AlpacaBroker": "from trading_bot.brokers.alpaca.adapter import AlpacaAdapter",
    r"from trading_bot\.brokers\.etrade import ETradeBroker": "from trading_bot.brokers.etrade.adapter import ETradeAdapter",
    
    # Strategy modules
    r"from trading_bot\.strategies import (\w+)Strategy": r"from trading_bot.strategies.\1.strategy import \1Strategy",
    
    # General imports for new architecture
    r"import trading_bot": "import trading_bot\nfrom trading_bot.core.event_bus import EventBus\nfrom trading_bot.brokers.multi_broker_manager import MultiBrokerManager",
}

# Code pattern replacement mappings
CODE_MAPPINGS = {
    # Bot initialization
    r"bot\s*=\s*TradingBot\(\s*trading_config=([^,]+),\s*broker_config=([^,]+),\s*strategy_config=([^,\)]+)": 
        "# Create unified config from legacy configs\n"
        "from utils.convert_config import merge_configs, migrate_old_config\n"
        "unified_config = merge_configs([\n"
        "    migrate_old_config(\\1, 'trading'),\n"
        "    migrate_old_config(\\2, 'broker'),\n"
        "    migrate_old_config(\\3, 'strategy')\n"
        "])\n\n"
        "# Initialize the bot with the unified configuration\n"
        "from trading_bot.cli.run_bot import create_bot\n"
        "bot = create_bot(config=unified_config, mode='live'",
    
    # Broker initialization
    r"broker\s*=\s*TradierBroker\(\s*token=([^,]+),\s*account_id=([^,\)]+)":
        "# Create event bus for broker communication\n"
        "event_bus = EventBus()\n\n"
        "# Create broker adapter\n"
        "from trading_bot.brokers.tradier.adapter import TradierAdapter\n"
        "broker = TradierAdapter(event_bus)\n\n"
        "# Connect and initialize the broker\n"
        "broker.connect(credentials={'token': \\1, 'account_id': \\2}",
    
    r"broker\s*=\s*AlpacaBroker\(\s*api_key=([^,]+),\s*api_secret=([^,\)]+)":
        "# Create event bus for broker communication\n"
        "event_bus = EventBus()\n\n"
        "# Create broker adapter\n"
        "from trading_bot.brokers.alpaca.adapter import AlpacaAdapter\n"
        "broker = AlpacaAdapter(event_bus)\n\n"
        "# Connect and initialize the broker\n"
        "broker.connect(credentials={'api_key': \\1, 'api_secret': \\2}",
    
    # Strategy initialization
    r"strategy\s*=\s*(\w+)Strategy\(([^\)]+)":
        "# Create strategy using the factory pattern\n"
        "from trading_bot.strategies import strategy_factory\n"
        "strategy_config = {\n"
        "    'name': '\\1'.lower(),\n"
        "    'class': 'trading_bot.strategies.\\1.strategy.\\1Strategy',\n"
        "    'parameters': {\n"
        "        # Convert parameters to the new format\n"
        "\\2\n"
        "    }\n"
        "}\n"
        "strategy = strategy_factory.create_strategy(strategy_config",
    
    # Backtest initialization
    r"backtester\s*=\s*Backtester\(\s*strategy=([^,]+),\s*initial_capital=([^,]+),\s*start_date=([^,]+),\s*end_date=([^,]+)":
        "# Create backtester with the new configuration approach\n"
        "backtest_config = {\n"
        "    'initial_balance': \\2,\n"
        "    'start_date': \\3,\n"
        "    'end_date': \\4,\n"
        "    'strategy': \\1\n"
        "}\n\n"
        "from trading_bot.simulation.backtest import Backtester\n"
        "backtester = Backtester(config=backtest_config",
    
    # MultiBrokerManager usage
    r"# Add multi-broker support\n":
        "# Initialize the multi-broker manager for better broker abstraction\n"
        "event_bus = EventBus()\n"
        "broker_manager = MultiBrokerManager(event_bus=event_bus)\n\n"
        "# You can add brokers as needed:\n"
        "# broker_manager.add_broker(\n"
        "#     broker_id='alpaca',\n"
        "#     broker=AlpacaAdapter(event_bus),\n"
        "#     credentials={'api_key': 'YOUR_KEY', 'api_secret': 'YOUR_SECRET'},\n"
        "#     make_primary=True\n"
        "# )\n",
}

# Deprecated function warnings with suggested alternatives
DEPRECATED_WARNINGS = {
    r"\.start_trading\(": "The .start_trading() method is deprecated. Use .run() instead.",
    r"\.get_historical_data\(": "Direct historical data methods are now unified under the DataManager interface.",
    r"\.place_market_order\(": "Specific order types are now unified under the .place_order() method with 'order_type' parameter.",
    r"\.get_account_balance\(": "Use .get_account_info() for comprehensive account information.",
}


def update_python_script(script_path: str, output_path: Optional[str] = None) -> str:
    """
    Update a Python script to use the new BensBot API.
    
    Args:
        script_path: Path to the original script
        output_path: Path to save the updated script (if None, return as string)
    
    Returns:
        The updated script content as a string
    """
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Add a migration notice at the top
    migration_notice = '\n'.join([
        '"""',
        'MIGRATION NOTICE:',
        'This script has been automatically updated to work with the new BensBot architecture.',
        'Please review the changes and ensure it works as expected.',
        '"""',
        ''
    ])
    
    # Apply import replacements
    for old_pattern, new_import in IMPORT_MAPPINGS.items():
        content = re.sub(old_pattern, new_import, content)
    
    # Apply code replacements
    for old_pattern, new_code in CODE_MAPPINGS.items():
        content = re.sub(old_pattern, new_code, content)
    
    # Add warnings for deprecated functions
    warnings = []
    for deprecated_pattern, warning in DEPRECATED_WARNINGS.items():
        if re.search(deprecated_pattern, content):
            warnings.append(f"# WARNING: {warning}")
    
    if warnings:
        content = '\n'.join(warnings + ['']) + content
    
    # Combine everything
    updated_content = migration_notice + content
    
    # Write to output file if specified
    if output_path:
        with open(output_path, 'w') as f:
            f.write(updated_content)
    
    return updated_content


def update_jupyter_notebook(notebook_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Update a Jupyter notebook to use the new BensBot API.
    
    Args:
        notebook_path: Path to the original notebook
        output_path: Path to save the updated notebook (if None, return as dict)
    
    Returns:
        The updated notebook as a dictionary
    """
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Add migration notice as a markdown cell at the top
    migration_notice_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# BensBot Migration Notice\n",
            "This notebook has been automatically updated to work with the new BensBot architecture.\n",
            "Please review the changes and ensure it works as expected.\n"
        ]
    }
    
    notebook['cells'].insert(0, migration_notice_cell)
    
    # Process code cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Join source lines into a single string for processing
            content = ''.join(cell['source'])
            
            # Apply import replacements
            for old_pattern, new_import in IMPORT_MAPPINGS.items():
                content = re.sub(old_pattern, new_import, content)
            
            # Apply code replacements
            for old_pattern, new_code in CODE_MAPPINGS.items():
                content = re.sub(old_pattern, new_code, content)
            
            # Add warnings for deprecated functions
            warnings = []
            for deprecated_pattern, warning in DEPRECATED_WARNINGS.items():
                if re.search(deprecated_pattern, content):
                    warnings.append(f"# WARNING: {warning}")
            
            if warnings:
                content = '\n'.join(warnings + ['']) + content
            
            # Split content back into lines
            cell['source'] = content.splitlines(True)
    
    # Write to output file if specified
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(notebook, f, indent=2)
    
    return notebook


def main():
    """Main entry point for the script updater."""
    parser = argparse.ArgumentParser(description='Update legacy BensBot example scripts')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--script', help='Path to a Python script to update')
    group.add_argument('--notebook', help='Path to a Jupyter notebook to update')
    parser.add_argument('--output', help='Output path for the updated file')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without saving')
    
    args = parser.parse_args()
    
    if args.script:
        input_path = args.script
        if not os.path.exists(input_path):
            print(f"Error: Script not found: {input_path}")
            sys.exit(1)
        
        if args.dry_run:
            updated_content = update_python_script(input_path)
            print(updated_content)
        else:
            output_path = args.output or input_path.replace('.py', '_updated.py')
            update_python_script(input_path, output_path)
            print(f"Updated script saved to: {output_path}")
    
    elif args.notebook:
        input_path = args.notebook
        if not os.path.exists(input_path):
            print(f"Error: Notebook not found: {input_path}")
            sys.exit(1)
        
        if args.dry_run:
            updated_notebook = update_jupyter_notebook(input_path)
            print(json.dumps(updated_notebook, indent=2))
        else:
            output_path = args.output or input_path.replace('.ipynb', '_updated.ipynb')
            update_jupyter_notebook(input_path, output_path)
            print(f"Updated notebook saved to: {output_path}")


if __name__ == "__main__":
    main()
