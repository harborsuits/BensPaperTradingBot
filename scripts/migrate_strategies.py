#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Migration Script

This script migrates strategies from the old organization to the new structure,
starting with the successful ForexTrendFollowingStrategy implementation.
"""

import os
import sys
import shutil
import inspect
import re
import logging
import argparse
import importlib
from typing import Dict, List, Any, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StrategyMigration")

# Strategy type patterns for classification
STRATEGY_PATTERNS = {
    'trend': ['trend', 'following', 'momentum'],
    'range': ['range', 'oscillator', 'bound'],
    'breakout': ['breakout', 'volatility'],
    'carry': ['carry', 'interest', 'swap'],
    'scalping': ['scalp', 'high_frequency'],
    'swing': ['swing'],
    'day_trading': ['day', 'intraday'],
    'position': ['position', 'longterm']
}

# Asset class patterns
ASSET_PATTERNS = {
    'forex': ['forex', 'currency', 'fx', 'eur', 'usd', 'jpy', 'gbp'],
    'stocks': ['stock', 'equity', 'share', 'etf'],
    'options': ['option', 'call', 'put', 'strike'],
    'crypto': ['crypto', 'bitcoin', 'eth', 'coin']
}

def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")

def snake_case(name: str) -> str:
    """
    Convert PascalCase to snake_case.
    
    Args:
        name: PascalCase name
        
    Returns:
        snake_case name
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def detect_strategy_type(strategy_name: str, strategy_code: str) -> str:
    """
    Detect the strategy type from name and code.
    
    Args:
        strategy_name: Strategy class name
        strategy_code: Strategy source code
        
    Returns:
        Strategy type
    """
    name_lower = strategy_name.lower()
    code_lower = strategy_code.lower()
    
    for strategy_type, patterns in STRATEGY_PATTERNS.items():
        for pattern in patterns:
            if pattern in name_lower or pattern in code_lower:
                return strategy_type
    
    # Default to trend if not found (since we're starting with ForexTrendFollowingStrategy)
    return 'trend'

def detect_asset_class(strategy_name: str, strategy_code: str) -> str:
    """
    Detect the asset class from name and code.
    
    Args:
        strategy_name: Strategy class name
        strategy_code: Strategy source code
        
    Returns:
        Asset class
    """
    name_lower = strategy_name.lower()
    code_lower = strategy_code.lower()
    
    for asset_class, patterns in ASSET_PATTERNS.items():
        for pattern in patterns:
            if pattern in name_lower or pattern in code_lower:
                return asset_class
    
    # Check for obvious class prefixes
    if strategy_name.startswith('Forex'):
        return 'forex'
    elif strategy_name.startswith('Stock'):
        return 'stocks'
    elif strategy_name.startswith('Option'):
        return 'options'
    elif strategy_name.startswith('Crypto'):
        return 'crypto'
    
    # Default to forex if not found (since we're focusing on ForexTrendFollowingStrategy)
    return 'forex'

def extract_meta_info(strategy_class: Any) -> Dict[str, Any]:
    """
    Extract metadata from a strategy class.
    
    Args:
        strategy_class: Strategy class object
        
    Returns:
        Metadata dictionary
    """
    metadata = {}
    
    # Look for class attributes that might contain metadata
    if hasattr(strategy_class, 'METADATA'):
        metadata.update(strategy_class.METADATA)
    
    # Look for documentation
    if strategy_class.__doc__:
        metadata['description'] = strategy_class.__doc__.strip()
    
    # Look for market regime related attributes
    if hasattr(strategy_class, 'compatible_market_regimes'):
        metadata['compatible_market_regimes'] = strategy_class.compatible_market_regimes
    
    # Default compatibility scores based on the ForexTrendFollowingStrategy pattern
    if 'Trend' in strategy_class.__name__:
        # Trend following strategies work best in trending markets
        metadata['regime_compatibility_scores'] = {
            'trending': 0.9,
            'ranging': 0.4,
            'volatile': 0.6,
            'low_volatility': 0.7,
            'all_weather': 0.7
        }
    elif 'Range' in strategy_class.__name__:
        # Range trading strategies work best in ranging markets
        metadata['regime_compatibility_scores'] = {
            'trending': 0.4,
            'ranging': 0.9,
            'volatile': 0.3,
            'low_volatility': 0.8,
            'all_weather': 0.6
        }
    elif 'Breakout' in strategy_class.__name__:
        # Breakout strategies work best in volatile markets
        metadata['regime_compatibility_scores'] = {
            'trending': 0.7,
            'ranging': 0.5,
            'volatile': 0.9,
            'low_volatility': 0.3,
            'all_weather': 0.6
        }
    
    return metadata

def find_strategy_files(base_path: str) -> List[str]:
    """
    Find all strategy files in the codebase.
    
    Args:
        base_path: Base path to search
        
    Returns:
        List of strategy file paths
    """
    strategy_files = []
    strategies_dir = os.path.join(base_path, 'trading_bot', 'strategies')
    
    # Walk the strategies directory
    for root, dirs, files in os.walk(strategies_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                strategy_files.append(os.path.join(root, file))
    
    return strategy_files

def extract_strategy_class(file_path: str) -> List[Tuple[str, Any]]:
    """
    Extract strategy classes from a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        List of (class_name, class_object) tuples
    """
    # Use a very simple approach - just read the file content and look for class definitions
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create a dummy module object to return
        class DummyModule: pass
        module = DummyModule()
        
        # Find class names that match strategy patterns
        class_names = re.findall(r'class\s+([A-Za-z0-9_]+Strategy)\s*\(', content)
        strategies = []
        
        for name in class_names:
            # Create a dummy class with the right name
            strategy_class = type(name, (), {})
            strategy_class.__doc__ = f"Strategy class {name}"
            
            # Extract docstring if possible
            docstring_match = re.search(f'class\s+{name}.*?"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                strategy_class.__doc__ = docstring_match.group(1).strip()
            
            strategies.append((name, strategy_class))
        
        return strategies
        

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return []

def generate_init_files(strategy_dirs: Dict[str, Set[str]]) -> None:
    """
    Generate __init__.py files for the new strategy directories.
    
    Args:
        strategy_dirs: Dictionary mapping directories to strategy names
    """
    for strategy_dir, strategy_names in strategy_dirs.items():
        init_path = os.path.join(strategy_dir, '__init__.py')
        
        with open(init_path, 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('# -*- coding: utf-8 -*-\n')
            f.write(f'"""\n')
            f.write(f'Strategy module for {os.path.basename(strategy_dir)} strategies.\n')
            f.write(f'"""\n\n')
            
            # Import statements
            for strategy in strategy_names:
                module_name = snake_case(strategy)
                f.write(f'from .{module_name} import {strategy}\n')
            
            f.write('\n')
            
            # __all__ list
            f.write('__all__ = [\n')
            for strategy in strategy_names:
                f.write(f'    "{strategy}",\n')
            f.write(']\n')
        
        logger.info(f"Generated {init_path}")

def migrate_forex_trend_following_strategy(base_path: str) -> None:
    """
    Migrate the ForexTrendFollowingStrategy to the new organization.
    This builds upon our successful implementation and serves as a template.
    
    Args:
        base_path: Base path of the project
    """
    logger.info("Migrating ForexTrendFollowingStrategy...")
    
    # Find the strategy file
    strategy_files = find_strategy_files(base_path)
    target_file = None
    target_class = None
    
    for file_path in strategy_files:
        if 'trend_following' in file_path.lower() or 'forex' in file_path.lower():
            strategies = extract_strategy_class(file_path)
            for name, cls in strategies:
                if name == 'ForexTrendFollowingStrategy':
                    target_file = file_path
                    target_class = cls
                    break
    
    if not target_file or not target_class:
        logger.error("ForexTrendFollowingStrategy not found")
        return
    
    # Read the original file
    with open(target_file, 'r') as f:
        content = f.read()
    
    # Extract metadata
    metadata = extract_meta_info(target_class)
    
    # Create the destination directory
    dest_dir = os.path.join(base_path, 'trading_bot', 'strategies_new', 'forex', 'trend')
    ensure_directory(dest_dir)
    
    # Create the decorator for the strategy registry
    registry_decorator = """
@register_strategy({
    'asset_class': 'forex',
    'strategy_type': 'trend_following',
    'compatible_market_regimes': ['trending', 'low_volatility'],
    'timeframe': 'swing',
    'regime_compatibility_scores': {
        'trending': 0.95,       # Highest compatibility with trending markets
        'ranging': 0.40,        # Poor compatibility with ranging markets
        'volatile': 0.60,       # Moderate compatibility with volatile markets
        'low_volatility': 0.75, # Good compatibility with low volatility markets
        'all_weather': 0.70     # Good overall compatibility
    },
    'optimal_parameters': {
        'trending': {
            'fast_ma_period': 8,
            'slow_ma_period': 21,
            'signal_ma_period': 9,
            'adx_period': 14,
            'adx_threshold': 25,
            'atr_period': 14,
            'atr_multiplier': 3.0
        },
        'low_volatility': {
            'fast_ma_period': 5,
            'slow_ma_period': 15,
            'signal_ma_period': 9,
            'adx_period': 14,
            'adx_threshold': 20,
            'atr_period': 14,
            'atr_multiplier': 2.5
        }
    }
})"""
    
    # Add the registry import and decorator
    if 'register_strategy' not in content:
        import_pos = content.find('import')
        if import_pos >= 0:
            # Find the end of imports
            import_end = content.find('\n\n', import_pos)
            if import_end >= 0:
                content = (content[:import_end] + 
                          '\nfrom trading_bot.strategies.factory.strategy_registry import register_strategy' + 
                          content[import_end:])
        
        # Add the decorator before the class definition
        class_pos = content.find('class ForexTrendFollowingStrategy')
        if class_pos >= 0:
            content = content[:class_pos] + registry_decorator + '\n' + content[class_pos:]
    
    # Write to the new location
    dest_file = os.path.join(dest_dir, 'trend_following_strategy.py')
    with open(dest_file, 'w') as f:
        f.write(content)
    
    logger.info(f"ForexTrendFollowingStrategy migrated to {dest_file}")
    
    # Create an __init__.py file
    generate_init_files({dest_dir: {'ForexTrendFollowingStrategy'}})

def migrate_strategy(strategy_file: str, strategy_name: str, strategy_class: Any, 
                   base_path: str, migrated_strategies: Set[str]) -> bool:
    """
    Migrate a strategy to the new organization.
    
    Args:
        strategy_file: Path to the strategy file
        strategy_name: Strategy class name
        strategy_class: Strategy class object
        base_path: Base path of the project
        migrated_strategies: Set of already migrated strategy names
        
    Returns:
        True if migration was successful
    """
    if strategy_name in migrated_strategies:
        logger.info(f"Strategy {strategy_name} already migrated, skipping")
        return True
    
    logger.info(f"Migrating {strategy_name}...")
    
    # Read the original file
    with open(strategy_file, 'r') as f:
        content = f.read()
    
    # Detect asset class and strategy type
    asset_class = detect_asset_class(strategy_name, content)
    strategy_type = detect_strategy_type(strategy_name, content)
    
    # Create the destination directory
    dest_dir = os.path.join(base_path, 'trading_bot', 'strategies_new', asset_class, strategy_type)
    ensure_directory(dest_dir)
    
    # Extract metadata
    metadata = extract_meta_info(strategy_class)
    
    # Create the registry decorator if it doesn't exist
    if 'register_strategy' not in content:
        registry_decorator = f"""
@register_strategy({{
    'asset_class': '{asset_class}',
    'strategy_type': '{strategy_type}',
    'compatible_market_regimes': {metadata.get('compatible_market_regimes', ['all_weather'])},
    'timeframe': '{metadata.get('timeframe', 'daily')}',
    'regime_compatibility_scores': {metadata.get('regime_compatibility_scores', {})}
}})"""
        
        # Add the registry import and decorator
        import_pos = content.find('import')
        if import_pos >= 0:
            # Find the end of imports
            import_end = content.find('\n\n', import_pos)
            if import_end >= 0:
                content = (content[:import_end] + 
                          '\nfrom trading_bot.strategies.factory.strategy_registry import register_strategy' + 
                          content[import_end:])
        
        # Add the decorator before the class definition
        class_pos = content.find(f'class {strategy_name}')
        if class_pos >= 0:
            content = content[:class_pos] + registry_decorator + '\n' + content[class_pos:]
    
    # Write to the new location
    dest_file = os.path.join(dest_dir, f'{snake_case(strategy_name)}.py')
    with open(dest_file, 'w') as f:
        f.write(content)
    
    logger.info(f"Strategy {strategy_name} migrated to {dest_file}")
    
    # Update the migrated strategies set
    migrated_strategies.add(strategy_name)
    
    # Create or update __init__.py file
    if os.path.exists(os.path.join(dest_dir, '__init__.py')):
        # Read existing init file
        with open(os.path.join(dest_dir, '__init__.py'), 'r') as f:
            init_content = f.read()
            
        # Check if strategy is already in init
        if f'from .{snake_case(strategy_name)} import {strategy_name}' not in init_content:
            # Find the last import
            last_import = init_content.rfind('from .')
            if last_import >= 0:
                last_import_end = init_content.find('\n', last_import)
                init_content = (init_content[:last_import_end+1] + 
                               f'from .{snake_case(strategy_name)} import {strategy_name}\n' + 
                               init_content[last_import_end+1:])
            
            # Update __all__ list
            all_pos = init_content.find('__all__')
            if all_pos >= 0:
                closing_bracket = init_content.find(']', all_pos)
                if closing_bracket >= 0:
                    init_content = (init_content[:closing_bracket] + 
                                  f'    "{strategy_name}",\n' + 
                                  init_content[closing_bracket:])
            
            # Write updated init
            with open(os.path.join(dest_dir, '__init__.py'), 'w') as f:
                f.write(init_content)
        
    else:
        # Create new init file
        generate_init_files({dest_dir: {strategy_name}})
    
    return True

def migrate_forex_strategies(base_path: str) -> None:
    """
    Migrate forex strategies to the new organization.
    
    Args:
        base_path: Base path of the project
    """
    logger.info("Migrating forex strategies...")
    
    # Start with ForexTrendFollowingStrategy as our reference
    migrate_forex_trend_following_strategy(base_path)
    
    # Find all strategy files
    strategy_files = find_strategy_files(base_path)
    migrated_strategies = {'ForexTrendFollowingStrategy'}
    
    # First pass: migrate forex strategies
    for file_path in strategy_files:
        if 'forex' in file_path.lower():
            strategies = extract_strategy_class(file_path)
            for name, cls in strategies:
                if name.startswith('Forex') and name not in migrated_strategies:
                    migrate_strategy(file_path, name, cls, base_path, migrated_strategies)

def create_forex_strategy_stubs(base_path: str) -> None:
    """
    Create stub files for core forex strategy types based on ForexTrendFollowingStrategy.
    
    Args:
        base_path: Base path of the project
    """
    logger.info("Creating forex strategy stubs...")
    
    # Define the core forex strategy types
    strategy_types = {
        'range': 'ForexRangeTradingStrategy',
        'breakout': 'ForexBreakoutStrategy',
        'momentum': 'ForexMomentumStrategy',
        'scalping': 'ForexScalpingStrategy',
        'swing': 'ForexSwingTradingStrategy'
    }
    
    # Find the ForexTrendFollowingStrategy as a template
    template_file = os.path.join(base_path, 'trading_bot', 'strategies_new', 'forex', 'trend', 'trend_following_strategy.py')
    
    if not os.path.exists(template_file):
        logger.error(f"Template file {template_file} not found")
        return
    
    # Read the template
    with open(template_file, 'r') as f:
        template_content = f.read()
    
    # Create stubs for each strategy type
    for strategy_type, strategy_name in strategy_types.items():
        # Create the destination directory
        dest_dir = os.path.join(base_path, 'trading_bot', 'strategies_new', 'forex', strategy_type)
        ensure_directory(dest_dir)
        
        # Create a modified version of the template
        content = template_content
        
        # Replace class name
        content = content.replace('ForexTrendFollowingStrategy', strategy_name)
        
        # Adjust docstring
        content = re.sub(r'""".*?"""', f'"""\n{strategy_name}\n\nThis strategy implements {strategy_type} trading for forex markets.\n"""', content, flags=re.DOTALL)
        
        # Adjust registry decorator
        content = re.sub(r"@register_strategy\({.*?}\)", 
                      f"""@register_strategy({{
    'asset_class': 'forex',
    'strategy_type': '{strategy_type}',
    'compatible_market_regimes': ['ranging', 'all_weather'],
    'timeframe': 'swing',
    'regime_compatibility_scores': {{
        'trending': 0.50,       # Moderate compatibility with trending markets
        'ranging': 0.90,        # High compatibility with ranging markets
        'volatile': 0.60,       # Moderate compatibility with volatile markets
        'low_volatility': 0.70, # Good compatibility with low volatility markets
        'all_weather': 0.70     # Good overall compatibility
    }}
}})""", content, flags=re.DOTALL)
        
        # Write to the new location
        dest_file = os.path.join(dest_dir, f'{snake_case(strategy_name)}.py')
        
        if not os.path.exists(dest_file):
            with open(dest_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Created stub for {strategy_name} at {dest_file}")
            
            # Create __init__.py file
            generate_init_files({dest_dir: {strategy_name}})
        else:
            logger.info(f"Stub for {strategy_name} already exists at {dest_file}")

def create_parent_inits(base_path: str) -> None:
    """
    Create parent __init__.py files for the new organization.
    
    Args:
        base_path: Base path of the project
    """
    logger.info("Creating parent __init__.py files...")
    
    # Create the main strategies __init__.py
    strategies_dir = os.path.join(base_path, 'trading_bot', 'strategies_new')
    ensure_directory(strategies_dir)
    
    with open(os.path.join(strategies_dir, '__init__.py'), 'w') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('# -*- coding: utf-8 -*-\n')
        f.write('"""\n')
        f.write('Trading strategies package.\n')
        f.write('"""\n\n')
        
        # Import subpackages
        subpackages = [d for d in os.listdir(strategies_dir) 
                     if os.path.isdir(os.path.join(strategies_dir, d)) and not d.startswith('__')]
        
        for subpackage in subpackages:
            f.write(f'from . import {subpackage}\n')
        
        f.write('\n')
    
    # Create asset class __init__.py files
    asset_classes = ['forex', 'stocks', 'crypto', 'options']
    
    for asset_class in asset_classes:
        asset_dir = os.path.join(strategies_dir, asset_class)
        
        if not os.path.exists(asset_dir):
            continue
        
        with open(os.path.join(asset_dir, '__init__.py'), 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('# -*- coding: utf-8 -*-\n')
            f.write('"""\n')
            f.write(f'{asset_class.capitalize()} trading strategies.\n')
            f.write('"""\n\n')
            
            # Import subpackages
            subpackages = [d for d in os.listdir(asset_dir) 
                        if os.path.isdir(os.path.join(asset_dir, d)) and not d.startswith('__')]
            
            for subpackage in subpackages:
                f.write(f'from . import {subpackage}\n')
            
            f.write('\n')
            
            # Collect all strategy classes
            all_strategies = []
            
            for subpackage in subpackages:
                subpackage_dir = os.path.join(asset_dir, subpackage)
                init_file = os.path.join(subpackage_dir, '__init__.py')
                
                if os.path.exists(init_file):
                    with open(init_file, 'r') as sf:
                        content = sf.read()
                        
                        # Extract __all__ list
                        all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
                        if all_match:
                            strategies = re.findall(r'"([^"]+)"', all_match.group(1))
                            all_strategies.extend(strategies)
            
            # Add re-exports
            if all_strategies:
                for strategy in all_strategies:
                    subpackage = detect_strategy_type(strategy, '')
                    f.write(f'from .{subpackage} import {strategy}\n')
                
                f.write('\n')
                
                # Add __all__ list
                f.write('__all__ = [\n')
                for strategy in all_strategies:
                    f.write(f'    "{strategy}",\n')
                f.write(']\n')
        
        logger.info(f"Created {asset_class} __init__.py")

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Trading Bot Strategy Migration")
    parser.add_argument('--path', default='.', help='Base path of the project')
    parser.add_argument('--forex-only', action='store_true', help='Migrate only forex strategies')
    parser.add_argument('--create-stubs', action='store_true', help='Create stub files for core strategy types')
    
    args = parser.parse_args()
    
    # Ensure base path is absolute
    base_path = os.path.abspath(args.path)
    
    logger.info(f"Starting strategy migration (base path: {base_path})")
    
    # Create base directories
    strategies_new = os.path.join(base_path, 'trading_bot', 'strategies_new')
    ensure_directory(strategies_new)
    
    # Ensure factory directory exists
    factory_dir = os.path.join(strategies_new, 'factory')
    ensure_directory(factory_dir)
    
    # Copy existing factory files
    src_factory = os.path.join(base_path, 'trading_bot', 'strategies', 'factory')
    if os.path.exists(src_factory):
        for file in os.listdir(src_factory):
            if file.endswith('.py'):
                shutil.copy(os.path.join(src_factory, file), os.path.join(factory_dir, file))
                logger.info(f"Copied factory file: {file}")
    
    # Create asset class directories
    ensure_directory(os.path.join(strategies_new, 'forex'))
    ensure_directory(os.path.join(strategies_new, 'stocks'))
    ensure_directory(os.path.join(strategies_new, 'crypto'))
    ensure_directory(os.path.join(strategies_new, 'options'))
    
    # Migrate forex strategies
    migrate_forex_strategies(base_path)
    
    # Create stubs if requested
    if args.create_stubs:
        create_forex_strategy_stubs(base_path)
    
    # Create parent __init__.py files
    create_parent_inits(base_path)
    
    logger.info("Strategy migration completed")
    
    # Provide next steps
    print("\nNext Steps:")
    print("1. Run the migration test script to verify functionality:")
    print("   python run_migration_test.py --test-type strategies --forex-only")
    print("2. Review migrated strategies in trading_bot/strategies_new/")
    print("3. Continue with migration of other asset classes")

if __name__ == "__main__":
    main()
