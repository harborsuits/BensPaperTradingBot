#!/usr/bin/env python3
"""
BensBot Strategy Import and Integration Demo

This script demonstrates how to:
1. Import strategies from BensBot
2. Apply meta-learning optimizations
3. Create EvoTrader-compatible strategy files
4. Register them with the evolutionary system
5. Run a test evolution with meta-learning guidance
"""

import os
import sys
import logging
import json
import argparse
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import components
from benbot.evotrader_bridge.strategy_importer import BenBotStrategyAdapter
from meta_learning_db import MetaLearningDB
from market_regime_detector import MarketRegimeDetector
from meta_learning_integration import MetaLearningIntegrator
from prop_strategy_registry import PropStrategyRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'import_demo.log')),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger('import_demo')

def import_benbot_strategies(source_path, 
                           output_dir=None, 
                           meta_db_path=None, 
                           recursive=True, 
                           register=True,
                           apply_meta_learning=True):
    """
    Import and integrate BensBot strategies with EvoTrader.
    
    Args:
        source_path: Path to BensBot strategy file or directory
        output_dir: Output directory for EvoTrader strategy files
        meta_db_path: Path to meta-learning database
        recursive: Search directory recursively
        register: Register imported strategies with PropStrategyRegistry
        apply_meta_learning: Apply meta-learning optimizations
        
    Returns:
        List of imported strategy information
    """
    try:
        # Default paths
        if not meta_db_path:
            meta_db_path = os.path.join(project_root, 'meta_learning', 'meta_db.sqlite')
        
        if not output_dir:
            output_dir = os.path.join(project_root, 'evotrader', 'strategies', 'imported')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create importer
        logger.info(f"Creating strategy importer with meta-db: {meta_db_path}")
        importer = BenBotStrategyAdapter(meta_db_path=meta_db_path if apply_meta_learning else None)
        
        # Create meta-learning components if needed
        meta_db = None
        regime_detector = None
        meta_integrator = None
        
        if apply_meta_learning:
            logger.info("Initializing meta-learning components")
            meta_db = MetaLearningDB(db_path=meta_db_path)
            regime_detector = MarketRegimeDetector()
            meta_integrator = MetaLearningIntegrator(
                meta_db=meta_db,
                regime_detector=regime_detector
            )
        
        # Import strategies
        logger.info(f"Importing strategies from: {source_path}")
        
        if os.path.isdir(source_path):
            imported_strategies = importer.import_strategy_from_directory(
                source_path, 
                recursive=recursive
            )
        else:
            imported_strategy = importer.import_strategy_from_path(source_path)
            imported_strategies = [imported_strategy] if imported_strategy else []
        
        logger.info(f"Imported {len(imported_strategies)} strategies")
        
        # Process each strategy
        processed_strategies = []
        
        for strategy_info in imported_strategies:
            if not strategy_info:
                continue
            
            strategy_id = strategy_info['strategy_id']
            strategy_name = strategy_info['strategy_name']
            
            logger.info(f"Processing strategy: {strategy_name} (ID: {strategy_id})")
            
            # Apply meta-learning optimizations if requested
            if apply_meta_learning and meta_integrator:
                logger.info(f"Applying meta-learning optimizations to {strategy_id}")
                
                # Get current market regime
                current_regime = regime_detector.detect_current_regime()
                
                # Apply meta-learning
                optimized_strategy = importer.apply_meta_learning(
                    strategy_id, 
                    market_regime=current_regime['regime']
                )
                
                logger.info(f"Applied meta-learning optimizations to {strategy_id} for regime: {current_regime['regime']}")
            else:
                optimized_strategy = strategy_info
            
            # Create EvoTrader strategy file
            file_path = importer.create_evotrader_strategy_file(
                strategy_id, 
                output_dir=output_dir
            )
            
            if file_path:
                logger.info(f"Created EvoTrader strategy file: {file_path}")
            else:
                logger.warning(f"Failed to create EvoTrader strategy file for: {strategy_id}")
                continue
            
            # Register strategy if requested
            if register:
                try:
                    logger.info(f"Registering strategy {strategy_id} with PropStrategyRegistry")
                    
                    # Initialize registry
                    registry = PropStrategyRegistry()
                    
                    # Create registration data
                    registration_data = {
                        'strategy_id': strategy_id,
                        'strategy_name': strategy_name,
                        'strategy_type': optimized_strategy['strategy_type'],
                        'strategy_file': file_path,
                        'parameters': optimized_strategy['parameters'],
                        'metadata': {
                            'source': 'benbot_import',
                            'imported_timestamp': datetime.now().isoformat(),
                            'original_source': optimized_strategy['file_path'],
                            'meta_learning_applied': apply_meta_learning,
                            'preferred_regimes': optimized_strategy['metadata'].get('preferred_regimes', []),
                            'supported_timeframes': optimized_strategy['metadata'].get('supported_timeframes', [])
                        },
                        'status': 'imported'
                    }
                    
                    # Register strategy
                    registry_result = registry.register_strategy(registration_data)
                    
                    if registry_result:
                        logger.info(f"Successfully registered strategy {strategy_id}")
                    else:
                        logger.warning(f"Failed to register strategy {strategy_id}")
                
                except Exception as e:
                    logger.error(f"Error registering strategy {strategy_id}: {e}")
            
            # Add to processed strategies
            processed_strategies.append({
                'strategy_id': strategy_id,
                'strategy_name': strategy_name,
                'strategy_type': optimized_strategy['strategy_type'],
                'file_path': file_path,
                'meta_learning_applied': apply_meta_learning,
                'registered': register
            })
        
        logger.info(f"Processed {len(processed_strategies)} strategies")
        return processed_strategies
    
    except Exception as e:
        logger.error(f"Error importing BensBot strategies: {e}")
        return []

def main():
    """Main function for the demo."""
    parser = argparse.ArgumentParser(description="BensBot Strategy Import and Integration Demo")
    parser.add_argument("--source", type=str, required=True, help="Path to BensBot strategy file or directory")
    parser.add_argument("--output", type=str, help="Output directory for EvoTrader strategy files")
    parser.add_argument("--meta-db", type=str, help="Path to meta-learning database")
    parser.add_argument("--recursive", action="store_true", help="Search directory recursively")
    parser.add_argument("--no-register", action="store_true", help="Don't register strategies with PropStrategyRegistry")
    parser.add_argument("--no-meta-learning", action="store_true", help="Don't apply meta-learning optimizations")
    
    args = parser.parse_args()
    
    # Import strategies
    imported = import_benbot_strategies(
        source_path=args.source,
        output_dir=args.output,
        meta_db_path=args.meta_db,
        recursive=args.recursive,
        register=not args.no_register,
        apply_meta_learning=not args.no_meta_learning
    )
    
    # Print summary
    print("\nImport Summary:")
    print(f"Total strategies processed: {len(imported)}")
    
    for idx, strategy in enumerate(imported, 1):
        print(f"\n{idx}. {strategy['strategy_name']} (ID: {strategy['strategy_id']})")
        print(f"   Type: {strategy['strategy_type']}")
        print(f"   File: {strategy['file_path']}")
        print(f"   Meta-learning applied: {strategy['meta_learning_applied']}")
        print(f"   Registered with PropStrategyRegistry: {strategy['registered']}")

if __name__ == "__main__":
    main()
