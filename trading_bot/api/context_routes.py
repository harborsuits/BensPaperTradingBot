from flask import Blueprint, jsonify, request, current_app
import logging
import json
from datetime import datetime

context_bp = Blueprint('context', __name__)
logger = logging.getLogger(__name__)

@context_bp.route('/api/market-context', methods=['GET'])
def get_market_context():
    """Get current market sentiment and context"""
    try:
        # Get query parameters
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        symbols = request.args.get('symbols')
        
        # Parse symbols if provided
        focus_symbols = None
        if symbols:
            focus_symbols = [s.strip().upper() for s in symbols.split(',')]
        
        # Get context analyzer
        context_analyzer = current_app.config['MARKET_CONTEXT_ANALYZER']
        
        # Get context
        context = context_analyzer.get_market_context(
            force_refresh=refresh,
            focus_symbols=focus_symbols
        )
        
        return jsonify({
            'status': 'success',
            'data': context
        })
    except Exception as e:
        logger.error(f"Error getting market context: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@context_bp.route('/api/strategy-recommendation', methods=['GET'])
def get_strategy_recommendation():
    """Get recommended trading strategies based on current market"""
    try:
        # Get query parameters
        symbols = request.args.get('symbols')
        
        # Parse symbols if provided
        focus_symbols = None
        if symbols:
            focus_symbols = [s.strip().upper() for s in symbols.split(',')]
        
        # Get context analyzer
        context_analyzer = current_app.config['MARKET_CONTEXT_ANALYZER']
        
        # Get context
        context = context_analyzer.get_market_context(focus_symbols=focus_symbols)
        
        # Extract strategies and reasoning
        strategies = context.get('suggested_strategies', [])
        triggers = context.get('triggers', [])
        reasoning = context.get('reasoning', '')
        
        return jsonify({
            'status': 'success',
            'data': {
                'market_bias': context.get('bias'),
                'confidence': context.get('confidence'),
                'recommended_strategies': strategies,
                'market_drivers': triggers,
                'analysis': reasoning,
                'timestamp': context.get('timestamp')
            }
        })
    except Exception as e:
        logger.error(f"Error getting strategy recommendations: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@context_bp.route('/api/context-scheduler/status', methods=['GET'])
def get_scheduler_status():
    """Get status of the adaptive context scheduler"""
    try:
        # Get scheduler from app config
        scheduler = current_app.config.get('CONTEXT_SCHEDULER')
        
        if not scheduler:
            return jsonify({
                'status': 'error',
                'message': 'Context scheduler not initialized'
            }), 404
        
        # Get scheduler status
        status = scheduler.get_status()
        
        return jsonify({
            'status': 'success',
            'data': status
        })
    except Exception as e:
        logger.error(f"Error getting scheduler status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@context_bp.route('/api/context-scheduler/update', methods=['POST'])
def trigger_context_update():
    """Manually trigger a context update"""
    try:
        # Get scheduler from app config
        scheduler = current_app.config.get('CONTEXT_SCHEDULER')
        
        if not scheduler:
            return jsonify({
                'status': 'error',
                'message': 'Context scheduler not initialized'
            }), 404
        
        # Get parameter for daily update
        is_daily = request.args.get('daily', 'false').lower() == 'true'
        
        # Trigger update
        output_path = scheduler.update_market_context(is_daily_update=is_daily)
        
        if not output_path:
            return jsonify({
                'status': 'error',
                'message': 'Failed to update context'
            }), 500
        
        return jsonify({
            'status': 'success',
            'message': f'Context updated successfully',
            'output_path': output_path
        })
    except Exception as e:
        logger.error(f"Error triggering context update: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@context_bp.route('/api/market-regime', methods=['GET'])
def get_market_regime():
    """Get current market regime details and forecasts"""
    try:
        # Get query parameters
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        symbols = request.args.get('symbols')
        timeframe = request.args.get('timeframe', 'daily')
        
        # Parse symbols if provided
        focus_symbols = None
        if symbols:
            focus_symbols = [s.strip().upper() for s in symbols.split(',')]
        
        # Get context analyzer
        context_analyzer = current_app.config['MARKET_CONTEXT_ANALYZER']
        
        # Get regime detector from context analyzer
        regime_detector = context_analyzer.market_regime_detector
        
        # Get current regime for each requested symbol or default markets
        regimes = {}
        
        if focus_symbols:
            for symbol in focus_symbols:
                try:
                    # Get market data for symbol
                    market_data = regime_detector.get_market_data(symbol, timeframe)
                    
                    # Detect regime
                    regime = regime_detector.detect_regime(market_data, symbol)
                    regimes[symbol] = regime
                except Exception as sym_error:
                    logger.error(f"Error detecting regime for {symbol}: {str(sym_error)}")
                    regimes[symbol] = {
                        "primary_regime": "unknown",
                        "confidence": 0,
                        "error": str(sym_error)
                    }
        else:
            # Get default indices regimes
            for index in regime_detector.default_indices:
                try:
                    # Get market data for index
                    market_data = regime_detector.get_market_data(index, timeframe)
                    
                    # Detect regime
                    regime = regime_detector.detect_regime(market_data, index)
                    regimes[index] = regime
                except Exception as idx_error:
                    logger.error(f"Error detecting regime for {index}: {str(idx_error)}")
                    regimes[index] = {
                        "primary_regime": "unknown",
                        "confidence": 0,
                        "error": str(idx_error)
                    }
        
        # Get aggregate market regime
        try:
            aggregate = regime_detector.get_aggregate_regime()
        except Exception as agg_error:
            logger.error(f"Error getting aggregate regime: {str(agg_error)}")
            aggregate = {
                "primary_regime": "unknown",
                "confidence": 0,
                "error": str(agg_error)
            }
        
        return jsonify({
            'status': 'success',
            'data': {
                'timestamp': datetime.now().isoformat(),
                'aggregate_regime': aggregate,
                'symbol_regimes': regimes,
                'regime_definitions': regime_detector.regime_definitions,
                'recommended_strategies': regime_detector.get_recommended_strategies(aggregate['primary_regime'])
            }
        })
    except Exception as e:
        logger.error(f"Error getting market regime: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@context_bp.route('/api/strategy-allocation', methods=['GET'])
def get_strategy_allocation():
    """Get current strategy allocation based on market context and performance"""
    try:
        # Get query parameters
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        performance_weight = float(request.args.get('performance_weight', '0.3'))
        
        # Get strategy prioritizer
        strategy_prioritizer = current_app.config.get('STRATEGY_PRIORITIZER')
        if not strategy_prioritizer:
            return jsonify({
                'status': 'error',
                'message': 'Strategy prioritizer not initialized'
            }), 404
            
        # Get context analyzer
        context_analyzer = current_app.config['MARKET_CONTEXT_ANALYZER']
        
        # Get market context
        market_context = context_analyzer.get_market_context(force_refresh=refresh)
        
        # Get performance data from the database
        # This is just a placeholder - in a real implementation you would fetch this from your database
        performance_data = strategy_prioritizer.get_strategy_performance()
        
        # Override performance weight if specified
        original_weight = strategy_prioritizer.performance_weight
        if performance_weight != original_weight:
            strategy_prioritizer.performance_weight = performance_weight
        
        # Get strategy allocation
        allocation = strategy_prioritizer.get_strategy_allocation(
            market_context=market_context,
            force_refresh=refresh,
            performance_data=performance_data
        )
        
        # Restore original weight
        if performance_weight != original_weight:
            strategy_prioritizer.performance_weight = original_weight
        
        return jsonify({
            'status': 'success',
            'data': {
                'timestamp': datetime.now().isoformat(),
                'allocation': allocation,
                'market_context': {
                    'regime': market_context.get('market_regime', {}),
                    'bias': market_context.get('bias'),
                    'volatility': market_context.get('volatility_level')
                },
                'performance_weight': performance_weight,
                'performance_metrics': {
                    strategy: {
                        metric: performance_data.get(strategy, {}).get(metric, 0)
                        for metric in ['sharpe_ratio', 'win_rate', 'avg_return']
                    }
                    for strategy in allocation.keys()
                }
            }
        })
    except Exception as e:
        logger.error(f"Error getting strategy allocation: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@context_bp.route('/api/context-scheduler/config', methods=['GET', 'POST'])
def scheduler_config():
    """Get or update scheduler configuration"""
    try:
        # Get scheduler from app config
        scheduler = current_app.config.get('CONTEXT_SCHEDULER')
        
        if not scheduler:
            return jsonify({
                'status': 'error',
                'message': 'Context scheduler not initialized'
            }), 404
        
        # Handle POST request to update config
        if request.method == 'POST':
            if not request.is_json:
                return jsonify({
                    'status': 'error',
                    'message': 'Request must be JSON'
                }), 400
            
            data = request.json
            
            # Update configurable parameters
            if 'market_hours_interval' in data:
                scheduler.market_hours_interval = int(data['market_hours_interval'])
            
            if 'after_hours_interval' in data:
                scheduler.after_hours_interval = int(data['after_hours_interval'])
            
            return jsonify({
                'status': 'success',
                'message': 'Configuration updated',
                'config': {
                    'market_hours_start': scheduler.market_hours_start,
                    'market_hours_end': scheduler.market_hours_end,
                    'market_hours_interval': scheduler.market_hours_interval,
                    'after_hours_interval': scheduler.after_hours_interval
                }
            })
        
        # Handle GET request to retrieve config
        return jsonify({
            'status': 'success',
            'config': {
                'market_hours_start': scheduler.market_hours_start,
                'market_hours_end': scheduler.market_hours_end,
                'market_hours_interval': scheduler.market_hours_interval,
                'after_hours_interval': scheduler.after_hours_interval,
                'is_market_hours': scheduler.is_market_hours()
            }
        })
    except Exception as e:
        logger.error(f"Error handling scheduler config: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 