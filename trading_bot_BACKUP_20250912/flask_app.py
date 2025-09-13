import os
import json
import logging
from flask import Flask, request, jsonify
from .webhook_handler import WebhookHandler

logger = logging.getLogger(__name__)

def create_app(config_path=None):
    """Create Flask application"""
    # Create Flask app
    app = Flask(__name__)
    
    # Initialize webhook handler
    webhook_handler = WebhookHandler(config_path)
    
    @app.route('/api/webhook/tradingview', methods=['POST'])
    def tradingview_webhook():
        """Handle webhooks from TradingView"""
        try:
            # Get webhook data
            if not request.is_json:
                return jsonify({"status": "error", "message": "Request must be JSON"}), 400
                
            webhook_data = request.json
            logger.info(f"Received TradingView webhook: {json.dumps(webhook_data)}")
            
            # Process the webhook
            result = webhook_handler.process_webhook(webhook_data, request.headers)
            
            # Return the result
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/signals', methods=['GET'])
    def get_signals():
        """Get signal history"""
        try:
            # Get query parameters
            limit = request.args.get('limit', default=10, type=int)
            strategy = request.args.get('strategy')
            symbol = request.args.get('symbol')
            
            # Get signals
            signals = webhook_handler.get_signals_history(
                limit=limit,
                strategy=strategy,
                symbol=symbol
            )
            
            return jsonify({
                "status": "success",
                "count": len(signals),
                "signals": signals
            })
            
        except Exception as e:
            logger.error(f"Error getting signals: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/stats', methods=['GET'])
    def get_stats():
        """Get signal processing statistics"""
        try:
            stats = webhook_handler.get_stats()
            
            return jsonify({
                "status": "success",
                "stats": stats
            })
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/market-context', methods=['GET'])
    def get_market_context():
        """Get current market context"""
        try:
            context = {
                "regime": webhook_handler.market_context.current_regime,
                "vix_level": webhook_handler.market_context.vix_level,
                "market_open": webhook_handler.market_context.market_open,
                "confidence": webhook_handler.market_context.regime_confidence,
                "last_update": webhook_handler.market_context.last_update.isoformat() 
                    if webhook_handler.market_context.last_update else None
            }
            
            return jsonify({
                "status": "success",
                "market_context": context
            })
            
        except Exception as e:
            logger.error(f"Error getting market context: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/healthcheck', methods=['GET'])
    def healthcheck():
        """Health check endpoint"""
        return jsonify({"status": "ok"})
    
    return app

def run_app(config_path=None, host='0.0.0.0', port=5000, debug=False):
    """Run Flask application"""
    app = create_app(config_path)
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get config path from environment
    config_path = os.environ.get('CONFIG_PATH')
    
    # Run the app
    run_app(
        config_path=config_path,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    ) 