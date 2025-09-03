
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def root():
    return jsonify({'message': 'Test API running', 'version': '1.0.0'})

@app.route('/api/portfolio')
def get_portfolio():
    return jsonify({
        'totalValue': 100000,
        'dailyChange': 1500,
        'dailyChangePercent': 1.5,
        'cash': 25000,
        'invested': 75000
    })

@app.route('/api/strategies')
def get_strategies():
    return jsonify([
        {
            'id': 'strat-1',
            'name': 'Momentum',
            'active': True,
            'exposure_pct': 0.12,
            'p_l_30d': 1250.50
        }
    ])

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)

