#!/usr/bin/env python3
"""
Simple test server to verify port and CORS configuration
"""
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "message": "Test server is running correctly"})

@app.route('/api/strategies')
def strategies():
    return jsonify([
        {"id": "test1", "name": "Test Strategy 1", "status": "active"},
        {"id": "test2", "name": "Test Strategy 2", "status": "pending_win"}
    ])

if __name__ == '__main__':
    print("Starting test server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
