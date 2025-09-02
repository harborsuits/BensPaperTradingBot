#!/usr/bin/env python3
"""Simple test server to verify Flask connectivity"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Flask Server Test</h1>
        <p>If you can see this, your Flask server is working correctly!</p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Starting test server on http://127.0.0.1:3000")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='127.0.0.1', port=3000) 