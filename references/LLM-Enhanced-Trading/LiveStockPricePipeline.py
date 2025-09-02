import websocket
import json
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
import pytz
import logging

class FinnhubWebSocket:
    def __init__(self, api_key, tickers, reconnect_delay=3):
        self.api_key = api_key
        self.tickers = tickers
        self.ws = None
        self.reconnect_delay = reconnect_delay
        self.cache = defaultdict(list)  # Cache to store price and volume data
        self.latest_vwap = {}  # Store the latest VWAP results for backend access
        self.last_minute = int(time.time() / 60)  # Track the current minute
        self.active = True  # Flag to control WebSocket activity

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("finnhub_websocket.log"),
            ],
        )

    def stop(self):
        """
        Gracefully stop the WebSocket connection.
        """
        self.active = False
        if self.ws:  # Close WebSocket if it's open
            self.ws.close()
            logging.info("WebSocket connection closed.")

    def convert_to_est(self, timestamp_ms):
        """
        Convert Unix timestamp in milliseconds to EST timezone.
        """
        timestamp_s = timestamp_ms / 1000  # Convert to seconds
        utc_time = datetime.utcfromtimestamp(timestamp_s)
        est_timezone = pytz.timezone("US/Eastern")
        est_time = utc_time.replace(tzinfo=pytz.utc).astimezone(est_timezone)
        return est_time.strftime("%Y-%m-%d %I:%M:%S %p %Z")

    def calculate_vwap(self, ticker_data):
        """
        Calculate VWAP for the given ticker data.
        """
        total_weighted_price = sum([entry["price"] * entry["volume"] for entry in ticker_data])
        total_volume = sum([entry["volume"] for entry in ticker_data])
        if total_volume > 0:
            return np.round(total_weighted_price / total_volume, 2)
        return None  # No trades for the ticker

    def dump_cache_to_file(self):
        """
        Dump the cache to a file, log the saved filename, retain the last key-value pair, and return VWAP results.
        """
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate a timestamp
        filename = f"{current_time}.txt"
        vwap_results = {}

        with open(filename, "w") as file:
            for ticker, data in self.cache.items():
                # Calculate VWAP
                vwap = self.calculate_vwap(data)
                vwap_results[ticker] = vwap
                file.write(f"Ticker: {ticker}, VWAP: {vwap}, Data: {data}\n")

        logging.info(f"Cache dumped and saved to {filename}")
        logging.info(f"VWAP results updated")

        # Retain only the last entry for each ticker in the cache
        for ticker, data in self.cache.items():
            if data:  # If there are trades, keep the last one
                self.cache[ticker] = [data[-1]]

        self.latest_vwap = vwap_results
        return vwap_results

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        if not self.active:
            return  # Ignore messages if WebSocket is not active

        data = json.loads(message)
        if "data" in data:
            for entry in data["data"]:
                ticker = entry["s"]
                price = entry["p"]
                volume = entry["v"]
                timestamp = entry["t"]

                # Convert timestamp to EST for logging
                est_time = self.convert_to_est(timestamp)

                # Store the trade in cache
                self.cache[ticker].append({"price": price, "volume": volume, "time": est_time})

        # Check if a minute has passed
        current_minute = int(time.time() / 60)
        if current_minute != self.last_minute:
            # Dump cache to file
            self.dump_cache_to_file()
            self.last_minute = current_minute

    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logging.error(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure."""
        if not self.active:
            return  # Do not attempt reconnection if WebSocket is stopped
        logging.warning("WebSocket closed. Attempting to reconnect...")
        time.sleep(self.reconnect_delay)
        self.start()  # Attempt to restart the WebSocket connection

    def on_open(self, ws):
        """Authenticate and subscribe to tickers on WebSocket open."""
        # Subscribe to tickers
        for ticker in self.tickers:
            subscribe_message = {
                "type": "subscribe",
                "symbol": ticker
            }
            ws.send(json.dumps(subscribe_message))
            logging.info(f"Subscribed to: {ticker}")

    def start(self):
        """Start the WebSocket connection."""
        self.active = True  # Set the WebSocket as active
        websocket.enableTrace(False)  # Disable WebSocket debugging for cleaner logs
        self.ws = websocket.WebSocketApp(
            f"wss://ws.finnhub.io?token={self.api_key}",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.on_open = self.on_open  # Assign the method, do not call it
        self.ws.run_forever()
