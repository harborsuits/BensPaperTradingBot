import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import bisect

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    """Order side in the order book"""
    BID = "bid"  # Buy orders
    ASK = "ask"  # Sell orders

class OrderBookEntry:
    """Entry in the order book at a specific price level"""
    def __init__(self, price: float, volume: float):
        self.price = price
        self.volume = volume
        self.orders = []  # Can track individual orders at this level

    def __repr__(self):
        return f"OrderBookEntry(price={self.price:.6f}, volume={self.volume:.2f})"

class OrderBook:
    """Simulated order book with bids and asks"""
    
    def __init__(self, symbol: str, tick_size: float = 0.01, depth: int = 10):
        """
        Initialize an order book
        
        Args:
            symbol: Symbol identifier
            tick_size: Minimum price increment
            depth: Number of price levels to track on each side
        """
        self.symbol = symbol
        self.tick_size = tick_size
        self.depth = depth
        
        # Initialize empty bid and ask books
        self.bids: List[OrderBookEntry] = []  # Sorted high to low
        self.asks: List[OrderBookEntry] = []  # Sorted low to high
        
        # Last trade information
        self.last_price: Optional[float] = None
        self.last_volume: Optional[float] = None
        self.last_timestamp: Optional[pd.Timestamp] = None
        
        # Daily OHLCV data
        self.open_price: Optional[float] = None
        self.high_price: Optional[float] = None
        self.low_price: Optional[float] = None
        self.close_price: Optional[float] = None
        self.total_volume: float = 0.0
        
        # Book imbalance metrics
        self.bid_volume: float = 0.0
        self.ask_volume: float = 0.0
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from best bid and ask"""
        if not self.bids or not self.asks:
            return self.last_price
        
        return (self.bids[0].price + self.asks[0].price) / 2
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        if not self.bids or not self.asks:
            return None
        
        return self.asks[0].price - self.bids[0].price
    
    @property
    def relative_spread(self) -> Optional[float]:
        """Calculate relative bid-ask spread as percentage of mid price"""
        if not self.bids or not self.asks or self.mid_price == 0:
            return None
        
        return self.spread / self.mid_price
    
    @property
    def imbalance(self) -> float:
        """Calculate order book imbalance (bid volume - ask volume)/(bid volume + ask volume)"""
        total_volume = self.bid_volume + self.ask_volume
        if total_volume == 0:
            return 0.0
        
        return (self.bid_volume - self.ask_volume) / total_volume
    
    def best_bid(self) -> Optional[OrderBookEntry]:
        """Get the best (highest) bid"""
        return self.bids[0] if self.bids else None
    
    def best_ask(self) -> Optional[OrderBookEntry]:
        """Get the best (lowest) ask"""
        return self.asks[0] if self.asks else None
    
    def bid_levels(self) -> List[OrderBookEntry]:
        """Get all bid levels (sorted high to low)"""
        return self.bids.copy()
    
    def ask_levels(self) -> List[OrderBookEntry]:
        """Get all ask levels (sorted low to high)"""
        return self.asks.copy()
    
    def add_order(self, side: OrderSide, price: float, volume: float) -> None:
        """
        Add a new order to the book
        
        Args:
            side: Order side (BID/ASK)
            price: Order price
            volume: Order volume
        """
        # Round price to tick size
        price = round(price / self.tick_size) * self.tick_size
        
        if side == OrderSide.BID:
            # Find the correct position to insert the order (bids sorted high to low)
            i = 0
            while i < len(self.bids) and price < self.bids[i].price:
                i += 1
                
            # If price level exists, just add volume
            if i < len(self.bids) and abs(self.bids[i].price - price) < 1e-10:
                self.bids[i].volume += volume
            else:
                # Create new price level
                new_entry = OrderBookEntry(price, volume)
                if i < len(self.bids):
                    self.bids.insert(i, new_entry)
                else:
                    self.bids.append(new_entry)
            
            # Update bid volume
            self.bid_volume += volume
            
            # Trim to max depth
            if len(self.bids) > self.depth:
                removed_entry = self.bids.pop()
                self.bid_volume -= removed_entry.volume
        
        elif side == OrderSide.ASK:
            # Find the correct position to insert the order (asks sorted low to high)
            i = 0
            while i < len(self.asks) and price > self.asks[i].price:
                i += 1
                
            # If price level exists, just add volume
            if i < len(self.asks) and abs(self.asks[i].price - price) < 1e-10:
                self.asks[i].volume += volume
            else:
                # Create new price level
                new_entry = OrderBookEntry(price, volume)
                if i < len(self.asks):
                    self.asks.insert(i, new_entry)
                else:
                    self.asks.append(new_entry)
            
            # Update ask volume
            self.ask_volume += volume
            
            # Trim to max depth
            if len(self.asks) > self.depth:
                removed_entry = self.asks.pop()
                self.ask_volume -= removed_entry.volume
    
    def remove_volume(self, side: OrderSide, price: float, volume: float) -> float:
        """
        Remove volume from the book at a given price level
        
        Args:
            side: Order side (BID/ASK)
            price: Price level
            volume: Volume to remove
            
        Returns:
            Volume that was actually removed
        """
        # Round price to tick size
        price = round(price / self.tick_size) * self.tick_size
        
        if side == OrderSide.BID:
            # Find the price level
            for i, bid in enumerate(self.bids):
                if abs(bid.price - price) < 1e-10:
                    # Remove as much volume as possible
                    removed = min(volume, bid.volume)
                    bid.volume -= removed
                    self.bid_volume -= removed
                    
                    # Remove price level if no volume left
                    if bid.volume < 1e-10:
                        self.bids.pop(i)
                    
                    return removed
            
            # Price level not found
            return 0.0
        
        elif side == OrderSide.ASK:
            # Find the price level
            for i, ask in enumerate(self.asks):
                if abs(ask.price - price) < 1e-10:
                    # Remove as much volume as possible
                    removed = min(volume, ask.volume)
                    ask.volume -= removed
                    self.ask_volume -= removed
                    
                    # Remove price level if no volume left
                    if ask.volume < 1e-10:
                        self.asks.pop(i)
                    
                    return removed
            
            # Price level not found
            return 0.0
    
    def match_market_order(self, side: OrderSide, volume: float) -> Tuple[float, float]:
        """
        Match a market order against the book
        
        Args:
            side: Order side (BID/ASK)
            volume: Order volume
            
        Returns:
            (matched_volume, average_price)
        """
        matched_volume = 0.0
        total_notional = 0.0
        remaining = volume
        
        if side == OrderSide.BID:
            # Buy market order matches against asks
            i = 0
            while i < len(self.asks) and remaining > 0:
                ask = self.asks[i]
                executed = min(remaining, ask.volume)
                
                # Update matched volume and notional
                matched_volume += executed
                total_notional += executed * ask.volume
                remaining -= executed
                
                # Update ask book
                self.asks[i].volume -= executed
                self.ask_volume -= executed
                
                # Remove level if no volume left
                if self.asks[i].volume < 1e-10:
                    self.asks.pop(i)
                else:
                    i += 1
        
        elif side == OrderSide.ASK:
            # Sell market order matches against bids
            i = 0
            while i < len(self.bids) and remaining > 0:
                bid = self.bids[i]
                executed = min(remaining, bid.volume)
                
                # Update matched volume and notional
                matched_volume += executed
                total_notional += executed * bid.volume
                remaining -= executed
                
                # Update bid book
                self.bids[i].volume -= executed
                self.bid_volume -= executed
                
                # Remove level if no volume left
                if self.bids[i].volume < 1e-10:
                    self.bids.pop(i)
                else:
                    i += 1
        
        # Calculate average execution price
        average_price = total_notional / matched_volume if matched_volume > 0 else None
        
        # Update last trade info if there was a match
        if matched_volume > 0 and average_price is not None:
            self.last_price = average_price
            self.last_volume = matched_volume
            self.last_timestamp = pd.Timestamp.now()
            self.total_volume += matched_volume
            
            # Update OHLCV
            if self.open_price is None:
                self.open_price = average_price
            self.high_price = max(self.high_price or 0, average_price)
            self.low_price = min(self.low_price or float('inf'), average_price)
            self.close_price = average_price
            
        return matched_volume, average_price
    
    def reset_daily_stats(self) -> None:
        """Reset daily OHLCV statistics"""
        self.open_price = None
        self.high_price = None
        self.low_price = None
        self.close_price = None
        self.total_volume = 0.0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert order book to a DataFrame for visualization"""
        # Create bid and ask DataFrames
        bid_data = []
        for bid in self.bids:
            bid_data.append({"side": "bid", "price": bid.price, "volume": bid.volume})
        
        ask_data = []
        for ask in self.asks:
            ask_data.append({"side": "ask", "price": ask.price, "volume": ask.volume})
        
        # Combine into one DataFrame
        if bid_data and ask_data:
            return pd.DataFrame(bid_data + ask_data)
        elif bid_data:
            return pd.DataFrame(bid_data)
        elif ask_data:
            return pd.DataFrame(ask_data)
        else:
            return pd.DataFrame(columns=["side", "price", "volume"])
    
    def __repr__(self) -> str:
        """String representation of the order book"""
        bid_str = "\n".join([f"{b.price:.4f}: {b.volume:.2f}" for b in self.bids[:5]])
        ask_str = "\n".join([f"{a.price:.4f}: {a.volume:.2f}" for a in self.asks[:5]])
        
        return (f"OrderBook({self.symbol})\n"
                f"Spread: {self.spread:.6f} ({self.relative_spread*100:.4f}%)\n"
                f"Imbalance: {self.imbalance:.4f}\n"
                f"Top 5 Bids:\n{bid_str}\n"
                f"Top 5 Asks:\n{ask_str}")

class OrderBookSimulator:
    """
    Simulator for realistic order book dynamics
    
    Features:
    - Builds and maintains order books for multiple symbols
    - Generates realistic order book shapes and dynamics
    - Simulates price impact for market orders
    - Simulates limit order execution
    - Provides realistic spread and depth profiles
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        tick_size: Union[float, Dict[str, float]] = 0.01,
        price_precision: Dict[str, int] = None,
        depth: int = 10,
        min_price: float = 1.0,
        price_anchoring: bool = True,
        mean_reversion_strength: float = 0.1,
        volatility_scaling: bool = True,
        random_seed: int = None,
        debug_mode: bool = False
    ):
        """
        Initialize order book simulator
        
        Args:
            symbols: List of symbols to track
            tick_size: Minimum price increment (per symbol or global)
            price_precision: Number of decimal places for prices by symbol
            depth: Number of price levels to track
            min_price: Minimum price to use
            price_anchoring: Whether to anchor book around OHLCV prices
            mean_reversion_strength: Strength of mean reversion (0-1)
            volatility_scaling: Whether to scale spreads by volatility
            random_seed: Random seed for reproducibility
            debug_mode: Enable detailed logging
        """
        self.depth = depth
        self.price_anchoring = price_anchoring
        self.mean_reversion_strength = mean_reversion_strength
        self.volatility_scaling = volatility_scaling
        self.debug_mode = debug_mode
        self.min_price = min_price
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Default symbols if none provided
        self.symbols = symbols or ["SPY", "AAPL", "MSFT", "AMZN", "GOOGL"]
        
        # Initialize order books
        self.order_books: Dict[str, OrderBook] = {}
        
        # Setup tick sizes
        if isinstance(tick_size, float):
            # Same tick size for all symbols
            self.tick_sizes = {symbol: tick_size for symbol in self.symbols}
        else:
            # Symbol-specific tick sizes
            self.tick_sizes = tick_size.copy()
            # Ensure all symbols have a tick size
            for symbol in self.symbols:
                if symbol not in self.tick_sizes:
                    self.tick_sizes[symbol] = 0.01  # Default
        
        # Setup price precision
        self.price_precision = price_precision or {}
        for symbol in self.symbols:
            if symbol not in self.price_precision:
                # Default precision based on tick size
                tick = self.tick_sizes.get(symbol, 0.01)
                if tick >= 1.0:
                    self.price_precision[symbol] = 0
                elif tick >= 0.1:
                    self.price_precision[symbol] = 1
                elif tick >= 0.01:
                    self.price_precision[symbol] = 2
                elif tick >= 0.001:
                    self.price_precision[symbol] = 3
                else:
                    self.price_precision[symbol] = 4
        
        # Historical data for volatility scaling
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.volatility_estimates: Dict[str, float] = {}
        
        # Create initial order books
        for symbol in self.symbols:
            self.order_books[symbol] = OrderBook(
                symbol=symbol,
                tick_size=self.tick_sizes.get(symbol, 0.01),
                depth=depth
            )
        
        logger.info(f"Initialized OrderBookSimulator with {len(self.symbols)} symbols")
    
    def update_from_ohlcv(self, symbol: str, ohlc_bar: pd.Series, timestamp: pd.Timestamp = None) -> None:
        """
        Update order book based on OHLCV bar data
        
        Args:
            symbol: Symbol to update
            ohlc_bar: OHLCV data in a pandas Series (open, high, low, close, volume)
            timestamp: Timestamp for the update
        """
        if symbol not in self.order_books:
            # Create order book if it doesn't exist
            self.order_books[symbol] = OrderBook(
                symbol=symbol,
                tick_size=self.tick_sizes.get(symbol, 0.01),
                depth=self.depth
            )
            
        # Extract OHLCV data
        open_price = ohlc_bar.get('open', 0)
        high_price = ohlc_bar.get('high', 0)
        low_price = ohlc_bar.get('low', 0)
        close_price = ohlc_bar.get('close', 0)
        volume = ohlc_bar.get('volume', 0)
        
        # Ensure valid prices
        if close_price <= 0:
            logger.warning(f"Invalid close price for {symbol}: {close_price}")
            return
        
        # Update order book with historical data if needed
        if symbol in self.historical_data:
            hist_data = self.historical_data[symbol]
            hist_data = pd.concat([hist_data, pd.DataFrame([ohlc_bar])])
            self.historical_data[symbol] = hist_data.tail(50)  # Keep last 50 bars
        else:
            self.historical_data[symbol] = pd.DataFrame([ohlc_bar])
        
        # Calculate volatility estimate
        if len(self.historical_data[symbol]) >= 5:
            returns = self.historical_data[symbol]['close'].pct_change().dropna()
            if len(returns) >= 5:
                self.volatility_estimates[symbol] = returns.std() * np.sqrt(252)
        
        # Reset daily stats
        self.order_books[symbol].reset_daily_stats()
        
        # Update order book's OHLCV references
        book = self.order_books[symbol]
        book.open_price = open_price
        book.high_price = high_price
        book.low_price = low_price
        book.close_price = close_price
        book.total_volume = volume
        book.last_price = close_price
        book.last_timestamp = timestamp or pd.Timestamp.now()
        
        # Clear existing book
        book.bids = []
        book.asks = []
        book.bid_volume = 0.0
        book.ask_volume = 0.0
        
        # Generate new order book shape
        self._generate_book_shape(symbol, close_price, volume)
    
    def _generate_book_shape(self, symbol: str, price: float, volume: float) -> None:
        """
        Generate realistic order book shape around a price
        
        Args:
            symbol: Symbol to generate for
            price: Current price to center book around
            volume: Volume to distribute in the book
        """
        book = self.order_books[symbol]
        tick_size = self.tick_sizes.get(symbol, 0.01)
        
        # Estimate spread based on price, volatility, and volume
        base_spread = max(tick_size, price * 0.0001)  # Default 1bp relative spread
        
        # Adjust spread based on volatility
        if self.volatility_scaling and symbol in self.volatility_estimates:
            vol = self.volatility_estimates[symbol]
            # Higher volatility = wider spread (typical relationship)
            vol_factor = max(1.0, min(3.0, 1.0 + 10 * vol))
            base_spread *= vol_factor
        
        # Adjust spread based on volume (higher volume = tighter spread)
        if volume > 0:
            volume_factor = max(0.5, min(2.0, 1.0 / np.log10(max(10, volume/1000))))
            base_spread *= volume_factor
        
        # Ensure minimum spread is at least one tick
        spread = max(tick_size, base_spread)
        
        # Calculate bid and ask prices
        mid_price = price
        bid_price = max(self.min_price, mid_price - (spread / 2))
        ask_price = mid_price + (spread / 2)
        
        # Round to tick size
        bid_price = round(bid_price / tick_size) * tick_size
        ask_price = round(ask_price / tick_size) * tick_size
        
        # Ensure minimum bid-ask spread
        if ask_price <= bid_price:
            ask_price = bid_price + tick_size
        
        # Estimate volume to distribute
        total_book_volume = max(100, volume * 2)  # Typical order book depth relative to trading volume
        
        # Create the shape of the book
        self._create_bid_levels(symbol, bid_price, total_book_volume * 0.5)
        self._create_ask_levels(symbol, ask_price, total_book_volume * 0.5)
        
        if self.debug_mode:
            logger.debug(f"Generated book for {symbol}: mid={mid_price:.4f}, spread={spread:.6f}")
            logger.debug(f"Bid levels: {len(book.bids)}, Ask levels: {len(book.asks)}")
    
    def _create_bid_levels(self, symbol: str, best_bid: float, total_volume: float) -> None:
        """Create bid side of the book with realistic shape"""
        book = self.order_books[symbol]
        tick_size = self.tick_sizes.get(symbol, 0.01)
        
        # Shape parameters
        depth = self.depth
        
        # Distribution parameters (higher alpha = more volume at the top of book)
        alpha = 1.5  # Shape parameter for volume distribution
        
        # Generate price levels
        prices = [best_bid - i * tick_size for i in range(depth)]
        
        # Generate volume distribution that decays with distance from top of book
        volumes = np.random.power(alpha, size=depth)
        
        # Normalize to total volume
        volumes = volumes / volumes.sum() * total_volume
        
        # Add random noise to volumes (5-15% variation)
        noise = np.random.uniform(0.85, 1.15, size=depth)
        volumes = volumes * noise
        
        # Create bid levels
        for i in range(depth):
            price = max(self.min_price, prices[i])
            volume = max(1.0, volumes[i])  # Ensure minimum volume
            book.add_order(OrderSide.BID, price, volume)
    
    def _create_ask_levels(self, symbol: str, best_ask: float, total_volume: float) -> None:
        """Create ask side of the book with realistic shape"""
        book = self.order_books[symbol]
        tick_size = self.tick_sizes.get(symbol, 0.01)
        
        # Shape parameters
        depth = self.depth
        
        # Distribution parameters (higher alpha = more volume at the top of book)
        alpha = 1.5  # Shape parameter for volume distribution
        
        # Generate price levels
        prices = [best_ask + i * tick_size for i in range(depth)]
        
        # Generate volume distribution that decays with distance from top of book
        volumes = np.random.power(alpha, size=depth)
        
        # Normalize to total volume
        volumes = volumes / volumes.sum() * total_volume
        
        # Add random noise to volumes (5-15% variation)
        noise = np.random.uniform(0.85, 1.15, size=depth)
        volumes = volumes * noise
        
        # Create ask levels
        for i in range(depth):
            price = prices[i]
            volume = max(1.0, volumes[i])  # Ensure minimum volume
            book.add_order(OrderSide.ASK, price, volume)
    
    def simulate_market_impact(self, symbol: str, side: OrderSide, volume: float) -> Tuple[float, float]:
        """
        Simulate market impact of an order
        
        Args:
            symbol: Symbol to trade
            side: Order side (BID/ASK)
            volume: Order volume
            
        Returns:
            (executed_volume, average_price)
        """
        if symbol not in self.order_books:
            logger.warning(f"No order book for symbol {symbol}")
            return 0.0, 0.0
        
        book = self.order_books[symbol]
        
        # Execute order against the book
        executed_volume, avg_price = book.match_market_order(side, volume)
        
        if executed_volume > 0:
            logger.debug(f"Executed {executed_volume:.2f} @ {avg_price:.4f} for {symbol} {side.value}")
        
        # After the execution, regenerate the shape of the impacted side
        # This simulates new orders flowing in to replace executed orders
        if side == OrderSide.BID:
            # Market buy affected the ask side, regenerate it
            if book.asks:
                new_best_ask = book.asks[0].price if book.asks else (book.last_price * 1.001)
                self._create_ask_levels(symbol, new_best_ask, book.ask_volume * 0.8)
        else:
            # Market sell affected the bid side, regenerate it
            if book.bids:
                new_best_bid = book.bids[0].price if book.bids else (book.last_price * 0.999)
                self._create_bid_levels(symbol, new_best_bid, book.bid_volume * 0.8)
        
        return executed_volume, avg_price
    
    def simulate_limit_order(self, symbol: str, side: OrderSide, price: float, volume: float, 
                            immediate_or_cancel: bool = False) -> Tuple[float, float]:
        """
        Simulate placing a limit order
        
        Args:
            symbol: Symbol to trade
            side: Order side (BID/ASK)
            price: Limit price
            volume: Order volume
            immediate_or_cancel: Whether order is IOC
            
        Returns:
            (executed_volume, average_price)
        """
        if symbol not in self.order_books:
            logger.warning(f"No order book for symbol {symbol}")
            return 0.0, 0.0
        
        book = self.order_books[symbol]
        executed_volume = 0.0
        total_notional = 0.0
        
        # Check if the limit order crosses the spread (would execute immediately)
        if side == OrderSide.BID and book.asks and price >= book.asks[0].price:
            # Buy limit crosses with ask side
            price_levels = book.ask_levels()
            remaining = volume
            
            for level in price_levels:
                if level.price > price:
                    break  # Price too high for this limit order
                
                # Calculate fill at this level
                fill_qty = min(remaining, level.volume)
                if fill_qty > 0:
                    executed_volume += fill_qty
                    total_notional += fill_qty * level.price
                    remaining -= fill_qty
                    
                    # Remove volume from the book
                    book.remove_volume(OrderSide.ASK, level.price, fill_qty)
                
                if remaining <= 0:
                    break
            
            # If IOC and not fully filled, don't place remaining
            if not immediate_or_cancel and remaining > 0:
                # Add remaining as a passive order
                book.add_order(side, price, remaining)
                
            # After execution, regenerate the ask side
            if executed_volume > 0 and book.asks:
                new_best_ask = book.asks[0].price if book.asks else (book.last_price * 1.001)
                self._create_ask_levels(symbol, new_best_ask, book.ask_volume * 0.8)
        
        elif side == OrderSide.ASK and book.bids and price <= book.bids[0].price:
            # Sell limit crosses with bid side
            price_levels = book.bid_levels()
            remaining = volume
            
            for level in price_levels:
                if level.price < price:
                    break  # Price too low for this limit order
                
                # Calculate fill at this level
                fill_qty = min(remaining, level.volume)
                if fill_qty > 0:
                    executed_volume += fill_qty
                    total_notional += fill_qty * level.price
                    remaining -= fill_qty
                    
                    # Remove volume from the book
                    book.remove_volume(OrderSide.BID, level.price, fill_qty)
                
                if remaining <= 0:
                    break
            
            # If IOC and not fully filled, don't place remaining
            if not immediate_or_cancel and remaining > 0:
                # Add remaining as a passive order
                book.add_order(side, price, remaining)
                
            # After execution, regenerate the bid side
            if executed_volume > 0 and book.bids:
                new_best_bid = book.bids[0].price if book.bids else (book.last_price * 0.999)
                self._create_bid_levels(symbol, new_best_bid, book.bid_volume * 0.8)
        
        else:
            # Limit order does not cross the spread, add to book
            if not immediate_or_cancel:
                book.add_order(side, price, volume)
        
        # Calculate average execution price
        avg_price = total_notional / executed_volume if executed_volume > 0 else 0.0
        
        # Update last trade if there was an execution
        if executed_volume > 0:
            book.last_price = avg_price
            book.last_volume = executed_volume
            book.last_timestamp = pd.Timestamp.now()
            book.total_volume += executed_volume
            
            logger.debug(f"Limit order executed {executed_volume:.2f} @ {avg_price:.4f} for {symbol} {side.value}")
        
        return executed_volume, avg_price
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get order book for a symbol"""
        return self.order_books.get(symbol)
    
    def get_best_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices for a symbol"""
        if symbol not in self.order_books:
            return None, None
        
        book = self.order_books[symbol]
        best_bid = book.bids[0].price if book.bids else None
        best_ask = book.asks[0].price if book.asks else None
        
        return best_bid, best_ask
    
    def get_mid_price(self, symbol: str) -> Optional[float]:
        """Get mid price for a symbol"""
        if symbol not in self.order_books:
            return None
        
        return self.order_books[symbol].mid_price
    
    def get_spread(self, symbol: str) -> Optional[float]:
        """Get spread for a symbol"""
        if symbol not in self.order_books:
            return None
        
        return self.order_books[symbol].spread
    
    def get_book_depth(self, symbol: str, side: OrderSide) -> List[Tuple[float, float]]:
        """
        Get book depth for a symbol
        
        Args:
            symbol: Symbol to query
            side: Order side (BID/ASK)
            
        Returns:
            List of (price, volume) tuples
        """
        if symbol not in self.order_books:
            return []
        
        book = self.order_books[symbol]
        
        if side == OrderSide.BID:
            return [(bid.price, bid.volume) for bid in book.bids]
        else:
            return [(ask.price, ask.volume) for ask in book.asks]
    
    def get_order_book_imbalance(self, symbol: str) -> float:
        """Get order book imbalance for a symbol"""
        if symbol not in self.order_books:
            return 0.0
        
        return self.order_books[symbol].imbalance
    
    def get_order_book_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get a complete snapshot of the order book"""
        if symbol not in self.order_books:
            return {}
        
        book = self.order_books[symbol]
        
        return {
            "symbol": symbol,
            "timestamp": book.last_timestamp,
            "mid_price": book.mid_price,
            "spread": book.spread,
            "relative_spread": book.relative_spread,
            "imbalance": book.imbalance,
            "bid_volume": book.bid_volume,
            "ask_volume": book.ask_volume,
            "last_price": book.last_price,
            "last_volume": book.last_volume,
            "total_volume": book.total_volume,
            "bids": [(bid.price, bid.volume) for bid in book.bids],
            "asks": [(ask.price, ask.volume) for ask in book.asks]
        }
    
    def reset(self) -> None:
        """Reset all order books"""
        for symbol in self.symbols:
            self.order_books[symbol] = OrderBook(
                symbol=symbol,
                tick_size=self.tick_sizes.get(symbol, 0.01),
                depth=self.depth
            )
        
        self.historical_data = {}
        self.volatility_estimates = {}
        
        logger.info("Reset all order books")

class CircuitBreakerLevel(Enum):
    """Circuit breaker levels with increasing severity"""
    NORMAL = 0
    LEVEL_1 = 1  # Mild stress, reduced position sizing
    LEVEL_2 = 2  # Severe stress, minimal trading
    LEVEL_3 = 3  # Extreme stress, trading halt

class CircuitBreaker:
    """
    Multi-level circuit breaker system for risk management
    
    Features:
    - Monitors drawdown, volatility, and other risk metrics
    - Implements graduated response with multiple trigger levels
    - Dynamically adjusts position sizing based on risk conditions
    - Detailed logging of circuit breaker events
    """
    
    def __init__(
        self,
        drawdown_thresholds: Dict[CircuitBreakerLevel, float] = None,
        volatility_thresholds: Dict[CircuitBreakerLevel, float] = None,
        allocation_multipliers: Dict[CircuitBreakerLevel, float] = None,
        window_size: int = 20,
        cooldown_bars: int = 5,
        enable_logging: bool = True
    ):
        """
        Initialize circuit breaker system
        
        Args:
            drawdown_thresholds: Maximum drawdown thresholds by level (e.g., {LEVEL_1: 0.05, LEVEL_2: 0.15, LEVEL_3: 0.30})
            volatility_thresholds: Volatility thresholds by level as multiples of baseline (e.g., {LEVEL_1: 1.5, LEVEL_2: 2.0, LEVEL_3: 3.0})
            allocation_multipliers: Position size multipliers by level (e.g., {LEVEL_1: 0.75, LEVEL_2: 0.25, LEVEL_3: 0.0})
            window_size: Rolling window size for calculating metrics (bars)
            cooldown_bars: Minimum bars before downgrading circuit breaker level
            enable_logging: Whether to enable detailed logging
        """
        # Default thresholds if none provided
        self.drawdown_thresholds = drawdown_thresholds or {
            CircuitBreakerLevel.LEVEL_1: 0.10,  # 10% drawdown
            CircuitBreakerLevel.LEVEL_2: 0.20,  # 20% drawdown
            CircuitBreakerLevel.LEVEL_3: 0.30   # 30% drawdown
        }
        
        self.volatility_thresholds = volatility_thresholds or {
            CircuitBreakerLevel.LEVEL_1: 1.5,   # 1.5x normal volatility
            CircuitBreakerLevel.LEVEL_2: 2.0,   # 2x normal volatility
            CircuitBreakerLevel.LEVEL_3: 3.0    # 3x normal volatility
        }
        
        self.allocation_multipliers = allocation_multipliers or {
            CircuitBreakerLevel.NORMAL: 1.0,    # Normal allocation
            CircuitBreakerLevel.LEVEL_1: 0.75,  # 75% of normal allocation
            CircuitBreakerLevel.LEVEL_2: 0.25,  # 25% of normal allocation
            CircuitBreakerLevel.LEVEL_3: 0.0    # No new positions
        }
        
        self.window_size = window_size
        self.cooldown_bars = cooldown_bars
        self.enable_logging = enable_logging
        
        # Runtime state
        self.current_level = CircuitBreakerLevel.NORMAL
        self.bars_at_current_level = 0
        self.highest_equity = 0.0
        self.baseline_volatility = 0.0
        self.returns_history = []
        self.equity_history = []
        self.circuit_breaker_history = []
        
        logger.info("Initialized CircuitBreaker with multi-level thresholds")
    
    def update(self, equity: float, returns: float = None) -> CircuitBreakerLevel:
        """
        Update circuit breaker status based on latest metrics
        
        Args:
            equity: Current portfolio equity value
            returns: Latest period return (if None, calculated from equity)
            
        Returns:
            Current circuit breaker level
        """
        # Update histories
        self.equity_history.append(equity)
        self.highest_equity = max(self.highest_equity, equity)
        
        # Calculate return if not provided
        if returns is None and len(self.equity_history) >= 2:
            returns = (self.equity_history[-1] / self.equity_history[-2]) - 1
        
        if returns is not None:
            self.returns_history.append(returns)
        
        # Only start checking after we have enough history
        if len(self.equity_history) < self.window_size:
            return self.current_level
        
        # Trim histories to window size
        if len(self.equity_history) > self.window_size * 2:
            self.equity_history = self.equity_history[-self.window_size*2:]
        
        if len(self.returns_history) > self.window_size * 2:
            self.returns_history = self.returns_history[-self.window_size*2:]
        
        # Calculate current drawdown
        current_drawdown = 1.0 - (equity / self.highest_equity)
        
        # Calculate current volatility
        current_volatility = np.std(self.returns_history[-self.window_size:])
        
        # Update baseline volatility (average of the previous window)
        if len(self.returns_history) >= self.window_size * 2:
            self.baseline_volatility = np.std(self.returns_history[-self.window_size*2:-self.window_size])
        elif self.baseline_volatility == 0:
            self.baseline_volatility = current_volatility
        
        # Calculate relative volatility
        relative_volatility = 1.0
        if self.baseline_volatility > 0:
            relative_volatility = current_volatility / self.baseline_volatility
        
        # Determine appropriate circuit breaker level
        new_level = self._determine_circuit_breaker_level(current_drawdown, relative_volatility)
        
        # Apply hysteresis logic for level transitions
        if new_level.value > self.current_level.value:
            # Instantly upgrade to higher level
            self._change_level(new_level, current_drawdown, relative_volatility)
        elif new_level.value < self.current_level.value:
            # Only downgrade after cooldown period
            self.bars_at_current_level += 1
            if self.bars_at_current_level >= self.cooldown_bars:
                self._change_level(new_level, current_drawdown, relative_volatility)
        else:
            # Same level, increment counter
            self.bars_at_current_level += 1
        
        return self.current_level
    
    def _determine_circuit_breaker_level(self, drawdown: float, relative_volatility: float) -> CircuitBreakerLevel:
        """Determine the appropriate circuit breaker level based on metrics"""
        # Check for Level 3 (most severe)
        if (drawdown >= self.drawdown_thresholds.get(CircuitBreakerLevel.LEVEL_3, float('inf')) or
            relative_volatility >= self.volatility_thresholds.get(CircuitBreakerLevel.LEVEL_3, float('inf'))):
            return CircuitBreakerLevel.LEVEL_3
            
        # Check for Level 2
        if (drawdown >= self.drawdown_thresholds.get(CircuitBreakerLevel.LEVEL_2, float('inf')) or
            relative_volatility >= self.volatility_thresholds.get(CircuitBreakerLevel.LEVEL_2, float('inf'))):
            return CircuitBreakerLevel.LEVEL_2
            
        # Check for Level 1
        if (drawdown >= self.drawdown_thresholds.get(CircuitBreakerLevel.LEVEL_1, float('inf')) or
            relative_volatility >= self.volatility_thresholds.get(CircuitBreakerLevel.LEVEL_1, float('inf'))):
            return CircuitBreakerLevel.LEVEL_1
            
        # Otherwise normal
        return CircuitBreakerLevel.NORMAL
    
    def _change_level(self, new_level: CircuitBreakerLevel, drawdown: float, relative_volatility: float) -> None:
        """Change circuit breaker level and log the event"""
        old_level = self.current_level
        self.current_level = new_level
        self.bars_at_current_level = 0
        
        event = {
            "timestamp": pd.Timestamp.now(),
            "old_level": old_level.name,
            "new_level": new_level.name,
            "drawdown": drawdown,
            "relative_volatility": relative_volatility,
            "current_equity": self.equity_history[-1] if self.equity_history else 0.0,
            "highest_equity": self.highest_equity,
            "allocation_multiplier": self.get_allocation_multiplier()
        }
        
        self.circuit_breaker_history.append(event)
        
        if self.enable_logging:
            if new_level.value > old_level.value:
                logger.warning(f"Circuit breaker UPGRADED to {new_level.name}: drawdown={drawdown:.2%}, volatility={relative_volatility:.2f}x")
            else:
                logger.info(f"Circuit breaker downgraded to {new_level.name}: drawdown={drawdown:.2%}, volatility={relative_volatility:.2f}x")
    
    def get_allocation_multiplier(self) -> float:
        """Get current position sizing multiplier based on circuit breaker level"""
        return self.allocation_multipliers.get(self.current_level, 1.0)
    
    def get_slippage_multiplier(self) -> float:
        """Get slippage multiplier based on circuit breaker level"""
        # Increase expected slippage as conditions worsen
        slippage_multipliers = {
            CircuitBreakerLevel.NORMAL: 1.0,
            CircuitBreakerLevel.LEVEL_1: 1.5,
            CircuitBreakerLevel.LEVEL_2: 2.0,
            CircuitBreakerLevel.LEVEL_3: 3.0
        }
        return slippage_multipliers.get(self.current_level, 1.0)
    
    def is_trading_halted(self) -> bool:
        """Check if trading should be completely halted"""
        return self.current_level == CircuitBreakerLevel.LEVEL_3
    
    def should_deleverage(self) -> bool:
        """Check if portfolio should be deleveraged"""
        return self.current_level.value >= CircuitBreakerLevel.LEVEL_2.value
    
    def reset(self) -> None:
        """Reset circuit breaker to normal state"""
        self.current_level = CircuitBreakerLevel.NORMAL
        self.bars_at_current_level = 0
        self.highest_equity = 0.0 if not self.equity_history else self.equity_history[-1]
        
        if self.enable_logging:
            logger.info("Circuit breaker reset to NORMAL")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report of circuit breaker system"""
        return {
            "current_level": self.current_level.name,
            "bars_at_current_level": self.bars_at_current_level,
            "drawdown_thresholds": {k.name: v for k, v in self.drawdown_thresholds.items()},
            "volatility_thresholds": {k.name: v for k, v in self.volatility_thresholds.items()},
            "allocation_multipliers": {k.name: v for k, v in self.allocation_multipliers.items()},
            "current_allocation_multiplier": self.get_allocation_multiplier(),
            "current_slippage_multiplier": self.get_slippage_multiplier(),
            "trading_halted": self.is_trading_halted(),
            "circuit_breaker_history": self.circuit_breaker_history
        }
    
class MarketRegime(Enum):
    """Market regime classifications"""
    UNKNOWN = 0
    BULL = 1      # Rising prices, low volatility
    BEAR = 2      # Falling prices, high volatility
    SIDEWAYS = 3  # Range-bound, moderate volatility
    HIGH_VOL = 4  # Extremely high volatility regardless of direction
    LOW_VOL = 5   # Extremely low volatility regardless of direction
    CRISIS = 6    # Extreme volatility, sharp declines, liquidity issues

class MarketRegimeDetector:
    """
    Detects market regimes based on price action and volatility
    
    Features:
    - Classifies market conditions into distinct regimes
    - Uses moving averages, volatility, and other indicators
    - Provides regime-specific position sizing multipliers
    - Implements hysteresis to prevent frequent regime switching
    """
    
    def __init__(
        self,
        short_ma_window: int = 10,
        long_ma_window: int = 50,
        vol_window: int = 20,
        high_vol_threshold: float = 2.0,
        low_vol_threshold: float = 0.5,
        regime_change_threshold: int = 3,
        regime_multipliers: Dict[MarketRegime, float] = None
    ):
        """
        Initialize market regime detector
        
        Args:
            short_ma_window: Short-term moving average window
            long_ma_window: Long-term moving average window
            vol_window: Volatility calculation window
            high_vol_threshold: Threshold for high volatility (multiple of baseline)
            low_vol_threshold: Threshold for low volatility (multiple of baseline)
            regime_change_threshold: Minimum bars before switching regimes
            regime_multipliers: Position sizing multipliers by regime
        """
        self.short_ma_window = short_ma_window
        self.long_ma_window = long_ma_window
        self.vol_window = vol_window
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.regime_change_threshold = regime_change_threshold
        
        # Default regime multipliers if none provided
        self.regime_multipliers = regime_multipliers or {
            MarketRegime.UNKNOWN: 1.0,
            MarketRegime.BULL: 1.0,
            MarketRegime.BEAR: 0.5,
            MarketRegime.SIDEWAYS: 0.75,
            MarketRegime.HIGH_VOL: 0.4,
            MarketRegime.LOW_VOL: 1.2,
            MarketRegime.CRISIS: 0.2
        }
        
        # Runtime state
        self.current_regime = MarketRegime.UNKNOWN
        self.potential_regime = MarketRegime.UNKNOWN
        self.bars_in_potential_regime = 0
        self.price_history = []
        self.returns_history = []
        self.short_ma = []
        self.long_ma = []
        self.volatility_history = []
        self.baseline_volatility = None
        self.regime_history = []
    
    def update(self, price: float, returns: float = None) -> MarketRegime:
        """
        Update regime detection with new price data
        
        Args:
            price: Current price
            returns: Current period return (calculated from price if None)
            
        Returns:
            Current market regime
        """
        # Update price history
        self.price_history.append(price)
        
        # Calculate returns if not provided
        if returns is None and len(self.price_history) >= 2:
            returns = (self.price_history[-1] / self.price_history[-2]) - 1
        
        if returns is not None:
            self.returns_history.append(returns)
        
        # Need sufficient history for regime detection
        if len(self.price_history) < self.long_ma_window or len(self.returns_history) < self.vol_window:
            return self.current_regime
        
        # Trim histories to manageable size
        max_history = max(self.long_ma_window, self.vol_window) * 3
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
        
        if len(self.returns_history) > max_history:
            self.returns_history = self.returns_history[-max_history:]
        
        # Update moving averages
        self.short_ma = self._calculate_ma(self.price_history, self.short_ma_window)
        self.long_ma = self._calculate_ma(self.price_history, self.long_ma_window)
        
        # Update volatility
        current_volatility = np.std(self.returns_history[-self.vol_window:]) * np.sqrt(252)  # Annualized
        self.volatility_history.append(current_volatility)
        
        # Establish baseline volatility if not set
        if self.baseline_volatility is None and len(self.volatility_history) >= 10:
            self.baseline_volatility = np.median(self.volatility_history)
        elif self.baseline_volatility is None:
            self.baseline_volatility = current_volatility
        
        # Update baseline volatility with long-term median
        if len(self.volatility_history) >= 50:
            self.baseline_volatility = np.median(self.volatility_history[-50:])
        
        # Detect regime
        new_regime = self._detect_regime(
            current_price=price,
            short_ma=self.short_ma[-1] if self.short_ma else price,
            long_ma=self.long_ma[-1] if self.long_ma else price,
            current_volatility=current_volatility
        )
        
        # Apply hysteresis - only change regime after consistent signals
        if new_regime != self.potential_regime:
            self.potential_regime = new_regime
            self.bars_in_potential_regime = 1
        else:
            self.bars_in_potential_regime += 1
            
            # If we've seen the same potential regime for enough bars, switch to it
            if self.bars_in_potential_regime >= self.regime_change_threshold and new_regime != self.current_regime:
                old_regime = self.current_regime
                self.current_regime = new_regime
                self.bars_in_potential_regime = 0
                
                # Log regime change
                regime_change = {
                    "timestamp": pd.Timestamp.now(),
                    "old_regime": old_regime.name,
                    "new_regime": new_regime.name,
                    "price": price,
                    "volatility": current_volatility,
                    "baseline_volatility": self.baseline_volatility,
                    "vol_ratio": current_volatility / self.baseline_volatility if self.baseline_volatility else 1.0,
                    "multiplier": self.get_regime_multiplier()
                }
                self.regime_history.append(regime_change)
                
                logger.info(f"Market regime changed from {old_regime.name} to {new_regime.name}")
        
        return self.current_regime
    
    def _calculate_ma(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average over the given window"""
        if len(data) < window:
            return []
        
        result = []
        for i in range(window-1, len(data)):
            result.append(sum(data[i-window+1:i+1]) / window)
        
        return result
    
    def _detect_regime(
        self, 
        current_price: float, 
        short_ma: float, 
        long_ma: float, 
        current_volatility: float
    ) -> MarketRegime:
        """Detect market regime based on price and volatility indicators"""
        # Calculate volatility ratio
        vol_ratio = 1.0
        if self.baseline_volatility and self.baseline_volatility > 0:
            vol_ratio = current_volatility / self.baseline_volatility
        
        # Check for crisis conditions first (extreme volatility)
        if vol_ratio > self.high_vol_threshold * 1.5:
            return MarketRegime.CRISIS
        
        # Check for high/low volatility regimes
        if vol_ratio > self.high_vol_threshold:
            return MarketRegime.HIGH_VOL
        
        if vol_ratio < self.low_vol_threshold:
            return MarketRegime.LOW_VOL
        
        # Check trend direction
        if short_ma > long_ma:
            # Bullish trend
            return MarketRegime.BULL
        elif short_ma < long_ma:
            # Bearish trend
            return MarketRegime.BEAR
        else:
            # Sideways market
            return MarketRegime.SIDEWAYS
    
    def get_regime_multiplier(self) -> float:
        """Get position sizing multiplier for current regime"""
        return self.regime_multipliers.get(self.current_regime, 1.0)
    
    def get_regime_slippage_factor(self) -> float:
        """Get slippage adjustment factor for current regime"""
        # Higher slippage in high volatility or crisis regimes
        slippage_factors = {
            MarketRegime.UNKNOWN: 1.0,
            MarketRegime.BULL: 1.0,
            MarketRegime.BEAR: 1.3,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.HIGH_VOL: 1.8,
            MarketRegime.LOW_VOL: 0.8,
            MarketRegime.CRISIS: 2.5
        }
        return slippage_factors.get(self.current_regime, 1.0)
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report of market regime detector"""
        return {
            "current_regime": self.current_regime.name,
            "potential_regime": self.potential_regime.name,
            "bars_in_potential_regime": self.bars_in_potential_regime,
            "regime_multiplier": self.get_regime_multiplier(),
            "current_volatility": self.volatility_history[-1] if self.volatility_history else None,
            "baseline_volatility": self.baseline_volatility,
            "vol_ratio": (self.volatility_history[-1] / self.baseline_volatility 
                        if self.volatility_history and self.baseline_volatility else None),
            "short_ma": self.short_ma[-1] if self.short_ma else None,
            "long_ma": self.long_ma[-1] if self.long_ma else None,
            "regime_history": self.regime_history
        }
    
    def reset(self) -> None:
        """Reset regime detector state"""
        self.current_regime = MarketRegime.UNKNOWN
        self.potential_regime = MarketRegime.UNKNOWN
        self.bars_in_potential_regime = 0
        
        # Keep history for reference but reset tracking
        self.baseline_volatility = None if not self.volatility_history else np.median(self.volatility_history)
        
        logger.info("Reset market regime detector")

class VolatilityBasedPositionSizer:
    """
    Dynamically sizes positions based on market volatility
    
    Features:
    - Volatility targeting: adjust position size to maintain constant risk
    - ATR-based sizing: scale positions based on Average True Range
    - Adaptive leverage: automatically adjust leverage based on volatility
    - Integrates with market regime and circuit breakers for layered risk management
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annualized
        vol_window: int = 20,
        max_leverage: float = 2.0,
        use_atr: bool = False,
        atr_window: int = 14,
        atr_risk_factor: float = 2.0,
        volatility_cap: float = 0.4,  # 40% annualized
        min_sizing: float = 0.1,
        max_sizing: float = 2.0
    ):
        """
        Initialize volatility-based position sizer
        
        Args:
            target_volatility: Target annualized volatility for the portfolio
            vol_window: Window for calculating realized volatility
            max_leverage: Maximum allowed leverage
            use_atr: Whether to use ATR-based sizing instead of volatility targeting
            atr_window: Window for calculating ATR
            atr_risk_factor: Risk factor for ATR-based sizing
            volatility_cap: Maximum volatility to consider (prevents extreme shrinkage)
            min_sizing: Minimum position size multiple
            max_sizing: Maximum position size multiple
        """
        self.target_volatility = target_volatility
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.use_atr = use_atr
        self.atr_window = atr_window
        self.atr_risk_factor = atr_risk_factor
        self.volatility_cap = volatility_cap
        self.min_sizing = min_sizing
        self.max_sizing = max_sizing
        
        # Runtime state
        self.price_history = []
        self.high_history = []
        self.low_history = []
        self.returns_history = []
        self.realized_volatility = None
        self.atr_values = []
        self.current_atr = None
        self.position_size_history = []
    
    def update(
        self, 
        close_price: float, 
        high_price: float = None, 
        low_price: float = None, 
        returns: float = None
    ) -> float:
        """
        Update position sizer with new price data
        
        Args:
            close_price: Current closing price
            high_price: Current high price (for ATR calculation)
            low_price: Current low price (for ATR calculation)
            returns: Current period return (calculated from price if None)
            
        Returns:
            Current position size multiplier
        """
        # Default high/low to close if not provided
        high_price = high_price if high_price is not None else close_price
        low_price = low_price if low_price is not None else close_price
        
        # Update price history
        self.price_history.append(close_price)
        self.high_history.append(high_price)
        self.low_history.append(low_price)
        
        # Calculate returns if not provided
        if returns is None and len(self.price_history) >= 2:
            returns = (self.price_history[-1] / self.price_history[-2]) - 1
        
        if returns is not None:
            self.returns_history.append(returns)
        
        # Need sufficient history
        if len(self.returns_history) < self.vol_window:
            return 1.0  # Default to normal sizing with insufficient history
        
        # Trim histories to manageable size
        max_history = max(self.vol_window, self.atr_window) * 3
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.high_history = self.high_history[-max_history:]
            self.low_history = self.low_history[-max_history:]
            self.returns_history = self.returns_history[-max_history:]
        
        # Calculate realized volatility (annualized)
        self.realized_volatility = np.std(self.returns_history[-self.vol_window:]) * np.sqrt(252)
        
        # Calculate ATR if needed
        if self.use_atr and len(self.price_history) > self.atr_window:
            self.current_atr = self._calculate_atr()
            self.atr_values.append(self.current_atr)
        
        # Calculate position size multiplier
        if self.use_atr:
            position_size = self._get_atr_position_size()
        else:
            position_size = self._get_volatility_position_size()
        
        # Apply min/max constraints
        position_size = max(self.min_sizing, min(self.max_sizing, position_size))
        
        # Store position size history
        self.position_size_history.append(position_size)
        
        return position_size
    
    def _calculate_atr(self) -> float:
        """Calculate Average True Range"""
        if len(self.price_history) <= self.atr_window:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(self.price_history)):
            high = self.high_history[i]
            low = self.low_history[i]
            prev_close = self.price_history[i-1]
            
            # True Range is the greatest of:
            # 1. Current High - Current Low
            # 2. |Current High - Previous Close|
            # 3. |Current Low - Previous Close|
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        # Calculate ATR as simple average of true ranges
        return np.mean(true_ranges[-self.atr_window:])
    
    def _get_volatility_position_size(self) -> float:
        """Calculate position size based on volatility targeting"""
        if not self.realized_volatility or self.realized_volatility < 0.001:
            return 1.0  # Default to normal sizing with negligible volatility
        
        # Cap volatility to prevent extreme shrinkage
        capped_volatility = min(self.realized_volatility, self.volatility_cap)
        
        # Position size = target vol / realized vol
        position_size = self.target_volatility / capped_volatility
        
        # Apply leverage constraints
        return min(position_size, self.max_leverage)
    
    def _get_atr_position_size(self) -> float:
        """Calculate position size based on ATR"""
        if not self.current_atr or self.current_atr < 0.0001 or not self.price_history:
            return 1.0  # Default to normal sizing with negligible ATR
        
        # ATR as percentage of price
        atr_pct = self.current_atr / self.price_history[-1]
        
        # Risk-based position size multiplier:
        # If ATR is high (volatile), decrease position size
        # If ATR is low (stable), increase position size
        position_size = self.target_volatility / (atr_pct * self.atr_risk_factor * np.sqrt(252))
        
        # Apply leverage constraints
        return min(position_size, self.max_leverage)
    
    def get_current_position_size(self) -> float:
        """Get current position size multiplier"""
        if not self.position_size_history:
            return 1.0
        return self.position_size_history[-1]
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report of position sizer"""
        return {
            "target_volatility": self.target_volatility,
            "realized_volatility": self.realized_volatility,
            "current_atr": self.current_atr,
            "current_position_size": self.get_current_position_size(),
            "max_leverage": self.max_leverage,
            "volatility_cap": self.volatility_cap,
            "sizing_method": "ATR" if self.use_atr else "Volatility Targeting"
        }
    
    def reset(self) -> None:
        """Reset position sizer state"""
        self.realized_volatility = None
        self.current_atr = None
        self.position_size_history = []
        
        logger.info("Reset volatility-based position sizer")

class BacktestCircuitBreakerManager:
    """
    Manager for applying circuit breaker rules during backtesting
    
    This class connects CircuitBreaker risk management to the OrderBookSimulator
    and applies the appropriate restrictions based on circuit breaker levels.
    """
    
    def __init__(
        self,
        simulator: OrderBookSimulator,
        circuit_breaker: CircuitBreaker = None,
        regime_detector: MarketRegimeDetector = None,
        position_sizer: VolatilityBasedPositionSizer = None,
        initial_equity: float = 100000.0,
        hard_dollar_stop: float = None
    ):
        """
        Initialize the circuit breaker manager
        
        Args:
            simulator: The order book simulator to manage
            circuit_breaker: Circuit breaker instance (created if None)
            regime_detector: Market regime detector (created if None)
            position_sizer: Volatility-based position sizer (created if None)
            initial_equity: Starting equity for tracking performance
            hard_dollar_stop: Absolute dollar loss threshold for emergency stop
        """
        self.simulator = simulator
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.position_sizer = position_sizer or VolatilityBasedPositionSizer()
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.equity_curve = [initial_equity]
        self.daily_returns = []
        self.trade_restrictions = {}
        self.hard_dollar_stop = hard_dollar_stop
        self.emergency_stop_triggered = False
        
        logger.info("Initialized BacktestCircuitBreakerManager with enhanced risk controls")

    def update_equity(self, current_equity: float, ohlc_data: Dict[str, float] = None) -> Tuple[CircuitBreakerLevel, MarketRegime, float]:
        """
        Update equity and all risk management systems
        
        Args:
            current_equity: Current portfolio value
            ohlc_data: Optional OHLC data for more accurate volatility calculations
            
        Returns:
            Tuple of (circuit_breaker_level, market_regime, position_size_multiplier)
        """
        # Calculate return
        if len(self.equity_curve) > 0:
            daily_return = (current_equity / self.equity_curve[-1]) - 1
            self.daily_returns.append(daily_return)
        
        # Check hard dollar stop
        if self.hard_dollar_stop is not None and self.initial_equity - current_equity >= self.hard_dollar_stop:
            if not self.emergency_stop_triggered:
                logger.warning(f"EMERGENCY STOP: Hard dollar loss threshold of ${self.hard_dollar_stop:.2f} exceeded")
                self.emergency_stop_triggered = True
        
        # Update equity tracking
        self.equity = current_equity
        self.equity_curve.append(current_equity)
        
        # Extract OHLC data if provided
        close_price = ohlc_data.get('close', None) if ohlc_data else None
        high_price = ohlc_data.get('high', None) if ohlc_data else None
        low_price = ohlc_data.get('low', None) if ohlc_data else None
        
        # If no price data available, use equity as a proxy
        if close_price is None:
            close_price = current_equity
        
        # Update all risk management systems
        circuit_level = self.circuit_breaker.update(current_equity, daily_return if 'daily_return' in locals() else None)
        market_regime = self.regime_detector.update(close_price, daily_return if 'daily_return' in locals() else None)
        vol_position_size = self.position_sizer.update(
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            returns=daily_return if 'daily_return' in locals() else None
        )
        
        return circuit_level, market_regime, vol_position_size
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        price: float = None,
        volume: float = None,
        original_volume: float = None,
        is_market_order: bool = True,
        immediate_or_cancel: bool = False
    ) -> Tuple[float, float]:
        """
        Place an order with all risk management constraints applied
        
        Args:
            symbol: Symbol to trade
            side: Order side (BID/ASK)
            price: Limit price (None for market order)
            volume: Order volume
            original_volume: Original intended volume before any constraints
            is_market_order: True for market order, False for limit
            immediate_or_cancel: Whether limit order is IOC
            
        Returns:
            (executed_volume, average_price)
        """
        # Store original volume if not provided
        if original_volume is None:
            original_volume = volume
        
        # Check for emergency stop
        if self.emergency_stop_triggered:
            logger.warning(f"Order rejected: emergency stop triggered by hard dollar loss threshold")
            self._log_trade_restriction(symbol, side, original_volume, 0.0, "EMERGENCY_STOP")
            return 0.0, 0.0
        
        # Check if trading is halted by circuit breaker
        if self.circuit_breaker.is_trading_halted():
            logger.warning(f"Order rejected: trading halted by LEVEL_3 circuit breaker")
            self._log_trade_restriction(symbol, side, original_volume, 0.0, "CIRCUIT_HALT")
            return 0.0, 0.0
        
        # Apply ALL position sizing constraints:
        
        # 1. Circuit breaker allocation multiplier
        cb_multiplier = self.circuit_breaker.get_allocation_multiplier()
        
        # 2. Market regime multiplier
        regime_multiplier = self.regime_detector.get_regime_multiplier()
        
        # 3. Volatility-based position sizer
        vol_multiplier = self.position_sizer.get_current_position_size()
        
        # Combine all multipliers - use the most conservative approach
        if self.circuit_breaker.current_level == CircuitBreakerLevel.LEVEL_1:
            # In mild stress, use average of all constraints
            combined_multiplier = (cb_multiplier + regime_multiplier + vol_multiplier) / 3
        else:
            # In higher stress, use minimum (most conservative)
            combined_multiplier = min(cb_multiplier, regime_multiplier, vol_multiplier)
        
        # Apply position sizing constraint
        adjusted_volume = volume * combined_multiplier
        
        # Log if volume was restricted
        if adjusted_volume < volume:
            logger.info(
                f"Order volume reduced from {volume:.2f} to {adjusted_volume:.2f} "
                f"[CB: {cb_multiplier:.2f}, Regime: {regime_multiplier:.2f}, Vol: {vol_multiplier:.2f}]"
            )
            self._log_trade_restriction(
                symbol, side, original_volume, adjusted_volume, 
                "RESIZE", 
                cb_multiplier, regime_multiplier, vol_multiplier
            )
        
        # If volume is too small after adjustment, reject the order
        if adjusted_volume < 1.0:
            logger.info(f"Order rejected: volume too small after risk adjustments")
            self._log_trade_restriction(
                symbol, side, original_volume, 0.0, 
                "REJECT", 
                cb_multiplier, regime_multiplier, vol_multiplier
            )
            return 0.0, 0.0
        
        # Calculate adjusted slippage for market orders
        cb_slippage = self.circuit_breaker.get_slippage_multiplier()
        regime_slippage = self.regime_detector.get_regime_slippage_factor()
        
        # Combined slippage factor - use the more conservative (higher) value
        slippage_multiplier = max(cb_slippage, regime_slippage)
        
        # Execute the order through the simulator
        if is_market_order:
            return self.simulator.simulate_market_impact(symbol, side, adjusted_volume)
        else:
            return self.simulator.simulate_limit_order(symbol, side, price, adjusted_volume, immediate_or_cancel)
    
    def _log_trade_restriction(
        self,
        symbol: str,
        side: OrderSide,
        original_volume: float,
        adjusted_volume: float,
        restriction_type: str,
        cb_multiplier: float = None,
        regime_multiplier: float = None,
        vol_multiplier: float = None
    ) -> None:
        """Log details of trade restrictions for analysis"""
        restriction = {
            "timestamp": pd.Timestamp.now(),
            "symbol": symbol,
            "side": side.value,
            "original_volume": original_volume,
            "adjusted_volume": adjusted_volume,
            "restriction_type": restriction_type,
            "circuit_breaker_level": self.circuit_breaker.current_level.name,
            "market_regime": self.regime_detector.current_regime.name,
            "cb_multiplier": cb_multiplier,
            "regime_multiplier": regime_multiplier,
            "vol_multiplier": vol_multiplier,
            "combined_multiplier": adjusted_volume / original_volume if original_volume > 0 else 0.0,
            "emergency_stop": self.emergency_stop_triggered
        }
        
        self.trade_restrictions[len(self.trade_restrictions)] = restriction
    
    def should_deleverage(self) -> bool:
        """Check if portfolio should be deleveraged based on risk systems"""
        # Deleverage if circuit breaker indicates, or we're in a crisis/bear market
        return (
            self.circuit_breaker.should_deleverage() or 
            self.regime_detector.current_regime in [MarketRegime.CRISIS, MarketRegime.BEAR, MarketRegime.HIGH_VOL] or
            self.emergency_stop_triggered
        )
    
    def get_overall_risk_score(self) -> float:
        """
        Calculate an overall risk score (0-100) based on all risk systems
        Higher values indicate higher risk and more conservative positioning
        """
        # Circuit breaker component (0-40)
        cb_level = self.circuit_breaker.current_level.value
        cb_score = cb_level * 40 / 3  # 0 for NORMAL, 40 for LEVEL_3
        
        # Regime component (0-30)
        regime_scores = {
            MarketRegime.UNKNOWN: 15,
            MarketRegime.BULL: 0,
            MarketRegime.SIDEWAYS: 10,
            MarketRegime.LOW_VOL: 5,
            MarketRegime.BEAR: 20,
            MarketRegime.HIGH_VOL: 25,
            MarketRegime.CRISIS: 30
        }
        regime_score = regime_scores.get(self.regime_detector.current_regime, 15)
        
        # Volatility component (0-30)
        if self.position_sizer.realized_volatility and self.position_sizer.target_volatility:
            vol_ratio = self.position_sizer.realized_volatility / self.position_sizer.target_volatility
            vol_score = min(30, max(0, (vol_ratio - 0.8) * 30))  # Scale from 0-30
        else:
            vol_score = 15  # Neutral with no data
        
        # Emergency stop override
        if self.emergency_stop_triggered:
            return 100
        
        # Combine scores
        return min(100, cb_score + regime_score + vol_score)
    
    def get_restrictions_report(self) -> pd.DataFrame:
        """Get a DataFrame of all trade restrictions applied"""
        if not self.trade_restrictions:
            return pd.DataFrame(columns=[
                "timestamp", "symbol", "side", "original_volume", 
                "adjusted_volume", "restriction_type", "circuit_breaker_level",
                "market_regime", "cb_multiplier", "regime_multiplier", "vol_multiplier"
            ])
            
        return pd.DataFrame.from_dict(self.trade_restrictions, orient="index")
    
    def get_risk_status_report(self) -> Dict[str, Any]:
        """Get comprehensive risk status report from all systems"""
        return {
            "equity": self.equity,
            "drawdown_pct": 1.0 - (self.equity / max(self.equity_curve)) if self.equity_curve else 0.0,
            "drawdown_amount": max(self.equity_curve) - self.equity if self.equity_curve else 0.0,
            "overall_risk_score": self.get_overall_risk_score(),
            "emergency_stop_triggered": self.emergency_stop_triggered,
            "circuit_breaker": self.circuit_breaker.get_status_report(),
            "market_regime": self.regime_detector.get_status_report(),
            "position_sizer": self.position_sizer.get_status_report(),
            "trading_halted": self.circuit_breaker.is_trading_halted() or self.emergency_stop_triggered,
            "should_deleverage": self.should_deleverage()
        }
    
    def reset(self) -> None:
        """Reset all risk management systems"""
        self.circuit_breaker.reset()
        self.regime_detector.reset()
        self.position_sizer.reset()
        self.trade_restrictions = {}
        self.emergency_stop_triggered = False
        
        logger.info("Reset all risk management systems") 

class ScenarioType(Enum):
    """Types of market stress scenarios for testing"""
    CRASH = "crash"                  # Sharp market decline (e.g., -20% in a week)
    VOLATILITY_SPIKE = "vol_spike"   # Significant increase in market volatility
    LIQUIDITY_CRISIS = "liquidity"   # Reduced market depth, wider spreads
    CORRELATION_BREAKDOWN = "correlation"  # Unexpected changes in asset correlations
    SECTOR_ROTATION = "rotation"     # Rapid shift between market sectors
    FLASH_CRASH = "flash_crash"      # Very rapid drop and recovery
    BEAR_MARKET = "bear"             # Extended downtrend
    REGIME_SHIFT = "regime_shift"    # Sudden change in market conditions
    CUSTOM = "custom"                # User-defined scenario

class Scenario:
    """
    Definition of a market stress scenario for testing
    
    This class encapsulates all parameters needed to simulate a specific
    market stress scenario, such as crash, volatility spike, liquidity crisis, etc.
    """
    
    def __init__(
        self,
        name: str,
        scenario_type: ScenarioType,
        duration_bars: int,
        start_bar: int = 50,
        description: str = "",
        severity: float = 1.0,
        params: Dict[str, Any] = None
    ):
        """
        Initialize a scenario definition
        
        Args:
            name: Unique scenario name
            scenario_type: Type of scenario (crash, vol spike, etc.)
            duration_bars: Number of bars the scenario lasts
            start_bar: Bar index when the scenario starts
            description: Detailed scenario description
            severity: Scenario severity multiplier (1.0 = normal, 2.0 = twice as severe)
            params: Additional scenario-specific parameters
        """
        self.name = name
        self.scenario_type = scenario_type
        self.duration_bars = duration_bars
        self.start_bar = start_bar
        self.description = description
        self.severity = severity
        self.params = params or {}
        
        # Validation
        if self.duration_bars <= 0:
            raise ValueError("Scenario duration must be positive")
        
        if self.start_bar < 0:
            raise ValueError("Scenario start bar must be non-negative")
        
        if self.severity <= 0:
            raise ValueError("Scenario severity must be positive")
    
    def __repr__(self) -> str:
        return f"Scenario({self.name}, type={self.scenario_type.value}, bars={self.duration_bars}, severity={self.severity:.2f})"

class ScenarioPerformanceMetrics:
    """Performance metrics specific to scenario testing"""
    
    def __init__(self):
        """Initialize scenario performance metrics"""
        # Drawdown metrics
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.drawdown_duration = 0
        
        # Recovery metrics
        self.recovery_bars = 0
        self.recovered = False
        
        # Return metrics
        self.scenario_return = 0.0
        self.scenario_return_annualized = 0.0
        
        # Risk-adjusted metrics
        self.stress_sharpe = 0.0
        self.stress_sortino = 0.0
        self.stress_calmar = 0.0
        
        # Strategy behavior
        self.avg_position_size = 0.0
        self.max_leverage = 0.0
        self.trade_count = 0
        self.rejection_count = 0
        self.circuit_breaker_activations = 0
        self.emergency_stop_activated = False
        
        # Liquidity metrics
        self.avg_slippage = 0.0
        self.max_slippage = 0.0
        self.partial_fills_pct = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting"""
        return {
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "drawdown_duration": self.drawdown_duration,
            "recovery_bars": self.recovery_bars,
            "recovered": self.recovered,
            "scenario_return": self.scenario_return,
            "scenario_return_annualized": self.scenario_return_annualized,
            "stress_sharpe": self.stress_sharpe,
            "stress_sortino": self.stress_sortino,
            "stress_calmar": self.stress_calmar,
            "avg_position_size": self.avg_position_size,
            "max_leverage": self.max_leverage,
            "trade_count": self.trade_count,
            "rejection_count": self.rejection_count,
            "circuit_breaker_activations": self.circuit_breaker_activations,
            "emergency_stop_activated": self.emergency_stop_activated,
            "avg_slippage": self.avg_slippage,
            "max_slippage": self.max_slippage,
            "partial_fills_pct": self.partial_fills_pct
        }

class ScenarioResult:
    """Results of a single scenario test"""
    
    def __init__(
        self,
        scenario: Scenario,
        equity_curve: List[float],
        returns: List[float],
        metrics: ScenarioPerformanceMetrics,
        trade_history: pd.DataFrame = None,
        risk_events: List[Dict[str, Any]] = None
    ):
        """
        Initialize scenario test results
        
        Args:
            scenario: The scenario that was tested
            equity_curve: Equity curve during the scenario
            returns: Daily returns during the scenario
            metrics: Performance metrics for the scenario
            trade_history: Trade execution history
            risk_events: Log of risk events (circuit breakers, etc.)
        """
        self.scenario = scenario
        self.equity_curve = equity_curve
        self.returns = returns
        self.metrics = metrics
        self.trade_history = trade_history
        self.risk_events = risk_events or []
        self.timestamp = pd.Timestamp.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for reporting"""
        return {
            "scenario": {
                "name": self.scenario.name,
                "type": self.scenario.scenario_type.value,
                "duration": self.scenario.duration_bars,
                "severity": self.scenario.severity,
                "description": self.scenario.description
            },
            "timestamp": self.timestamp,
            "metrics": self.metrics.to_dict(),
            "equity_final": self.equity_curve[-1] if self.equity_curve else None,
            "equity_change_pct": ((self.equity_curve[-1] / self.equity_curve[0]) - 1) if len(self.equity_curve) > 1 else 0.0,
            "risk_event_count": len(self.risk_events)
        }
    
    def save_to_file(self, directory: str) -> str:
        """
        Save scenario results to files
        
        Args:
            directory: Directory to save results
            
        Returns:
            Path to the saved results
        """
        import os
        import json
        from datetime import datetime
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Create scenario-specific subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_dir = os.path.join(directory, f"{self.scenario.name}_{timestamp}")
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Save summary as JSON
        summary_path = os.path.join(scenario_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        # Save equity curve as CSV
        equity_path = os.path.join(scenario_dir, "equity_curve.csv")
        pd.DataFrame({
            "equity": self.equity_curve,
            "returns": self.returns
        }).to_csv(equity_path, index=False)
        
        # Save trade history if available
        if self.trade_history is not None:
            trades_path = os.path.join(scenario_dir, "trade_history.csv")
            self.trade_history.to_csv(trades_path, index=False)
        
        # Save risk events as CSV
        if self.risk_events:
            events_path = os.path.join(scenario_dir, "risk_events.csv")
            pd.DataFrame(self.risk_events).to_csv(events_path, index=False)
        
        logger.info(f"Scenario results saved to {scenario_dir}")
        return scenario_dir

class ScenarioInjector:
    """
    Injects scenario events into market data
    
    This class modifies market data according to scenario specifications,
    creating synthetic stress conditions or replaying historical events.
    """
    
    def __init__(self, scenario: Scenario):
        """
        Initialize scenario injector
        
        Args:
            scenario: Scenario to inject
        """
        self.scenario = scenario
        self.current_bar = 0
        self.in_scenario = False
        self.scenario_bar = 0
    
    def update(self, bar_index: int) -> bool:
        """
        Update scenario state based on current bar
        
        Args:
            bar_index: Current bar index
            
        Returns:
            True if currently in scenario, False otherwise
        """
        self.current_bar = bar_index
        
        # Check if we're entering the scenario
        if not self.in_scenario and bar_index >= self.scenario.start_bar:
            self.in_scenario = True
            self.scenario_bar = 0
            logger.info(f"Entering scenario: {self.scenario.name}")
        
        # Check if we're exiting the scenario
        if self.in_scenario:
            if self.scenario_bar >= self.scenario.duration_bars:
                self.in_scenario = False
                logger.info(f"Exiting scenario: {self.scenario.name}")
            else:
                self.scenario_bar += 1
        
        return self.in_scenario
    
    def modify_ohlc(self, ohlc_data: Dict[str, float]) -> Dict[str, float]:
        """
        Modify OHLC data according to scenario
        
        Args:
            ohlc_data: Original OHLC data
            
        Returns:
            Modified OHLC data
        """
        if not self.in_scenario:
            return ohlc_data
        
        modified_ohlc = ohlc_data.copy()
        
        # Apply scenario-specific modifications
        if self.scenario.scenario_type == ScenarioType.CRASH:
            # Market crash: decrease prices steadily over the scenario duration
            progress = self.scenario_bar / self.scenario.duration_bars
            daily_change = self._get_crash_daily_change(progress)
            
            for key in ['open', 'high', 'low', 'close']:
                if key in modified_ohlc:
                    modified_ohlc[key] *= (1.0 + daily_change)
        
        elif self.scenario.scenario_type == ScenarioType.VOLATILITY_SPIKE:
            # Volatility spike: increase high-low range while maintaining same close direction
            vol_multiplier = self._get_vol_multiplier()
            mid_price = (modified_ohlc.get('high', 0) + modified_ohlc.get('low', 0)) / 2
            range_half = (modified_ohlc.get('high', 0) - modified_ohlc.get('low', 0)) / 2
            
            # Expand the range
            if 'high' in modified_ohlc:
                modified_ohlc['high'] = mid_price + range_half * vol_multiplier
            if 'low' in modified_ohlc:
                modified_ohlc['low'] = mid_price - range_half * vol_multiplier
        
        elif self.scenario.scenario_type == ScenarioType.LIQUIDITY_CRISIS:
            # No direct price changes, but will affect execution in the order book
            pass
        
        elif self.scenario.scenario_type == ScenarioType.FLASH_CRASH:
            # Flash crash: sharp drop followed by partial recovery
            progress = self.scenario_bar / self.scenario.duration_bars
            daily_change = self._get_flash_crash_daily_change(progress)
            
            for key in ['open', 'high', 'low', 'close']:
                if key in modified_ohlc:
                    modified_ohlc[key] *= (1.0 + daily_change)
        
        return modified_ohlc
    
    def modify_order_book(self, simulator: OrderBookSimulator, symbol: str) -> None:
        """
        Modify order book according to scenario
        
        Args:
            simulator: Order book simulator instance
            symbol: Symbol to modify
        """
        if not self.in_scenario:
            return
        
        # Get the order book for the symbol
        order_book = simulator.get_order_book(symbol)
        if not order_book:
            return
        
        # Apply scenario-specific modifications to the order book
        if self.scenario.scenario_type == ScenarioType.LIQUIDITY_CRISIS:
            # Reduce order book depth and widen spreads
            progress = self.scenario_bar / self.scenario.duration_bars
            self._apply_liquidity_crisis(order_book, progress)
        
        elif self.scenario.scenario_type == ScenarioType.VOLATILITY_SPIKE:
            # Widen spread and increase order book imbalance
            self._apply_volatility_to_book(order_book)
    
    def _get_crash_daily_change(self, progress: float) -> float:
        """Calculate daily price change for crash scenario"""
        # Default params for a crash scenario
        total_decline = self.scenario.params.get('total_decline', -0.20)  # 20% total decline
        front_loaded = self.scenario.params.get('front_loaded', True)  # More decline at the beginning
        
        # Apply severity multiplier
        total_decline *= self.scenario.severity
        
        if front_loaded:
            # More decline in the early stages
            daily_change = total_decline * (1.0 - progress) * 2.5 / self.scenario.duration_bars
        else:
            # Linear decline
            daily_change = total_decline / self.scenario.duration_bars
        
        return daily_change
    
    def _get_flash_crash_daily_change(self, progress: float) -> float:
        """Calculate daily price change for flash crash scenario"""
        # Default params for a flash crash scenario
        max_decline = self.scenario.params.get('max_decline', -0.15)  # 15% max decline
        recovery_pct = self.scenario.params.get('recovery_pct', 0.6)  # 60% recovery
        crash_point = self.scenario.params.get('crash_point', 0.3)  # When the crash bottoms
        
        # Apply severity multiplier
        max_decline *= self.scenario.severity
        
        if progress < crash_point:
            # Accelerating decline phase
            phase_progress = progress / crash_point
            return max_decline * phase_progress * 2.0 / self.scenario.duration_bars
        else:
            # Recovery phase
            phase_progress = (progress - crash_point) / (1.0 - crash_point)
            total_recovery = max_decline * recovery_pct
            return total_recovery * phase_progress * 2.0 / self.scenario.duration_bars
    
    def _get_vol_multiplier(self) -> float:
        """Calculate volatility multiplier based on scenario progress"""
        # Default params for volatility spike scenario
        base_multiplier = self.scenario.params.get('base_multiplier', 2.0)  # Double normal range
        peak_multiplier = self.scenario.params.get('peak_multiplier', 3.0)  # Triple at peak
        peak_position = self.scenario.params.get('peak_position', 0.5)  # Peak in the middle
        
        # Apply severity multiplier
        base_multiplier = 1.0 + (base_multiplier - 1.0) * self.scenario.severity
        peak_multiplier = 1.0 + (peak_multiplier - 1.0) * self.scenario.severity
        
        # Calculate position relative to peak
        progress = self.scenario_bar / self.scenario.duration_bars
        dist_from_peak = abs(progress - peak_position)
        peak_width = self.scenario.params.get('peak_width', 0.2)
        
        # Calculate multiplier using a bell curve shape
        if dist_from_peak <= peak_width:
            # Near the peak
            peak_factor = 1.0 - (dist_from_peak / peak_width)
            return base_multiplier + (peak_multiplier - base_multiplier) * peak_factor
        else:
            # Away from peak
            return base_multiplier
    
    def _apply_liquidity_crisis(self, order_book: OrderBook, progress: float) -> None:
        """Apply liquidity crisis effects to order book"""
        # Default params for liquidity crisis
        depth_reduction = self.scenario.params.get('depth_reduction', 0.7)  # 70% reduction at peak
        spread_multiplier = self.scenario.params.get('spread_multiplier', 3.0)  # 3x spread at peak
        
        # Apply severity multiplier
        depth_reduction *= self.scenario.severity
        spread_multiplier = 1.0 + (spread_multiplier - 1.0) * self.scenario.severity
        
        # Calculate reduction based on scenario progress
        # Triangular shape: rises to a peak then falls
        phase_factor = 1.0 - abs(2.0 * progress - 1.0)
        current_depth_reduction = depth_reduction * phase_factor
        current_spread_multiplier = 1.0 + (spread_multiplier - 1.0) * phase_factor
        
        # Reduce book depth by removing volume
        for i, bid in enumerate(order_book.bids):
            order_book.bids[i].volume *= (1.0 - current_depth_reduction)
        
        for i, ask in enumerate(order_book.asks):
            order_book.asks[i].volume *= (1.0 - current_depth_reduction)
        
        # Widen spread by moving best ask up and best bid down
        if order_book.bids and order_book.asks:
            mid_price = order_book.mid_price
            if mid_price:
                current_spread = order_book.spread or (mid_price * 0.001)  # Default 0.1% spread
                target_spread = current_spread * current_spread_multiplier
                
                # Adjust best bid and ask to achieve target spread
                spread_adjustment = (target_spread - current_spread) / 2
                
                if order_book.bids:
                    order_book.bids[0].price -= spread_adjustment
                
                if order_book.asks:
                    order_book.asks[0].price += spread_adjustment
    
    def _apply_volatility_to_book(self, order_book: OrderBook) -> None:
        """Apply volatility effects to order book"""
        vol_multiplier = self._get_vol_multiplier()
        
        # Increase bid-ask spread
        if order_book.bids and order_book.asks:
            mid_price = order_book.mid_price
            if mid_price:
                current_spread = order_book.spread or (mid_price * 0.001)
                target_spread = current_spread * vol_multiplier
                
                # Adjust best bid and ask to achieve target spread
                spread_adjustment = (target_spread - current_spread) / 2
                
                if order_book.bids:
                    order_book.bids[0].price -= spread_adjustment
                
                if order_book.asks:
                    order_book.asks[0].price += spread_adjustment
        
        # Increase order book imbalance
        imbalance_direction = self.scenario.params.get('imbalance_direction', 1)  # 1 for bid-heavy, -1 for ask-heavy
        imbalance_strength = self.scenario.params.get('imbalance_strength', 0.3) * self.scenario.severity
        
        if imbalance_direction > 0:
            # Increase bid volume relative to ask
            for bid in order_book.bids:
                bid.volume *= (1.0 + imbalance_strength)
            
            for ask in order_book.asks:
                ask.volume *= (1.0 - imbalance_strength / 2)
        else:
            # Increase ask volume relative to bid
            for bid in order_book.bids:
                bid.volume *= (1.0 - imbalance_strength / 2)
            
            for ask in order_book.asks:
                ask.volume *= (1.0 + imbalance_strength)

class ScenarioTester:
    """
    Framework for testing trading strategies under stress scenarios
    
    Features:
    - Run various market stress scenarios to test strategy robustness
    - Historical replay of crisis periods or synthetic scenario generation
    - Detailed metrics on strategy performance during stress events
    - Visualization and reporting tools for scenario analysis
    """
    
    def __init__(
        self,
        simulator: OrderBookSimulator,
        risk_manager: BacktestCircuitBreakerManager = None,
        scenarios_dir: str = "scenarios",
        results_dir: str = "scenario_results"
    ):
        """
        Initialize scenario tester
        
        Args:
            simulator: Order book simulator
            risk_manager: Risk management system
            scenarios_dir: Directory for scenario definitions
            results_dir: Directory for scenario results
        """
        self.simulator = simulator
        self.risk_manager = risk_manager or BacktestCircuitBreakerManager(simulator)
        self.scenarios_dir = scenarios_dir
        self.results_dir = results_dir
        
        # Collection of predefined scenarios
        self.scenarios: Dict[str, Scenario] = {}
        
        # Results from scenario tests
        self.results: Dict[str, ScenarioResult] = {}
        
        # Active scenario tracking
        self.active_scenario = None
        self.active_injector = None
        self.current_bar = 0
        
        # Performance data
        self.equity_curve = [self.risk_manager.equity]
        self.returns = []
        self.trade_history = []
        self.risk_events = []
        
        logger.info("Initialized ScenarioTester")
        
        # Load predefined scenarios
        self._load_predefined_scenarios()
    
    def _load_predefined_scenarios(self) -> None:
        """Load predefined scenario library"""
        # Market crash scenarios
        self.add_scenario(Scenario(
            name="moderate_crash",
            scenario_type=ScenarioType.CRASH,
            duration_bars=10,
            description="Moderate market crash (-15% over 10 days)",
            params={"total_decline": -0.15, "front_loaded": True}
        ))
        
        self.add_scenario(Scenario(
            name="severe_crash",
            scenario_type=ScenarioType.CRASH,
            duration_bars=15,
            description="Severe market crash (-30% over 15 days)",
            severity=2.0,
            params={"total_decline": -0.15, "front_loaded": True}
        ))
        
        # Volatility scenarios
        self.add_scenario(Scenario(
            name="volatility_spike",
            scenario_type=ScenarioType.VOLATILITY_SPIKE,
            duration_bars=20,
            description="Sharp increase in market volatility (2-3x normal levels)",
            params={"base_multiplier": 2.0, "peak_multiplier": 3.0}
        ))
        
        # Liquidity scenarios
        self.add_scenario(Scenario(
            name="liquidity_crisis",
            scenario_type=ScenarioType.LIQUIDITY_CRISIS,
            duration_bars=15,
            description="Market liquidity dries up (70% reduction in depth, 3x spreads)",
            params={"depth_reduction": 0.7, "spread_multiplier": 3.0}
        ))
        
        # Flash crash scenarios
        self.add_scenario(Scenario(
            name="flash_crash",
            scenario_type=ScenarioType.FLASH_CRASH,
            duration_bars=5,
            description="Flash crash with partial recovery (-15% drop, 60% recovery)",
            params={"max_decline": -0.15, "recovery_pct": 0.6, "crash_point": 0.3}
        ))
    
    def add_scenario(self, scenario: Scenario) -> None:
        """
        Add a scenario to the testing library
        
        Args:
            scenario: Scenario definition
        """
        self.scenarios[scenario.name] = scenario
        logger.info(f"Added scenario: {scenario.name}")
    
    def create_custom_scenario(
        self,
        name: str,
        scenario_type: ScenarioType,
        duration_bars: int,
        params: Dict[str, Any] = None,
        severity: float = 1.0,
        description: str = ""
    ) -> Scenario:
        """
        Create a custom scenario
        
        Args:
            name: Unique scenario name
            scenario_type: Type of scenario
            duration_bars: Number of bars the scenario lasts
            params: Scenario-specific parameters
            severity: Scenario severity multiplier
            description: Detailed description
            
        Returns:
            Created scenario
        """
        scenario = Scenario(
            name=name,
            scenario_type=scenario_type,
            duration_bars=duration_bars,
            description=description,
            severity=severity,
            params=params or {}
        )
        
        self.add_scenario(scenario)
        return scenario
    
    def run_scenario(
        self,
        scenario_name: str,
        ohlc_data: pd.DataFrame,
        symbol: str,
        initial_equity: float = None,
        max_bars: int = 200,
        post_scenario_bars: int = 50
    ) -> ScenarioResult:
        """
        Run a specific scenario
        
        Args:
            scenario_name: Name of scenario to run
            ohlc_data: OHLC data to use as baseline
            symbol: Trading symbol
            initial_equity: Starting equity (if None, use risk_manager's current equity)
            max_bars: Maximum number of bars to run
            post_scenario_bars: Number of bars to run after scenario ends
            
        Returns:
            Scenario test results
        """
        # Get the scenario
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenarios[scenario_name]
        
        # Reset state
        self._reset_state(initial_equity)
        
        # Create scenario injector
        self.active_scenario = scenario
        self.active_injector = ScenarioInjector(scenario)
        
        # Calculate total bars to run
        total_bars = min(
            max_bars,
            scenario.start_bar + scenario.duration_bars + post_scenario_bars
        )
        
        # Ensure we have enough data
        if len(ohlc_data) < total_bars:
            logger.warning(f"OHLC data has {len(ohlc_data)} bars, but scenario requires {total_bars}")
            total_bars = len(ohlc_data)
        
        # Run the backtest
        logger.info(f"Running scenario '{scenario_name}' for {total_bars} bars")
        
        for bar in range(total_bars):
            # Update current bar
            self.current_bar = bar
            
            # Get original OHLC data for this bar
            orig_ohlc = ohlc_data.iloc[bar].to_dict()
            
            # Update scenario injector
            in_scenario = self.active_injector.update(bar)
            
            # Apply scenario modifications if active
            if in_scenario:
                # Modify OHLC data according to scenario
                modified_ohlc = self.active_injector.modify_ohlc(orig_ohlc)
                
                # Modify order book according to scenario
                self.active_injector.modify_order_book(self.simulator, symbol)
            else:
                modified_ohlc = orig_ohlc
            
            # Update simulator with the (possibly modified) OHLC data
            self.simulator.update_from_ohlcv(symbol, pd.Series(modified_ohlc))
            
            # Update risk management systems
            cb_level, market_regime, position_size = self.risk_manager.update_equity(
                self.risk_manager.equity,
                modified_ohlc
            )
            
            # Record equity and returns
            self.equity_curve.append(self.risk_manager.equity)
            if len(self.equity_curve) >= 2:
                daily_return = (self.equity_curve[-1] / self.equity_curve[-2]) - 1.0
                self.returns.append(daily_return)
            
            # Log risk events
            self._log_risk_event(bar, cb_level, market_regime, position_size, in_scenario)
            
            # Update progress
            if bar % 10 == 0:
                logger.info(f"Scenario progress: {bar}/{total_bars} bars")
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(scenario)
        
        # Create result object
        result = ScenarioResult(
            scenario=scenario,
            equity_curve=self.equity_curve,
            returns=self.returns,
            metrics=metrics,
            trade_history=pd.DataFrame(self.trade_history) if self.trade_history else None,
            risk_events=self.risk_events
        )
        
        # Store result
        self.results[scenario_name] = result
        
        # Save result
        result.save_to_file(self.results_dir)
        
        logger.info(f"Completed scenario '{scenario_name}'")
        return result
    
    def run_all_scenarios(
        self,
        ohlc_data: pd.DataFrame,
        symbol: str,
        initial_equity: float = None,
        max_bars: int = 200,
        post_scenario_bars: int = 50
    ) -> Dict[str, ScenarioResult]:
        """
        Run all defined scenarios
        
        Args:
            ohlc_data: OHLC data to use as baseline
            symbol: Trading symbol
            initial_equity: Starting equity
            max_bars: Maximum number of bars to run
            post_scenario_bars: Number of bars to run after scenario ends
            
        Returns:
            Dictionary of scenario results
        """
        results = {}
        
        for scenario_name in self.scenarios:
            logger.info(f"Running scenario batch: {scenario_name}")
            result = self.run_scenario(
                scenario_name=scenario_name,
                ohlc_data=ohlc_data,
                symbol=symbol,
                initial_equity=initial_equity,
                max_bars=max_bars,
                post_scenario_bars=post_scenario_bars
            )
            
            results[scenario_name] = result
        
        # Generate comparative report
        self._generate_comparative_report(results)
        
        return results
    
    def _reset_state(self, initial_equity: float = None) -> None:
        """Reset tester state for a new scenario run"""
        # Reset simulator and risk manager
        self.simulator.reset()
        self.risk_manager.reset()
        
        # Set initial equity if provided
        if initial_equity is not None:
            self.risk_manager.equity = initial_equity
            self.risk_manager.initial_equity = initial_equity
            self.risk_manager.equity_curve = [initial_equity]
        
        # Reset tracking variables
        self.active_scenario = None
        self.active_injector = None
        self.current_bar = 0
        self.equity_curve = [self.risk_manager.equity]
        self.returns = []
        self.trade_history = []
        self.risk_events = []
    
    def _log_risk_event(
        self,
        bar: int,
        circuit_level: CircuitBreakerLevel,
        market_regime: MarketRegime,
        position_size: float,
        in_scenario: bool
    ) -> None:
        """Log risk management events"""
        # Check for significant risk events
        is_new_event = False
        
        # Check if this is the first entry or if any key risk metric has changed
        if not self.risk_events:
            is_new_event = True
        else:
            last_event = self.risk_events[-1]
            if (last_event['circuit_level'] != circuit_level.name or
                last_event['market_regime'] != market_regime.name or
                abs(last_event['position_size'] - position_size) > 0.1):
                is_new_event = True
        
        # Only log when there's a significant change
        if is_new_event:
            event = {
                'bar': bar,
                'timestamp': pd.Timestamp.now(),
                'equity': self.risk_manager.equity,
                'circuit_level': circuit_level.name,
                'market_regime': market_regime.name,
                'position_size': position_size,
                'in_scenario': in_scenario,
                'risk_score': self.risk_manager.get_overall_risk_score()
            }
            self.risk_events.append(event)
    
    def _calculate_metrics(self, scenario: Scenario) -> ScenarioPerformanceMetrics:
        """Calculate performance metrics for the scenario"""
        metrics = ScenarioPerformanceMetrics()
        
        # Define scenario period
        scenario_start = scenario.start_bar
        scenario_end = scenario.start_bar + scenario.duration_bars
        
        # Get equity curve segments
        if len(self.equity_curve) <= scenario_start:
            logger.warning("Equity curve too short for scenario analysis")
            return metrics
        
        pre_scenario = self.equity_curve[:scenario_start+1]
        in_scenario = self.equity_curve[scenario_start:min(scenario_end+1, len(self.equity_curve))]
        post_scenario = self.equity_curve[min(scenario_end+1, len(self.equity_curve)):]
        
        # Drawdown metrics
        if len(in_scenario) >= 2:
            peak = max(pre_scenario)
            trough = min(in_scenario)
            metrics.max_drawdown = peak - trough
            metrics.max_drawdown_pct = (peak - trough) / peak if peak > 0 else 0.0
            
            # Find drawdown duration
            below_peak = False
            duration = 0
            for equity in in_scenario:
                if equity < peak:
                    below_peak = True
                    duration += 1
                elif below_peak:
                    # Reset if we reached a new peak
                    break
            metrics.drawdown_duration = duration
        
        # Recovery metrics
        if len(post_scenario) > 0 and len(in_scenario) > 0:
            scenario_end_equity = in_scenario[-1]
            for i, equity in enumerate(post_scenario):
                if equity >= scenario_end_equity:
                    metrics.recovered = True
                    metrics.recovery_bars = i + 1
                    break
        
        # Return metrics
        if len(in_scenario) >= 2:
            metrics.scenario_return = (in_scenario[-1] / in_scenario[0]) - 1.0
            metrics.scenario_return_annualized = (
                (1.0 + metrics.scenario_return) ** (252 / len(in_scenario)) - 1.0
            )
        
        # Risk-adjusted metrics
        scenario_returns = self.returns[scenario_start:min(scenario_end, len(self.returns))]
        if len(scenario_returns) > 0:
            avg_return = np.mean(scenario_returns)
            std_return = np.std(scenario_returns)
            downside_returns = [r for r in scenario_returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else std_return
            
            # Sharpe and Sortino ratios (assuming 0% risk-free rate)
            metrics.stress_sharpe = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
            metrics.stress_sortino = avg_return / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
            
            # Calmar ratio
            if metrics.max_drawdown_pct > 0:
                metrics.stress_calmar = metrics.scenario_return_annualized / metrics.max_drawdown_pct
        
        # Get strategy behavior metrics from risk events
        scenario_risk_events = [e for e in self.risk_events if 
                               scenario_start <= e['bar'] <= scenario_end]
        
        if scenario_risk_events:
            position_sizes = [e['position_size'] for e in scenario_risk_events]
            metrics.avg_position_size = np.mean(position_sizes)
            
            # Count circuit breaker activations
            cb_levels = [e['circuit_level'] for e in scenario_risk_events]
            metrics.circuit_breaker_activations = sum(1 for level in cb_levels 
                                                    if level != CircuitBreakerLevel.NORMAL.name)
            
            # Check for emergency stop
            metrics.emergency_stop_activated = any(
                e.get('emergency_stop', False) for e in scenario_risk_events
            )
        
        return metrics
    
    def _generate_comparative_report(self, results: Dict[str, ScenarioResult]) -> None:
        """Generate a comparative report of all scenario results"""
        import os
        import json
        from datetime import datetime
        
        # Create report directory
        os.makedirs(self.results_dir, exist_ok=True)
        report_path = os.path.join(self.results_dir, f"comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Prepare comparative data
        comparative_data = {
            "timestamp": datetime.now().isoformat(),
            "scenario_count": len(results),
            "scenarios": {}
        }
        
        # Add summary of each scenario
        for name, result in results.items():
            comparative_data["scenarios"][name] = {
                "type": result.scenario.scenario_type.value,
                "equity_change_pct": ((result.equity_curve[-1] / result.equity_curve[0]) - 1) if len(result.equity_curve) > 1 else 0.0,
                "max_drawdown_pct": result.metrics.max_drawdown_pct,
                "recovery_bars": result.metrics.recovery_bars,
                "recovered": result.metrics.recovered,
                "circuit_breaker_activations": result.metrics.circuit_breaker_activations,
                "emergency_stop_activated": result.metrics.emergency_stop_activated
            }
        
        # Save comparative report
        with open(report_path, 'w') as f:
            json.dump(comparative_data, f, indent=2, default=str)
        
        logger.info(f"Comparative scenario report saved to {report_path}")
    
    def visualize_scenario(self, scenario_name: str) -> None:
        """
        Visualize results from a specific scenario
        
        Args:
            scenario_name: Name of the scenario to visualize
        """
        if scenario_name not in self.results:
            logger.warning(f"No results found for scenario '{scenario_name}'")
            return
        
        # This would typically generate plots, but we'll just print summary stats for now
        result = self.results[scenario_name]
        scenario = result.scenario
        metrics = result.metrics
        
        print(f"Scenario: {scenario.name} ({scenario.scenario_type.value})")
        print(f"Duration: {scenario.duration_bars} bars, Severity: {scenario.severity:.2f}")
        print(f"Description: {scenario.description}")
        print("\nPerformance Metrics:")
        print(f"  Equity Change: {((result.equity_curve[-1] / result.equity_curve[0]) - 1)*100:.2f}%")
        print(f"  Max Drawdown: {metrics.max_drawdown_pct*100:.2f}%")
        print(f"  Recovery Bars: {metrics.recovery_bars} (Recovered: {metrics.recovered})")
        print(f"  Stress Sharpe: {metrics.stress_sharpe:.4f}")
        print(f"  Stress Sortino: {metrics.stress_sortino:.4f}")
        print(f"  Stress Calmar: {metrics.stress_calmar:.4f}")
        print("\nRisk Management:")
        print(f"  Avg Position Size: {metrics.avg_position_size:.2f}")
        print(f"  Circuit Breaker Activations: {metrics.circuit_breaker_activations}")
        print(f"  Emergency Stop Activated: {metrics.emergency_stop_activated}")
        
        logger.info(f"Visualized scenario '{scenario_name}'")
    
    def get_scenario_library(self) -> List[Dict[str, Any]]:
        """Get information about all available scenarios"""
        return [
            {
                "name": scenario.name,
                "type": scenario.scenario_type.value,
                "duration": scenario.duration_bars,
                "severity": scenario.severity,
                "description": scenario.description
            }
            for scenario in self.scenarios.values()
        ]