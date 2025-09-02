#!/usr/bin/env python3
"""
VWAP (Volume-Weighted Average Price) Order Executor for BensBot

This module implements the VWAP algorithm for order execution,
distributing order chunks based on historical or expected volume profiles.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd

from trading_bot.core.advanced_orders import (
    VWAPOrder, AlgorithmicChunkPlaced, AlgorithmicOrderProgress
)
from trading_bot.core.algo_execution.base_executor import (
    AlgorithmicExecutionBase, AlgorithmicOrderTracker
)
from trading_bot.event_system.event_bus import EventBus


class VWAPExecutor(AlgorithmicExecutionBase):
    """
    VWAP (Volume-Weighted Average Price) Order Executor
    
    Implements the VWAP algorithm, distributing order chunks based on
    historical or expected volume profiles to achieve execution close
    to the market VWAP.
    """
    
    def __init__(self, broker_manager, event_bus: EventBus, market_data_provider=None):
        """
        Initialize the VWAP executor
        
        Args:
            broker_manager: Multi-broker manager for order routing
            event_bus: EventBus for order events
            market_data_provider: Provider for market data (required for VWAP)
        """
        super().__init__(broker_manager, event_bus)
        self.market_data_provider = market_data_provider
        self.logger = logging.getLogger(__name__)
        
        # Default volume profiles (to use if market data provider cannot provide them)
        self.default_volume_profiles = self._load_default_volume_profiles()
    
    def _load_default_volume_profiles(self) -> Dict[str, List[float]]:
        """
        Load default volume profiles
        
        Returns:
            Dictionary of volume profiles
        """
        # These are generic market volume profiles for different market types
        # Percentages of daily volume per hour in a typical trading day
        
        # U.S. Equities (9:30 AM - 4:00 PM ET)
        us_equities = [
            0.0850,  # 9:30-10:30 - Opening volume spike
            0.0730,  # 10:30-11:30
            0.0690,  # 11:30-12:30
            0.0630,  # 12:30-13:30 - Lunch dip
            0.0670,  # 13:30-14:30
            0.0730,  # 14:30-15:30
            0.1700   # 15:30-16:00 - Closing volume spike (30 min)
        ]
        
        # Normalize to hourly for the last entry (which is only 30 min)
        us_equities[-1] = us_equities[-1] * 2  # Convert to hourly rate for calculations
        
        # Forex (24 hours, but with volume patterns)
        forex = [
            0.0350,  # 00:00-01:00 UTC
            0.0250,  # 01:00-02:00
            0.0200,  # 02:00-03:00
            0.0180,  # 03:00-04:00
            0.0200,  # 04:00-05:00
            0.0350,  # 05:00-06:00
            0.0650,  # 06:00-07:00 London open
            0.0850,  # 07:00-08:00
            0.0950,  # 08:00-09:00
            0.0870,  # 09:00-10:00
            0.0800,  # 10:00-11:00
            0.0750,  # 11:00-12:00
            0.0850,  # 12:00-13:00 New York open
            0.0900,  # 13:00-14:00
            0.0850,  # 14:00-15:00
            0.0650,  # 15:00-16:00
            0.0500,  # 16:00-17:00
            0.0400,  # 17:00-18:00
            0.0300,  # 18:00-19:00
            0.0250,  # 19:00-20:00
            0.0220,  # 20:00-21:00
            0.0200,  # 21:00-22:00
            0.0250,  # 22:00-23:00
            0.0280   # 23:00-00:00
        ]
        
        # Futures (based on a generic futures trading session)
        futures = [
            0.0800,  # Opening hour
            0.0700,
            0.0650,
            0.0600,
            0.0550,
            0.0500,
            0.0550,
            0.0600,
            0.0650,
            0.0700,
            0.0900   # Closing hour
        ]
        
        # Cryptocurrencies (24 hours)
        crypto = [0.0417] * 24  # Roughly equal throughout the day
        
        return {
            "us_equities": us_equities,
            "forex": forex,
            "futures": futures,
            "crypto": crypto
        }
    
    def _create_order_tracker(self, order: VWAPOrder) -> 'VWAPOrderTracker':
        """
        Create a tracker for a VWAP order
        
        Args:
            order: VWAP order
            
        Returns:
            VWAP order tracker
        """
        return VWAPOrderTracker(order, self.event_bus)
    
    def _execute_algorithm(self, order_id: str):
        """
        Execute a VWAP order
        
        This method runs in a separate thread and manages the execution
        of the VWAP algorithm, including volume profile analysis, chunk
        scheduling, and placement.
        
        Args:
            order_id: Order ID
        """
        with self.lock:
            tracker = self.active_orders.get(order_id)
            
            if not tracker:
                self.logger.error(f"VWAP order {order_id} not found")
                return
        
        order = tracker.order
        
        try:
            # Generate execution schedule based on volume profile
            self._generate_execution_schedule(tracker)
            
            # Execute chunks according to schedule
            self._execute_chunks_according_to_schedule(tracker)
            
        except Exception as e:
            self.logger.error(f"Error executing VWAP order {order_id}: {str(e)}")
            
            # Fail the order
            remaining = tracker.get_remaining_quantity()
            self._fail_order(order_id, f"Error executing VWAP: {str(e)}", remaining)
        
        # Clean up thread
        with self.lock:
            if order_id in self.executor_threads:
                del self.executor_threads[order_id]
    
    def _generate_execution_schedule(self, tracker: 'VWAPOrderTracker'):
        """
        Generate execution schedule for a VWAP order based on volume profile
        
        Args:
            tracker: VWAP order tracker
        """
        order = tracker.order
        
        # Get order parameters
        total_quantity = order.total_quantity
        start_time = datetime.now()
        
        # Calculate end time based on duration
        end_time = start_time + timedelta(seconds=order.duration_seconds)
        
        # Get volume profile
        volume_profile = self._get_volume_profile(order, start_time, end_time)
        
        # Store volume profile for reference
        tracker.set_volume_profile(volume_profile)
        
        # Calculate number of chunks
        if order.num_chunks and order.num_chunks > 0:
            num_chunks = order.num_chunks
        else:
            # Default to number of volume profile segments or minimum of 10
            num_chunks = max(10, len(volume_profile))
        
        # Generate schedule
        schedule = []
        
        # Calculate time interval per chunk
        total_seconds = (end_time - start_time).total_seconds()
        seconds_per_chunk = total_seconds / num_chunks
        
        # Keep track of total allocation to handle rounding errors
        total_allocated = 0.0
        
        for i in range(num_chunks):
            # Calculate scheduled time
            chunk_start_offset = i * seconds_per_chunk
            scheduled_time = start_time + timedelta(seconds=chunk_start_offset)
            
            # Calculate chunk quantity based on volume profile
            current_time = scheduled_time
            profile_index = self._get_profile_index(current_time, start_time, end_time, len(volume_profile))
            volume_ratio = volume_profile[profile_index]
            
            # Calculate quantity for this chunk
            chunk_quantity = total_quantity * volume_ratio
            
            # Ensure we don't exceed total quantity with rounding
            total_allocated += chunk_quantity
            if i == num_chunks - 1:  # Last chunk
                chunk_quantity = total_quantity - (total_allocated - chunk_quantity)
            
            # Create chunk entry
            chunk = {
                "chunk_index": i,
                "quantity": chunk_quantity,
                "scheduled_time": scheduled_time,
                "status": "scheduled",
                "volume_ratio": volume_ratio
            }
            
            schedule.append(chunk)
        
        # Save schedule to tracker
        tracker.set_schedule(schedule)
        
        self.logger.info(
            f"Generated VWAP schedule for {order.order_id}: {num_chunks} chunks "
            f"over {order.duration_seconds} seconds based on volume profile"
        )
    
    def _get_volume_profile(self, order: VWAPOrder, start_time: datetime, 
                           end_time: datetime) -> List[float]:
        """
        Get volume profile for the order's time window
        
        Args:
            order: VWAP order
            start_time: Start time
            end_time: End time
            
        Returns:
            List of volume ratios
        """
        # Check if custom profile is provided
        if order.custom_volume_profile:
            return order.custom_volume_profile
        
        # Try to get historical volume profile from market data provider
        if self.market_data_provider:
            try:
                profile = self.market_data_provider.get_volume_profile(
                    symbol=order.symbol,
                    start_time=start_time,
                    end_time=end_time,
                    num_buckets=order.num_chunks or 10
                )
                
                if profile:
                    # Normalize to ensure sum is 1.0
                    total = sum(profile)
                    if total > 0:
                        return [v / total for v in profile]
            
            except Exception as e:
                self.logger.warning(f"Error getting volume profile from market data provider: {str(e)}")
        
        # Fall back to default profiles based on asset type
        if order.asset_class:
            asset_class = order.asset_class.lower()
            
            if asset_class in ["stock", "equity", "etf"]:
                profile_key = "us_equities"
            elif asset_class in ["forex", "fx"]:
                profile_key = "forex"
            elif asset_class in ["futures", "future"]:
                profile_key = "futures"
            elif asset_class in ["crypto", "cryptocurrency"]:
                profile_key = "crypto"
            else:
                profile_key = "us_equities"  # Default
                
            if profile_key in self.default_volume_profiles:
                return self._interpolate_profile(
                    self.default_volume_profiles[profile_key],
                    start_time, end_time, order.num_chunks or 10
                )
        
        # Default to uniform distribution if all else fails
        num_chunks = order.num_chunks or 10
        return [1.0 / num_chunks] * num_chunks
    
    def _interpolate_profile(self, base_profile: List[float], 
                            start_time: datetime, end_time: datetime,
                            num_chunks: int) -> List[float]:
        """
        Interpolate a base profile to the required number of chunks
        
        Args:
            base_profile: Base volume profile
            start_time: Start time
            end_time: End time
            num_chunks: Number of chunks
            
        Returns:
            Interpolated profile
        """
        # Convert to numpy array for interpolation
        profile = np.array(base_profile)
        
        # Normalize
        profile = profile / profile.sum()
        
        # Create time indices for original profile
        x_old = np.linspace(0, 1, len(profile))
        
        # Create time indices for new profile
        x_new = np.linspace(0, 1, num_chunks)
        
        # Interpolate
        interpolated = np.interp(x_new, x_old, profile)
        
        # Normalize again to ensure sum is 1.0
        interpolated = interpolated / interpolated.sum()
        
        return interpolated.tolist()
    
    def _get_profile_index(self, current_time: datetime, start_time: datetime, 
                          end_time: datetime, profile_length: int) -> int:
        """
        Get the index in the volume profile for the current time
        
        Args:
            current_time: Current time
            start_time: Start time
            end_time: End time
            profile_length: Length of volume profile
            
        Returns:
            Profile index
        """
        if current_time >= end_time:
            return profile_length - 1
        
        if current_time <= start_time:
            return 0
        
        # Calculate position within the time range (0.0 to 1.0)
        total_seconds = (end_time - start_time).total_seconds()
        elapsed_seconds = (current_time - start_time).total_seconds()
        position = elapsed_seconds / total_seconds
        
        # Map to profile index
        index = int(position * profile_length)
        
        # Clamp to valid range
        return max(0, min(profile_length - 1, index))
    
    def _execute_chunks_according_to_schedule(self, tracker: 'VWAPOrderTracker'):
        """
        Execute chunks according to the schedule
        
        Args:
            tracker: VWAP order tracker
        """
        order = tracker.order
        schedule = tracker.get_schedule()
        
        # Get broker ID
        broker_id = order.broker_id or self.broker_manager.get_preferred_broker(
            symbol=order.symbol,
            order_type=order.order_type,
            side=order.side
        )
        
        # Execute each chunk at its scheduled time
        for chunk in schedule:
            # Check if order was cancelled or failed
            with self.lock:
                if order.order_id not in self.active_orders:
                    self.logger.info(f"VWAP order {order.order_id} was cancelled or completed")
                    break
            
            # Wait until scheduled time
            now = datetime.now()
            scheduled_time = datetime.fromisoformat(chunk["scheduled_time"]) if isinstance(chunk["scheduled_time"], str) else chunk["scheduled_time"]
            
            if now < scheduled_time:
                wait_seconds = (scheduled_time - now).total_seconds()
                
                # Wait with periodic checks for cancellation
                wait_start = time.time()
                while time.time() - wait_start < wait_seconds:
                    # Check for cancellation every second
                    with self.lock:
                        if order.order_id not in self.active_orders:
                            break
                    
                    # Wait up to 1 second
                    time.sleep(min(1.0, wait_seconds - (time.time() - wait_start)))
                    
                    # Check if we've waited long enough
                    if time.time() - wait_start >= wait_seconds:
                        break
            
            # Check again if order was cancelled
            with self.lock:
                if order.order_id not in self.active_orders:
                    break
            
            # Place chunk order
            chunk_index = chunk["chunk_index"]
            quantity = chunk["quantity"]
            
            self.logger.debug(
                f"Placing VWAP chunk {chunk_index} for order {order.order_id}, "
                f"quantity: {quantity}, scheduled: {scheduled_time.isoformat()}"
            )
            
            # Get current market data for price adjustment
            market_data = None
            if self.market_data_provider:
                try:
                    market_data = self.market_data_provider.get_market_data(order.symbol)
                except Exception as e:
                    self.logger.warning(f"Error getting market data: {str(e)}")
            
            # Determine price for this chunk
            price = self._determine_chunk_price(order, market_data)
            
            # Generate chunk order ID
            chunk_order_id = f"{order.order_id}_chunk_{chunk_index}"
            
            try:
                # Place the order with broker manager
                result = self.broker_manager.place_order(
                    broker_id=broker_id,
                    symbol=order.symbol,
                    quantity=quantity,
                    side=order.side,
                    order_type=order.order_type,
                    price=price,
                    time_in_force=order.time_in_force,
                    metadata={
                        "algo_order_id": order.order_id,
                        "chunk_index": chunk_index,
                        "algorithm": "VWAP"
                    }
                )
                
                # Get assigned order ID
                assigned_order_id = result.get("order_id")
                
                if not assigned_order_id:
                    error_msg = f"Failed to place VWAP chunk {chunk_index}: {result.get('message', 'No order ID returned')}"
                    self.logger.error(error_msg)
                    
                    # Add retry logic here if needed
                    retry_count = tracker.increment_retry_count(chunk_index)
                    
                    if retry_count >= order.max_retries:
                        remaining = tracker.get_remaining_quantity()
                        self._fail_order(order.order_id, error_msg, remaining)
                        break
                    
                    # Wait before retry
                    time.sleep(1.0)
                    continue
                
                # Update tracker with chunk order
                with self.lock:
                    # Map chunk to parent order
                    self.chunk_map[assigned_order_id] = order.order_id
                    
                    # Record chunk
                    tracker.add_chunk(
                        chunk_index=chunk_index,
                        chunk_order_id=assigned_order_id,
                        quantity=quantity,
                        scheduled_time=scheduled_time
                    )
                
                # Emit chunk placed event
                self.event_bus.emit(AlgorithmicChunkPlaced(
                    order_id=order.order_id,
                    chunk_index=chunk_index,
                    quantity=quantity,
                    remaining_quantity=tracker.get_remaining_quantity(),
                    chunk_order_id=assigned_order_id,
                    algorithm="VWAP"
                ))
                
                # Record market conditions at time of placement
                if market_data:
                    tracker.update_performance_metric(f"chunk_{chunk_index}_market_price", market_data.get("last_price"))
                    tracker.update_performance_metric(f"chunk_{chunk_index}_bid_ask_spread", 
                                                     market_data.get("ask", 0) - market_data.get("bid", 0))
                
                # Update current benchmark VWAP
                if self.market_data_provider:
                    try:
                        benchmark_vwap = self.market_data_provider.get_vwap(
                            symbol=order.symbol,
                            start_time=tracker.start_time,
                            end_time=datetime.now()
                        )
                        
                        if benchmark_vwap:
                            tracker.update_performance_metric("current_benchmark_vwap", benchmark_vwap)
                    except Exception as e:
                        self.logger.warning(f"Error updating benchmark VWAP: {str(e)}")
                
                # Emit progress event
                self._emit_progress_event(tracker)
                
            except Exception as e:
                self.logger.error(f"Error executing VWAP chunk {chunk_index}: {str(e)}")
                
                # Add retry logic if needed
                retry_count = tracker.increment_retry_count(chunk_index)
                
                if retry_count >= order.max_retries:
                    remaining = tracker.get_remaining_quantity()
                    self._fail_order(order.order_id, f"Error executing VWAP chunk: {str(e)}", remaining)
                    break
                
                # Wait before retry
                time.sleep(1.0)
    
    def _determine_chunk_price(self, order: VWAPOrder, market_data: Optional[Dict[str, Any]]) -> Optional[float]:
        """
        Determine price for a chunk based on market data
        
        Args:
            order: VWAP order
            market_data: Current market data
            
        Returns:
            Price or None for market orders
        """
        # For market orders, return None
        if order.order_type.lower() == "market":
            return None
        
        # Use provided price if no market data or adaptation disabled
        if not market_data or not order.adapt_to_market:
            return order.price
        
        # Get current market price
        current_price = market_data.get("last_price")
        if not current_price:
            return order.price
        
        # Adjust based on order side
        if order.side.lower() == "buy":
            # For buy limit orders, adjust to stay competitive but below maximum acceptable price
            max_price = order.price * (1 + order.price_buffer_percent/100) if order.price else current_price
            
            if order.order_type.lower() == "limit":
                # Set at or slightly above current bid to be competitive
                bid = market_data.get("bid", current_price)
                price = min(max_price, bid * (1 + 0.001))  # 0.1 bp above bid
            else:
                # For other order types, use provided price
                price = min(max_price, current_price)
                
        else:  # Sell
            # For sell limit orders, adjust to stay competitive but above minimum acceptable price
            min_price = order.price * (1 - order.price_buffer_percent/100) if order.price else current_price
            
            if order.order_type.lower() == "limit":
                # Set at or slightly below current ask to be competitive
                ask = market_data.get("ask", current_price)
                price = max(min_price, ask * (1 - 0.001))  # 0.1 bp below ask
            else:
                # For other order types, use provided price
                price = max(min_price, current_price)
        
        return price
    
    def _get_performance_metrics(self, tracker: 'VWAPOrderTracker') -> Dict[str, Any]:
        """
        Get performance metrics for a VWAP order
        
        Args:
            tracker: VWAP order tracker
            
        Returns:
            Performance metrics
        """
        metrics = super()._get_performance_metrics(tracker)
        
        # Add VWAP-specific performance metrics
        
        # Calculate performance vs benchmark (if market data available)
        if self.market_data_provider:
            try:
                # Get VWAP benchmark for the order's time window
                start_time = tracker.start_time
                end_time = datetime.now()
                
                benchmark_vwap = self.market_data_provider.get_vwap(
                    symbol=tracker.order.symbol,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if benchmark_vwap:
                    # Calculate our VWAP
                    our_vwap = tracker.get_average_fill_price()
                    
                    # Performance vs benchmark
                    if our_vwap and benchmark_vwap:
                        if tracker.order.side.lower() == "buy":
                            # For buys, lower is better
                            metrics["vs_benchmark_bps"] = (benchmark_vwap - our_vwap) / benchmark_vwap * 10000
                        else:
                            # For sells, higher is better
                            metrics["vs_benchmark_bps"] = (our_vwap - benchmark_vwap) / benchmark_vwap * 10000
                        
                        metrics["benchmark_vwap"] = benchmark_vwap
                        metrics["our_vwap"] = our_vwap
            
            except Exception as e:
                self.logger.warning(f"Error calculating VWAP benchmark metrics: {str(e)}")
        
        return metrics


class VWAPOrderTracker(AlgorithmicOrderTracker):
    """
    Tracks the status of a VWAP order
    """
    
    def __init__(self, order: VWAPOrder, event_bus: EventBus):
        """
        Initialize the VWAP order tracker
        
        Args:
            order: VWAP order
            event_bus: EventBus
        """
        super().__init__(order, event_bus)
        
        # Additional VWAP-specific tracking
        self.volume_profile = []  # Volume profile
    
    def set_volume_profile(self, profile: List[float]):
        """
        Set the volume profile
        
        Args:
            profile: Volume profile
        """
        self.volume_profile = profile
        
        # Store in performance metrics
        self.update_performance_metric("volume_profile", profile)
