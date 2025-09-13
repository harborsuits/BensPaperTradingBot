#!/usr/bin/env python3
"""
PnL Repository

This module provides the repository implementation for profit and loss records.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from trading_bot.persistence.mongo_repository import MongoRepository
from trading_bot.persistence.redis_repository import RedisRepository
from trading_bot.persistence.connection_manager import ConnectionManager


class PnLModel:
    """Data model for profit and loss persistence"""
    
    def __init__(
        self,
        timestamp: datetime,
        total_equity: float,
        unrealized_pnl: float,
        realized_pnl: float,
        cash_balance: Optional[float] = None,
        broker: Optional[str] = None,
        daily_pnl: Optional[float] = None,
        record_type: Optional[str] = None,  # 'snapshot', 'eod', 'manual', etc.
        equity_high_watermark: Optional[float] = None,
        drawdown: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        _id: Optional[str] = None
    ):
        """
        Initialize a PnL model.
        
        Args:
            timestamp: Record timestamp
            total_equity: Total account equity
            unrealized_pnl: Unrealized profit/loss
            realized_pnl: Realized profit/loss
            cash_balance: Cash balance (optional)
            broker: Broker identifier (optional)
            daily_pnl: Daily profit/loss (optional)
            record_type: Type of record (optional)
            equity_high_watermark: Equity high watermark (optional)
            drawdown: Current drawdown from high watermark (optional)
            drawdown_pct: Current drawdown percentage (optional)
            metadata: Additional metadata (optional)
            _id: MongoDB document ID
        """
        self.timestamp = timestamp
        self.total_equity = total_equity
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.cash_balance = cash_balance
        self.broker = broker
        self.daily_pnl = daily_pnl
        self.record_type = record_type or 'snapshot'
        self.equity_high_watermark = equity_high_watermark
        self.drawdown = drawdown
        self.drawdown_pct = drawdown_pct
        self.metadata = metadata or {}
        self._id = _id
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PnLModel':
        """Create a PnLModel from a dictionary"""
        # Convert ObjectId to string if present
        if '_id' in data and not isinstance(data['_id'], str):
            data['_id'] = str(data['_id'])
            
        # Handle timestamp conversion
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        result = {
            'timestamp': self.timestamp.isoformat(),
            'total_equity': self.total_equity,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'record_type': self.record_type,
            'metadata': self.metadata
        }
        
        # Add optional fields if present
        if self.cash_balance is not None:
            result['cash_balance'] = self.cash_balance
        if self.broker:
            result['broker'] = self.broker
        if self.daily_pnl is not None:
            result['daily_pnl'] = self.daily_pnl
        if self.equity_high_watermark is not None:
            result['equity_high_watermark'] = self.equity_high_watermark
        if self.drawdown is not None:
            result['drawdown'] = self.drawdown
        if self.drawdown_pct is not None:
            result['drawdown_pct'] = self.drawdown_pct
        
        if self._id:
            result['_id'] = self._id
            
        return result
    
    @classmethod
    def from_portfolio_update(cls, equity: float, unrealized_pnl: float, realized_pnl: float, 
                              cash: Optional[float] = None, broker: Optional[str] = None) -> 'PnLModel':
        """Create a PnLModel from portfolio update values"""
        return cls(
            timestamp=datetime.now(),
            total_equity=equity,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            cash_balance=cash,
            broker=broker,
            record_type='snapshot'
        )
    
    def calculate_drawdown(self, high_watermark: float) -> Tuple[float, float]:
        """
        Calculate drawdown from high watermark.
        
        Args:
            high_watermark: Equity high watermark
            
        Returns:
            Tuple of (drawdown amount, drawdown percentage)
        """
        if high_watermark <= 0:
            return 0.0, 0.0
            
        drawdown = max(0, high_watermark - self.total_equity)
        drawdown_pct = (drawdown / high_watermark) * 100.0 if high_watermark > 0 else 0.0
        
        return drawdown, drawdown_pct


class PnLRepository:
    """Repository for PnL persistence"""
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the PnL repository.
        
        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager
        self.mongo_repo = MongoRepository(connection_manager, 'pnl_records')
        self.redis_repo = RedisRepository(connection_manager, 'pnl_latest')
        self.logger = logging.getLogger(__name__)
        
        # Cache for high watermark
        self._high_watermark = None
    
    def record_snapshot(self, pnl: Union[PnLModel, Dict[str, Any]]) -> str:
        """
        Record a PnL snapshot.
        
        Args:
            pnl: PnL model or dictionary
            
        Returns:
            MongoDB document ID
        """
        if not isinstance(pnl, PnLModel):
            pnl = PnLModel.from_dict(pnl)
            
        try:
            # Calculate drawdown if high watermark available
            if self._high_watermark is None:
                self._high_watermark = self._get_high_watermark()
                
            if self._high_watermark is not None:
                if self.total_equity > self._high_watermark:
                    self._high_watermark = self.total_equity
                    pnl.equity_high_watermark = self._high_watermark
                    pnl.drawdown = 0.0
                    pnl.drawdown_pct = 0.0
                else:
                    pnl.equity_high_watermark = self._high_watermark
                    pnl.drawdown, pnl.drawdown_pct = pnl.calculate_drawdown(self._high_watermark)
            
            # Save to MongoDB
            result = self.mongo_repo.save(pnl)
            
            # Update latest in Redis
            try:
                latest_key = 'latest'
                if pnl.broker:
                    latest_key = f"latest:{pnl.broker}"
                    
                self.redis_repo.save_with_key(latest_key, pnl)
            except Exception as e:
                self.logger.warning(f"Failed to update latest PnL in Redis: {str(e)}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to record PnL snapshot: {str(e)}")
            raise
    
    def record_eod_snapshot(self, pnl: Union[PnLModel, Dict[str, Any]]) -> str:
        """
        Record an end-of-day PnL snapshot.
        
        Args:
            pnl: PnL model or dictionary
            
        Returns:
            MongoDB document ID
        """
        if not isinstance(pnl, PnLModel):
            pnl = PnLModel.from_dict(pnl)
            
        # Mark as EOD record
        pnl.record_type = 'eod'
        
        try:
            # Calculate daily P&L if possible
            yesterday = pnl.timestamp.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            yesterday_close = self.get_eod_snapshot(yesterday)
            
            if yesterday_close:
                pnl.daily_pnl = pnl.total_equity - yesterday_close.total_equity
            
            # Save to MongoDB
            return self.mongo_repo.save(pnl)
            
        except Exception as e:
            self.logger.error(f"Failed to record EOD PnL snapshot: {str(e)}")
            raise
    
    def get_latest_snapshot(self, broker: Optional[str] = None) -> Optional[PnLModel]:
        """
        Get the latest PnL snapshot.
        
        Args:
            broker: Optional broker filter
            
        Returns:
            Latest PnLModel if available
        """
        try:
            # Try Redis first
            try:
                latest_key = 'latest'
                if broker:
                    latest_key = f"latest:{broker}"
                    
                pnl = self.redis_repo.find_by_key(latest_key)
                if pnl:
                    if isinstance(pnl, dict):
                        return PnLModel.from_dict(pnl)
                    return pnl
            except Exception:
                # Redis error, continue to MongoDB
                pass
            
            # Try MongoDB
            query = {'record_type': 'snapshot'}
            if broker:
                query['broker'] = broker
                
            # Get most recent snapshot
            records = self.mongo_repo.find_by_query(
                query,
                sort_field='timestamp',
                sort_direction=-1,
                limit=1
            )
            
            if records and len(records) > 0:
                if isinstance(records[0], dict):
                    return PnLModel.from_dict(records[0])
                return records[0]
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest PnL snapshot: {str(e)}")
            raise
    
    def get_eod_snapshot(self, date: datetime) -> Optional[PnLModel]:
        """
        Get the end-of-day PnL snapshot for a specific date.
        
        Args:
            date: Date to get EOD snapshot for
            
        Returns:
            EOD PnLModel if available
        """
        try:
            # Normalize date to start of day
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)
            
            # Query MongoDB
            records = self.mongo_repo.find_by_query(
                {
                    'record_type': 'eod',
                    'timestamp': {
                        '$gte': start_of_day.isoformat(),
                        '$lte': end_of_day.isoformat()
                    }
                },
                sort_field='timestamp',
                sort_direction=-1,
                limit=1
            )
            
            if records and len(records) > 0:
                if isinstance(records[0], dict):
                    return PnLModel.from_dict(records[0])
                return records[0]
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get EOD PnL snapshot: {str(e)}")
            raise
    
    def get_snapshots_in_range(self, start_time: datetime, end_time: datetime, 
                               broker: Optional[str] = None, limit: int = 1000) -> List[PnLModel]:
        """
        Get PnL snapshots within a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            broker: Optional broker filter
            limit: Maximum records to return
            
        Returns:
            List of PnLModel
        """
        try:
            # Build query
            query = {
                'timestamp': {
                    '$gte': start_time.isoformat(),
                    '$lte': end_time.isoformat()
                }
            }
            
            if broker:
                query['broker'] = broker
                
            # Query MongoDB
            records = self.mongo_repo.find_by_query(
                query,
                sort_field='timestamp',
                sort_direction=1,
                limit=limit
            )
            
            # Convert to PnLModel if needed
            result = []
            for record in records:
                if isinstance(record, dict):
                    result.append(PnLModel.from_dict(record))
                else:
                    result.append(record)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get PnL snapshots in range: {str(e)}")
            raise
    
    def get_daily_snapshots(self, days: int = 30, broker: Optional[str] = None) -> List[PnLModel]:
        """
        Get daily PnL snapshots for the specified number of days.
        
        Args:
            days: Number of days to look back
            broker: Optional broker filter
            
        Returns:
            List of daily PnLModel
        """
        try:
            # Calculate date range
            end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
            start_date = (end_date - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Build query
            query = {
                'record_type': 'eod',
                'timestamp': {
                    '$gte': start_date.isoformat(),
                    '$lte': end_date.isoformat()
                }
            }
            
            if broker:
                query['broker'] = broker
                
            # Query MongoDB
            records = self.mongo_repo.find_by_query(
                query,
                sort_field='timestamp',
                sort_direction=1,
                limit=days
            )
            
            # Convert to PnLModel if needed
            result = []
            for record in records:
                if isinstance(record, dict):
                    result.append(PnLModel.from_dict(record))
                else:
                    result.append(record)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get daily PnL snapshots: {str(e)}")
            raise
    
    def calculate_returns(self, period_days: int = 30) -> Dict[str, float]:
        """
        Calculate returns statistics.
        
        Args:
            period_days: Number of days to analyze
            
        Returns:
            Dictionary of return statistics
        """
        try:
            # Get snapshots
            daily_snapshots = self.get_daily_snapshots(period_days)
            
            if not daily_snapshots:
                return {
                    'daily_return_avg': 0.0,
                    'total_return': 0.0,
                    'total_return_pct': 0.0,
                    'max_drawdown': 0.0,
                    'max_drawdown_pct': 0.0
                }
                
            # Calculate returns
            daily_returns_pct = []
            high_watermark = daily_snapshots[0].total_equity
            max_drawdown = 0.0
            max_drawdown_pct = 0.0
            
            for i in range(1, len(daily_snapshots)):
                prev_equity = daily_snapshots[i-1].total_equity
                curr_equity = daily_snapshots[i].total_equity
                
                # Update high watermark
                high_watermark = max(high_watermark, curr_equity)
                
                # Calculate daily return
                if prev_equity > 0:
                    daily_return_pct = ((curr_equity - prev_equity) / prev_equity) * 100.0
                    daily_returns_pct.append(daily_return_pct)
                
                # Calculate drawdown
                drawdown = max(0, high_watermark - curr_equity)
                drawdown_pct = (drawdown / high_watermark) * 100.0 if high_watermark > 0 else 0.0
                
                max_drawdown = max(max_drawdown, drawdown)
                max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
            
            # Calculate statistics
            daily_return_avg = sum(daily_returns_pct) / len(daily_returns_pct) if daily_returns_pct else 0.0
            
            initial_equity = daily_snapshots[0].total_equity
            final_equity = daily_snapshots[-1].total_equity
            total_return = final_equity - initial_equity
            total_return_pct = (total_return / initial_equity) * 100.0 if initial_equity > 0 else 0.0
            
            return {
                'daily_return_avg': daily_return_avg,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate returns: {str(e)}")
            raise
    
    def _get_high_watermark(self) -> Optional[float]:
        """
        Get the current equity high watermark.
        
        Returns:
            High watermark if available
        """
        try:
            # Query MongoDB for records with high watermark
            records = self.mongo_repo.find_by_query(
                {'equity_high_watermark': {'$exists': True}},
                sort_field='equity_high_watermark',
                sort_direction=-1,
                limit=1
            )
            
            if records and len(records) > 0:
                record = records[0]
                if isinstance(record, dict):
                    return record.get('equity_high_watermark')
                return record.equity_high_watermark
                
            # If no high watermark records, use highest total equity
            records = self.mongo_repo.find_by_query(
                {},
                sort_field='total_equity',
                sort_direction=-1,
                limit=1
            )
            
            if records and len(records) > 0:
                record = records[0]
                if isinstance(record, dict):
                    return record.get('total_equity')
                return record.total_equity
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get high watermark: {str(e)}")
            return None
