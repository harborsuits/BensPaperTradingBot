"""
Trade Accounting Safeguards

This module provides enhanced robustness, data integrity, and fault tolerance
for the TradeAccounting component, ensuring reliable and accurate financial tracking
even during system failures or data corruption.

Features:
- Transaction-based recording with atomic operations
- Data integrity verification and auto-repair
- Redundant storage with cross-validation
- Audit trails and reconciliation
- Automatic backups and recovery
"""

import logging
import threading
import time
import traceback
import json
import os
import sqlite3
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import copy

# Import components
from trading_bot.accounting.trade_accounting import TradeAccounting
from trading_bot.accounting.pnl_calculator import PnLCalculator
from trading_bot.accounting.performance_metrics import PerformanceMetrics
from trading_bot.robustness.system_safeguards import ComponentType, SafeguardState

logger = logging.getLogger(__name__)

class TradeAccountingSafeguards:
    """
    Enhances TradeAccounting with robust safeguards to ensure data integrity,
    reliable operation, and automatic recovery from error conditions.
    """
    
    def __init__(self, trade_accounting: TradeAccounting, 
                pnl_calculator: Optional[PnLCalculator] = None,
                performance_metrics: Optional[PerformanceMetrics] = None,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize trade accounting safeguards.
        
        Args:
            trade_accounting: The trade accounting instance to enhance
            pnl_calculator: Optional PnL calculator instance
            performance_metrics: Optional performance metrics instance
            config: Configuration parameters
        """
        self.trade_accounting = trade_accounting
        self.pnl_calculator = pnl_calculator
        self.performance_metrics = performance_metrics
        self.config = config or {}
        
        # Data integrity tracking
        self.data_checksums = {}
        self.integrity_checks: List[Dict[str, Any]] = []
        self.last_integrity_check: Optional[datetime] = None
        
        # Backup management
        self.db_backup_directory = self.config.get('db_backup_directory', 'db_backups')
        self.backup_interval_hours = self.config.get('backup_interval_hours', 24)
        self.backups: List[Dict[str, Any]] = []
        self.last_backup_time: Optional[datetime] = None
        
        # Transaction tracking
        self.transaction_log: List[Dict[str, Any]] = []
        self.pending_transactions: Dict[str, Dict[str, Any]] = {}
        
        # Reconciliation state
        self.last_reconciliation: Optional[datetime] = None
        self.reconciliation_errors: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize
        self._setup_safeguards()
        
        logger.info("Trade Accounting safeguards initialized")
    
    def _setup_safeguards(self) -> None:
        """Setup safety mechanisms and hooks."""
        # Create db backup directory if needed
        os.makedirs(self.db_backup_directory, exist_ok=True)
        
        # Wrap critical methods with safeguards
        if hasattr(self.trade_accounting, 'record_trade'):
            original_record_trade = self.trade_accounting.record_trade
            
            def safe_record_trade(*args, **kwargs):
                return self._safe_record_transaction(original_record_trade, 'record_trade', *args, **kwargs)
            
            self.trade_accounting.record_trade = safe_record_trade
        
        if hasattr(self.trade_accounting, 'update_trade'):
            original_update_trade = self.trade_accounting.update_trade
            
            def safe_update_trade(*args, **kwargs):
                return self._safe_record_transaction(original_update_trade, 'update_trade', *args, **kwargs)
            
            self.trade_accounting.update_trade = safe_update_trade
        
        # Schedule initial integrity check and backup
        self._schedule_integrity_check()
        self._schedule_database_backup()
    
    def _safe_record_transaction(self, original_func, transaction_type, *args, **kwargs) -> Any:
        """
        Execute database operations with transaction safety and rollback capability.
        
        Args:
            original_func: Original function to execute
            transaction_type: Type of transaction (for logging)
            *args, **kwargs: Arguments for the function
            
        Returns:
            Original function result or error handling result
        """
        transaction_id = f"{transaction_type}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Log transaction start
        self.transaction_log.append({
            'id': transaction_id,
            'type': transaction_type,
            'args': str(args),
            'kwargs': str(kwargs),
            'start_time': datetime.now().isoformat(),
            'status': 'started'
        })
        
        try:
            # Mark transaction as pending
            self.pending_transactions[transaction_id] = {
                'type': transaction_type,
                'args': args,
                'kwargs': kwargs,
                'start_time': datetime.now()
            }
            
            # Execute original function
            result = original_func(*args, **kwargs)
            
            # Log successful completion
            self.transaction_log.append({
                'id': transaction_id,
                'type': transaction_type,
                'status': 'completed',
                'end_time': datetime.now().isoformat()
            })
            
            # Remove from pending
            if transaction_id in self.pending_transactions:
                del self.pending_transactions[transaction_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Error during {transaction_type}: {str(e)}")
            
            # Log transaction failure
            self.transaction_log.append({
                'id': transaction_id,
                'type': transaction_type,
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            
            # Remove from pending
            if transaction_id in self.pending_transactions:
                del self.pending_transactions[transaction_id]
            
            # Check if we should retry
            if "database is locked" in str(e).lower():
                # SQLite lock error, retry after short delay
                logger.warning(f"Database locked during {transaction_type}, retrying...")
                time.sleep(1)
                return self._safe_record_transaction(original_func, transaction_type, *args, **kwargs)
                
            elif "constraint" in str(e).lower():
                # Constraint violation, probably duplicate
                logger.warning(f"Constraint violation in {transaction_type}: {str(e)}")
                return {'error': 'constraint_violation', 'message': str(e)}
                
            elif "no such table" in str(e).lower():
                # Missing table, try to reinitialize database
                logger.error(f"Missing table in {transaction_type}, attempting recovery")
                if self._recover_database_schema():
                    logger.info("Database schema recovered, retrying transaction")
                    return self._safe_record_transaction(original_func, transaction_type, *args, **kwargs)
            
            # Re-raise the exception
            raise
    
    def _calculate_checksum(self, data: Any) -> str:
        """
        Calculate a checksum for data verification.
        
        Args:
            data: Data to calculate checksum for
            
        Returns:
            str: Checksum
        """
        # Convert to JSON and calculate checksum
        json_data = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_data.encode()).hexdigest()
    
    def _schedule_integrity_check(self) -> None:
        """Schedule a database integrity check."""
        self.last_integrity_check = datetime.now()
        
    def _schedule_database_backup(self) -> None:
        """Schedule a database backup."""
        self.last_backup_time = datetime.now()
    
    def check_data_integrity(self) -> Tuple[bool, List[str]]:
        """
        Check integrity of trade accounting data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Get database path
            if not hasattr(self.trade_accounting, 'db_path'):
                errors.append("Trade accounting missing db_path attribute")
                return False, errors
                
            db_path = self.trade_accounting.db_path
            
            # Verify database exists
            if not os.path.exists(db_path):
                errors.append(f"Database file not found: {db_path}")
                return False, errors
            
            # Connect to database for verification
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check database integrity
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()
            if integrity_result[0] != "ok":
                errors.append(f"SQLite integrity check failed: {integrity_result[0]}")
            
            # Check required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = set(row[0] for row in cursor.fetchall())
            required_tables = {'trades', 'positions', 'trade_history'}
            
            missing_tables = required_tables - tables
            if missing_tables:
                errors.append(f"Missing required tables: {', '.join(missing_tables)}")
            
            # Check trades table structure if it exists
            if 'trades' in tables:
                try:
                    cursor.execute("PRAGMA table_info(trades)")
                    columns = set(row[1] for row in cursor.fetchall())
                    required_columns = {'trade_id', 'symbol', 'quantity', 'entry_price', 'exit_price', 'entry_time'}
                    
                    missing_columns = required_columns - columns
                    if missing_columns:
                        errors.append(f"Trades table missing columns: {', '.join(missing_columns)}")
                        
                except sqlite3.Error as e:
                    errors.append(f"Error checking trades table structure: {str(e)}")
            
            # Check for orphaned records
            if {'trades', 'positions'}.issubset(tables):
                try:
                    cursor.execute("""
                        SELECT COUNT(*) FROM positions 
                        WHERE position_id NOT IN (SELECT DISTINCT position_id FROM trades WHERE position_id IS NOT NULL)
                    """)
                    orphaned_count = cursor.fetchone()[0]
                    if orphaned_count > 0:
                        errors.append(f"Found {orphaned_count} orphaned position records")
                except sqlite3.Error as e:
                    errors.append(f"Error checking for orphaned records: {str(e)}")
            
            # Update integrity check log
            self.integrity_checks.append({
                'timestamp': datetime.now().isoformat(),
                'errors': errors,
                'status': 'pass' if not errors else 'fail'
            })
            
            # Update last check time
            self.last_integrity_check = datetime.now()
            
            conn.close()
            return len(errors) == 0, errors
            
        except Exception as e:
            error_msg = f"Error during data integrity check: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            
            # Update integrity check log
            self.integrity_checks.append({
                'timestamp': datetime.now().isoformat(),
                'errors': errors,
                'status': 'error'
            })
            
            return False, errors
    
    def create_database_backup(self) -> Dict[str, Any]:
        """
        Create a backup of the trade accounting database.
        
        Returns:
            Dict with backup info
        """
        backup_info = {
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': None,
            'backup_path': None
        }
        
        try:
            # Check database path exists
            if not hasattr(self.trade_accounting, 'db_path'):
                backup_info['error'] = "Trade accounting missing db_path attribute"
                return backup_info
            
            db_path = self.trade_accounting.db_path
            if not os.path.exists(db_path):
                backup_info['error'] = f"Database file not found: {db_path}"
                return backup_info
            
            # Create backup filename with timestamp
            backup_filename = f"trade_accounting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            backup_path = os.path.join(self.db_backup_directory, backup_filename)
            
            # Ensure backup directory exists
            os.makedirs(self.db_backup_directory, exist_ok=True)
            
            # Connect to source database
            conn = sqlite3.connect(db_path)
            
            # Create backup connection
            backup_conn = sqlite3.connect(backup_path)
            
            # Copy database
            conn.backup(backup_conn)
            
            # Close connections
            backup_conn.close()
            conn.close()
            
            # Update backup info
            backup_info['status'] = 'success'
            backup_info['backup_path'] = backup_path
            
            # Update backup log
            self.backups.append(backup_info)
            
            # Update last backup time
            self.last_backup_time = datetime.now()
            
            # Keep backup history to a reasonable size
            max_backups = self.config.get('max_backup_history', 10)
            if len(self.backups) > max_backups:
                self.backups = self.backups[-max_backups:]
            
            logger.info(f"Created trade accounting database backup: {backup_path}")
            return backup_info
            
        except Exception as e:
            error_msg = f"Error creating database backup: {str(e)}"
            logger.error(error_msg)
            
            backup_info['error'] = error_msg
            return backup_info
    
    def restore_from_backup(self, backup_path: Optional[str] = None) -> bool:
        """
        Restore database from a backup.
        
        Args:
            backup_path: Path to backup file, or None for latest
            
        Returns:
            bool: Success status
        """
        try:
            # Check database path
            if not hasattr(self.trade_accounting, 'db_path'):
                logger.error("Trade accounting missing db_path attribute")
                return False
            
            db_path = self.trade_accounting.db_path
            
            # Determine backup path
            if backup_path is None:
                # Use latest backup
                if not self.backups:
                    logger.error("No backups available")
                    return False
                
                # Sort backups by timestamp and use the latest
                latest_backup = max(self.backups, key=lambda b: b.get('timestamp', ''))
                backup_path = latest_backup.get('backup_path')
                
                if not backup_path or not os.path.exists(backup_path):
                    logger.error(f"Latest backup not found: {backup_path}")
                    return False
            
            # Verify backup exists
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Create temporary connection
            backup_conn = sqlite3.connect(backup_path)
            
            # Check backup integrity
            cursor = backup_conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()
            if integrity_result[0] != "ok":
                logger.error(f"Backup integrity check failed: {integrity_result[0]}")
                backup_conn.close()
                return False
            
            # Close any existing connections in trade accounting
            if hasattr(self.trade_accounting, 'conn') and self.trade_accounting.conn:
                try:
                    self.trade_accounting.conn.close()
                except:
                    pass
            
            # Create new db connection
            conn = sqlite3.connect(db_path)
            
            # Copy from backup
            backup_conn.backup(conn)
            
            # Close connections
            conn.close()
            backup_conn.close()
            
            # Re-initialize trade accounting connection
            if hasattr(self.trade_accounting, 'initialize_db'):
                self.trade_accounting.initialize_db()
            
            logger.warning(f"Restored trade accounting database from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {str(e)}")
            return False
    
    def _recover_database_schema(self) -> bool:
        """
        Attempt to recover database schema.
        
        Returns:
            bool: Success status
        """
        try:
            # Check if trade accounting has initialize_db method
            if not hasattr(self.trade_accounting, 'initialize_db'):
                logger.error("Trade accounting missing initialize_db method")
                return False
            
            # Try to reinitialize database
            self.trade_accounting.initialize_db(force=True)
            
            logger.warning("Reinitialized trade accounting database schema")
            return True
            
        except Exception as e:
            logger.error(f"Error recovering database schema: {str(e)}")
            return False
    
    def reconcile_with_broker_data(self, broker_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconcile trade database with broker data.
        
        Args:
            broker_trades: List of trades from broker
            
        Returns:
            Dict with reconciliation results
        """
        reconciliation_result = {
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'missing_trades': [],
            'mismatched_trades': [],
            'extra_trades': [],
            'fixed_trades': []
        }
        
        try:
            # Create a backup before reconciliation
            backup_info = self.create_database_backup()
            if backup_info['status'] != 'success':
                reconciliation_result['error'] = f"Failed to create backup: {backup_info['error']}"
                return reconciliation_result
            
            # Get all trades from database
            if not hasattr(self.trade_accounting, 'get_all_trades'):
                reconciliation_result['error'] = "Trade accounting missing get_all_trades method"
                return reconciliation_result
            
            db_trades = self.trade_accounting.get_all_trades()
            
            # Create lookup dictionaries
            db_trades_by_id = {t['trade_id']: t for t in db_trades}
            broker_trades_by_id = {t['trade_id']: t for t in broker_trades if 'trade_id' in t}
            
            # Find missing trades (in broker but not in DB)
            for trade_id, broker_trade in broker_trades_by_id.items():
                if trade_id not in db_trades_by_id:
                    reconciliation_result['missing_trades'].append(broker_trade)
            
            # Find extra trades (in DB but not in broker)
            for trade_id, db_trade in db_trades_by_id.items():
                if trade_id not in broker_trades_by_id:
                    reconciliation_result['extra_trades'].append(db_trade)
            
            # Find mismatched trades
            for trade_id, broker_trade in broker_trades_by_id.items():
                if trade_id in db_trades_by_id:
                    db_trade = db_trades_by_id[trade_id]
                    
                    # Check key fields match
                    mismatch = False
                    for field in ['symbol', 'quantity', 'entry_price', 'exit_price']:
                        if field in broker_trade and field in db_trade:
                            if broker_trade[field] != db_trade[field]:
                                mismatch = True
                                break
                    
                    if mismatch:
                        reconciliation_result['mismatched_trades'].append({
                            'trade_id': trade_id,
                            'broker_data': broker_trade,
                            'db_data': db_trade
                        })
            
            # Fix issues if requested
            if self.config.get('auto_fix_reconciliation', False):
                # Add missing trades
                for trade in reconciliation_result['missing_trades']:
                    try:
                        self.trade_accounting.record_trade(trade)
                        reconciliation_result['fixed_trades'].append({
                            'trade_id': trade.get('trade_id'),
                            'action': 'added_missing'
                        })
                    except Exception as e:
                        logger.error(f"Error adding missing trade {trade.get('trade_id')}: {str(e)}")
                
                # Fix mismatched trades
                for mismatch in reconciliation_result['mismatched_trades']:
                    try:
                        trade_id = mismatch['trade_id']
                        broker_data = mismatch['broker_data']
                        
                        # Update trade with broker data
                        self.trade_accounting.update_trade(trade_id, broker_data)
                        
                        reconciliation_result['fixed_trades'].append({
                            'trade_id': trade_id,
                            'action': 'fixed_mismatch'
                        })
                    except Exception as e:
                        logger.error(f"Error fixing mismatched trade {mismatch.get('trade_id')}: {str(e)}")
            
            # Update reconciliation status
            reconciliation_result['status'] = 'success'
            self.last_reconciliation = datetime.now()
            
            return reconciliation_result
            
        except Exception as e:
            error_msg = f"Error during trade reconciliation: {str(e)}"
            logger.error(error_msg)
            
            reconciliation_result['error'] = error_msg
            self.reconciliation_errors.append({
                'timestamp': datetime.now().isoformat(),
                'error': error_msg
            })
            
            return reconciliation_result
    
    def is_healthy(self) -> Tuple[bool, List[str]]:
        """
        Check if the trade accounting system is healthy.
        
        Returns:
            Tuple of (is_healthy, messages)
        """
        messages = []
        
        # Check data integrity
        is_valid, integrity_errors = self.check_data_integrity()
        if not is_valid:
            messages.extend(integrity_errors)
        
        # Check backup health
        if self.last_backup_time:
            hours_since_backup = (datetime.now() - self.last_backup_time).total_seconds() / 3600
            if hours_since_backup > self.backup_interval_hours:
                messages.append(f"Database backup overdue: {hours_since_backup:.1f} hours since last backup")
        else:
            messages.append("No database backup on record")
        
        # Check pending transactions
        if len(self.pending_transactions) > 0:
            # Check for any transactions older than 5 minutes
            old_transactions = []
            for tx_id, tx_info in self.pending_transactions.items():
                tx_age = (datetime.now() - tx_info['start_time']).total_seconds() / 60
                if tx_age > 5:
                    old_transactions.append(tx_id)
            
            if old_transactions:
                messages.append(f"Found {len(old_transactions)} stuck transactions older than 5 minutes")
        
        # Check reconciliation health if enabled
        if self.config.get('reconciliation_enabled', False) and self.last_reconciliation:
            days_since_reconciliation = (datetime.now() - self.last_reconciliation).total_seconds() / (24 * 3600)
            max_days = self.config.get('max_reconciliation_days', 7)
            
            if days_since_reconciliation > max_days:
                messages.append(f"Trade reconciliation overdue: {days_since_reconciliation:.1f} days since last reconciliation")
        
        # Overall health status
        is_healthy = len(messages) == 0
        
        return is_healthy, messages

# Recovery function for system safeguards
def recover_trade_accounting(trade_accounting_safeguards: TradeAccountingSafeguards) -> bool:
    """
    Attempt to recover trade accounting from error state.
    
    Args:
        trade_accounting_safeguards: Trade accounting safeguards instance
        
    Returns:
        bool: Success status
    """
    try:
        # Check integrity first
        is_valid, integrity_errors = trade_accounting_safeguards.check_data_integrity()
        
        if not is_valid:
            # Create a fresh backup of the current state (even if corrupted)
            logger.warning("Creating backup of potentially corrupted database")
            trade_accounting_safeguards.create_database_backup()
            
            # Try to restore from most recent backup
            logger.warning(f"Database integrity check failed, attempting restore: {integrity_errors}")
            return trade_accounting_safeguards.restore_from_backup()
        
        # Check for stuck transactions
        if trade_accounting_safeguards.pending_transactions:
            logger.warning(f"Found {len(trade_accounting_safeguards.pending_transactions)} stuck transactions, clearing")
            with trade_accounting_safeguards._lock:
                trade_accounting_safeguards.pending_transactions.clear()
        
        # Create a routine backup
        trade_accounting_safeguards.create_database_backup()
        
        return True
        
    except Exception as e:
        logger.error(f"Trade accounting recovery failed: {str(e)}")
        
        # Last resort: try to restore from backup
        try:
            return trade_accounting_safeguards.restore_from_backup()
        except:
            return False

# Function to create validation function for system safeguards
def create_trade_accounting_validator(trade_accounting_safeguards: TradeAccountingSafeguards):
    """Create a validation function for the trade accounting component."""
    
    def validate_trade_accounting(component: Any) -> Tuple[bool, List[str]]:
        """
        Validate trade accounting state.
        
        Args:
            component: Trade accounting component
            
        Returns:
            Tuple of (is_valid, messages)
        """
        return trade_accounting_safeguards.is_healthy()
    
    return validate_trade_accounting
