"""
Configuration Hot-Reload System

Watches configuration files for changes and reloads them automatically.
Uses watchdog to efficiently monitor file system events.
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Set, List
import json

# Optional watchdog import - will degrade gracefully if not available
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = object  # type: ignore
    FileSystemEventHandler = object  # type: ignore

from trading_bot.core.simple_config import load_config, ConfigError

# Configure logging
logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """Handles file system events for configuration files"""
    
    def __init__(self, callback: Callable[[str], None], watched_files: Set[str]):
        """
        Initialize the handler
        
        Args:
            callback: Function to call when a config file changes
            watched_files: Set of absolute paths to watch
        """
        self.callback = callback
        self.watched_files = watched_files
        self.last_modified_times: Dict[str, float] = {}
        
        # Initialize last modified times
        for path in watched_files:
            try:
                self.last_modified_times[path] = os.path.getmtime(path)
            except (FileNotFoundError, PermissionError):
                logger.warning(f"Could not get modification time for {path}")
    
    def on_modified(self, event):
        """Called when a file is modified"""
        if not hasattr(event, 'src_path'):
            return
            
        path = os.path.abspath(event.src_path)
        if path not in self.watched_files:
            return
            
        # Check if the file was actually modified (some editors trigger multiple events)
        try:
            current_mtime = os.path.getmtime(path)
            last_mtime = self.last_modified_times.get(path, 0)
            
            if current_mtime > last_mtime:
                self.last_modified_times[path] = current_mtime
                logger.debug(f"Configuration file modified: {path}")
                self.callback(path)
        except (FileNotFoundError, PermissionError):
            logger.warning(f"Could not check modification time for {path}")


class ConfigPoller:
    """Polls configuration files for changes when watchdog is not available"""
    
    def __init__(self, callback: Callable[[str], None], watched_files: Set[str], interval_seconds: int = 30):
        """
        Initialize the poller
        
        Args:
            callback: Function to call when a config file changes
            watched_files: Set of absolute paths to watch
            interval_seconds: How often to check for changes
        """
        self.callback = callback
        self.watched_files = watched_files
        self.interval_seconds = interval_seconds
        self.last_modified_times: Dict[str, float] = {}
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Initialize last modified times
        for path in watched_files:
            try:
                self.last_modified_times[path] = os.path.getmtime(path)
            except (FileNotFoundError, PermissionError):
                logger.warning(f"Could not get modification time for {path}")
    
    def start(self):
        """Start polling for changes"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started polling for config changes every {self.interval_seconds} seconds")
    
    def stop(self):
        """Stop polling for changes"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def _poll_loop(self):
        """Poll for changes in a loop"""
        while self.running:
            for path in self.watched_files:
                try:
                    current_mtime = os.path.getmtime(path)
                    last_mtime = self.last_modified_times.get(path, 0)
                    
                    if current_mtime > last_mtime:
                        self.last_modified_times[path] = current_mtime
                        logger.debug(f"Configuration file modified: {path}")
                        self.callback(path)
                except (FileNotFoundError, PermissionError):
                    continue
            
            # Sleep for the interval
            time.sleep(self.interval_seconds)


class ConfigWatcher:
    """
    Watches configuration files for changes and reloads them automatically
    """
    
    def __init__(self, 
                 main_config_path: str, 
                 reload_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 interval_seconds: int = 60):
        """
        Initialize the configuration watcher
        
        Args:
            main_config_path: Path to the main configuration file
            reload_callback: Function to call with new config when reloaded
            interval_seconds: How often to check for changes if watchdog is not available
        """
        self.main_config_path = os.path.abspath(main_config_path)
        self.reload_callback = reload_callback
        self.interval_seconds = interval_seconds
        self.config: Dict[str, Any] = {}
        self.watched_files: Set[str] = {self.main_config_path}
        self.watcher: Optional[Observer] = None
        self.poller: Optional[ConfigPoller] = None
        self.handler: Optional[ConfigFileHandler] = None
        self.running = False
        
        # Load initial configuration
        try:
            self.config = load_config(self.main_config_path)
            logger.info(f"Loaded initial configuration from {self.main_config_path}")
            
            # Find referenced config files
            self._find_referenced_config_files()
        except ConfigError as e:
            logger.error(f"Error loading initial configuration: {str(e)}")
            raise
    
    def _find_referenced_config_files(self):
        """Find additional config files referenced in the main config"""
        referenced_files = []
        
        # List of common config file references
        ref_keys = [
            "market_regime_config_path",
            "broker_config_path",
            "market_data_config_path",
            "strategy_config_path",
            "notification_config_path",
            "logging_config_path"
        ]
        
        for key in ref_keys:
            if key in self.config and isinstance(self.config[key], str):
                path = self.config[key]
                if not os.path.isabs(path):
                    # Make relative paths absolute based on main config location
                    base_dir = os.path.dirname(self.main_config_path)
                    path = os.path.abspath(os.path.join(base_dir, path))
                
                if os.path.exists(path):
                    referenced_files.append(path)
                    logger.debug(f"Found referenced config file: {path}")
        
        # Add all found files to watched files
        for path in referenced_files:
            self.watched_files.add(path)
    
    def _handle_config_change(self, file_path: str):
        """
        Handle a configuration file change
        
        Args:
            file_path: Path to the changed file
        """
        logger.info(f"Detected change in configuration file: {file_path}")
        
        try:
            # If main config changed, reload it and update references
            if file_path == self.main_config_path:
                new_config = load_config(self.main_config_path)
                
                # Keep track of old referenced files
                old_files = self.watched_files.copy()
                
                # Update the config
                self.config = new_config
                
                # Find new referenced files
                old_watched = self.watched_files.copy()
                self._find_referenced_config_files()
                
                # Update file watchers if using watchdog
                if WATCHDOG_AVAILABLE and self.watcher and self.handler:
                    new_files = self.watched_files - old_watched
                    for new_file in new_files:
                        dir_path = os.path.dirname(new_file)
                        if os.path.exists(dir_path):
                            # Start watching the directory if not already watched
                            self.watcher.schedule(self.handler, dir_path, recursive=False)
                            logger.debug(f"Started watching directory for new config: {dir_path}")
            
            # If a referenced file changed, reload it
            else:
                # Simply note the change - the callback will handle any necessary reloading
                logger.info(f"Referenced configuration file changed: {file_path}")
            
            # Call the reload callback if provided
            if self.reload_callback:
                self.reload_callback(self.config)
            
            logger.info("Configuration reload complete")
            
        except ConfigError as e:
            logger.error(f"Error reloading configuration: {str(e)}")
            logger.warning("Continuing with previous configuration")
        except Exception as e:
            logger.error(f"Unexpected error reloading configuration: {str(e)}")
            logger.warning("Continuing with previous configuration")
    
    def start(self):
        """Start watching for configuration changes"""
        if self.running:
            return
        
        self.running = True
        
        if WATCHDOG_AVAILABLE:
            try:
                # Create observer and handler
                self.watcher = Observer()
                self.handler = ConfigFileHandler(
                    callback=self._handle_config_change,
                    watched_files=self.watched_files
                )
                
                # Watch directories containing config files
                watched_dirs = set()
                for file_path in self.watched_files:
                    dir_path = os.path.dirname(file_path)
                    if dir_path not in watched_dirs and os.path.exists(dir_path):
                        self.watcher.schedule(self.handler, dir_path, recursive=False)
                        watched_dirs.add(dir_path)
                
                # Start the observer
                self.watcher.start()
                logger.info(f"Started watching {len(self.watched_files)} configuration files for changes")
                return
            except Exception as e:
                logger.error(f"Error setting up watchdog: {str(e)}")
                logger.warning("Falling back to polling for configuration changes")
        
        # Fall back to polling if watchdog is not available or failed
        self.poller = ConfigPoller(
            callback=self._handle_config_change,
            watched_files=self.watched_files,
            interval_seconds=self.interval_seconds
        )
        self.poller.start()
    
    def stop(self):
        """Stop watching for configuration changes"""
        if not self.running:
            return
        
        self.running = False
        
        if self.watcher:
            self.watcher.stop()
            self.watcher.join(timeout=1.0)
            self.watcher = None
        
        if self.poller:
            self.poller.stop()
            self.poller = None
        
        logger.info("Stopped watching for configuration changes")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration"""
        return self.config


# Global configuration watcher instance
_config_watcher: Optional[ConfigWatcher] = None


def init_config_watcher(
    config_path: str,
    reload_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    interval_seconds: int = 60
) -> ConfigWatcher:
    """
    Initialize the global configuration watcher
    
    Args:
        config_path: Path to the main configuration file
        reload_callback: Function to call with new config when reloaded
        interval_seconds: How often to check for changes if watchdog is not available
        
    Returns:
        ConfigWatcher instance
    """
    global _config_watcher
    
    if _config_watcher:
        _config_watcher.stop()
    
    _config_watcher = ConfigWatcher(
        main_config_path=config_path,
        reload_callback=reload_callback,
        interval_seconds=interval_seconds
    )
    
    return _config_watcher


def get_config_watcher() -> Optional[ConfigWatcher]:
    """
    Get the global configuration watcher instance
    
    Returns:
        ConfigWatcher instance or None if not initialized
    """
    return _config_watcher
