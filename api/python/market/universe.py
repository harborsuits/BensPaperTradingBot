#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universe Module

This module provides the Universe class for defining tradable assets and filtering them.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Callable, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class Universe:
    """
    A comprehensive framework for defining, managing, and filtering tradable asset collections.
    
    The Universe class provides a structured approach to asset selection and management,
    serving as a core component in the trading system's architecture. It enables the creation
    of precisely defined investment universes based on multiple criteria, with dynamic 
    filtering capabilities to adapt to changing market conditions.
    
    Key capabilities:
    1. Defines collections of tradable symbols with rich metadata
    2. Provides robust symbol management (addition, removal, batch operations)
    3. Implements a sophisticated filtering system with composable filter functions
    4. Supports both static and dynamic universe definitions
    5. Enables hierarchical universe construction (sectors, industries, strategies)
    6. Facilitates universe persistence and serialization
    
    The Universe class serves multiple critical functions in the trading system:
    - Constrains strategy execution to appropriate asset sets
    - Enables efficient market data retrieval for only relevant assets
    - Provides the foundation for specialized strategy implementations
    - Supports systematic backtesting across defined asset collections
    - Facilitates risk management through controlled asset exposure
    
    Implementation considerations:
    - Universe definitions can be static (fixed set) or dynamic (criteria-based)
    - Filtering can incorporate fundamental, technical, and market structure data
    - Filter composition allows for increasingly refined asset selection
    - Efficient set operations optimize performance for large universes
    - Metadata storage enables rich context for universe members
    
    Typical usage patterns:
    - Creating sector/industry/theme-based universes
    - Defining liquidity-filtered investment universes
    - Building strategy-specific candidate pools
    - Constructing test and validation datasets
    - Managing multiple market segment exposures
    """
    
    def __init__(self, 
                name: str = "", 
                symbols: Optional[List[str]] = None,
                description: str = "",
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a Universe instance with configurable properties.
        
        Creates a new Universe instance with the specified name, symbols, and metadata.
        The universe starts with the provided symbols and empty filter list, ready
        for further refinement through filtering operations.
        
        Parameters:
            name (str): A descriptive identifier for the universe, useful for tracking
                multiple universes in a system. Examples: "SP500", "Tech_Stocks", 
                "Liquid_Small_Caps"
            symbols (Optional[List[str]]): Initial list of ticker symbols to include
                in the universe. If None, an empty universe is created.
            description (str): Detailed description of the universe's purpose and
                composition for documentation and reference.
            metadata (Optional[Dict[str, Any]]): Additional contextual information
                about the universe, such as creation date, source, criteria used,
                or any custom attributes for application-specific uses.
        
        Notes:
            - Symbol lists are converted to sets internally for efficient membership testing
            - Universe size is only constrained by system memory
            - Empty universes are valid and can be populated later
            - Metadata can be used to store creation timestamp, data sources, or custom attributes
        """
        self.name = name
        self.symbols = set(symbols or [])
        self.description = description
        self.metadata = metadata or {}
        self.filters = []
        self.symbol_metadata = {}  # Store metadata for individual symbols
        
        logger.info(f"Created universe '{name}' with {len(self.symbols)} symbols")
    
    def add_symbol(self, symbol: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a single symbol to the universe with optional metadata.
        
        This method adds the specified symbol to the universe if it's not already present,
        and associates optional metadata with the symbol for future reference.
        
        Parameters:
            symbol (str): The ticker symbol to add to the universe. Should follow
                the convention used by the connected data sources (e.g., "AAPL", "BTC-USD").
            metadata (Optional[Dict[str, Any]]): Symbol-specific metadata to store,
                such as sector, industry, market cap category, or custom attributes.
                
        Notes:
            - If the symbol already exists in the universe, only the metadata is updated
            - Symbol addition is idempotent - adding the same symbol multiple times has no effect
            - Symbol case sensitivity follows the convention used by data providers
            - No validation is performed on symbol format or existence in external data sources
        """
        if symbol not in self.symbols:
            self.symbols.add(symbol)
            logger.debug(f"Added symbol {symbol} to universe '{self.name}'")
        
        # Add or update symbol metadata if provided
        if metadata:
            if symbol not in self.symbol_metadata:
                self.symbol_metadata[symbol] = {}
            self.symbol_metadata[symbol].update(metadata)
    
    def add_symbols(self, symbols: List[str], metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Add multiple symbols to the universe efficiently.
        
        This method provides a batch operation to add multiple symbols at once,
        optimizing performance for large additions. Optionally, symbol-specific
        metadata can be provided as a dictionary.
        
        Parameters:
            symbols (List[str]): List of ticker symbols to add to the universe
            metadata (Optional[Dict[str, Dict[str, Any]]]): Dictionary mapping symbols
                to their respective metadata dictionaries. Only symbols included in
                this dictionary will have metadata added.
                
        Notes:
            - Significantly more efficient than calling add_symbol() in a loop for large lists
            - Only new symbols (not already in the universe) are logged as added
            - Symbol-specific metadata is updated for all provided symbols
            - Existing symbols will have their metadata updated if included in the metadata dict
            - Does not raise exceptions for duplicate symbols
        """
        new_symbols = [s for s in symbols if s not in self.symbols]
        if new_symbols:
            self.symbols.update(new_symbols)
            logger.debug(f"Added {len(new_symbols)} symbols to universe '{self.name}'")
        
        # Add metadata for symbols if provided
        if metadata:
            for symbol, symbol_meta in metadata.items():
                if symbol in self.symbols:
                    if symbol not in self.symbol_metadata:
                        self.symbol_metadata[symbol] = {}
                    self.symbol_metadata[symbol].update(symbol_meta)
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a single symbol from the universe.
        
        This method removes the specified symbol from the universe if it exists,
        including any associated symbol metadata.
        
        Parameters:
            symbol (str): The ticker symbol to remove from the universe
            
        Returns:
            bool: True if the symbol was found and removed, False if it wasn't in the universe
            
        Notes:
            - Returns False if the symbol doesn't exist in the universe (no exception raised)
            - Also removes any associated symbol metadata
            - Does not affect universe metadata or filters
            - Removal is idempotent - removing a non-existent symbol has no effect
        """
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            # Also remove symbol metadata if present
            if symbol in self.symbol_metadata:
                del self.symbol_metadata[symbol]
            logger.debug(f"Removed symbol {symbol} from universe '{self.name}'")
            return True
        return False
    
    def clear(self) -> None:
        """
        Remove all symbols from the universe while preserving filters and metadata.
        
        This method completely empties the universe of all symbols while maintaining
        the universe's structure, name, description, metadata, and defined filters.
        
        Notes:
            - Resets the universe to an empty state without deleting the instance
            - Preserves all filters, which will be applied to any newly added symbols
            - Maintains universe metadata but clears all symbol-specific metadata
            - Useful for rebuilding a universe from scratch while keeping its definition
            - Logs the number of symbols removed for auditing purposes
        """
        count = len(self.symbols)
        self.symbols.clear()
        self.symbol_metadata.clear()
        logger.debug(f"Cleared {count} symbols from universe '{self.name}'")
    
    def add_filter(self, filter_func: Callable[[str, Dict[str, Any]], bool], 
                  name: Optional[str] = None,
                  description: Optional[str] = None) -> None:
        """
        Add a filtering function to the universe for dynamic symbol selection.
        
        This method registers a callable filter function that will be used to
        dynamically filter the universe when apply_filters() is called. Filters
        are applied in the order they're added, allowing for progressive refinement.
        
        Parameters:
            filter_func (Callable[[str, Dict[str, Any]], bool]): A function that takes
                a symbol and a data dictionary and returns True if the symbol should
                be included in the filtered universe, False otherwise.
            name (Optional[str]): A descriptive name for the filter function, used
                in logging and debugging. If None, a default name is generated.
            description (Optional[str]): Detailed description of the filter's purpose
                and criteria for documentation.
                
        Filter Function Interface:
            The filter_func must accept two parameters:
            - symbol (str): The symbol being evaluated
            - data (Dict[str, Any]): A dictionary containing data for the symbol
            
            And must return:
            - bool: True if the symbol passes the filter, False otherwise
            
        Examples of filter functions:
            - Market cap filters: lambda s, data: data.get('market_cap', 0) > 1e9
            - Price filters: lambda s, data: data.get('close', 0) > 10.0
            - Volume filters: lambda s, data: data.get('volume', 0) > 1e6
            - Volatility filters: lambda s, data: data.get('atr', 0) / data.get('close', 1) < 0.05
            
        Notes:
            - Filters are stored and applied in the order they're added
            - Each filter can progressively narrow down the universe
            - Filters are only applied when apply_filters() is explicitly called
            - Filters don't modify the base universe until apply_filters() is called
            - Filters can incorporate any data provided in the data dictionary
        """
        filter_name = name or f"filter_{len(self.filters) + 1}"
        filter_desc = description or "No description provided"
        self.filters.append((filter_name, filter_func, filter_desc))
        logger.debug(f"Added filter '{filter_name}' to universe '{self.name}'")
    
    def apply_filters(self, data: Dict[str, Dict[str, Any]], inplace: bool = False) -> Set[str]:
        """
        Apply all registered filters to the universe symbols.
        
        This method runs all registered filter functions against the provided data,
        progressively narrowing down the set of symbols to those that pass all filters.
        The filtering process can either create a new filtered set or modify the
        universe in place.
        
        Parameters:
            data (Dict[str, Dict[str, Any]]): A dictionary mapping symbols to their
                respective data dictionaries. Each symbol's data dictionary will be
                passed to the filter functions.
            inplace (bool): If True, the universe is modified in place by removing
                symbols that don't pass the filters. If False, a new set of filtered
                symbols is returned without modifying the universe.
            
        Returns:
            Set[str]: A set of symbols that passed all filters. If inplace=True, this
                will be the new contents of self.symbols.
        
        Filter Application Process:
        1. Starts with a copy of all symbols in the universe
        2. For each filter function:
           a. Applies the function to each symbol with its data
           b. Keeps only symbols that return True
           c. Logs the number of symbols removed by the filter
        3. If inplace=True, updates the universe symbols to the filtered set
        4. Returns the final set of filtered symbols
        
        Notes:
            - Missing data for a symbol automatically excludes it from results
            - Filter application order can affect results if filters are interdependent
            - Performance depends on the complexity of filter functions and data size
            - Detailed logging provides insight into each filter's impact
            - Symbol metadata is preserved for remaining symbols when inplace=True
        """
        # Start with all symbols
        filtered_symbols = self.symbols.copy()
        
        # Apply each filter
        for filter_name, filter_func, _ in self.filters:
            # Filter symbols
            before_count = len(filtered_symbols)
            filtered_symbols = {
                symbol for symbol in filtered_symbols
                if symbol in data and filter_func(symbol, data[symbol])
            }
            after_count = len(filtered_symbols)
            
            logger.debug(f"Filter '{filter_name}' removed {before_count - after_count} symbols")
        
        logger.info(f"Applied {len(self.filters)} filters: {len(filtered_symbols)} symbols remain")
        
        # Update universe if inplace is True
        if inplace and filtered_symbols != self.symbols:
            removed_symbols = self.symbols - filtered_symbols
            self.symbols = filtered_symbols
            
            # Remove metadata for symbols no longer in the universe
            for symbol in removed_symbols:
                if symbol in self.symbol_metadata:
                    del self.symbol_metadata[symbol]
            
            logger.info(f"Updated universe '{self.name}' in place with {len(self.symbols)} symbols")
        
        return filtered_symbols
    
    def get_symbols(self) -> List[str]:
        """
        Get a list of all symbols in the universe.
        
        Returns:
            List[str]: A list of all ticker symbols currently in the universe
            
        Notes:
            - Returns a new list, not a reference to the internal set
            - Order is not guaranteed due to set implementation
            - For sorted results, use: sorted(universe.get_symbols())
        """
        return list(self.symbols)
    
    def add_metadata(self, symbol: str, key: str, value: Any) -> None:
        """
        Add or update metadata for a specific symbol in the universe.
        
        Parameters:
            symbol (str): The symbol to add metadata for
            key (str): The metadata attribute name
            value (Any): The value to store for the attribute
            
        Notes:
            - If the symbol doesn't exist in the universe, a warning is logged
            - Creates the metadata dictionary for the symbol if it doesn't exist
            - Updates existing metadata if the key already exists
        """
        if symbol not in self.symbols:
            logger.warning(f"Adding metadata for symbol {symbol} not in universe '{self.name}'")
            
        if symbol not in self.symbol_metadata:
            self.symbol_metadata[symbol] = {}
            
        self.symbol_metadata[symbol][key] = value
    
    def get_metadata(self, symbol: str, key: str, default: Any = None) -> Any:
        """
        Retrieve metadata for a specific symbol.
        
        Parameters:
            symbol (str): The symbol to get metadata for
            key (str): The metadata attribute to retrieve
            default (Any): Value to return if the key or symbol doesn't exist
            
        Returns:
            Any: The stored metadata value or the default value
        """
        if symbol not in self.symbol_metadata:
            return default
            
        return self.symbol_metadata[symbol].get(key, default)
    
    def __len__(self) -> int:
        """Return the number of symbols in the universe."""
        return len(self.symbols)
    
    def __contains__(self, symbol: str) -> bool:
        """Check if a symbol is in the universe."""
        return symbol in self.symbols
    
    def __iter__(self):
        """Iterate over symbols in the universe."""
        return iter(self.symbols)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the universe to a dictionary representation for serialization.
        
        Creates a complete representation of the universe including all symbols,
        metadata, filters, and properties suitable for serialization or storage.
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - name: Universe name
                - description: Universe description
                - symbols: List of all symbols in the universe
                - num_filters: Number of registered filters
                - metadata: Universe-level metadata
                - symbol_metadata: Dictionary of symbol-specific metadata
                - created_at: Universe creation timestamp if available
                
        Notes:
            - Filter functions are not serialized as they cannot be properly represented
            - Only the count of filters is included in the dictionary
            - For full serialization including filters, consider using pickle or custom serialization
        """
        return {
            "name": self.name,
            "description": self.description,
            "symbols": list(self.symbols),
            "num_filters": len(self.filters),
            "metadata": self.metadata,
            "symbol_metadata": self.symbol_metadata,
            "created_at": self.metadata.get("created_at", datetime.now().isoformat())
        }
    
    @classmethod
    def from_list(cls, name: str, symbols: List[str], description: str = "") -> 'Universe':
        """
        Create a universe from a list of symbols.
        
        Factory method to conveniently create a new universe from a list of symbols
        without additional metadata or filters.
        
        Parameters:
            name (str): Name for the universe
            symbols (List[str]): List of symbols to include
            description (str): Description of the universe
            
        Returns:
            Universe: New Universe instance containing the provided symbols
            
        Notes:
            - Convenience method for simple universe creation
            - Removes duplicate symbols automatically
            - No additional metadata or filters are applied
        """
        return cls(name=name, symbols=symbols, description=description)
    
    @classmethod
    def from_file(cls, filepath: str, name: Optional[str] = None, 
                 description: str = "", delimiter: str = ",") -> 'Universe':
        """
        Create a universe from a file containing symbols.
        
        Factory method to create a new universe by reading symbols from a text file.
        The file can be a simple list with one symbol per line or a CSV with additional
        metadata columns.
        
        Parameters:
            filepath (str): Path to file with symbols (one per line or CSV)
            name (Optional[str]): Name for the universe (defaults to filename if None)
            description (str): Description of the universe
            delimiter (str): Delimiter for parsing CSV files with metadata columns
            
        Returns:
            Universe: New Universe instance containing the symbols from the file
            
        Notes:
            - For simple files: one symbol per line, ignores blank lines and whitespace
            - For CSV files: first column must be the symbol, other columns can contain metadata
            - If the first line contains headers, they will be used as metadata keys
            - Automatically derives the universe name from the filename if not provided
            - Raises FileNotFoundError if the file doesn't exist
        """
        # Get name from filename if not provided
        if name is None:
            import os
            name = os.path.splitext(os.path.basename(filepath))[0]
        
        universe = cls(name=name, description=description)
        
        # Read symbols from file
        try:
            with open(filepath, 'r') as f:
                # Check if this appears to be a CSV with headers
                first_line = f.readline().strip()
                has_headers = ',' in first_line and not first_line.split(',')[0].isalpha()
                
                # If CSV with headers, parse as CSV with metadata
                if delimiter in first_line and has_headers:
                    import csv
                    f.seek(0)  # Reset to file beginning
                    reader = csv.DictReader(f, delimiter=delimiter)
                    symbol_field = reader.fieldnames[0] if reader.fieldnames else 'symbol'
                    
                    symbols = []
                    metadata = {}
                    for row in reader:
                        symbol = row.pop(symbol_field, '').strip()
                        if symbol:
                            symbols.append(symbol)
                            metadata[symbol] = {k: v for k, v in row.items() if v.strip()}
                    
                    universe.add_symbols(symbols, metadata)
                else:
                    # Simple file with one symbol per line
                    f.seek(0)  # Reset to file beginning
                    symbols = [line.strip() for line in f if line.strip()]
                    universe.add_symbols(symbols)
        
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error reading universe file {filepath}: {str(e)}")
            raise
            
        logger.info(f"Created universe '{name}' from file with {len(universe.symbols)} symbols")
        return universe 