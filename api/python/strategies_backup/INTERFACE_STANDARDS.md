# Strategy Interface Standards

This document outlines the standardized interface that all trading strategies should follow to ensure consistency and interoperability within the trading bot system.

## Base Classes

All strategies should inherit from one of the following base classes:

1. **Asset-specific base strategies**:
   - `StockBaseStrategy` for equity strategies
   - `OptionsBaseStrategy` for options strategies
   - `CryptoBaseStrategy` for cryptocurrency strategies
   - `ForexBaseStrategy` for forex strategies

2. **If no asset-specific base exists**, use:
   - `StrategyOptimizable` for strategies that support parameter optimization
   - `StrategyTemplate` for basic strategies

## Required Methods

### For All Strategies

Every strategy must implement these methods:

```python
def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
    """
    Generate trading signals based on the provided data.
    
    Args:
        data: Dictionary mapping symbols to DataFrames with market data
        
    Returns:
        Dictionary mapping symbols to Signal objects
    """
```

### For Optimizable Strategies

Optimizable strategies must additionally implement:

```python
def get_parameter_space(self) -> Dict[str, List[Any]]:
    """
    Define the parameter space for optimization.
    
    Returns:
        Dictionary mapping parameter names to lists of possible values
    """
    
def _calculate_performance_score(self, signals: Dict[str, Signal], 
                               data: Dict[str, Any]) -> float:
    """
    Calculate a performance score for the generated signals.
    
    Args:
        signals: Generated signals
        data: Input data
        
    Returns:
        Performance score (higher is better)
    """
```

### Recommended Methods

These methods are recommended for all strategies:

```python
def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate technical indicators for all symbols.
    
    Args:
        data: Dictionary mapping symbols to DataFrames with market data
        
    Returns:
        Dictionary of calculated indicators for each symbol
    """
    
def regime_compatibility(self, regime: MarketRegime) -> float:
    """
    Get compatibility score for this strategy in the given market regime.
    
    Args:
        regime: Market regime
        
    Returns:
        Compatibility score (0-1, higher is better)
    """
```

## Parameter Handling

All strategies should:

1. Define default parameters as a class constant:
   ```python
   DEFAULT_PARAMS = {
       "param1": value1,
       "param2": value2,
       ...
   }
   ```

2. Merge parameters in the constructor:
   ```python
   def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
       # Start with default parameters
       strategy_params = self.DEFAULT_PARAMS.copy()
       
       # Override with provided parameters
       if parameters:
           strategy_params.update(parameters)
       
       # Initialize parent class
       super().__init__(name=name, parameters=strategy_params, metadata=metadata)
   ```

3. Access parameters using `self.parameters` dictionary:
   ```python
   value = self.parameters["param_name"]
   ```

## Return Types

1. **Signal Generation**:
   - Always return `Dict[str, Signal]` where key is symbol and value is a Signal object
   - Use the Signal class from `strategy_template.py` to ensure consistency

2. **Indicator Calculation**:
   - Return `Dict[str, Dict[str, pd.DataFrame]]` with structure:
     ```
     {
         "symbol1": {
             "indicator1": pd.DataFrame({"indicator1": series}),
             "indicator2": pd.DataFrame({"indicator2": series}),
             ...
         },
         "symbol2": {
             ...
         }
     }
     ```

## Input Data Format

All strategies should accept input data in the following format:

```python
{
    "symbol1": pd.DataFrame({
        "open": [...],
        "high": [...],
        "low": [...],
        "close": [...],
        "volume": [...],
        ...
    }),
    "symbol2": pd.DataFrame({...}),
    ...
}
```

## Error Handling

1. Implement proper error handling in all methods
2. Use logging instead of print statements
3. Catch and log exceptions, avoid letting them propagate to the caller when possible
4. Return empty dictionaries or default values upon error

## Performance Considerations

1. Avoid recalculating the same indicators multiple times
2. Use vectorized operations when possible
3. Implement filtering early to avoid unnecessary calculations
4. Consider caching results for frequently accessed calculations

## Documentation

1. All strategy classes should have a detailed docstring explaining:
   - Purpose of the strategy
   - Main parameters
   - Market conditions where the strategy works best
   - Any special considerations

2. All methods should be documented with:
   - Clear purpose description
   - Parameter descriptions
   - Return value descriptions
   - Any side effects

By following these standards, we ensure that all strategies in the system are consistent, maintainable, and interchangeable. 