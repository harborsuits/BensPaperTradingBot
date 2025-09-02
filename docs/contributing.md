# Contributing Guidelines

Thank you for considering contributing to the BensBot Trading System! This document outlines the process for contributing to the project and coding standards to follow.

## Ways to Contribute

There are many ways to contribute to the project:

1. **Code Contributions**: Implementing new features or fixing bugs
2. **Documentation**: Improving or adding documentation
3. **Testing**: Adding test cases or improving test coverage
4. **Bug Reports**: Reporting issues you encounter
5. **Feature Requests**: Suggesting new features or enhancements
6. **Strategy Development**: Creating new trading strategies

## Development Process

### 1. Setting Up Your Development Environment

Follow the [Development Setup](development-setup.md) guide to set up your local development environment.

### 2. Finding an Issue to Work On

- Check the [Issues](https://github.com/TheClitCommander/BensBot/issues) page for open tasks
- Look for issues labeled `good-first-issue` if you're new to the project
- Comment on an issue to express interest before starting work

### 3. Branching Strategy

We follow a feature branch workflow:

- `main` branch contains the stable, released code
- `develop` branch is the integration branch for new features
- Feature branches should be created from `develop`

Branch naming conventions:
- `feature/short-description` for new features
- `bugfix/issue-description` for bug fixes
- `docs/what-you-are-documenting` for documentation changes

### 4. Development Workflow

1. Create a new branch from `develop`:
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards

3. Add tests for your changes

4. Run the test suite to make sure everything passes:
   ```bash
   python -m pytest
   ```

5. Commit your changes with meaningful commit messages:
   ```bash
   git commit -m "feat: Add new momentum strategy implementation"
   ```

6. Push your branch to GitHub:
   ```bash
   git push -u origin feature/your-feature-name
   ```

7. Create a pull request to merge your changes into `develop`

### 5. Pull Request Process

1. Fill out the PR template with details about your changes
2. Link any relevant issues your PR addresses
3. Ensure all CI checks pass
4. Request a review from maintainers
5. Address any feedback or requested changes
6. Once approved, a maintainer will merge your PR

## Coding Standards

### Python Coding Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style
- Line length should be limited to 100 characters
- Use 4 spaces for indentation (no tabs)
- Use meaningful variable and function names

### Documentation

- Use Google-style docstrings for functions and classes:
  ```python
  def function_name(param1, param2):
      """Short description of function.
      
      Longer description explaining details.
      
      Args:
          param1: Description of param1
          param2: Description of param2
          
      Returns:
          Description of return value
          
      Raises:
          ExceptionType: When and why this exception is raised
      """
  ```

- Add comments for complex logic
- Keep documentation up to date with code changes

### Testing

- Write unit tests for all new functionality
- Aim for at least 80% test coverage for new code
- Include both happy path and error case tests
- Use fixtures and mocks appropriately

### Type Annotations

- Use type hints for function parameters and return values:
  ```python
  def calculate_position_size(symbol: str, price: float, risk_pct: float) -> int:
      """Calculate the position size based on risk percentage."""
  ```

- Use Optional, Union, etc. for complex types
- Use the typing module for container types (List, Dict, etc.)

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Changes that don't affect the code's meaning (formatting, etc.)
- `refactor`: Code changes that neither fix a bug nor add a feature
- `test`: Adding or fixing tests
- `chore`: Changes to build process, tools, etc.

Example:
```
feat(strategy): Add RSI-based entry signals to momentum strategy

Implement relative strength index (RSI) as an additional entry signal
for the momentum strategy. This allows for better detection of
overbought/oversold conditions.

Fixes #123
```

## Code Review Process

Pull requests are reviewed by maintainers based on:

1. Correctness: Does the code work as intended?
2. Code quality: Is the code well-written and maintainable?
3. Testing: Are there sufficient tests for the changes?
4. Documentation: Are the changes well-documented?
5. Performance: Will the code perform well in production?

Reviewers may request changes before approving the PR.

## Creating a New Strategy

When contributing a new trading strategy:

1. Create a new file in `trading_bot/strategies/`
2. Implement the Strategy interface
3. Register your strategy with the StrategyFactory
4. Add comprehensive unit tests
5. Include backtesting results in your PR
6. Document the strategy's approach and parameters

Example strategy implementation:

```python
from trading_bot.strategies.strategy_base import Strategy
from trading_bot.strategies.strategy_factory import StrategyFactory

class RSIStrategy(Strategy):
    """RSI-based mean reversion strategy."""
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data):
        # Calculate RSI
        # Generate buy/sell signals
        # Return signals dictionary
    
    def get_parameters(self):
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought
        }
    
    def set_parameters(self, parameters):
        self.rsi_period = parameters.get("rsi_period", self.rsi_period)
        self.oversold = parameters.get("oversold", self.oversold)
        self.overbought = parameters.get("overbought", self.overbought)

# Register the strategy
StrategyFactory.register("rsi", RSIStrategy)
```

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Questions?

If you have questions about contributing, feel free to open an issue with the `question` label.
