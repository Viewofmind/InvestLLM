# InvestLLM Tests

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=investllm

# Run specific test file
pytest tests/test_collectors.py

# Run with verbose output
pytest -v
```

## Test Structure

```
tests/
├── test_collectors.py    # Data collector tests
├── test_models.py        # ML model tests
├── test_features.py      # Feature engineering tests
├── test_backtesting.py   # Backtesting tests
└── conftest.py           # Pytest fixtures
```
