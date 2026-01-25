# InvestLLM Notebooks

This directory contains Jupyter notebooks for research and analysis.

## Structure

```
notebooks/
├── exploration/       # Data exploration
├── training/          # Model training experiments
├── backtesting/       # Strategy backtesting
└── analysis/          # Result analysis
```

## Usage

Start Jupyter Lab:
```bash
docker-compose up jupyter
# Access at http://localhost:8888
# Token: investllm
```

Or locally:
```bash
jupyter lab --notebook-dir=notebooks
```
