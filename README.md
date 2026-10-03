# Allocation Lab

How should a fixed monthly contribution be deployed across IDX stocks, the S&P 500, gold and crypto, and
when should cash waiting in a money-market fund move into equities? This project builds the
data pipeline, a backtester with realistic costs, and a dashboard to answer that, with
out-of-sample validation.

Status: Phase 0 (setup and decisions). See [plan.md](plan.md) for the phases and
[docs/decisions.md](docs/decisions.md) for the choices the code depends on.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
pip install -e ".[dev]"
pytest
ruff check .
```

## Disclaimer

Research project. Nothing here is financial advice, and backtest results do not predict
future returns.
