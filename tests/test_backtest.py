"""End-to-end checks on the course data. Skipped when data/raw is absent."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import monthly_returns
from baselines import BASELINE_STRATEGIES
from engine import FuturesCostModel, ZeroCostModel, run_backtest
from panel import Panel
from strategy import USER_STRATEGIES

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ASSET_MAP_PATH = ROOT / "data" / "processed" / "AssetMapCsv.csv"
# keep in sync with main.py
DROPPED_SYMBOLS = {"EN", "ES", "SC", "BC", "BG", "ZH", "FF", "RL", "YM"}
SAMPLE_START = "1976-02"

STRATEGIES = {**BASELINE_STRATEGIES, **USER_STRATEGIES}
BASELINE = "TSMOM 12m (baseline)"
STATIC = "Static benchmark"

pytestmark = pytest.mark.skipif(
    not RAW_DIR.exists(), reason="course data not in data/raw"
)


@pytest.fixture(scope="module")
def panel():
    return Panel(RAW_DIR, ASSET_MAP_PATH, DROPPED_SYMBOLS)


def sharpe(r: pd.Series) -> float:
    return r.mean() / r.std() * np.sqrt(12)


def test_baseline_results_without_costs(panel):
    # frictionless, trading at the signal close: the course baseline is about
    # Sharpe 1.0, the static benchmark about 0.5, and they are barely correlated
    run = lambda name: monthly_returns(
        run_backtest(panel, STRATEGIES[name](), cost_model=ZeroCostModel(), exec_lag=1)
    ).loc[SAMPLE_START:]
    tsmom, static = run(BASELINE), run(STATIC)
    assert sharpe(tsmom) == pytest.approx(1.01, abs=0.10)
    assert sharpe(static) == pytest.approx(0.47, abs=0.10)
    assert tsmom.corr(static) == pytest.approx(0.10, abs=0.10)


def test_costs_only_reduce_returns(panel):
    free = run_backtest(panel, STRATEGIES[BASELINE](), cost_model=ZeroCostModel())
    paid = run_backtest(panel, STRATEGIES[BASELINE](), cost_model=FuturesCostModel())
    pd.testing.assert_frame_equal(
        free.weights, paid.weights
    )  # decisions don't depend on costs
    assert (
        free.gross_pnl.corr(paid.gross_pnl) > 0.999
    )  # gross differs only through weight drift
    assert (paid.costs >= 0).all() and paid.costs.sum() > 0
    assert paid.net_pnl.sum() < free.net_pnl.sum()
    assert paid.turnover.sum() > 0
    assert (paid.turnover >= 0).all()


def test_later_exec_lag_does_not_help_a_daily_strategy(panel):
    name = "Daily 252d TSMOM, inverse vol"
    first = run_backtest(panel, STRATEGIES[name](), exec_lag=1)
    second = run_backtest(panel, STRATEGIES[name](), exec_lag=2)
    assert sharpe(monthly_returns(second)) <= sharpe(monthly_returns(first)) + 0.05


@pytest.mark.parametrize("name", ["Vol target 11%", "Daily 252d TSMOM, inverse vol"])
def test_no_lookahead(panel, tmp_path, name):
    # cut every file at a month end: nothing decided up to that day may change
    cutoff = pd.Timestamp("1999-12-31")
    for file in RAW_DIR.glob("*.csv"):
        prices = pd.read_csv(file, index_col=0, parse_dates=True).loc[:cutoff]
        if len(
            prices
        ):  # Panel cannot read an empty file (its column becomes object dtype)
            prices.to_csv(tmp_path / file.name)
    short_panel = Panel(tmp_path, ASSET_MAP_PATH, DROPPED_SYMBOLS)

    model = FuturesCostModel()
    full = run_backtest(panel, STRATEGIES[name](), cost_model=model)
    short = run_backtest(short_panel, STRATEGIES[name](), cost_model=model)

    later_symbols = full.weights.columns.difference(
        short.weights.columns
    )  # no data before the cutoff
    assert (full.weights.loc[:cutoff, later_symbols] == 0).all().all()
    pd.testing.assert_frame_equal(
        full.weights.loc[:cutoff, short.weights.columns],
        short.weights,
        check_freq=False,
    )
    pd.testing.assert_series_equal(
        full.net_pnl.loc[:cutoff], short.net_pnl, check_freq=False
    )
