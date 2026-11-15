"""End-to-end checks on the full-sample course data. Skipped when data/full_raw is absent."""

import re

import numpy as np
import pandas as pd
import pytest

from analysis import monthly_returns
from baselines import BASELINE_STRATEGIES
from config import ASSET_MAP_PATH, DROPPED_SYMBOLS, SAMPLE_END, SAMPLE_START, SUCCESSORS
from engine import FuturesCostModel, ZeroCostModel, run_backtest
from panel import Panel
from strategy import USER_STRATEGIES

STRATEGIES = {**BASELINE_STRATEGIES, **USER_STRATEGIES}
BASELINE = "TSMOM 12m (baseline)"
STATIC = "Static benchmark"


def sharpe(r: pd.Series) -> float:
    return r.mean() / r.std() * np.sqrt(12)


def test_baseline_results_without_costs(real_panel):
    # frictionless, trading at the signal close: the course baseline is about
    # Sharpe 1.0, the static benchmark about 0.5, and they are barely correlated
    run = lambda name: monthly_returns(
        run_backtest(
            real_panel, STRATEGIES[name](), cost_model=ZeroCostModel(), exec_lag=1
        )
    ).loc[SAMPLE_START:SAMPLE_END]
    tsmom, static = run(BASELINE), run(STATIC)
    assert sharpe(tsmom) == pytest.approx(1.01, abs=0.10)
    assert sharpe(static) == pytest.approx(0.47, abs=0.10)
    assert tsmom.corr(static) == pytest.approx(0.10, abs=0.10)


def test_costs_only_reduce_returns(real_panel):
    free = run_backtest(real_panel, STRATEGIES[BASELINE](), cost_model=ZeroCostModel())
    paid = run_backtest(
        real_panel, STRATEGIES[BASELINE](), cost_model=FuturesCostModel()
    )
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


def test_later_exec_lag_does_not_help_a_daily_strategy(real_panel):
    name = "Daily 252d TSMOM, inverse vol"
    first = run_backtest(real_panel, STRATEGIES[name](), exec_lag=1)
    second = run_backtest(real_panel, STRATEGIES[name](), exec_lag=2)
    assert sharpe(monthly_returns(second)) <= sharpe(monthly_returns(first)) + 0.05


@pytest.mark.parametrize("name", ["Vol target 11%", *USER_STRATEGIES])
def test_no_lookahead(real_panel, full_raw_dir, tmp_path, read_rad, write_rad, name):
    # cut every file mid-month and rebuild the panel on the shorter data
    cutoff = pd.Timestamp("1999-12-15")
    raw = tmp_path / "raw"
    raw.mkdir()
    kept = set()
    for file in full_raw_dir.iterdir():
        prices = read_rad(file).loc[:cutoff]
        if len(prices):  # a contract with no data before the cut does not exist yet
            symbol = re.sub("_RAD$", "", file.stem, flags=re.I)
            write_rad(raw, symbol, prices)
            kept.add(symbol)
    asset_map = pd.read_csv(ASSET_MAP_PATH, index_col="ID", keep_default_na=False)
    asset_map.loc[asset_map.index.isin(kept)].to_csv(tmp_path / "AssetMapCsv.csv")
    successors = {b: m for b, m in SUCCESSORS.items() if b in kept and m in kept}
    short_panel = Panel(raw, tmp_path / "AssetMapCsv.csv", DROPPED_SYMBOLS, successors)

    model = FuturesCostModel()
    full = run_backtest(real_panel, STRATEGIES[name](), cost_model=model)
    short = run_backtest(short_panel, STRATEGIES[name](), cost_model=model)

    # A contract whose last bar before the cut precedes the cut looks delisted in the short panel but
    # not in the full one, so only compare up to the earliest such last bar
    last_bar = short_panel.close_df.apply(pd.Series.last_valid_index)
    after_cut = real_panel.close_df.loc[cutoff + pd.Timedelta(days=1) :]
    safe = min(last_bar[s] for s in short_panel.symbols if after_cut[s].notna().any())
    assert len(short.weights.loc[:safe]) > 250  # still compares something

    later_symbols = full.weights.columns.difference(
        short.weights.columns
    )  # no data before the cut
    assert (full.weights.loc[:safe, later_symbols] == 0).all().all()
    pd.testing.assert_frame_equal(
        full.weights.loc[:safe, short.weights.columns],
        short.weights.loc[:safe],
        check_freq=False,
    )
    pd.testing.assert_series_equal(
        full.net_pnl.loc[:safe], short.net_pnl.loc[:safe], check_freq=False
    )


def test_trend_profile_earns_nothing_once_returns_are_scrambled(
    full_raw_dir, tmp_path, read_rad, write_rad
):
    # flip the sign of each day's returns at random, the same flip for every contract:
    # vol and correlations survive, anything predictable does not
    days = pd.date_range("1960-01-01", "2050-12-31")
    flip = pd.Series(
        np.random.default_rng(0).choice([-1.0, 1.0], len(days)), index=days
    )
    for file in full_raw_dir.iterdir():
        prices = read_rad(file)
        prices = prices[
            (prices[["open", "high", "low", "close"]] != 0).any(axis=1)
        ]  # placeholder rows
        if prices.empty:
            continue
        close = prices["close"]
        ret = close.pct_change().fillna(0.0) * flip.reindex(close.index)
        scrambled = close.iloc[0] * (1.0 + ret).cumprod()
        for column in ["open", "high", "low", "close"]:
            prices[column] = scrambled
        write_rad(tmp_path, re.sub("_RAD$", "", file.stem, flags=re.I), prices)
    scrambled_panel = Panel(tmp_path, ASSET_MAP_PATH, DROPPED_SYMBOLS, SUCCESSORS)

    res = run_backtest(scrambled_panel, STRATEGIES["Trend profile"](), exec_lag=2)
    gross = monthly_returns(res, gross=True).loc[SAMPLE_START:SAMPLE_END]
    assert abs(sharpe(gross)) < 0.5  # 1.85 on the real data
