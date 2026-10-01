from __future__ import annotations

import pandas as pd

from common import HistoryContext, Strategy
from panel import Panel


def turnover_cost(positions: pd.DataFrame, cost_bps: float) -> pd.Series:
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    return turnover * (cost_bps / 1e4)


class BacktestResult:
    def __init__(
        self,
        weights: pd.DataFrame,
        positions: pd.DataFrame,
        gross_pnl: pd.Series,
        costs: pd.Series,
        net_pnl: pd.Series,
    ):
        self.weights = weights
        self.positions = positions
        self.gross_pnl = gross_pnl
        self.costs = costs
        self.net_pnl = net_pnl
        self.nav = (1.0 + net_pnl.fillna(0.0)).cumprod()


def run_backtest(
    panel: Panel,
    strategy: Strategy,
    cost_bps: float = 0.0,
    exec_lag: int = 1,
) -> BacktestResult:
    if exec_lag < 1:
        raise ValueError("exec_lag must be >= 1")

    aligned_returns = (
        panel.returns.reindex(columns=panel.symbols).fillna(0.0).to_numpy()
    )

    context = HistoryContext(
        returns_arr=aligned_returns,
        dates=panel.all_dates,
        symbols=panel.symbols,
    )

    weights_history = {}
    last_w = None

    for idx, (t, bars) in enumerate(panel.timeline):
        context.update_idx(idx)

        w = strategy.on_bar(bars, context)
        last_w = w if w is not None else last_w
        weights_history[t] = last_w if last_w is not None else {}

    weights_df = (
        pd.DataFrame(weights_history)
        .T.reindex(index=panel.all_dates, columns=panel.symbols)
        .fillna(0.0)
    )
    positions = weights_df.shift(exec_lag).fillna(0.0)
    gross_pnl = (positions * aligned_returns).sum(axis=1)
    costs = turnover_cost(positions, cost_bps)

    return BacktestResult(weights_df, positions, gross_pnl, costs, gross_pnl - costs)
