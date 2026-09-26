from __future__ import annotations

import pandas as pd

from common import Strategy
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

    def sliced(
        self, start: str | None = None, end: str | None = None
    ) -> BacktestResult:
        return BacktestResult(
            self.weights.loc[start:end],
            self.positions.loc[start:end],
            self.gross_pnl.loc[start:end],
            self.costs.loc[start:end],
            self.net_pnl.loc[start:end],
        )


def run_backtest(
    panel: Panel,
    strategy: Strategy,
    cost_bps: float = 0.0,
    exec_lag: int = 1,
) -> BacktestResult:
    weights_history = {}
    last_w = None

    for t, bars in panel.timeline:
        w = strategy.on_bar(bars)
        last_w = w if w is not None else last_w
        weights_history[t] = last_w if last_w is not None else {}

    weights_df = (
        pd.DataFrame(weights_history)
        .T.reindex(index=panel.all_dates, columns=panel.symbols)
        .fillna(0.0)
    )
    positions = weights_df.shift(exec_lag).fillna(0.0)
    gross_pnl = (positions * panel.returns.reindex(columns=panel.symbols)).sum(axis=1)
    costs = turnover_cost(positions, cost_bps)

    return BacktestResult(weights_df, positions, gross_pnl, costs, gross_pnl - costs)
