from __future__ import annotations

import numpy as np
import pandas as pd

from common import HistoryContext, Strategy
from panel import Panel


class BacktestResult:
    def __init__(
        self,
        weights: pd.DataFrame,
        positions: pd.DataFrame,
        gross_pnl: pd.Series,
        costs: pd.Series,
        net_pnl: pd.Series,
    ):
        # weights: target weights as decided by the strategy, carried forward
        # positions: weights actually held into each day's return as prices drift
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

    returns = panel.returns.reindex(columns=panel.symbols).fillna(0.0).to_numpy()
    n_days, n_assets = returns.shape
    col = {s: j for j, s in enumerate(panel.symbols)}

    context = HistoryContext(
        returns_arr=returns,
        dates=panel.all_dates,
        symbols=panel.symbols,
    )

    held = np.zeros(n_assets)
    orders: dict[int, np.ndarray] = {}  # execution bar -> target weights
    positions = np.zeros((n_days, n_assets))
    gross = np.zeros(n_days)
    costs = np.zeros(n_days)
    weights_history = {}
    last_w = None

    for idx, (t, bars) in enumerate(panel.timeline):
        # trade into the target decided exec_lag bars ago
        if idx in orders:
            target = orders.pop(idx)
            costs[idx] = np.abs(target - held).sum() * (cost_bps / 1e4)
            held = target

        # earn today's return, then let the weights drift with prices
        positions[idx] = held
        gross[idx] = held @ returns[idx]
        net = gross[idx] - costs[idx]
        held = held * (1.0 + returns[idx]) / (1.0 + net)

        context.update_idx(idx)
        w = strategy.on_bar(bars, context)
        if w is not None:
            last_w = w
            if idx + exec_lag < n_days:
                target = np.zeros(n_assets)
                for s, x in w.items():
                    target[col[s]] = x
                orders[idx + exec_lag] = target
        weights_history[t] = last_w if last_w is not None else {}

    weights_df = (
        pd.DataFrame(weights_history)
        .T.reindex(index=panel.all_dates, columns=panel.symbols)
        .fillna(0.0)
    )
    positions_df = pd.DataFrame(positions, index=panel.all_dates, columns=panel.symbols)
    gross_pnl = pd.Series(gross, index=panel.all_dates)
    cost_s = pd.Series(costs, index=panel.all_dates)

    return BacktestResult(
        weights_df, positions_df, gross_pnl, cost_s, gross_pnl - cost_s
    )
