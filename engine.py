"""Backtest engine: cost models and `run_backtest`, which runs a `Strategy` over a `Panel`."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import Bar, HistoryContext, Strategy
from panel import Panel

# Transaction Cost Models


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Trading, roll and financing cost of a period, as fractions of NAV."""

    trading: float = 0.0
    roll: float = 0.0
    financing: float = 0.0

    @property
    def total(self) -> float:
        """Sum of trading, roll and financing costs."""
        return self.trading + self.roll + self.financing


class CostModel(abc.ABC):
    """Interface for transaction and holding cost calculation."""

    @abc.abstractmethod
    def trading_cost(
        self,
        traded_weights: np.ndarray,
        bars: dict[str, Bar],
        symbols: list[str],
    ) -> float:
        """Cost incurred when executing weight changes |w_target - w_held|."""

    @abc.abstractmethod
    def holding_costs(self, held_weights: np.ndarray) -> tuple[float, float]:
        """Daily holding costs: returns (roll_cost, financing_cost)."""


class ZeroCostModel(CostModel):
    """No transaction costs."""

    def trading_cost(
        self, traded_weights: np.ndarray, bars: dict[str, Bar], symbols: list[str]
    ) -> float:
        """Always 0."""
        return 0.0

    def holding_costs(self, held_weights: np.ndarray) -> tuple[float, float]:
        """Always (0, 0)."""
        return 0.0, 0.0


class FuturesCostModel(CostModel):
    """
    1. Trading & Market Impact Cost: bps per unit of turnover.
    2. Roll Cost: bps per roll * rolls per year, accrued daily on gross notional |w|.
    3. Leverage Financing Cost: bps per year on gross leverage exceeding
       the embedded margin-free threshold.
    """

    def __init__(
        self,
        trading_cost_bps: float = 1.5,
        roll_cost_bps: float = 1.5,
        rolls_per_year: float = 4.0,
        financing_cost_bps: float = 50.0,
        free_leverage_threshold: float = 1.0,
        periods_per_year: int = 252,
    ):
        self.trading_rate = trading_cost_bps * 1e-4
        self.daily_roll_rate = (
            roll_cost_bps * 1e-4 * rolls_per_year
        ) / periods_per_year
        self.daily_financing_rate = (financing_cost_bps * 1e-4) / periods_per_year
        self.free_leverage_threshold = free_leverage_threshold

    def trading_cost(
        self,
        traded_weights: np.ndarray,
        bars: dict[str, Bar],
        symbols: list[str],
    ) -> float:
        """Turnover (sum of |traded weights|) times the trading rate."""
        turnover = float(np.abs(traded_weights).sum())
        return turnover * self.trading_rate

    def holding_costs(self, held_weights: np.ndarray) -> tuple[float, float]:
        """Daily (roll, financing): roll on gross leverage, financing on gross leverage above the free threshold."""
        gross_leverage = float(np.abs(held_weights).sum())
        roll_cost = gross_leverage * self.daily_roll_rate
        excess_leverage = max(0.0, gross_leverage - self.free_leverage_threshold)
        financing_cost = excess_leverage * self.daily_financing_rate
        return roll_cost, financing_cost


# Backtest Engine


class BacktestResult:
    """Output of `run_backtest`: weights, positions, P&L, costs, turnover and NAV, all indexed by date.

    `weights` are the targets the strategy chose; `positions` are the weights actually held into each day's return."""

    def __init__(
        self,
        weights: pd.DataFrame,
        positions: pd.DataFrame,
        gross_pnl: pd.Series,
        trading_costs: pd.Series,
        roll_costs: pd.Series,
        financing_costs: pd.Series,
        turnover: pd.Series,
    ):
        # weights: target weights decided by the strategy, carried forward
        # positions: actual weights held into each day's return (after execution & drift)
        self.weights = weights
        self.positions = positions
        self.gross_pnl = gross_pnl
        self.trading_costs = trading_costs
        self.roll_costs = roll_costs
        self.financing_costs = financing_costs
        # executed |traded weights| per day (excludes weight drift)
        self.turnover = turnover
        self.costs = trading_costs + roll_costs + financing_costs
        self.net_pnl = gross_pnl - self.costs
        self.nav = (1.0 + self.net_pnl.fillna(0.0)).cumprod()


def run_backtest(
    panel: Panel,
    strategy: Strategy,
    cost_model: CostModel | None = None,
    cost_bps: float | None = None,
    exec_lag: int = 1,
) -> BacktestResult:
    """Run `strategy` over `panel` one day at a time and return a `BacktestResult`.

    Orders decided at day t's close execute `exec_lag` bars later (must be at least 1, so no look-ahead),
    on the first day each contract has a bar. Positions in contracts past their final bar are closed.

    Args:
        panel: Aligned price data.
        strategy: Strategy called once per day with that day's bars.
        cost_model: Cost model. Defaults to `FuturesCostModel` with `cost_bps` trading cost, else zero cost.
        cost_bps: Shortcut for trading cost only when `cost_model` is not given.
        exec_lag: Bars between decision and execution."""
    if exec_lag < 1:
        raise ValueError("exec_lag must be >= 1 to prevent lookahead bias")

    if cost_model is None:
        if cost_bps is not None and cost_bps > 0.0:
            cost_model = FuturesCostModel(
                trading_cost_bps=cost_bps,
                roll_cost_bps=0.0,
                financing_cost_bps=0.0,
            )
        else:
            cost_model = ZeroCostModel()

    returns = panel.returns.reindex(columns=panel.symbols).fillna(0.0).to_numpy()
    n_days, n_assets = returns.shape
    col = {s: j for j, s in enumerate(panel.symbols)}

    observed_ret = panel.observed_returns.reindex(columns=panel.symbols).to_numpy()
    observed_close = panel.close_df.reindex(columns=panel.symbols).to_numpy()

    context = HistoryContext(
        returns_arr=observed_ret,
        close_arr=observed_close,
        dates=panel.all_dates,
        symbols=panel.symbols,
        first_valid_idx=panel.first_valid_idx,
        last_valid_idx=panel.last_valid_idx,
    )

    held = np.zeros(n_assets)
    scheduled_orders: dict[
        int, np.ndarray
    ] = {}  # execution idx -> full target weight vector
    pending_targets: dict[
        int, float
    ] = {}  # asset col idx -> target weight waiting for tradable bar

    positions = np.zeros((n_days, n_assets))
    gross = np.zeros(n_days)
    t_costs = np.zeros(n_days)
    r_costs = np.zeros(n_days)
    f_costs = np.zeros(n_days)
    turnover = np.zeros(n_days)
    weights_mat = np.zeros((n_days, n_assets))
    last_w_vec = np.zeros(n_assets)

    for idx, (t, bars) in enumerate(panel.timeline):
        # 1. Promote any order scheduled for today into pending_targets
        if idx in scheduled_orders:
            target_vec = scheduled_orders.pop(idx)
            for j in range(n_assets):
                pending_targets[j] = float(target_vec[j])

        traded_diff = np.zeros(n_assets)

        # A contract past its final bar can never trade again. Close the position at the last price
        # (its return is zero from here on) as a normal trade, so turnover and cost are booked,
        # and drop its pending order. This runs after the promotion above so that no stale order survives.
        for j in np.flatnonzero(panel.last_valid_idx < idx):
            traded_diff[j] = -held[j]
            held[j] = 0.0
            pending_targets.pop(int(j), None)

        # Execute pending targets ONLY for symbols that are tradable today (present in `bars`)
        if pending_targets and bars:
            executed_cols = []
            for j, target_val in pending_targets.items():
                sym = panel.symbols[j]
                # Execute if symbol trades today, or if we are already holding 0 and target is 0
                if sym in bars or (held[j] == 0.0 and target_val == 0.0):
                    traded_diff[j] = target_val - held[j]
                    held[j] = target_val
                    executed_cols.append(j)

            for j in executed_cols:
                del pending_targets[j]

        if np.any(traded_diff != 0.0):
            turnover[idx] = float(np.abs(traded_diff).sum())
            t_costs[idx] = cost_model.trading_cost(traded_diff, bars, panel.symbols)

        # 2. Settle today's return, holding costs (roll + financing), and weight drift
        positions[idx] = held
        gross[idx] = float(held @ returns[idx])
        r_costs[idx], f_costs[idx] = cost_model.holding_costs(held)

        total_cost_today = t_costs[idx] + r_costs[idx] + f_costs[idx]
        net = gross[idx] - total_cost_today
        if (1.0 + net) > 1e-8:
            held = held * (1.0 + returns[idx]) / (1.0 + net)

        # 3. Reveal today's close data to context
        context._advance_to(idx)

        # 4. Run strategy logic at today's close; schedule resulting orders for idx + exec_lag
        w = strategy.on_bar(bars, context)
        if w is not None:
            # a strategy may still carry weights for contracts whose final bar has passed (e.g. a cached sleeve)
            alive = panel.last_valid_idx >= idx
            target = np.zeros(n_assets)
            for s, x in w.items():
                if s in col and pd.notna(x) and alive[col[s]]:
                    target[col[s]] = float(x)
            last_w_vec = target
            if idx + exec_lag < n_days:
                scheduled_orders[idx + exec_lag] = target

        weights_mat[idx] = last_w_vec

    weights_df = pd.DataFrame(weights_mat, index=panel.all_dates, columns=panel.symbols)
    positions_df = pd.DataFrame(positions, index=panel.all_dates, columns=panel.symbols)
    gross_pnl = pd.Series(gross, index=panel.all_dates)
    trading_cost_s = pd.Series(t_costs, index=panel.all_dates)
    roll_cost_s = pd.Series(r_costs, index=panel.all_dates)
    financing_cost_s = pd.Series(f_costs, index=panel.all_dates)
    turnover_s = pd.Series(turnover, index=panel.all_dates)

    return BacktestResult(
        weights=weights_df,
        positions=positions_df,
        gross_pnl=gross_pnl,
        trading_costs=trading_cost_s,
        roll_costs=roll_cost_s,
        financing_costs=financing_cost_s,
        turnover=turnover_s,
    )
