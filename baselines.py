"""Course baseline strategies: monthly time-series momentum (TSMOM) and a static long-only benchmark."""

import numpy as np
import pandas as pd

from common import (
    AlphaModel,
    Bar,
    EqualRiskPCM,
    FirstAllTradableDayMonthlySchedule,
    HistoryContext,
    ModularStrategy,
    RiskModel,
)
from indicators import MonthlyReturn

# Baseline Components


class MonthlyTsmomAlpha(AlphaModel):
    """Sign of past completed calendar-month returns (single or blended lookbacks)."""

    def __init__(self, lookbacks: tuple[int, ...] = (12,)):
        self.lookbacks = lookbacks
        self._max_lb = max(lookbacks)
        self._monthly: dict[str, MonthlyReturn] = {}

    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        """Update each traded symbol's monthly-return indicator and emit a signal on the bar that closes a month."""
        signals = {}
        for symbol, bar in bars.items():
            if symbol not in self._monthly:
                self._monthly[symbol] = MonthlyReturn(self._max_lb)
            ind = self._monthly[symbol]
            ind.update(bar)
            if ind.just_closed_month and ind.is_ready:
                signals[symbol] = self.direction(list(ind.completed_returns))
        return signals

    def direction(self, completed_returns: list[float]) -> float:
        """Average over `lookbacks` of the sign of the summed last k completed monthly returns.

    >>> MonthlyTsmomAlpha((1, 2)).direction([0.05, -0.01])
    0.0
    >>> MonthlyTsmomAlpha((1,)).direction([-0.02])
    -1.0"""
        signs = [np.sign(sum(completed_returns[-k:])) for k in self.lookbacks]
        return float(np.mean(signs))


class StaticLongAlpha(MonthlyTsmomAlpha):
    """Static long-only benchmark: enters each contract once the same history as TSMOM exists."""

    def direction(self, completed_returns: list[float]) -> float:
        """Always +1: the position is long whenever enough history exists."""
        return 1.0


def _completed_monthly_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Fast compounding of daily returns into completed calendar months using integer year*100+month."""
    if daily_returns.empty:
        return daily_returns
    ym = daily_returns.index.year * 100 + daily_returns.index.month
    # Keep only completed months strictly prior to the current bar's month
    if daily_returns.index[-1].day <= 7:
        mask = ym != ym[-1]
        daily_returns = daily_returns.loc[mask]
        ym = ym[mask]
    return (1.0 + daily_returns).groupby(ym).prod(min_count=1) - 1.0


class ExpandingMonthlyRiskModel(RiskModel):
    """Expanding-window covariance of completed monthly returns."""

    def __init__(self, min_months: int = 36):
        self.min_months = min_months

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Annualised (x12) covariance of completed monthly returns, NaN below `min_months` observations."""
        return (
            _completed_monthly_returns(returns).cov(min_periods=self.min_months) * 12.0
        )


# Baseline & Benchmark Strategies


class BaselineTsmomStrategy(ModularStrategy):
    """Monthly 12m TSMOM, expanding monthly vol, 40% equal risk, 1/N (the midterm baseline)."""

    def __init__(self, lookbacks: tuple[int, ...] = (12,)):
        super().__init__(
            alpha_model=MonthlyTsmomAlpha(lookbacks=lookbacks),
            risk_model=ExpandingMonthlyRiskModel(),
            pcm=EqualRiskPCM(position_vol=0.40),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


class StaticBenchmarkStrategy(ModularStrategy):
    """Static Long-Only Benchmark: identical risk sizing & universe entry as the monthly 12m TSMOM."""

    def __init__(self):
        super().__init__(
            alpha_model=StaticLongAlpha(lookbacks=(12,)),
            risk_model=ExpandingMonthlyRiskModel(),
            pcm=EqualRiskPCM(position_vol=0.40),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


BASELINE_STRATEGIES = {
    "Baseline TSMOM": BaselineTsmomStrategy,
    "Static benchmark": StaticBenchmarkStrategy,
}
