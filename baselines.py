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
    VolTargetPCM,
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
        signs = [np.sign(sum(completed_returns[-k:])) for k in self.lookbacks]
        return float(np.mean(signs))


class StaticLongAlpha(MonthlyTsmomAlpha):
    """Static long-only benchmark: enters each contract once the same history as TSMOM exists."""

    def direction(self, completed_returns: list[float]) -> float:
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
        return (
            _completed_monthly_returns(returns).cov(min_periods=self.min_months) * 12.0
        )


class EwmaMonthlyRiskModel(RiskModel):
    """Exponentially weighted covariance of completed monthly returns."""

    def __init__(self, halflife: float = 12.0, min_months: int = 36):
        self.halflife = halflife
        self.min_months = min_months

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        monthly = _completed_monthly_returns(returns)
        has_data = monthly.notna().to_numpy(dtype=float)
        x = (
            (monthly - monthly.ewm(halflife=self.halflife).mean().iloc[-1])
            .fillna(0.0)
            .to_numpy()
        )
        age = np.arange(len(monthly))[::-1]
        weight = (0.5 ** (age / self.halflife))[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = (weight * x).T @ x / ((weight * has_data).T @ has_data)
        enough = has_data.sum(axis=0) >= self.min_months
        cov[~enough, :] = np.nan
        cov[:, ~enough] = np.nan
        return pd.DataFrame(cov * 12.0, index=monthly.columns, columns=monthly.columns)


class EwmaDailyRiskModel(RiskModel):
    """Moskowitz, Ooi and Pedersen (2012): diagonal EWMA of daily returns (center of mass = 60d)."""

    def __init__(self, com: float = 60.0, min_days: int = 60, days_per_year: int = 261):
        self.com = com
        self.min_days = min_days
        self.days_per_year = days_per_year

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        var = (
            returns.ewm(com=self.com, min_periods=self.min_days).var().iloc[-1]
            * self.days_per_year
        )
        return pd.DataFrame(
            np.diag(var), index=returns.columns, columns=returns.columns
        )


# Baseline & Benchmark Strategies


class BaselineTsmomStrategy(ModularStrategy):
    """Course Baseline: 12m monthly TSMOM, expanding monthly vol, 40% equal risk, 1/N."""

    def __init__(self, lookbacks: tuple[int, ...] = (12,)):
        super().__init__(
            alpha_model=MonthlyTsmomAlpha(lookbacks=lookbacks),
            risk_model=ExpandingMonthlyRiskModel(),
            pcm=EqualRiskPCM(position_vol=0.40),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


class StaticBenchmarkStrategy(ModularStrategy):
    """Static Long-Only Benchmark: identical risk sizing & universe entry as Baseline TSMOM."""

    def __init__(self):
        super().__init__(
            alpha_model=StaticLongAlpha(lookbacks=(12,)),
            risk_model=ExpandingMonthlyRiskModel(),
            pcm=EqualRiskPCM(position_vol=0.40),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


class VolTargetTsmomStrategy(ModularStrategy):
    """12m TSMOM scaled to 11% ex-ante portfolio volatility using expanding monthly covariance."""

    def __init__(self, target_vol: float = 0.11):
        super().__init__(
            alpha_model=MonthlyTsmomAlpha((12,)),
            risk_model=ExpandingMonthlyRiskModel(),
            pcm=VolTargetPCM(EqualRiskPCM(0.40), target_vol=target_vol),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


class EwmaVolTargetTsmomStrategy(ModularStrategy):
    """12m TSMOM scaled to 11% ex-ante portfolio volatility using EWMA monthly covariance."""

    def __init__(self, target_vol: float = 0.11):
        super().__init__(
            alpha_model=MonthlyTsmomAlpha((12,)),
            risk_model=EwmaMonthlyRiskModel(),
            pcm=VolTargetPCM(EqualRiskPCM(0.40), target_vol=target_vol),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


class BlendTsmomStrategy(ModularStrategy):
    """Blended 1/3/6/12m monthly TSMOM with expanding monthly volatility."""

    def __init__(self):
        super().__init__(
            alpha_model=MonthlyTsmomAlpha((1, 3, 6, 12)),
            risk_model=ExpandingMonthlyRiskModel(),
            pcm=EqualRiskPCM(0.40),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


class BlendEwmaVolTargetStrategy(ModularStrategy):
    """Blended 1/3/6/12m TSMOM with EWMA monthly covariance and 11% portfolio vol target."""

    def __init__(self, target_vol: float = 0.11):
        super().__init__(
            alpha_model=MonthlyTsmomAlpha((1, 3, 6, 12)),
            risk_model=EwmaMonthlyRiskModel(),
            pcm=VolTargetPCM(EqualRiskPCM(0.40), target_vol=target_vol),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


class MopDailySizingTsmomStrategy(ModularStrategy):
    """Monthly 12m TSMOM signal sized using MOP (2012) 60-day EWMA daily volatility."""

    def __init__(self):
        super().__init__(
            alpha_model=MonthlyTsmomAlpha((12,)),
            risk_model=EwmaDailyRiskModel(),
            pcm=EqualRiskPCM(0.40),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


BASELINE_STRATEGIES = {
    "TSMOM 12m (baseline)": BaselineTsmomStrategy,
    "Static benchmark": StaticBenchmarkStrategy,
    "Vol target 11%": VolTargetTsmomStrategy,
    "Vol target 11%, EWMA risk": EwmaVolTargetTsmomStrategy,
    "Blend 1/3/6/12m": BlendTsmomStrategy,
    "Blend + EWMA vol target": BlendEwmaVolTargetStrategy,
    "Daily EWMA sizing (MOP 2012)": MopDailySizingTsmomStrategy,
}
