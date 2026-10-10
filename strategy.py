import numpy as np
import pandas as pd

from common import (
    AlphaModel,
    Bar,
    HistoryContext,
    PortfolioConstructionModel,
    RiskModel,
    Strategy,
)
from indicators.momentum import Momentum
from indicators.monthly_returns import MonthlyReturn

# pass as risk_lookback when a risk model needs every day of history (get_returns clips at day 0)
ALL_HISTORY = 1_000_000


class ModularStrategy(Strategy):
    def __init__(
        self,
        alpha_model: AlphaModel,
        risk_model: RiskModel,
        pcm: PortfolioConstructionModel,
        risk_lookback: int = 60,
        rebalance: str = "monthly",
    ):
        if rebalance not in ("daily", "monthly"):
            raise ValueError(f"Unsupported rebalance: {rebalance}")
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.pcm = pcm
        self.risk_lookback = risk_lookback
        self.rebalance = rebalance
        self._signals: dict[str, float] = {}

    def on_bar(
        self, bars: dict[str, Bar], context: HistoryContext
    ) -> dict[str, float] | None:
        if not bars:
            return None

        # alpha runs on every bar; symbols it does not return keep their last signal
        self._signals.update(self.alpha_model.update(bars, context))

        if not self._is_rebalance_bar(bars):
            return None

        cov = self.risk_model.covariance(context.get_returns(self.risk_lookback))
        return self.pcm.create_targets(self._signals, cov)

    def _is_rebalance_bar(self, bars: dict[str, Bar]) -> bool:
        if self.rebalance == "daily":
            return True
        return any(b.is_calendar_month_end for b in bars.values())


class TsmomAlpha(AlphaModel):
    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self._momentum: dict[str, Momentum] = {}

    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        signals = {}
        for symbol, bar in bars.items():
            if symbol not in self._momentum:
                self._momentum[symbol] = Momentum(self.lookback)
            ind = self._momentum[symbol]
            ind.update(bar)
            if ind.is_ready:
                signals[symbol] = float(np.sign(ind.value))
        return signals


class SampleCovarianceRiskModel(RiskModel):
    def __init__(self, periods_per_year: int = 252, min_periods: int = 20):
        self.periods_per_year = periods_per_year
        self.min_periods = min_periods

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        # pairwise-complete: NaN days are skipped, thin history gives NaN
        return returns.cov(min_periods=self.min_periods) * self.periods_per_year


class InverseVolPCM(PortfolioConstructionModel):
    def __init__(self, target_gross_leverage: float = 1.0):
        self.target_gross_leverage = target_gross_leverage

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        vol = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)

        raw = {}
        for symbol, signal in signals.items():
            if signal == 0.0 or symbol not in vol.index:
                continue
            if pd.notna(vol[symbol]) and vol[symbol] > 1e-4:
                raw[symbol] = signal / vol[symbol]

        gross = sum(abs(w) for w in raw.values())
        if gross == 0.0:
            return {}

        scale = self.target_gross_leverage / gross
        return {s: w * scale for s, w in raw.items()}


# ---- monthly TSMOM models ------------------------------------------------------------


class MonthlyTsmomAlpha(AlphaModel):
    """Sign of the summed past monthly returns, updated on each symbol's own month end.

    With several lookbacks the signal is the average of their signs,
    e.g. (1, 3, 6, 12) gives -1, -0.5, 0, 0.5 or 1.
    """

    def __init__(self, lookbacks: tuple[int, ...] = (12,)):
        self.lookbacks = lookbacks
        self._monthly: dict[str, MonthlyReturn] = {}

    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        signals = {}
        for symbol, bar in bars.items():
            if symbol not in self._monthly:
                self._monthly[symbol] = MonthlyReturn(max(self.lookbacks))
            monthly = self._monthly[symbol]
            monthly.update(bar)
            if bar.is_month_end and monthly.is_ready:
                signals[symbol] = self.direction(list(monthly.returns))
        return signals

    def direction(self, monthly_returns: list[float]) -> float:
        signs = [np.sign(sum(monthly_returns[-k:])) for k in self.lookbacks]
        return float(np.mean(signs))


class StaticLongAlpha(MonthlyTsmomAlpha):
    """Static benchmark: always long, once the same history as TSMOM exists."""

    def direction(self, monthly_returns: list[float]) -> float:
        return 1.0


def monthly_from_daily(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound daily returns into calendar-month returns (NaN if a symbol had no bar that month)."""
    months = daily_returns.index.to_period("M")
    return (1 + daily_returns).groupby(months).prod(min_count=1) - 1


class ExpandingMonthlyRiskModel(RiskModel):
    """Covariance of all past monthly returns (expanding window). Needs risk_lookback=ALL_HISTORY."""

    def __init__(self, min_months: int = 36):
        self.min_months = min_months

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        return monthly_from_daily(returns).cov(min_periods=self.min_months) * 12


class EwmaMonthlyRiskModel(RiskModel):
    """Exponentially weighted covariance of monthly returns. Reacts faster than the expanding window."""

    def __init__(self, halflife: float = 12, min_months: int = 36):
        self.halflife = halflife
        self.min_months = min_months

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        monthly = monthly_from_daily(returns)
        has_data = monthly.notna().to_numpy(dtype=float)
        x = (monthly - monthly.ewm(halflife=self.halflife).mean().iloc[-1]).fillna(0).to_numpy()
        age = np.arange(len(monthly))[::-1]
        weight = (0.5 ** (age / self.halflife))[:, None]
        # pairwise: each pair only uses the months where both symbols have data (0/0 -> NaN if never)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = (weight * x).T @ x / ((weight * has_data).T @ has_data)
        enough = has_data.sum(axis=0) >= self.min_months
        cov[~enough, :] = np.nan
        cov[:, ~enough] = np.nan
        return pd.DataFrame(cov * 12, index=monthly.columns, columns=monthly.columns)


class EwmaDailyRiskModel(RiskModel):
    """Moskowitz, Ooi and Pedersen (2012): EWMA of daily returns, centre of mass 60 days.

    Diagonal only (no correlations), so use it for position sizing rather than vol targeting.
    """

    def __init__(self, com: float = 60, min_days: int = 60, days_per_year: int = 261):
        self.com = com
        self.min_days = min_days
        self.days_per_year = days_per_year

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        var = returns.ewm(com=self.com, min_periods=self.min_days).var().iloc[-1] * self.days_per_year
        return pd.DataFrame(np.diag(var), index=returns.columns, columns=returns.columns)


class EqualRiskPCM(PortfolioConstructionModel):
    """Each position targets `position_vol`, then 1/N across the live symbols.

    Stays flat until at least `min_assets` symbols have both a signal and a volatility.
    """

    def __init__(self, position_vol: float = 0.40, min_assets: int = 10):
        self.position_vol = position_vol
        self.min_assets = min_assets

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        vol = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)
        live = {s: x for s, x in signals.items() if s in vol.index and vol[s] > 1e-4}
        if len(live) < self.min_assets:
            return {}
        n = len(live)
        return {s: x * self.position_vol / vol[s] / n for s, x in live.items()}


class VolTargetPCM(PortfolioConstructionModel):
    """Scales another PCM's weights so the ex-ante portfolio vol sqrt(w' cov w) equals `target_vol`."""

    def __init__(self, pcm: PortfolioConstructionModel, target_vol: float = 0.11):
        self.pcm = pcm
        self.target_vol = target_vol

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        targets = self.pcm.create_targets(signals, cov)
        if not targets:
            return targets
        vol = exante_vol(targets, cov)
        if not vol > 0:
            return {}
        return {s: w * self.target_vol / vol for s, w in targets.items()}


def exante_vol(weights: dict[str, float], cov: pd.DataFrame) -> float:
    """Annualised sqrt(w' cov w) for an annualised covariance."""
    w = pd.Series(weights)
    q = cov.reindex(index=w.index, columns=w.index).fillna(0)
    return float(np.sqrt(max(w @ q @ w, 0.0)))
