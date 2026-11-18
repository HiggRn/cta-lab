"""Core interfaces and shared building blocks for the backtest.

Defines `Bar`, the `Indicator`, `AlphaModel`, `RiskModel`, `PortfolioConstructionModel`,
`RebalanceSchedule` and `Strategy` interfaces, the `ModularStrategy` that composes them, the
engine-fed `HistoryContext`, and common risk models and portfolio-construction models.
"""

import abc
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Bar:
    """One day's open, high, low, close, volume and open interest for a single contract."""

    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float


class Indicator(abc.ABC):
    """Streaming statistic fed one `Bar` at a time through `update`."""

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Whether enough bars have been seen for `value` to be meaningful."""

    @property
    @abc.abstractmethod
    def value(self) -> float:
        """Only meaningful once is_ready is True."""

    @abc.abstractmethod
    def update(self, bar: Bar) -> None:
        """Feed one bar. Called only on days the symbol traded."""


class HistoryContext:
    """Read-only view of market history up to the current day, handed to strategies by the engine.

    Only data up to and including today is exposed, so strategy code cannot look ahead."""

    __slots__ = (
        "_returns_arr",
        "_close_arr",
        "_dates",
        "_symbols",
        "_first_idx",
        "_last_idx",
        "_current_idx",
    )

    def __init__(
        self,
        returns_arr: np.ndarray,
        close_arr: np.ndarray,
        dates: pd.DatetimeIndex,
        symbols: list[str],
        first_valid_idx: np.ndarray,
        last_valid_idx: np.ndarray,
    ):
        self._returns_arr = returns_arr
        self._close_arr = close_arr
        self._dates = dates
        self._symbols = symbols
        self._first_idx = first_valid_idx
        self._last_idx = last_valid_idx
        self._current_idx = 0

    def _advance_to(self, idx: int) -> None:
        """Internal engine method. Never call from user strategy code."""
        self._current_idx = idx

    @property
    def current_date(self) -> pd.Timestamp:
        """Date of the bar currently being processed."""
        return self._dates[self._current_idx]

    @property
    def prev_date(self) -> pd.Timestamp | None:
        """Date of the previous bar, or None on the first day."""
        if self._current_idx == 0:
            return None
        return self._dates[self._current_idx - 1]

    @property
    def active_universe(self) -> set[str]:
        """Symbols that have started trading and whose final bar is not in the past.

        The dataset's last bar is taken as the delisting date, so a symbol drops out the day after
        its last bar. This is reference data from the future, but a contract is never dropped before
        its last bar: truncating the data at any date leaves everything before the cut unchanged.
        """
        idx = self._current_idx
        active_mask = (self._first_idx <= idx) & (idx <= self._last_idx)
        return {
            self._symbols[i] for i, is_active in enumerate(active_mask) if is_active
        }

    def get_returns(self, lookback: int | None = None) -> pd.DataFrame:
        """Observed daily returns up to today (inclusive). NaN on non-trading days."""
        end_idx = self._current_idx + 1
        if lookback is None:
            start_idx = 0
        else:
            if lookback <= 0:
                raise ValueError(
                    f"lookback {lookback} must be a positive integer or None"
                )
            start_idx = max(0, end_idx - lookback)

        sub_arr = self._returns_arr[start_idx:end_idx]
        sub_dates = self._dates[start_idx:end_idx]
        return pd.DataFrame(sub_arr, index=sub_dates, columns=self._symbols, copy=True)

    def get_closes(self, lookback: int | None = None) -> pd.DataFrame:
        """Observed daily close prices up to today (inclusive). NaN on non-trading days."""
        end_idx = self._current_idx + 1
        if lookback is None:
            start_idx = 0
        else:
            if lookback <= 0:
                raise ValueError(
                    f"lookback {lookback} must be a positive integer or None"
                )
            start_idx = max(0, end_idx - lookback)

        sub_arr = self._close_arr[start_idx:end_idx]
        sub_dates = self._dates[start_idx:end_idx]
        return pd.DataFrame(sub_arr, index=sub_dates, columns=self._symbols, copy=True)


# Rebalance Schedules


class RebalanceSchedule(abc.ABC):
    """Determines whether the strategy should rebalance on the current bar."""

    @abc.abstractmethod
    def should_rebalance(
        self, bars: dict[str, Bar], context: HistoryContext
    ) -> bool:
        """Return True if the strategy should rebalance on the current bar."""


class DailySchedule(RebalanceSchedule):
    """Rebalance on every trading day."""

    def should_rebalance(self, bars: dict[str, Bar], context: HistoryContext) -> bool:
        """True on every day with at least one bar."""
        return bool(bars)


class WeeklySchedule(RebalanceSchedule):
    """Rebalance on the first trading day of each new ISO calendar week."""

    def should_rebalance(self, bars: dict[str, Bar], context: HistoryContext) -> bool:
        """True on the first bar of each new ISO week (never on the first day)."""
        if not bars or context.prev_date is None:
            return False
        t, prev = context.current_date, context.prev_date
        return (t.isocalendar().year, t.isocalendar().week) != (
            prev.isocalendar().year,
            prev.isocalendar().week,
        )


class FirstAllTradableDayMonthlySchedule(RebalanceSchedule):
    """Rebalance once a month, on the first day every active symbol has a bar.

    If that does not happen within `max_wait_days` trading days of the new month, rebalance anyway."""

    def __init__(self, max_wait_days: int = 5):
        self.max_wait_days = max_wait_days
        self._pending_ym: tuple[int, int] | None = None
        self._last_rebalanced_ym: tuple[int, int] | None = None
        self._wait_days: int = 0

    def should_rebalance(self, bars: dict[str, Bar], context: HistoryContext) -> bool:
        """True once per calendar month, when all active symbols trade or `max_wait_days` have passed."""
        if not bars:
            return False

        t = context.current_date
        cur_ym = (t.year, t.month)

        if self._last_rebalanced_ym is None and context.prev_date is None:
            self._last_rebalanced_ym = cur_ym
            return False

        if cur_ym != self._last_rebalanced_ym and self._pending_ym != cur_ym:
            self._pending_ym = cur_ym
            self._wait_days = 0

        if self._pending_ym == cur_ym:
            self._wait_days += 1
            active = context.active_universe
            if (
                bool(active) and active.issubset(bars.keys())
            ) or self._wait_days >= self.max_wait_days:
                self._last_rebalanced_ym = cur_ym
                self._pending_ym = None
                return True

        return False


# Core Strategy Interfaces


class AlphaModel(abc.ABC):
    """Turns new bars into directional signals per symbol (positive long, negative short, 0 flat)."""

    @abc.abstractmethod
    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        """Return updated signals. Absent symbols keep their previous signal; 0.0 means flat."""


class RiskModel(abc.ABC):
    """Estimates the covariance matrix of returns."""

    @abc.abstractmethod
    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Annualized covariance matrix, indexed and columned by symbol."""


class PortfolioConstructionModel(abc.ABC):
    """Turns signals and a covariance matrix into target portfolio weights."""

    @abc.abstractmethod
    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        """Return target weights dict. Absent symbols are treated as 0.0 (flat)."""


class Strategy(abc.ABC):
    """A trading strategy: given each day's bars, decides when to rebalance and to what weights."""

    @abc.abstractmethod
    def on_bar(
        self, bars: dict[str, Bar], context: HistoryContext
    ) -> dict[str, float] | None:
        """Called at the close of each trading day.
        Return a dict of new target weights to trigger a rebalance, or None to hold current positions.
        """


class ModularStrategy(Strategy):
    """Composable strategy skeleton combining AlphaModel, RiskModel, PCM, and RebalanceSchedule."""

    def __init__(
        self,
        alpha_model: AlphaModel,
        risk_model: RiskModel,
        pcm: PortfolioConstructionModel,
        schedule: RebalanceSchedule | None = None,
        risk_lookback: int | None = None,
    ):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.pcm = pcm
        self.schedule = schedule or FirstAllTradableDayMonthlySchedule()
        self.risk_lookback = risk_lookback
        self._signals: dict[str, float] = {}

    def on_bar(
        self, bars: dict[str, Bar], context: HistoryContext
    ) -> dict[str, float] | None:
        """Update signals, drop those of delisted contracts, and on a scheduled day return the PCM's targets."""
        if not bars:
            return None

        self._signals.update(self.alpha_model.update(bars, context))
        # Drop signals of contracts whose final bar has passed before the PCM sizes the book,
        # otherwise their risk still counts in vol targeting and scales the live contracts down
        active = context.active_universe
        self._signals = {
            symbol: signal
            for symbol, signal in self._signals.items()
            if symbol in active
        }
        if not self.schedule.should_rebalance(bars, context):
            return None

        cov = self.risk_model.covariance(context.get_returns(self.risk_lookback))
        return self.pcm.create_targets(self._signals, cov)


# Common Models


def exante_vol(weights: dict[str, float], cov: pd.DataFrame) -> float:
    """Annualised ex-ante portfolio volatility sqrt(w' cov w).

    >>> import pandas as pd
    >>> cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
    >>> round(exante_vol({"A": 1.0, "B": 1.0}, cov), 4)
    0.3606
    """
    w = pd.Series(weights)
    q = cov.reindex(index=w.index, columns=w.index).fillna(0.0)
    return float(np.sqrt(max(w @ q @ w, 0.0)))


class SampleCovarianceRiskModel(RiskModel):
    """Standard rolling/expanding daily sample covariance matrix."""

    def __init__(self, periods_per_year: int = 252, min_periods: int = 20):
        self.periods_per_year = periods_per_year
        self.min_periods = min_periods

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Annualised sample covariance of `returns`."""
        return returns.cov(min_periods=self.min_periods) * self.periods_per_year


class EqualRiskPCM(PortfolioConstructionModel):
    """Each active position targets `position_vol`, weighted 1/N across live symbols."""

    def __init__(self, position_vol: float = 0.40, min_assets: int = 10):
        self.position_vol = position_vol
        self.min_assets = min_assets

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        """Weight w = signal * position_vol / vol / N for each live symbol; empty when fewer than `min_assets` are live."""
        vol = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)
        live = {
            s: x
            for s, x in signals.items()
            if x != 0.0 and s in vol.index and pd.notna(vol[s]) and vol[s] > 1e-4
        }
        if len(live) < self.min_assets:
            return {}
        n = len(live)
        return {s: x * self.position_vol / vol[s] / n for s, x in live.items()}


class VolTargetPCM(PortfolioConstructionModel):
    """Wraps any PCM and scales weights so ex-ante portfolio vol equals `target_vol`."""

    def __init__(self, base_pcm: PortfolioConstructionModel, target_vol: float = 0.11):
        self.base_pcm = base_pcm
        self.target_vol = target_vol

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        """Scale the base PCM's weights so the ex-ante portfolio volatility equals `target_vol`."""
        targets = self.base_pcm.create_targets(signals, cov)
        if not targets:
            return targets
        vol = exante_vol(targets, cov)
        if not vol > 0.0:
            return {}
        return {s: w * self.target_vol / vol for s, w in targets.items()}


class InverseVolPCM(PortfolioConstructionModel):
    """Sizes positions inversely proportional to volatility, scaled to `target_gross_leverage`."""

    def __init__(self, target_gross_leverage: float = 1.0):
        self.target_gross_leverage = target_gross_leverage

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        """Weights proportional to signal / vol, scaled so gross leverage equals `target_gross_leverage`."""
        vol = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)
        raw = {
            s: sig / vol[s]
            for s, sig in signals.items()
            if sig != 0.0 and s in vol.index and pd.notna(vol[s]) and vol[s] > 1e-4
        }
        gross = sum(abs(w) for w in raw.values())
        if gross == 0.0:
            return {}
        scale = self.target_gross_leverage / gross
        return {s: w * scale for s, w in raw.items()}
