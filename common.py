import abc
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Bar:
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float


class Indicator(abc.ABC):
    @property
    @abc.abstractmethod
    def is_ready(self) -> bool: ...

    @property
    @abc.abstractmethod
    def value(self) -> float:
        """Only meaningful once is_ready is True."""

    @abc.abstractmethod
    def update(self, bar: Bar) -> None:
        """Feed one bar. Called only on days the symbol traded."""


class HistoryContext:
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
        return self._dates[self._current_idx]

    @property
    def prev_date(self) -> pd.Timestamp | None:
        if self._current_idx == 0:
            return None
        return self._dates[self._current_idx - 1]

    @property
    def active_universe(self) -> set[str]:
        """Symbols that have started trading on or before today and have not permanently delisted."""
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
    ) -> bool: ...


class DailySchedule(RebalanceSchedule):
    """Rebalance on every trading day."""

    def should_rebalance(self, bars: dict[str, Bar], context: HistoryContext) -> bool:
        return bool(bars)


class WeeklySchedule(RebalanceSchedule):
    """Rebalance on the first trading day of each new ISO calendar week."""

    def should_rebalance(self, bars: dict[str, Bar], context: HistoryContext) -> bool:
        if not bars or context.prev_date is None:
            return False
        t, prev = context.current_date, context.prev_date
        return (t.isocalendar().year, t.isocalendar().week) != (
            prev.isocalendar().year,
            prev.isocalendar().week,
        )


class FirstAllTradableDayMonthlySchedule(RebalanceSchedule):
    """After a calendar month rolls over, wait until the first day where all active
    symbols in the universe are tradable (have a valid Bar today) to rebalance together.
    Includes a `max_wait_days` safety valve in case a contract has an abnormal data gap.
    """

    def __init__(self, max_wait_days: int = 5):
        self.max_wait_days = max_wait_days
        self._pending_month: pd.Period | None = None
        self._last_rebalanced_month: pd.Period | None = None
        self._wait_days: int = 0

    def should_rebalance(self, bars: dict[str, Bar], context: HistoryContext) -> bool:
        if not bars:
            return False

        cur_month = context.current_date.to_period("M")

        # Initialize on the very first bar so we don't immediately fire on day 0
        if self._last_rebalanced_month is None and context.prev_date is None:
            self._last_rebalanced_month = cur_month
            return False

        # Detect rollover into a new calendar month
        if (
            cur_month != self._last_rebalanced_month
            and self._pending_month != cur_month
        ):
            self._pending_month = cur_month
            self._wait_days = 0

        if self._pending_month == cur_month:
            self._wait_days += 1
            active = context.active_universe
            all_tradable = bool(active) and active.issubset(bars.keys())

            if all_tradable or self._wait_days >= self.max_wait_days:
                self._last_rebalanced_month = cur_month
                self._pending_month = None
                return True

        return False


# Core Strategy Interfaces


class AlphaModel(abc.ABC):
    @abc.abstractmethod
    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        """Return updated signals. Absent symbols keep their previous signal; 0.0 means flat."""


class RiskModel(abc.ABC):
    @abc.abstractmethod
    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Annualized covariance matrix, indexed and columned by symbol."""


class PortfolioConstructionModel(abc.ABC):
    @abc.abstractmethod
    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        """Return target weights dict. Absent symbols are treated as 0.0 (flat)."""


class Strategy(abc.ABC):
    @abc.abstractmethod
    def on_bar(
        self, bars: dict[str, Bar], context: HistoryContext
    ) -> dict[str, float] | None:
        """Called at the close of each trading day.
        Return a dict of new target weights to trigger a rebalance, or None to hold current positions.
        """
