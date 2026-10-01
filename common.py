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
    is_month_end: bool = False
    is_calendar_month_end: bool = False


class HistoryContext:
    __slots__ = ("_returns_arr", "_dates", "_symbols", "_current_idx")

    def __init__(
        self,
        returns_arr: np.ndarray,
        dates: pd.DatetimeIndex,
        symbols: list[str],
    ):
        self._returns_arr = returns_arr
        self._dates = dates
        self._symbols = symbols
        self._current_idx = 0

    def update_idx(self, idx: int) -> None:
        self._current_idx = idx

    def get_returns(self, lookback: int) -> pd.DataFrame:
        """Last `lookback` rows up to today. NaN where a symbol had no bar."""
        if lookback <= 0:
            raise ValueError(f"lookback {lookback} is not a positive integer")

        end_idx = self._current_idx + 1
        start_idx = max(0, end_idx - lookback)

        sub_arr = self._returns_arr[start_idx:end_idx]
        sub_dates = self._dates[start_idx:end_idx]
        return pd.DataFrame(sub_arr, index=sub_dates, columns=self._symbols, copy=True)


class AlphaModel(abc.ABC):
    @abc.abstractmethod
    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        """New signals only. Absent symbols keep their last signal; 0.0 means flat."""


class RiskModel(abc.ABC):
    @abc.abstractmethod
    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Annualized covariance, indexed and columned by symbol."""


class PortfolioConstructionModel(abc.ABC):
    @abc.abstractmethod
    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        """Target weights. Symbols that are absent are treated as flat."""


class Strategy(abc.ABC):
    @abc.abstractmethod
    def on_bar(
        self, bars: dict[str, Bar], context: HistoryContext
    ) -> dict[str, float] | None:
        """Return new target weights, or None to keep the previous ones."""
