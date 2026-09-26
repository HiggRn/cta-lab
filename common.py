import abc
from dataclasses import dataclass

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
    is_month_end: bool = False  # per symbol
    is_calendar_month_end: bool = False  # across panel


class Indicator(abc.ABC):
    @abc.abstractmethod
    def on_bar(self, bar: Bar) -> None: ...

    @property
    @abc.abstractmethod
    def value(self) -> float | None: ...


class Strategy(abc.ABC):
    @abc.abstractmethod
    def on_bar(self, bars: dict[str, Bar]) -> dict[str, float] | None:
        """
        bars: {symbol: Bar} bars of current day
        return target weights {symbol: weight}, or None (no change in weight)
        """
        ...
