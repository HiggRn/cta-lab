from collections import deque

import pandas as pd

from common import Bar, Indicator


class MonthlyReturn(Indicator):
    """Tracks calendar-month returns in real time without any lookahead flags.

    - Within the same calendar month, each new bar updates the current month's return in-place.
    - When a bar from a new calendar month arrives, the previous month's final close is locked
      as the new baseline and the previous month's return is permanently sealed in `completed_returns`.
    """

    def __init__(self, lookback: int = 12):
        self.lookback = lookback
        self.completed_returns: deque[float] = deque(maxlen=lookback)
        self._current_month: pd.Period | None = None
        self._prev_month_close: float | None = None
        self._current_close: float | None = None
        self._current_month_ret: float | None = None

    @property
    def returns(self) -> list[float]:
        """Up to `lookback` monthly returns, including the in-progress current month at the end."""
        if self._current_month_ret is None:
            return list(self.completed_returns)
        combined = list(self.completed_returns) + [self._current_month_ret]
        return combined[-self.lookback :]

    @property
    def is_ready(self) -> bool:
        """True once we have `lookback` completed months (or `lookback` total months)."""
        return len(self.completed_returns) == self.lookback

    @property
    def value(self) -> float:
        """Sum of the last `lookback` completed monthly returns."""
        return float(sum(self.completed_returns))

    def update(self, bar: Bar) -> None:
        bar_month = bar.date.to_period("M")

        if self._current_month is None:
            self._current_month = bar_month
            self._current_close = bar.close
            return

        if bar_month != self._current_month:
            # New calendar month started: seal the previous month's return
            if self._current_month_ret is not None:
                self.completed_returns.append(self._current_month_ret)
            self._prev_month_close = self._current_close
            self._current_month = bar_month

        self._current_close = bar.close
        if self._prev_month_close is not None and self._prev_month_close > 0:
            self._current_month_ret = bar.close / self._prev_month_close - 1.0
