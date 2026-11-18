"""`MonthlyReturn`: returns of completed calendar months, from month-end closes."""

from collections import deque

from common import Bar, Indicator


class MonthlyReturn(Indicator):
    """Close-to-close return of each calendar month, keeping the last `lookback` completed months.

    A month's return is measured from the previous month's last close to its own last close.
    `just_closed_month` is True on the bar that starts a new month."""

    def __init__(self, lookback: int = 12):
        self.lookback = lookback
        self.completed_returns: deque[float] = deque(maxlen=lookback)
        self._current_ym: tuple[int, int] | None = None
        self._prev_month_close: float | None = None
        self._current_close: float | None = None
        self._current_month_ret: float | None = None
        self.just_closed_month: bool = False

    @property
    def returns(self) -> list[float]:
        """Completed monthly returns plus the month in progress, at most `lookback` of them."""
        if self._current_month_ret is None:
            return list(self.completed_returns)
        combined = list(self.completed_returns) + [self._current_month_ret]
        return combined[-self.lookback :]

    @property
    def is_ready(self) -> bool:
        """True once `lookback` months have completed."""
        return len(self.completed_returns) == self.lookback

    @property
    def value(self) -> float:
        """Sum of the completed monthly returns."""
        return float(sum(self.completed_returns))

    def update(self, bar: Bar) -> None:
        """Feed one bar, closing the previous month if the month changed."""
        ym = (bar.date.year, bar.date.month)
        self.just_closed_month = False

        if self._current_ym is None:
            self._current_ym = ym
            self._current_close = bar.close
            return

        if ym != self._current_ym:
            if self._current_month_ret is not None:
                self.completed_returns.append(self._current_month_ret)
                self.just_closed_month = True
            self._prev_month_close = self._current_close
            self._current_ym = ym

        self._current_close = bar.close
        if self._prev_month_close is not None and self._prev_month_close > 0:
            self._current_month_ret = bar.close / self._prev_month_close - 1.0
