from collections import deque

from common import Bar, Indicator


class MonthlyReturn(Indicator):
    """The last `lookback` month-end to month-end returns of one symbol.

    Months are closed on the symbol's own last bar of the month (`Bar.is_month_end`).
    """

    def __init__(self, lookback: int = 12):
        self.returns: deque[float] = deque(maxlen=lookback)
        self._last_close: float | None = None

    @property
    def is_ready(self) -> bool:
        return len(self.returns) == self.returns.maxlen

    @property
    def value(self) -> float:
        """Sum of the stored monthly returns."""
        return sum(self.returns)

    def update(self, bar: Bar) -> None:
        if not bar.is_month_end:
            return
        if self._last_close is not None and self._last_close > 0:
            self.returns.append(bar.close / self._last_close - 1.0)
        self._last_close = bar.close
