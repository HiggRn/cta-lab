from collections import deque

import pandas as pd

from common import Bar, Indicator


class MonthlyReturn(Indicator):
    def __init__(self, lookback: int = 1):
        self.lookback = lookback
        self._last_month_end_close: float | None = None
        self.history: deque[float] = deque(maxlen=lookback)
        self.latest_return: float | None = None

    def on_bar(self, bar: Bar) -> None:
        if not bar.is_month_end:
            return None

        price = bar.close
        if pd.isna(price) or price <= 0:
            return None

        self.latest_return = None
        if self._last_month_end_close is not None and self._last_month_end_close > 0:
            ret = (price / self._last_month_end_close) - 1.0
            self.history.append(ret)
            self.latest_return = ret

        self._last_month_end_close = price

    @property
    def value(self) -> float | None:
        return self.latest_return
