import math
from collections import deque

import numpy as np

from common import Bar, Indicator


class YangZhangVolatility(Indicator):
    """Rolling annualized Yang-Zhang OHLC volatility estimator."""

    def __init__(
        self,
        lookback: int = 60,
        min_periods: int = 20,
        periods_per_year: int = 252,
    ):
        self.lookback = lookback
        self.min_periods = max(min_periods, 2)
        self.periods_per_year = periods_per_year
        self._prev_close: float | None = None
        self._o: deque[float] = deque(maxlen=lookback)
        self._c: deque[float] = deque(maxlen=lookback)
        self._rs: deque[float] = deque(maxlen=lookback)
        self._vol: float = 0.0

    @property
    def is_ready(self) -> bool:
        return len(self._o) >= self.min_periods and self._vol > 0.0

    @property
    def value(self) -> float:
        return self._vol

    def update(self, bar: Bar) -> None:
        c = bar.close
        if math.isnan(c) or c <= 0.0:
            return

        o = bar.open if (not math.isnan(bar.open) and bar.open > 0.0) else c
        h = bar.high if (not math.isnan(bar.high) and bar.high > 0.0) else max(o, c)
        l = bar.low if (not math.isnan(bar.low) and bar.low > 0.0) else min(o, c)
        h = max(h, o, c)
        l = min(l, o, c)

        if self._prev_close is not None and self._prev_close > 0.0:
            self._o.append(math.log(o / self._prev_close))
            self._c.append(math.log(c / o))
            self._rs.append(
                math.log(h / c) * math.log(h / o) + math.log(l / c) * math.log(l / o)
            )

        self._prev_close = c
        n = len(self._o)
        if n < self.min_periods:
            return

        o_arr = np.fromiter(self._o, dtype=float, count=n)
        c_arr = np.fromiter(self._c, dtype=float, count=n)
        rs_arr = np.fromiter(self._rs, dtype=float, count=n)

        var_o = float(np.var(o_arr, ddof=1))
        var_c = float(np.var(c_arr, ddof=1))
        var_rs = float(np.mean(rs_arr))

        k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
        yz_var = max(var_o + k * var_c + (1.0 - k) * var_rs, 0.0)
        self._vol = math.sqrt(yz_var * self.periods_per_year)
