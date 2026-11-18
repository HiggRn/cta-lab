"""`TrendTStat`: volatility-adjusted trend strength."""

import math
from collections import deque

import numpy as np

from common import Bar, Indicator


class TrendTStat(Indicator):
    """Log return over the last `lookback` bars divided by the volatility of those bars.

    Same thing as the t-statistic of the average daily return, so it says how clear the
    trend is and not only which way it points.
    """

    def __init__(self, lookback: int = 252):
        self._logs: deque[float] = deque(maxlen=lookback + 1)

    @property
    def is_ready(self) -> bool:
        """True once `lookback` + 1 closes have been seen."""
        return len(self._logs) == self._logs.maxlen

    @property
    def value(self) -> float:
        """Trend t-statistic over the window; 0 when volatility is zero."""
        daily = np.diff(self._logs)
        vol = daily.std(ddof=1)
        return float(daily.sum() / (vol * math.sqrt(len(daily)))) if vol > 0.0 else 0.0

    def update(self, bar: Bar) -> None:
        """Append the log of the bar's close (non-positive closes are ignored)."""
        if bar.close > 0.0:
            self._logs.append(math.log(bar.close))
