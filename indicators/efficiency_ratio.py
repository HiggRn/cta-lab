from collections import deque

from common import Bar, Indicator


class EfficiencyRatio(Indicator):
    """Kaufman Efficiency Ratio using True Range in the denominator: |C_t - C_{t-N}| / sum(TR)."""

    def __init__(self, lookback: int = 20):
        self._closes: deque[float] = deque(maxlen=lookback + 1)
        self._trs: deque[float] = deque(maxlen=lookback)

    @property
    def is_ready(self) -> bool:
        return len(self._closes) == self._closes.maxlen

    @property
    def value(self) -> float:
        total_tr = sum(self._trs)
        if total_tr <= 1e-12:
            return 0.0
        return min(1.0, abs(self._closes[-1] - self._closes[0]) / total_tr)

    def update(self, bar: Bar) -> None:
        c = bar.close
        if self._closes:
            prev_c = self._closes[-1]
            h = bar.high if bar.high > 0.0 else c
            l = bar.low if bar.low > 0.0 else c
            self._trs.append(max(h, prev_c) - min(l, prev_c))
        self._closes.append(c)
