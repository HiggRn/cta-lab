from collections import deque

from common import Bar, Indicator


class Momentum(Indicator):
    """Return over the last `lookback` bars of one symbol."""

    def __init__(self, lookback: int):
        self._closes: deque[float] = deque(maxlen=lookback + 1)

    @property
    def is_ready(self) -> bool:
        return len(self._closes) == self._closes.maxlen

    @property
    def value(self) -> float:
        return self._closes[-1] / self._closes[0] - 1.0

    def update(self, bar: Bar) -> None:
        self._closes.append(bar.close)
