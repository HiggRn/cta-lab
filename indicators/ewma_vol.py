import math

from common import Bar, Indicator


class EwmaVolatility(Indicator):
    """Exponentially weighted daily volatility of log returns (not annualised).

    Same weights and small-sample correction as pandas `ewm(span).std()`.
    """

    def __init__(self, span: int = 60, min_periods: int = 20):
        self.decay = 1.0 - 2.0 / (span + 1.0)
        self.min_periods = max(min_periods, 2)
        self._prev_close: float | None = None
        self._n = 0
        self._w = 0.0  # sum of weights
        self._w2 = 0.0  # sum of squared weights
        self._sx = 0.0  # weighted sum of returns
        self._sxx = 0.0  # weighted sum of squared returns
        self._vol = 0.0

    @property
    def is_ready(self) -> bool:
        return self._n >= self.min_periods and self._vol > 0.0

    @property
    def value(self) -> float:
        return self._vol

    def update(self, bar: Bar) -> None:
        c = bar.close
        if math.isnan(c) or c <= 0.0:
            return
        if self._prev_close is not None:
            x = math.log(c / self._prev_close)
            d = self.decay
            self._w = d * self._w + 1.0
            self._w2 = d * d * self._w2 + 1.0
            self._sx = d * self._sx + x
            self._sxx = d * self._sxx + x * x
            self._n += 1
            if self._n >= 2:
                mean = self._sx / self._w
                var = self._sxx / self._w - mean * mean
                var *= self._w**2 / (self._w**2 - self._w2)
                self._vol = math.sqrt(max(var, 0.0))
        self._prev_close = c
