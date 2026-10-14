from common import Bar, Indicator


class ATR(Indicator):
    """Exponentially weighted Average True Range."""

    def __init__(self, span: int = 32):
        self._alpha = 2.0 / (span + 1.0)
        self._min_bars = span // 2
        self._count = 0
        self._prev_close: float | None = None
        self._value = 0.0

    @property
    def is_ready(self) -> bool:
        return self._count >= self._min_bars and self._value > 0.0

    @property
    def value(self) -> float:
        return self._value

    def update(self, bar: Bar) -> None:
        c = bar.close
        if self._prev_close is None:
            self._prev_close = c
            return

        h = bar.high if bar.high > 0.0 else c
        l = bar.low if bar.low > 0.0 else c
        tr = max(h, self._prev_close) - min(l, self._prev_close)

        self._prev_close = c
        self._count += 1
        self._value = (
            tr
            if self._count == 1
            else (1.0 - self._alpha) * self._value + self._alpha * tr
        )
